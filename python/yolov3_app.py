from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import matplotlib.pyplot as plt
import time

from utils import *

anchors = [(116,90), (156,198), (373,326), (30,61), (62,45), (59,119), (10,13), (16,30), (33,23)]
color_map = [(255,0,0), (0,255,0), (0,0,255)]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def read_img_scaling(img_path, size, scale):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    org_size = image.shape
    image = cv2.resize(image, (size, size))
    
    return image, org_size

def preprocess_fn(image_path, size=416, fix_scale=1.0):
    #print(f'size={size}')
    #print(f'fix_scale={fix_scale}')
    image, org_size = read_img_scaling(image_path, size, fix_scale)
    image = image.astype(np.float32)
    image /= 255.
    image *= fix_scale
    image = image.astype(np.int8)
    return image, org_size[:2]

def get_child_subgraph_dpu(graph: "Graph"):
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

def show_bbox(img_path, size, bbox_list):
    image,_ = read_img_scaling(img_path, size, 1)
    for (x0,y0,x1,y1,i_map, _, _) in bbox_list:
        cv2.rectangle(image,(int(x0), int(y0)),(int(x1), int(y1)), color=color_map[int(i_map)], thickness=1,lineType=cv2.LINE_4)

    return image

def show_bbox_nms(img_path, size, bbox_list):
    image,_ = read_img_scaling(img_path, size, 1)
    for (x0,y0,x1,y1, _, _) in bbox_list:
        cv2.rectangle(image,(int(x0), int(y0)),(int(x1), int(y1)), color=color_map[i_map], thickness=1,lineType=cv2.LINE_4)

    return image

def make_parser():
    parser = argparse.ArgumentParser("yolov3 detection parser")
    parser.add_argument("-o", "--out_dir", type=str, default="./", help="output of jpeg file with bboxes")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="batch size")
    default_model = '/usr/share/vitis_ai_library/models/yolov3_coco_416_tf2/yolov3_coco_416_tf2.xmodel'
    parser.add_argument("-m", "--model", default=default_model, type=str, help="*.xmodel")
    parser.add_argument("-i", "--img_path", type=str, default='./000000079588.jpg', help="input image name to detect objects")

    # Tuning parameters
    parser.add_argument('-c','--conf_the', type=float, default=0.5, help='confidence map threshold')
    parser.add_argument('-n','--nms_the', type=float, default=0.1, help='nms threshold')

    return parser

def main(args):
    # time measurement
    #start_time = time.time()
    ## make graph from model ##
    g = xir.Graph.deserialize(args.model)
    subgraphs = get_child_subgraph_dpu(g)

    subgraph0 = vart.Runner.create_runner(subgraphs[0], "run")
    # child subgraphs:
    #    <xir.Subgraph named 'subgraph_YOLOX__YOLOX_QuantStub_quant_in__input_1'>,
    #    <xir.Subgraph named 'subgraph_YOLOX__YOLOX_YOLOPAFPN_backbone__BaseConv_lateral_conv0__Conv2d_conv__input_277'>,
    #    <xir.Subgraph named 'subgraph_YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_2__inputs_fix_'>,
    #    <xir.Subgraph named 'subgraph_YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_1__inputs_5_fix_'>,
    #    <xir.Subgraph named 'subgraph_YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__inputs_3_fix_'> 

    # input tensor
    model0_input_tensors = subgraph0.get_input_tensors()

    model0_input_ndim = tuple(model0_input_tensors[0].dims)
    model0_input_fixpos = model0_input_tensors[0].get_attr("fix_point")
    model0_input_scale = 2**model0_input_fixpos
    #print(f'input scale={model0_input_scale}')

    # output tensor
    model0_output_tensors = subgraph0.get_output_tensors()

    model0_output0_ndim = tuple(model0_output_tensors[0].dims)
    #print(f'model0_output0_ndim={model0_output0_ndim}')
    model0_output1_ndim = tuple(model0_output_tensors[1].dims)
    #print(f'model0_output1_ndim={model0_output1_ndim}')
    model0_output2_ndim = tuple(model0_output_tensors[2].dims)
    #print(f'model0_output2_ndim={model0_output2_ndim}')

    model0_output0_fixpos = model0_output_tensors[0].get_attr("fix_point")
    model0_output0_scale = 1 / (2**model0_output0_fixpos)
    model0_output1_fixpos = model0_output_tensors[1].get_attr("fix_point")
    model0_output1_scale = 1 / (2**model0_output1_fixpos)
    model0_output2_fixpos = model0_output_tensors[2].get_attr("fix_point")
    model0_output2_scale = 1 / (2**model0_output2_fixpos)
    #print(f'output0 scale={model0_output0_scale}')
    #print(f'output1 scale={model0_output1_scale}')
    #print(f'output2 scale={model0_output2_scale}')

    # preprocessing
    start_time = time.time()
    input_size = 416
    input_img, org_size = preprocess_fn(args.img_path, size=input_size, fix_scale=model0_input_scale)
    #np.save('input_img.npy', input_img)
    pre_process_time = time.time()
    print(f'preprocess time={(pre_process_time-start_time)*1000.0:.2f}')
    # preparation
    model0_output = []
    for out_tensor in model0_output_tensors:
        model0_output.append(np.empty(tuple(out_tensor.dims), dtype=np.int8, order='C'))

    # execution
    job_id = subgraph0.execute_async([input_img], model0_output)
    #print(f'job_id={job_id}')

    subgraph0.wait(job_id)
    dpu_process_time = time.time()
    print(f'dpu time={(dpu_process_time-pre_process_time)*1000.0:.2f}')

    model0_output0, model0_output1, model0_output2 = model0_output

    # output scaling
    model0_output0 = model0_output0.astype(np.float32)
    model0_output0 = model0_output0 * model0_output0_scale
    model0_output0 = model0_output0.reshape(model0_output0_ndim)
    model0_output1 = model0_output1.astype(np.float32)
    model0_output1 = model0_output1 * model0_output1_scale
    model0_output1 = model0_output1.reshape(model0_output1_ndim)
    model0_output2 = model0_output2.astype(np.float32)
    model0_output2 = model0_output2 * model0_output2_scale
    model0_output2 = model0_output2.reshape(model0_output2_ndim)    

    #print(model0_output)a

    #np.save('out0.npy',model0_output0)
    #np.save('out1.npy',model0_output1)
    #np.save('out2.npy',model0_output2)
 
    out0 = model0_output0.squeeze().transpose(2,0,1)
    out1 = model0_output1.squeeze().transpose(2,0,1)
    out2 = model0_output2.squeeze().transpose(2,0,1)
   
    #print(f'time before postprocessing ={time.time()-start_time}')
    #print(f'time for DPU: {time.time()-start_time}')

    # make bboxes
    img_size = input_img.shape[0]
    pred = [out0, out1, out2]
    class_n = 80
    bbox_list = []
    conf_the = 0.5
    for j in range(len(pred)): # index showing output feature map resolution
        #print('#### feature map:{pred[j].shape} ####')
        grid = img_size/pred[j].shape[2]
        #print(f'grid={grid}')
        for i in range(3): # 3 kinds of anchors is defined for each feature map 
            anchs_idx=int(len(anchors)/3*j+i)
            anchor = anchors[anchs_idx]
            y_pred_cut = pred[j][i*(4 + 1 + class_n):(i+1)*(4 + 1 + class_n)]
            y_pred_conf = sigmoid(y_pred_cut[4,:,:])
            index = np.where(y_pred_conf > args.conf_the)
            #print(f'--- anchor type #{anchs_idx}:({anchor[0],anchor[1]}) ---')
            for y,x in zip(index[0],index[1]): # index shows number of anchors having conf lenvel > threshold
                cx = x*grid + sigmoid(y_pred_cut[0,y,x])*grid
                cy = y*grid + sigmoid(y_pred_cut[1,y,x])*grid
                width = anchor[0]*np.exp(y_pred_cut[2,y,x])
                height = anchor[1]*np.exp(y_pred_cut[3,y,x])
                #print(f'cx, cy = {cx,cy}')
                #print(f'width, height = {width, height}')
                xmin,ymin,xmax,ymax = cx - width/2 , cy - height/2 ,cx + width/2 , cy + height/2
                p_class = np.zeros((class_n,1), dtype=np.float32)
                for k in range(class_n):
                    p_class[k] = y_pred_cut[4,y,x]*sigmoid(y_pred_cut[k+5,y,x])
                ip_max = np.argmax(p_class) + 1
                bbox_list.append([xmin,ymin,xmax,ymax,j,float(p_class[ip_max-1]),ip_max])            

    # show bbox without NMS
    img_bb = show_bbox(args.img_path, input_size, bbox_list)
    img_bb = cv2.resize(img_bb, org_size)
    #print(f'num of bbox before nms = {len(bbox_list)}') 
    #plt.imshow(img_bb)
    fname = os.path.splitext(args.img_path)[0]
    fout1 = fname + '_bbox.jpg'
    cv2.imwrite(fout1, img_bb)

    # apply NMS
    bbox_array = np.array(bbox_list)
    bbox_arr_copy = bbox_array.copy()
    bbox_arr = np.delete(bbox_arr_copy, 4, axis=1) # delete 4th row
    final_idx = non_max_suppression(bbox_arr, args.nms_the)
    #bbox = bbox_arr[final_idx]
    bbox = bbox_array[final_idx]

    # show final results
    #print(f'num of bbox after nms = {bbox.shape}')    
    img_bb = show_bbox(args.img_path, input_size, bbox)
    img_bb = cv2.resize(img_bb, org_size)
    #plt.imshow(img_bb)
    fname = os.path.splitext(args.img_path)[0]
    fout2 = fname + '_bbox_nms.jpg'
    img_bb = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fout2, img_bb)
    end_time = time.time()
    print(f'post process time={(end_time-dpu_process_time)*1000.0:.2f}')
    print(f'total time={(end_time-start_time)*1000.0:.2f}')

if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
