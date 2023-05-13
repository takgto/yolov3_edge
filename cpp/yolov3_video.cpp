#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
//#include <dnndk/dnndk.h>
#include <opencv2/opencv.hpp>

// next include files from VART program
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <xir/graph/graph.hpp>
#include "vitis/ai/collection_helper.hpp"
#include "common.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

chrono::system_clock::time_point start_time, end_time, pre_end_time, dpu_end_time;

#define INPUT_NODE "layer0_conv"

int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reding input frame
bool bExiting = false;

typedef pair<int, Mat> imagePair;
class paircomp {
 public:
  bool operator()(const imagePair& n1, const imagePair& n2) const {
    if (n1.first == n2.first) {
      return (n1.first > n2.first);
    }

    return n1.first > n2.first;
  }
};

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protecFtion of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;

GraphInfo shapes;
//int inHeight = 416;
//int inWidth = 416;
TensorShape inshapes[1];
TensorShape outshapes[3];

void readFrame(const char* fileName) {
  static int loop = 3;
  VideoCapture video;
  string videoFile = fileName;
  start_time = chrono::system_clock::now();

  while (loop > 0) {
    loop--;
    if (!video.open(videoFile)) {
      cout << "Fail to open specified video file:" << videoFile << endl;
      exit(-1);
    }

    while (true) {
      usleep(20000);
      Mat img;
      if (queueInput.size() < 30) {
        if (!video.read(img)) {
          break;
        }
        //cvtColor(img, img, cv::COLOR_BGR2RGB);
        //Mat image2 = cv::Mat(416, 416, CV_8SC3); // CV_8SC3 means 3ch singed char data type 
        //cv::resize(img, image2, Size(416, 416), 0, 0, cv::INTER_LINEAR);

        mtxQueueInput.lock();
        //queueInput.push(make_pair(idxInputImage++, image2));
        queueInput.push(make_pair(idxInputImage++, img));
        mtxQueueInput.unlock();
      } else {
        usleep(10);
      }
    }

    video.release();
  }
  bExiting = true;
}

void displayFrame() {
  Mat frame;

  while (true) {
    if (bExiting) break;
    mtxQueueShow.lock();

    if (queueShow.empty()) {
      mtxQueueShow.unlock();
      usleep(10);
    } else if (idxShowImage == queueShow.top().first) {
      auto show_time = chrono::system_clock::now();
      stringstream buffer;
      frame = queueShow.top().second;
      if (frame.rows <= 0 || frame.cols <= 0) {
        mtxQueueShow.unlock();
        continue;
      }
      auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
      buffer << fixed << setprecision(1)
             << (float)queueShow.top().first / (dura / 1000000.f);
      string a = buffer.str() + " FPS";
      cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
      cv::imshow("ADAS Detection@Xilinx DPU", frame);

      idxShowImage++;
      queueShow.pop();
      mtxQueueShow.unlock();
      if (waitKey(1) == 'q') {
        bReading = false;
        exit(0);
      }
    } else {
      mtxQueueShow.unlock();
    }
  }
}
// for debug
void write_output(const string& name, const int8_t* result, const int& size0) {
    ofstream ofs ( name, ios_base::binary );
    if (!ofs) {
        cout << "Cannot open output file!!" << endl;
    } else {
        ofs.write( (char*)&result[0], sizeof(int8_t)*size0 );
    }
}

void post_process(Mat& img, const vector<int8_t*>& out, const GraphInfo& shapes, 
		const float& scale, const int& sHeight, const int& sWidth) {
// sHeight, sWidth = 416, 416
    vector<vector<float>> boxes;
    char fname[256];
    for (size_t i=0; i < out.size(); i++) {
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        //cout << "width, height = " << width << ", " << height << endl; // debug
        //cout << "channel, sizeOut = " << channel << ", " << sizeOut << endl; // debug

        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
	//scaling_cast(out[i], scale, result);

	detect(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
        
	/* debug
	sprintf(fname, "out%d.dat", i);
	cout << "binary file output : " << fname << endl;
	cout << "" << endl;

	write_binary(fname, result);
	*/
    }

    /* Restore the correct coordinate frame of the original image */
    //correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight); // don't know why this func is needed

    /* Apply the computation for NMS */
    //cout << "boxes size: " << boxes.size() << endl; // debug
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    //cout << "boxes size after NMS: " << res.size() << endl;
    //cout << "class conf, class, xmin, ymin, xmax, ymax" << endl;  
    float h = img.rows;
    float w = img.cols;
    for(size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;
	
	//cout<<res[i][res[i][4] + 6]<<" "; // (res[i][4]=class#)+6(offset) means class_score due to the results of apply NMS.
        //cout<<res[i][4] << " "; // most confident class number	
	//cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;


        if(res[i][res[i][4] + 6] > CONF ) {
            int type = res[i][4];

            if (type==0) {
		    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), 
				                               Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type==1) {
		    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), 
				                               Scalar(255, 0, 0), 1, 1, 0);
            }
            else {
		    rectangle(img, Point(xmin, ymin), Point(xmax, ymax), 
				                               Scalar(0 ,255, 255), 1, 1, 0);
            }
        }
    }


}
/*
void setInputImageForYOLO(int8_t* data, const Mat& frame, float* mean, 
		float input_scale) {
  Mat img_copy;
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;
  int size = shapes.inTensorList[0].size;
  image img_new = load_image_cv(frame);
  image img_yolo = letterbox_image(img_new, width, height);

  vector<float> bb(size);
  for (int b = 0; b < height; ++b) {
    for (int c = 0; c < width; ++c) {
      for (int a = 0; a < 3; ++a) {
        bb[b * width * 3 + c * 3 + a] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }

  float scale = pow(2, 7);
  for (int i = 0; i < size; ++i) {
    data[i] = (int8_t)(bb.data()[i] * input_scale);
    if (data[i] < 0) data[i] = (int8_t)((float)(127 / scale) * input_scale);
  }
  free_image(img_new);
  free_image(img_yolo);
}
*/
void setInputPointer(vart::Runner *runner, const Mat& frame, 
		int8_t* data, const float* mean, const int& scale) {
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    Mat img = frame.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat image2 = cv::Mat(height, width, CV_8SC3); // CV_8SC3 means 3ch singed char data type 
    cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

    //unsigned char* imdata = img.data;
    unsigned char* imdata = image2.data;
    for(int i = 0; i < size; ++i) {
        float dataf = static_cast<float>(imdata[i]);
        data[i] = static_cast<int>( (dataf*static_cast<float>(scale)/256.0) );
        if(data[i] < 0) data[i] = 127;
    }
}

void runYOLO(vart::Runner *runner) {
    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());

    // set input pointer
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;
    int inChannel = 3; // fixed
    int batchSize = 1; // fixed
    int inSize = inHeight * inWidth * inChannel;
    int8_t* imageInputs = new int8_t[inSize * batchSize];
    
    //Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3); // CV_8SC3 means 3ch singed char data type 
    //cv::resize(img, image2, Size(inWidth, inHeight), 0, 0, cv::INTER_LINEAR);
    //setInputPointer(, imageInputs, inHeight, inWidth, inChannel, input_scale);
    /*
    cout << "in_height = " << inHeight << endl;
    cout << "in_width = " << inWidth << endl;
    cout << "in_channel = " << inChannel << endl;
    cout << "batch size = " << batchSize << endl;
    cout << "\n";
    */
    // set output pointer
    vector<int> output_mapping = shapes.output_mapping;
    auto conf_output_scale =
      get_output_scale(runner->get_output_tensors()[output_mapping[1]]);

    // make output vector 
    const int size0 = shapes.outTensorList[0].size;
    //cout << "size0 = " << size0 << endl; // debug
    int8_t* result0 = new int8_t[size0*batchSize];
    const int size1 = shapes.outTensorList[1].size;
    //cout << "size1 = " << size1 << endl; // debug
    int8_t* result1 = new int8_t[size1*batchSize];
    const int size2 = shapes.outTensorList[2].size;
    //cout << "size2 = " << size2 << endl; // debug
    int8_t* result2 = new int8_t[size2*batchSize];


    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);
    float mean[3] = {0, 0, 0};

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    while (true) {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if (bExiting) break;
            if (bReading) {
                continue;
            } else {
                break;
            }
        } else {
        /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }

        setInputPointer(runner, pairIndexImage.second, imageInputs, mean, input_scale);

        // preparation for execute
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            imageInputs, inputTensors[0].get()));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result0, outputTensors[output_mapping[0]].get()));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result1, outputTensors[output_mapping[1]].get()));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result2, outputTensors[output_mapping[2]].get()));
    
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());
        outputsPtr.push_back(outputs[1].get());
        outputsPtr.push_back(outputs[2].get());
        //pre_end_time = std::chrono::system_clock::now();

        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        //cout << "Done execution" << endl; // debug

        //dpu_end_time = std::chrono::system_clock::now();

        // check output
        /* debug
        write_output("out0.bin", result0, size0);
        write_output("out1.bin", result1, size1);
        write_output("out2.bin", result2, size2);
        */
        vector<int8_t *> results = {result0, result1, result2};
        post_process(pairIndexImage.second, results, shapes, conf_output_scale, inHeight, inWidth);
        //cv::imwrite("result.jpg", image2);
	mtxQueueShow.lock();

        /* push the image into display frame queue */
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
        inputs.clear();
        outputs.clear();
        inputsPtr.clear();
        outputsPtr.clear();
    }
    delete imageInputs;
    delete[] result0;
    delete[] result1;
    delete[] result2;
    /*
    end_time = std::chrono::system_clock::now();
    cout << "\npre_process time = " << 
	    std::chrono::duration_cast<std::chrono::milliseconds>(pre_end_time - start_time).count() 
	    << " [mS]" << endl; 
    cout << "DPU time = " << 
	    std::chrono::duration_cast<std::chrono::milliseconds>(dpu_end_time - pre_end_time).count() 
	    << " [mS]" << endl; 
    cout << "post_process time = " << 
	    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - dpu_end_time).count() 
	    << " [mS]" << endl;
    cout << "-------------------------------------------" << endl; 
    cout << "total proc. time = " << 
	    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() 
	    << " [mS]" << endl; 

    cout << "\nDone yolov3." << endl;*/

}

int main(const int argc, const char** argv) {

    // Check args
    if (argc != 3) {
        cout << "Usage of yolov3: ./resnet50 [model_file] [jpg image]" << endl;
        // #of images = batch size 
        return -1;
    }
    auto xmodel_file = std::string(argv[1]);

    // read input images
    //cv::Mat img = cv::imread(argv[2]);
    //cv::imwrite("input.jpg", img);
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //if (img.empty()) {
    //    std::cout << "Cannot load image : " << argv[2] << std::endl;
    //    return -1;
    //}

    // create dpu runner
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u)
          << "yolov3 should have one and only one dpu subgraph." << endl;
    cout << "create running for subgraph: " << subgraph[0]->get_name() << endl;

    auto attrs = xir::Attrs::create();
    auto runner =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner1 =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner2 =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner3 =
        vart::Runner::create_runner(subgraph[0], "run");
    //auto runner = vart::Runner::create_runner(subgraph[0], "run");
    //start_time = chrono::system_clock::now();

    //write_output("input.bin", imageInputs, inSize); // for debug
    // get in/out tenosrs
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    // init the shape info
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;    // get output size
    getTensorShape(runner.get(), &shapes, 1,
        {"quant_conv2d_58_fix", "quant_conv2d_66_fix", "quant_conv2d_74_fix"});
   
    array<thread, 6> threadsList = {
        thread(readFrame, argv[2]), thread(displayFrame),
        // thread(runYOLO, runner),
        thread(runYOLO, runner.get()), thread(runYOLO, runner1.get()),
        thread(runYOLO, runner2.get()), thread(runYOLO, runner3.get())
    };

    for (int i = 0; i < 6; i++) {
        threadsList[i].join();
    }

    return 0;
}
