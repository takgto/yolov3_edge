# yolov3_edge
All of yolov3 codes for woking it on edge device. Kria KV260 was used to check working properly.

## yolov3_app.cpp
Read one jpeg image and detect object (default 3 classes), then result jpeg with bounding box (bbox) is saved in "result.jpg". ./build.sh makes executable file (default yolov3).
Usage:
yolov3 (*.xmoel) (jpeg file)

## yolov3_file.cpp
Read jpeg files reading from input file  and detect object, then resulting bouding box results are saved in "result.txt". ./build.sh can be used by renaming source file.
Usage:
yolov3 (*.xmodel) (input file)

## yolov3_video.cpp
Read video file (*.webm only at this moment) and detect object, then output video with detected bbox. Currently, FSP is dominated by imshow and waitkey command (that is bottle neck), which would be imporved by OpenGL libray in near future. Also ./build.sh can be used by renaming source file.
Usage:
yolov3 (*.xmodel) (input movie, *.webm file only)

yolov3_video2 -- exec file with threads programming using concurrent queue. thread_test2.cpp would be helpful to understand the process. 
