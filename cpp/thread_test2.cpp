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
#include <condition_variable>
#include <future>
#include <utility>
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
//bool bExiting = false; // Not use

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
queue<pair<int, Mat>> queueInput; // queue of FIFO
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow; // priority queue by index comp.

GraphInfo shapes;
//int inHeight = 416;
//int inWidth = 416;
TensorShape inshapes[1];
TensorShape outshapes[3];

template<typename T>
class concurrent_queue {
public:
    typedef typename std::queue<T>::size_type size_type;
private:
    std::queue<T> queue_;
    size_type capacity_;

    std::mutex mtx_;
    std::condition_variable can_pop_;
    std::condition_variable can_push_;
public:
    concurrent_queue(size_type capacity) : capacity_(capacity) {
        if ( capacity_ == 0 ) {
            throw std::invalid_argument("capacity cannot be zero");
        }
    }     

    void push(const T& value) {
        std::unique_lock<std::mutex> guard(mtx_); 
        // wait 'can set'
        can_push_.wait(guard, [this]() { return queue_.size() < capacity_; });
        queue_.push(value);
        // notify 'can get'
        can_pop_.notify_one();
    }   

    T pop() { 
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can get' 
        can_pop_.wait(guard, [this]() { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        // notify 'can set'
        can_push_.notify_one();
        return value;
    }
    int size() {return queue_.size();}
};


void readFrame(const char* fileName, concurrent_queue<imagePair>& out) {
  static int loop = 3; // video end of three times play
  VideoCapture video;
  string videoFile = fileName;
  start_time = chrono::system_clock::now();
  if (!video.open(videoFile)) {
      cout << "Fail to open specified video file:" << videoFile << endl;
      exit(-1);
  }
  Mat img;

  while (loop > 0) {
      //loop--; //infinite loop
      if (!video.open(videoFile)) {
          cout << "Fail to open specified video file:" << videoFile << endl;
          exit(-1);
      }

      while (true) {
          //usleep(20000); // No performanec increase if 20000 --> 2000
          Mat img;
          if (!video.read(img)) {
              cout << "failed to read video." << endl;
              break;
          }  
          //cvtColor(img, img, cv::COLOR_BGR2RGB);
          //Mat image2 = cv::Mat(416, 416, CV_8SC3); // CV_8SC3 means 3ch singed char data type 
          //cv::resize(img, image2, Size(416, 416), 0, 0, cv::INTER_LINEAR);

          //mtxQueueInput.lock();
          //queueInput.push(make_pair(idxInputImage++, image2));
          auto pair = make_pair(idxInputImage++, img);
          out.push(pair);
          cout << "index=" << idxInputImage << "\n" << flush;
          //cout << "q size=" << out.size() << "\n" << flush;
      }

      video.release();
  }
      exit(0);
}

void displayFrame(concurrent_queue<imagePair>& in) {
  Mat frame;
  while (true) {
      auto pairIndexImg = in.pop();
      frame = pairIndexImg.second;
      if (frame.rows <= 0 || frame.cols <= 0) {
          mtxQueueShow.unlock();
          continue;
      }

      auto show_time = chrono::system_clock::now();
      auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
      stringstream buffer;
      buffer << fixed << setprecision(1)
             << (float)pairIndexImg.first / (dura / 1000000.f);
      string a = buffer.str() + " FPS";
      putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
      //cout << "FPS=" << buffer.str() << "\n" << flush;
      //imshow("YOLOv3 Detection@Xilinx DPU", frame);
      usleep(10000); // instead of imshow
      cout << "idx show=" << pairIndexImg.first << ", FPS=" << buffer.str() << "\n" << flush;
      if (waitKey(1) == 'q') {
          bReading = false; // usually true, set false only when 'q' key is pushed.
          exit(0);
      }
      //waitKey(1); // a little bit faster than above if condition.
    } 
}

void runYOLO(int i, concurrent_queue<imagePair>& in, concurrent_queue<imagePair>& out) {

   while (true) {
     auto pairIndexImage = in.pop();
     usleep(100000);// 60 mS
     //Mat img = Mat::eye(416,416, CV_8U);
     //auto pairIndexImage = make_pair((int)100, img);
     cout << "run# = " << i << ", YOLO index=" << pairIndexImage.first << "\n" << flush;
     //cout << "run# = " << i << "\n" << flush;
     out.push(pairIndexImage);
   }
}

int main(const int argc, const char** argv) {
    cout << "concurrency = " << std::thread::hardware_concurrency() << std::endl;
    // Check args
    if (argc != 2) {
        cout << "Usage of test_thread: ./yolov3 [webm video]" << endl;
        // #of images = batch size 
        return -1;
    }

    concurrent_queue<imagePair> fr(30), shw(30);
    array<thread, 6> threadsList = {
        thread(readFrame, argv[1], ref(fr)),
	thread(displayFrame, ref(shw)),
        thread(runYOLO, 1, ref(fr), ref(shw)), thread(runYOLO, 2, ref(fr), ref(shw)),
        thread(runYOLO, 4, ref(fr), ref(shw)), thread(runYOLO, 3, ref(fr), ref(shw))
    };


    for (int i = 0; i < 6; i++) {
        threadsList[i].join();
    }

    return 0;
}
