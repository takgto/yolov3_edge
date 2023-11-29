/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/demo.hpp>

#include "./process_result2.hpp"
using namespace std;

int main(int argc, char* argv[]) {
  std::string model = argv[1];
  std::string file = argv[2];

  // read image filename from the list file.
  ifstream ifs(argv[2]);
  if (ifs.fail()) {
      cerr << "failed to open list file." << endl;
      return -1;
  }
  // output file
  ofstream ofs("results.txt");
  if (ofs.fail()) {
      cerr << "failed to open resut file" << endl;
  }
  string str;

  while (getline(ifs, str)) {
      //cout << str << endl;
      size_t eth = str.find_last_of(".");
      //cout << "eth = " << eth << endl;
      string ext = str.substr(eth, str.size());

      if ( (ext != ".jpg") & (ext != ".JPG") & (ext != ".png") & (ext != ".PNG") ) {
          cerr << "file is not image file : " << str << endl;
          return -1;
      }
      size_t pth = str.find_last_of("/");
      string basename = str.substr(pth+1, eth-pth-1);
      string outname = "./mAP/input/detection-results/" + basename + ".txt";
      //cout << "outname = " << outname << endl;

      // put input images
      strcpy(argv[2], const_cast<char*>(str.c_str()));
      //cout << "input file = " << argv[2] << endl;

      vitis::ai::main_for_jpeg_demo(
      argc, argv,
          [model] {
            return vitis::ai::YOLOv3::create(model);
          },
          process_result, 2);

      // change filename
      const int buf_size = 255;
      char buf[buf_size];

      ofstream of0(outname);
      ifstream ifs("det_result.txt");
      while (ifs.getline(buf, buf_size)) {
          of0 << buf << endl;
      }

  }
  return 0;
}
