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
#include <opencv2/opencv.hpp>

// new include file related to thread handlig
#include <future>
#include <condition_variable>
#include <utility>

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
#include "benchmark.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

bool Lbox_on = false;

// Global Logger
static CSVLogger logger("bench_tmp.csv", "tid,func,frame,start,latency");
static std::chrono::high_resolution_clock::time_point program_start;
int maxFrame = 100; // maximum number of frames to process, set to 0 for no limit

chrono::system_clock::time_point start_time, end_time;

int idxInputImage = 0; // frame index of input video
int idxShowImage = 0;  // next frame index to be displayed
bool bReading = true;  // flag of reding input frame

typedef pair<int, Mat> imagePair;
class paircomp
{
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const
    {
        if (n1.first == n2.first)
        {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

queue<imagePair> queueInput; // queue of FIFO
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow; // priority queue by index comp.

GraphInfo shapes;
TensorShape inshapes[1];
TensorShape outshapes[3];

template <typename T>
class concurrent_queue
{
public:
    typedef typename std::queue<T>::size_type size_type;

private:
    std::queue<T> queue_;
    size_type capacity_;

    std::mutex mtx_;
    std::condition_variable can_pop_;
    std::condition_variable can_push_;

public:
    concurrent_queue(size_type capacity) : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("capacity cannot be zero");
        }
    }

    void push(const T &value)
    {
        // ScopeTimer timer_push("concurrent_queue::push"); // start timer for push function
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can set'
        can_push_.wait(guard, [this]()
                       { return queue_.size() < capacity_; });
        queue_.push(value);
        // notify 'can get'
        can_pop_.notify_one();
    }

    T pop()
    {
        // ScopeTimer timer_pop("concurrent_queue::pop"); // start timer for pop function
        std::unique_lock<std::mutex> guard(mtx_);
        // wait 'can get'
        can_pop_.wait(guard, [this]()
                      { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        // notify 'can set'
        can_push_.notify_one();
        return value;
    }
    int size() { return queue_.size(); }
    bool empty() { return queue_.empty(); }
};

void readFrame(const char *fileName, concurrent_queue<imagePair> &out)
{
    static int loop = 1; // video end of three times play
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();

    while (loop > 0)
    {
        loop--; //infinite loop
        if (!video.open(videoFile))
        {
            cout << "Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (true)
        {
            Mat img;
            if (!video.read(img))
            {
                break;
            }
            auto pair = make_pair(idxInputImage, img);
            end_time = chrono::system_clock::now();

            out.push(pair);
        }
        video.release();
    }
    exit(0);
}

void readOneFrame(VideoCapture &video, concurrent_queue<imagePair> &out, int &idxInputImage)
{
    // ScopeTimer timer_read("readOneFrame"); // start timer for readOneFrame function
    auto t0 = chrono::high_resolution_clock::now();
    Mat img;
    if (!video.read(img))
    {
        cout << "End of video stream." << endl;
        return;
    }

    auto pair = make_pair(idxInputImage, img);
    out.push(pair);
    auto t1 = chrono::high_resolution_clock::now();
    auto read_dur = duration_cast<microseconds>(t1 - t0).count();
    auto read_start = (duration_cast<microseconds>(t1 - program_start)).count();
    logger.logRow("readFrame", {pair.first, read_start, read_dur});
}

void displayFrame(concurrent_queue<imagePair> &in)
{
    Mat frame;
    int index;
    static const int fontFace = FONT_HERSHEY_SIMPLEX;
    static const double fontScale = 1.0;
    static const int thickness = 1;
    static const Scalar color{0,0,240};
    char buf[32];

    while (true)
    {
        auto pairIndexImg = in.pop();
        frame = pairIndexImg.second;
        index = pairIndexImg.first;
        if (frame.rows <= 0 || frame.cols <= 0)
        {
            continue;
        }

        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        double fps = index / (dura * 1e-6);
        int len = std::snprintf(buf, sizeof(buf), "%.1f FPS", fps);
        cv::putText(frame, buf, Point(10,15),
                    fontFace, fontScale, color, thickness);
        if (index % 2)
        {
            imshow("YOLOv3 Detection@Xilinx DPU", frame);
            waitKey(1); // wait for 1 ms to show the frame
            // imwrite("result.jpg", frame);
            // if (waitKey(1) == 'q')
            // {
            //     bReading = false; // usually true, set false only when 'q' key is pushed.
            //     exit(0);
            // }
        }
    }
}

void displayOneFrame(concurrent_queue<imagePair> &in)
{
    // ScopeTimer timer_display("displayOnFrame"); // start timer for displayOnFrame function
    static const int fontFace = FONT_HERSHEY_SIMPLEX;
    static const double fontScale = 1.0;
    static const int thickness = 1;
    static const Scalar color{0,0,240};
    char buf[32];

    auto t0 = chrono::high_resolution_clock::now();
    auto pairIndexImg = in.pop();
    Mat frame = pairIndexImg.second;
    int index = pairIndexImg.first;
    if (frame.rows <= 0 || frame.cols <= 0)
    {
        cout << "Invalid frame size: " << frame.rows << "x" << frame.cols << endl;
        return;
    }

    auto show_time = chrono::system_clock::now();
    auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
    double fps = index / (dura * 1e-6);
    std::snprintf(buf, sizeof(buf), "%.1f FPS", fps);
    cv::putText(frame, buf, Point(10, 15),
                fontFace, fontScale, color, thickness);

    // imwrite("result.jpg", frame);
    imshow("YOLOv3 Detection@Xilinx DPU", frame);
    auto key = waitKey(1);
    auto t1 = chrono::high_resolution_clock::now();
    auto display_dur = duration_cast<microseconds>(t1 - t0).count();
    auto display_start = (duration_cast<microseconds>(t0 - program_start)).count();
    logger.logRow("displayFrame", {index, display_start, display_dur});
}

Mat post_process(const Mat &frame, const vector<int8_t *> &out, const GraphInfo &shapes, const float &scale, const int &sHeight, const int &sWidth)
{
    // sHeight, sWidth = 416, 416
    auto img = frame.clone();
    vector<vector<float>> boxes;
    char fname[256];
    const int outSize = out.size();
    for (size_t i = 0; i < outSize; i++)
    {
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        // cout << "width, height = " << width << ", " << height << endl; // debug
        // cout << "channel, sizeOut = " << channel << ", " << sizeOut << endl; // debug

        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        // scaling_cast(out[i], scale, result);

        detect(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);

        /* debug
        sprintf(fname, "out%d.dat", i);
        cout << "binary file output : " << fname << endl;
        cout << "" << endl;

        write_binary(fname, result);
        */
    }
    /* Restore the correct coordinate frame of the original image */
    if (Lbox_on)
    {
        cout << "Lbox_on" << endl;                    // debug
        cout << img.cols << ", " << img.rows << endl; // debug
        cout << sWidth << ", " << sHeight << endl;    // debug
        correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight);
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            cout << boxes[i][0] << ", " << boxes[i][1] << endl; // debug
            cout << boxes[i][2] << ", " << boxes[i][3] << endl; // debug
        }
    }
    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl; // debug
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    // cout << "boxes size after NMS: " << res.size() << endl;
    // cout << "class conf, class, xmin, ymin, xmax, ymax" << endl;
    float h = img.rows;
    float w = img.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

        // cout<<res[i][res[i][4] + 6]<<" "; // (res[i][4]=class#)+6(offset) means class_score due to the results of apply NMS.
        // cout<<res[i][4] << " "; // most confident class number
        // cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;

        if (res[i][res[i][4] + 6] > CONF)
        {
            int type = res[i][4];

            if (type == 0)
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type == 1)
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(255, 0, 0), 1, 1, 0);
            }
            else
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 255, 255), 1, 1, 0);
            }
        }
    }
    return img;
}

void postProcess_OpenCV(const Mat &frame, const vector<int8_t *> &out, const GraphInfo &shapes,
                       const float &scale, const int &sHeight, const int &sWidth)
{
    // auto t0 = chrono::system_clock::now();
    // sHeight, sWidth = 416, 416
    vector<vector<float>> boxes;
    for (size_t i = 0; i < out.size(); i++)
    {
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        detect(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
        // detect_yolov4(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
    }

    // auto t1 = chrono::system_clock::now();

    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl; // debug
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    // auto t2 = chrono::system_clock::now();

    float h = frame.rows;
    float w = frame.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

        if (res[i][res[i][4] + 6] > CONF)
        {
            int type = res[i][4];

            if (type == 0)
            {
                rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type == 1)
            {
                rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(255, 0, 0), 1, 1, 0);
            }
            else
            {
                rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(0, 255, 255), 1, 1, 0);
            }
        }
    }
    // return img;
}

// original setInputPointer function
void setInputPointer(const Mat &frame, int8_t *data,
                     const int &scale)
{
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    Mat img = frame.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat image2 = cv::Mat(height, width, CV_8SC3); // CV_8SC3 means 3ch singed char data type
    cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

    // unsigned char* imdata = img.data;
    unsigned char *imdata = image2.data;
    for (int i = 0; i < size; ++i)
    {
        float dataf = static_cast<float>(imdata[i]);
        data[i] = static_cast<int>((dataf * static_cast<float>(scale) / 256.0));
        if (data[i] < 0)
            data[i] = 127;
    }
}

// OpenCV version of setInputPointer function
static int fromTo[] = {0,2 , 1, 1, 2, 0}; // BGR to RGB
void setInputPointer_OpenCV(const Mat &frame, int8_t *data,
                                 float input_scale)
{
    // setInput_start_time = chrono::system_clock::now();
    // 1) リサイズ＋BGR→RGB
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    static Mat image, image2, rgb, quantized;
    image     .create(frame.size(), frame.type());
    image2    .create(Size(width,height), CV_8SC3);
    rgb       .create(Size(width, height), CV_8UC3);
    quantized .create(Size(width,height), CV_8SC3);

    image = frame.clone();
    cv::resize(image, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);
    mixChannels(&image2, 1, &rgb, 1, fromTo, 3); // BGR to RGB conversion

    // 3) scale を反映しつつ int8 へ量子化, padding
    rgb.convertTo(quantized, CV_8S, input_scale / 256.0, 0);
    // 4) メモリコピー
    std::memcpy(data, quantized.data, width * height * 3);
}

class YoloContext
{
public:
    YoloContext(vart::Runner *runner_, const GraphInfo &shapes_) : runner(runner_), shapes(shapes_)
    {
        // 1) cloneTensorBuffer はコンストラクタで一度だけ
        inputTensors = cloneTensorBuffer(runner->get_input_tensors());
        outputTensors = cloneTensorBuffer(runner->get_output_tensors());
        // 2) スケール係数の取得 (１回だけ)
        inputScale = get_input_scale(runner->get_input_tensors()[0]);
        conf_output_scale =
            get_output_scale(runner->get_output_tensors()[shapes.output_mapping[1]]);

        // 3) サイズ計算
        const auto &inShape = shapes.inTensorList[0];
        inSize = inShape.height * inShape.width * /*channel=*/3;
        size0 = shapes.outTensorList[0].size;
        size1 = shapes.outTensorList[1].size;
        size2 = shapes.outTensorList[2].size;

        // 4) メモリ確保
        posix_memalign(reinterpret_cast<void **>(&imageInputs), 64, inSize);
        posix_memalign(reinterpret_cast<void **>(&result0), 64, size0);
        posix_memalign(reinterpret_cast<void **>(&result1), 64, size1);
        posix_memalign(reinterpret_cast<void **>(&result2), 64, size2);

        // 5) TensorBuffer の生成も一度だけ
        inputs.emplace_back(
            std::make_unique<CpuFlatTensorBuffer>(imageInputs, inputTensors[0].get()));
        outputs.emplace_back(
            std::make_unique<CpuFlatTensorBuffer>(result0, outputTensors[shapes.output_mapping[0]].get()));
        outputs.emplace_back(
            std::make_unique<CpuFlatTensorBuffer>(result1, outputTensors[shapes.output_mapping[1]].get()));
        outputs.emplace_back(
            std::make_unique<CpuFlatTensorBuffer>(result2, outputTensors[shapes.output_mapping[2]].get()));

        // 6) ポインタ配列は std::array など固定長で保持
        inputsPtr.reserve(1);
        inputsPtr.emplace_back(inputs[0].get());
        outputsPtr.reserve(3);
        outputsPtr = {outputs[0].get(), outputs[1].get(), outputs[2].get()};
    }

    ~YoloContext()
    {
        std::free(imageInputs);
        std::free(result0);
        std::free(result1);
        std::free(result2);
    }

    Mat infer(const imagePair ip) // 推論 + メトリクス
    {                                                       
        auto t0 = chrono::high_resolution_clock::now();
        Mat frame = ip.second;
        int index = ip.first;
        setInputPointer_OpenCV(frame, imageInputs, inputScale);
        auto t1 = chrono::high_resolution_clock::now();
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        auto t2 = chrono::high_resolution_clock::now();
        runner->wait(job_id.first, -1);
        auto t3 = chrono::high_resolution_clock::now();

        // ポスト処理へ
        std::vector<int8_t *> results = {result0, result1, result2};
        postProcess_OpenCV(frame, results, shapes, conf_output_scale,
                                            shapes.inTensorList[0].height, shapes.inTensorList[0].width);
        auto t4 = chrono::high_resolution_clock::now();
        auto setInput_dur = duration_cast<microseconds>(t1 - t0).count();
        auto execute_dur = duration_cast<microseconds>(t2 - t1).count();
        auto wait_dur = duration_cast<microseconds>(t3 - t2).count();
        auto postProcess_dur = duration_cast<microseconds>(t4 - t3).count();
        auto setInput_start = (duration_cast<microseconds>(t0 - program_start)).count();
        auto execute_start = (duration_cast<microseconds>(t1 - program_start)).count();
        auto wait_start = (duration_cast<microseconds>(t2 - program_start)).count();
        auto postProcess_start = (duration_cast<microseconds>(t3 - program_start)).count();
        logger.logRow("setInputPointer", {index, setInput_start, setInput_dur});
        logger.logRow("pre_process", {index,  execute_start, 0});
        logger.logRow("exec_async", {index, execute_start, execute_dur});
        logger.logRow("wait", {index, wait_start, wait_dur});
        logger.logRow("post_process", {index, postProcess_start, postProcess_dur});
        return frame;
    }

private:
    vart::Runner *runner;
    GraphInfo shapes;

    std::vector<std::unique_ptr<xir::Tensor>> inputTensors, outputTensors;
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer *> inputsPtr;
    std::vector<vart::TensorBuffer *> outputsPtr;

    int8_t *imageInputs = nullptr, *result0 = nullptr, *result1 = nullptr, *result2 = nullptr;
    size_t inSize, size0, size1, size2;
    float inputScale;
    float conf_output_scale;
};

void runYOLO(vart::Runner *runner, concurrent_queue<imagePair> &in, concurrent_queue<imagePair> &out)
{
    static YoloContext yoloContext(runner, shapes);
    imagePair pairIndexImage = in.pop();
    pairIndexImage.second = yoloContext.infer(pairIndexImage);
    out.push(pairIndexImage);
}

int main(const int argc, const char **argv)
{
    program_start = chrono::high_resolution_clock::now();
    cout << "concurrency = " << std::thread::hardware_concurrency() << std::endl;
    // Check args
    if (argc != 3)
    {
        cout << "Usage of yolov3: ./resnet50 [model_file] [jpg image]" << endl;
        // #of images = batch size
        return -1;
    }
    auto xmodel_file = std::string(argv[1]);

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
    // auto runner2 =
    //     vart::Runner::create_runner(subgraph[0], "run");
    // auto runner3 =
    //     vart::Runner::create_runner(subgraph[0], "run");
    
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes; // get output size
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    concurrent_queue<imagePair> fr(30), shw(30);

    idxInputImage = 0; // reset frame index of input video
    VideoCapture video = VideoCapture(argv[2]);
    start_time = chrono::system_clock::now();
    while(idxInputImage < maxFrame) {
        readOneFrame(video, fr, idxInputImage);
        runYOLO(runner.get(), fr, shw);
        displayOneFrame(shw);
        ++idxInputImage;
    }
    return 0;
}
