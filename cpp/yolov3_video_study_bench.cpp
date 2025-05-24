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
int maxFrame = 20; // maximum number of frames to process, set to 0 for no limit

chrono::system_clock::time_point start_time, end_time, pre_end_time, dpu_end_time;

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

// input frames queue
queue<pair<int, Mat>> queueInput; // queue of FIFO
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
    ScopeTimer timer_all("readFrame_total"); // start timer for readFrame function
    static int loop = 1;                     // video end of three times play
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();

    while (loop > 0)
    {
        loop--; // Otherwirse infinite loop
        if (!video.open(videoFile))
        {
            cout << "Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (idxInputImage < maxFrame)
        {
            auto t0 = high_resolution_clock::now();
            Mat img;
            if (!video.read(img))
            {
                break;
            }

            auto pair = make_pair(idxInputImage, img);
            out.push(pair);
            auto t1 = high_resolution_clock::now();
            auto read_dur = duration_cast<microseconds>(t1 - t0).count();
            auto read_start = (duration_cast<microseconds>(t1 - program_start)).count();

            logger.logRow("readFrame", {idxInputImage, read_start, read_dur});
            ++idxInputImage;
        }
        cout << "Read " << idxInputImage << " frames from video file: " << videoFile << endl;
        video.release();
    }
    cout << "Finished reading frames from video file: " << videoFile << endl;
    bReading = false; // usually true, set false only when 'q' key is pushed.
}

void displayFrame(concurrent_queue<imagePair> &in)
{
    ScopeTimer timer_all("displayFrame_total"); // start timer for displayFrame function
    Mat frame;
    int index;
    while (true)
    {
        auto t0 = high_resolution_clock::now();
        auto pairIndexImg = in.pop();
        frame = pairIndexImg.second;
        index = pairIndexImg.first;
        if (frame.rows <= 0 || frame.cols <= 0)
        {
            continue;
        }

        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        stringstream buffer;
        buffer << fixed << setprecision(1)
               << (float)pairIndexImg.first / (dura / 1000000.f);
        string a = buffer.str() + " FPS";
        putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
        // cout << "FPS=" << buffer.str() << "\n" << flush;
        // cout << "index = " << index << flush;
        if (index % 2)
        {
            // imshow("YOLOv3 Detection@Xilinx DPU", frame);
            // auto key = waitKey(1);
            // if (key == 27)
            // {
            //     bReading = false; // usually true, set false only when 'q' key is pushed.
            //     exit(0);
            // }
        }
        auto t1 = high_resolution_clock::now();
        auto display_dur = duration_cast<microseconds>(t1 - t0).count();
        auto display_start = (duration_cast<microseconds>(t1 - program_start)).count();
        logger.logRow("displayFrame", {index, display_start, display_dur});
        if (index > maxFrame-2)
        {
            break;
        }
    }
}

void post_process(Mat &img, const vector<int8_t *> &out, const GraphInfo &shapes,
                  const float &scale, const int &sHeight, const int &sWidth)
{
    // ScopeTimer timer_all("post_process"); // start timer for post_process function

    // sHeight, sWidth = 416, 416
    vector<vector<float>> boxes;
    char fname[256];
    for (size_t i = 0; i < out.size(); i++)
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

    /* Apply the computation for NMS */
    // cout << "boxes size: " << boxes.size() << endl; // debug
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = img.rows;
    float w = img.cols;
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
}

void setInputPointer(vart::Runner *runner, const Mat &frame, int8_t *data,
                     const int &scale)
{
    // ScopeTimer timer_all("setInputPointer"); // start timer for setInputPointer function
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    Mat img = frame.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat image2 = cv::Mat(height, width, CV_8SC3); // CV_8SC3 means 3ch singed char data type
    cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

    unsigned char *imdata = image2.data;
    for (int i = 0; i < size; ++i)
    {
        float dataf = static_cast<float>(imdata[i]);
        data[i] = static_cast<int>((dataf * static_cast<float>(scale) / 256.0));
        if (data[i] < 0)
            data[i] = 127;
    }
}

void runYOLO(vart::Runner *runner, concurrent_queue<imagePair> &in, concurrent_queue<imagePair> &out)
{
    auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());

    // set input pointer
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;
    int inChannel = 3; // fixed
    int batchSize = 1; // fixed
    int inSize = inHeight * inWidth * inChannel;
    int8_t *imageInputs = new int8_t[inSize * batchSize];

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
    // cout << "size0 = " << size0 << endl; // debug
    int8_t *result0 = new int8_t[size0 * batchSize];
    const int size1 = shapes.outTensorList[1].size;
    // cout << "size1 = " << size1 << endl; // debug
    int8_t *result1 = new int8_t[size1 * batchSize];
    const int size2 = shapes.outTensorList[2].size;
    // cout << "size2 = " << size2 << endl; // debug
    int8_t *result2 = new int8_t[size2 * batchSize];

    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);
    float mean[3] = {0, 0, 0};

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
    while (bReading || !in.empty())
    {
        auto pairIndexImage = in.pop();
        auto t0 = high_resolution_clock::now();

        auto yolo_start_time = std::chrono::system_clock::now(); // runYOLO starttime
        setInputPointer(runner, pairIndexImage.second, imageInputs, input_scale);

        auto t1 = high_resolution_clock::now();
        
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
        
        auto t2 = high_resolution_clock::now();
    
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);

        auto t3 = high_resolution_clock::now();

        runner->wait(job_id.first, -1);

        auto t4 = high_resolution_clock::now();

        vector<int8_t *> results = {result0, result1, result2};
        post_process(pairIndexImage.second, results, shapes, conf_output_scale, inHeight, inWidth);
        // cv::imwrite("result.jpg", image2);
        out.push(pairIndexImage);
        auto yolo_end_time = std::chrono::system_clock::now();


        inputs.clear();
        outputs.clear();
        inputsPtr.clear();
        outputsPtr.clear();
        auto t5 = high_resolution_clock::now();

        long long setInput_dur = duration_cast<microseconds>(t1 - t0).count();
        long long setInput_start = duration_cast<microseconds>(t0 - program_start).count();
        logger.logRow("setInputPointer", {pairIndexImage.first, setInput_start, setInput_dur});
        long long pre_process_dur = duration_cast<microseconds>(t2 - t1).count();
        long long pre_process_start = duration_cast<microseconds>(t1 - program_start).count();
        logger.logRow("pre_process", {pairIndexImage.first, pre_process_start, pre_process_dur});
        long long dpu_dur = duration_cast<microseconds>(t3 - t2).count();
        long long dpu_start = duration_cast<microseconds>(t2 - program_start).count();
        logger.logRow("exec_async", {pairIndexImage.first, dpu_start, dpu_dur});
        long long wait_dur = duration_cast<microseconds>(t4 - t3).count();
        long long wait_start = duration_cast<microseconds>(t3 - program_start).count();
        logger.logRow("wait", {pairIndexImage.first, wait_start, wait_dur});
        long long post_process_dur = duration_cast<microseconds>(t5 - t4).count();
        long long post_process_start = duration_cast<microseconds>(t4 - program_start).count();
        logger.logRow("post_process", {pairIndexImage.first, post_process_start, post_process_dur});
    }
    delete imageInputs;
    delete[] result0;
    delete[] result1;
    delete[] result2;
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
    auto runner2 =
        vart::Runner::create_runner(subgraph[0], "run");
    auto runner3 =
        vart::Runner::create_runner(subgraph[0], "run");

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
    array<thread, 4> threadsList = {
        thread(readFrame, argv[2], ref(fr)),
        thread(displayFrame, ref(shw)),
        thread(runYOLO, runner.get(), ref(fr), ref(shw)),
        thread(runYOLO, runner1.get(), ref(fr), ref(shw)),
    };

    for (int i = 0; i < 4; i++)
    {
        threadsList[i].join();
    }

    return 0;
}
