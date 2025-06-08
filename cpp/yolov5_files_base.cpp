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
#include <numeric>
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
int maxFrame = 1000; // maximum number of frames to process, set to 0 for no limit

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
    auto read_start = (duration_cast<microseconds>(t0 - program_start)).count();
    logger.logRow("0_readFrame", {pair.first, read_start, read_dur});
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
    static const int fontFace = 1;
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
    waitKey(1);
    auto t1 = chrono::high_resolution_clock::now();
    auto display_dur = duration_cast<microseconds>(t1 - t0).count();
    auto display_start = (duration_cast<microseconds>(t0 - program_start)).count();
    logger.logRow("9_displayFrame", {index, display_start, display_dur});
}

void postProcess_OpenCV(int &idxInputImage, const Mat &frame, const vector<int8_t *> &out, const GraphInfo &shapes,
                       const float &scale, const int &sHeight, const int &sWidth)
{
    auto t0 = chrono::system_clock::now();
    // sHeight, sWidth = 416, 416
    vector<vector<float>> boxes;
    for (size_t i = 0; i < out.size(); ++i)
    {
        int channel = shapes.outTensorList[i].channel;
        int width = shapes.outTensorList[i].width;
        int height = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        // cout << "channel, width, height = " << channel << ", " << width << ", " << height << endl; // debug
        // cout << "sizeOut = " << sizeOut << endl; // debug
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);
           
        detect_yolov5(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
    }

    auto t1 = chrono::system_clock::now();

    /* Apply the computation for NMS */
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    auto t2 = chrono::system_clock::now();

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
    auto t3 = chrono::system_clock::now();
    auto post_detect = duration_cast<microseconds>(t1 - t0).count();
    auto nms_dur = duration_cast<microseconds>(t2 - t1).count();
    auto draw_dur = duration_cast<microseconds>(t3 - t2).count();
    auto post_start = (duration_cast<microseconds>(t0 - program_start)).count();
    logger.logRow("6_detect", {idxInputImage, post_start, post_detect});
    logger.logRow("7_nms", {idxInputImage, post_start + post_detect, nms_dur});
    logger.logRow("8_draw", {idxInputImage, post_start + post_detect + nms_dur, draw_dur});

    // return img;
}

void quantize_u8c3_to_i8c3_neon_optimized(const Mat& in, Mat& out)
{
    // CV_Assert(in.type() == CV_8UC3);
    // CV_Assert(out.type() == CV_8SC3);
    // CV_Assert(in.size() == out.size());
    // CV_Assert(in.step == out.step);  // 1行あたりのバイト数は完全に一致している必要あり

    const int rows = in.rows;
    const int cols = in.cols;
    // 1行あたりの総バイト数：cols × 3 (チャンネル数)
    const int width_bytes = cols * 3;

    for (int y = 0; y < rows; y++)
    {
        // 行の先頭アドレスを取得
        const uint8_t*  in_ptr  = in.ptr<uint8_t>(y);
        int8_t*         out_ptr = out.ptr<int8_t>(y);

        int remaining = width_bytes;

        // 行先頭のプリフェッチ（次の行読み込みを想定して先読みする場合）
        // __builtin_prefetch(in_ptr + 64, 0, 1);

        // --------------------------------------------------------------------
        // (A) まず「32バイトずつ (vld1q_u8×2, vshrq_n_u8×2, vst1q×2)」で処理
        // --------------------------------------------------------------------
        while (remaining >= 32)
        {
            // プリフェッチ：もう少し先のデータを先読みしたい場合
            // __builtin_prefetch(in_ptr + 128, 0, 1);

            // メモリから 16 バイト読み込み
            uint8x16_t vin1 = vld1q_u8(in_ptr);
            // さらに次の 16 バイト読み込み
            uint8x16_t vin2 = vld1q_u8(in_ptr + 16);

            // 1 ビット右シフト (0..255 → 0..127)
            uint8x16_t vq1 = vshrq_n_u8(vin1, 1);
            uint8x16_t vq2 = vshrq_n_u8(vin2, 1);

            // それぞれをそのまま int8x16_t としてストア
            vst1q_s8(out_ptr,         vreinterpretq_s8_u8(vq1));
            vst1q_s8(out_ptr + 16,    vreinterpretq_s8_u8(vq2));

            // ポインタを 32 バイトだけ進める
            in_ptr  += 32;
            out_ptr += 32;
            remaining -= 32;
        }

        // --------------------------------------------------------------------
        // (B) 続いて「残り 16 バイト以上なら 16 バイトずつ」処理
        // --------------------------------------------------------------------
        if (remaining >= 16)
        {
            uint8x16_t vin = vld1q_u8(in_ptr);
            uint8x16_t vq  = vshrq_n_u8(vin, 1);
            vst1q_s8(out_ptr, vreinterpretq_s8_u8(vq));

            in_ptr  += 16;
            out_ptr += 16;
            remaining -= 16;
        }

        // --------------------------------------------------------------------
        // (C) 最後に「<16 バイト」の余り部分をスカラーループで処理
        // --------------------------------------------------------------------
        for (int i = 0; i < remaining; i++)
        {
            // 1ビット右シフトして 0..127 に量子化
            *out_ptr++ = static_cast<int8_t>((*in_ptr++) >> 1);
        }
    }
}

// OpenCV version of setInputPointer function
static int fromTo[] = {0,2 , 1,1, 2,0}; // BGR to RGB
void setInputPointer_OpenCV(const Mat &frame, int8_t *data,
                                 float input_scale)
{
    // setInput_start_time = chrono::system_clock::now();
    auto t0 = chrono::high_resolution_clock::now();
    // 1) リサイズ＋BGR→RGB
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    static Mat image, image2, rgb, quantized;
    image     .create(frame.size(), frame.type());
    image2    .create(Size(width,height), CV_8UC3);
    rgb       .create(Size(width, height), CV_8UC3);
    quantized .create(Size(width,height), CV_8SC3);

    image = frame.clone();
    // cv::resize(image, image2, Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::resize(image, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);
    auto t1 = chrono::high_resolution_clock::now();
    // 2) BGR→RGB
    mixChannels(&image2, 1, &rgb, 1, fromTo, 3);
    auto t2 = chrono::high_resolution_clock::now();

    // 3) scale を反映しつつ int8 へ量子化, padding
    // rgb.convertTo(quantized, CV_8S, input_scale / 128.0, 0); // 128 ? 256 ?
    quantize_u8c3_to_i8c3_neon_optimized(rgb, quantized);
    // 4) メモリコピー
    std::memcpy(data, quantized.data, width * height * 3);
    auto t3 = chrono::high_resolution_clock::now();

    // metrics
    auto resize_dur = duration_cast<microseconds>(t1 - t0).count();
    auto bgr2rgb_dur = duration_cast<microseconds>(t2 - t1).count();
    auto quantize_dur = duration_cast<microseconds>(t3 - t2).count();
    auto setInput_start = (duration_cast<microseconds>(t0 - program_start)).count();
    logger.logRow("1_resize", {idxInputImage, setInput_start, resize_dur});
    logger.logRow("2_bgr2rgb", {idxInputImage, setInput_start + resize_dur, bgr2rgb_dur});
    logger.logRow("3_quantize", {idxInputImage, setInput_start + resize_dur + bgr2rgb_dur, quantize_dur});
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
        Mat frame = ip.second;

        int index = ip.first;
        setInputPointer_OpenCV(frame, imageInputs, inputScale);
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);

        // ポスト処理へ
        std::vector<int8_t *> results = {result0, result1, result2};
        postProcess_OpenCV(index, frame, results, shapes, conf_output_scale,
                                            shapes.inTensorList[0].height, shapes.inTensorList[0].width);
        return frame;
    }

    vector<vector<float>> infer_bboxes(const imagePair &ip) // 推論 + bboxes
    {
        Mat frame = ip.second;
        int index = ip.first;

        // 1) 入力画像の前処理
        setInputPointer_OpenCV(frame, imageInputs, inputScale);

        // 2) 推論実行
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);

        // 3) 出力結果の取得
        std::vector<int8_t *> results = {result0, result1, result2};

        vector<vector<float>> boxes;
        for (size_t i = 0; i < results.size(); ++i)
        {
            int channel = shapes.outTensorList[i].channel;
            int width = shapes.outTensorList[i].width;
            int height = shapes.outTensorList[i].height;
            int sizeOut = shapes.outTensorList[i].size;
            // cout << "channel, width, height = " << channel << ", " << width << ", " << height << endl; // debug
            // cout << "sizeOut = " << sizeOut << endl; // debug
            vector<float> result(sizeOut);
            boxes.reserve(sizeOut);
            
            detect_yolov5(boxes, results[i], channel, height, width, i, shapes.inTensorList[0].height, shapes.inTensorList[0].width, conf_output_scale);
        }
        /* Apply the computation for NMS */
        vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);
        return res;
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


vector<vector<float>> readGroundTruth(const string &gtFile)
{
    vector<vector<float>> gts;
    ifstream file(gtFile);
    if (!file.is_open())
    {
        cerr << "Error opening ground truth file: " << gtFile << endl;
        return gts;
    }

    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        vector<float> box(5);
        for (int i = 0; i < 5; ++i)
        {
            iss >> box[i];
        }
        gts.push_back(box);
    }
    file.close();
    return gts;
}

/**
 * IoU を計算する関数（YOLO形式: [class_id, cx, cy, w, h, (conf)] または [class_id, cx, cy, w, h]）
 * 
 * @param a 予測ボックス配列（float[5] 以上）。a[1]=cx, a[2]=cy, a[3]=w, a[4]=h を使用。
 * @param b GT ボックス配列（float[5]）。b[1]=cx, b[2]=cy, b[3]=w, b[4]=h を使用。
 * @return 2つのボックス間の IoU（0.0～1.0）
 */
float IoU(const array<float,6>& a, const array<float,5>& b) {
    float x1_min = a[1] - a[3] / 2.0f;
    float y1_min = a[2] - a[4] / 2.0f;
    float x1_max = a[1] + a[3] / 2.0f;
    float y1_max = a[2] + a[4] / 2.0f;

    float x2_min = b[1] - b[3] / 2.0f;
    float y2_min = b[2] - b[4] / 2.0f;
    float x2_max = b[1] + b[3] / 2.0f;
    float y2_max = b[2] + b[4] / 2.0f;

    float inter_w = max(0.0f, min(x1_max, x2_max) - max(x1_min, x2_min));
    float inter_h = max(0.0f, min(y1_max, y2_max) - max(y1_min, y2_min));
    float inter_area = inter_w * inter_h;

    float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    float union_area = area1 + area2 - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

/**
 * 11点補間による AP（Average Precision）を計算する関数
 * 
 * @param precisions  各検出閾値ステップでの Precision の配列（降順ソート済み検出に対応）
 * @param recalls     各検出閾値ステップでの Recall の配列
 * @return            AP 値（0.0～1.0）
 */
float compute_ap(const vector<float>& precisions, const vector<float>& recalls) {
    float ap = 0.0f;
    // Recall の閾値 t = 0.0, 0.1, 0.2, ..., 1.0
    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        float p_max = 0.0f;
        for (size_t i = 0; i < recalls.size(); ++i) {
            if (recalls[i] >= t) {
                p_max = max(p_max, precisions[i]);
            }
        }
        ap += p_max;
    }
    return ap / 11.0f;
}

/**
 * 複数画像・複数クラス対応の mAP@0.5 計算関数
 * 
 * @param preds         予測結果リスト
 *                      各要素: pair<image_id, float[6]>  
 *                         float[6] = { class_id, cx, cy, w, h, confidence }
 * @param gts           Ground Truth リスト
 *                      各要素: pair<image_id, float[5]>
 *                         float[5] = { class_id, cx, cy, w, h }
 * @param num_classes   クラス数
 * @return              mAP@0.5（0.0～1.0）
 */
float calculate_mAP50_multi_image(
    const vector<pair<int, array<float,6>>>& preds,
    const vector<pair<int, array<float,5>>>& gts,
    int num_classes
) {
    const float iou_thresh = 0.5f;
    vector<float> ap_list;  // 各クラスごとの AP を格納

    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    // 1. クラスごとに処理
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    for (int cls = 0; cls < num_classes; ++cls) {
        // 2. クラス c に属する GT を画像ID別にまとめる
        //    map<image_id, vector<GT_box_ptr>>
        map<int, vector<array<float,5>>> cls_gts_per_img;
        for (const auto& gt : gts) {
            int img_id = gt.first;
            const array<float,5>& gt_box = gt.second;  // 型は float[5] だが、配列はポインタに劣化する
            int gt_cls = static_cast<int>(gt_box[0]);
            
            if (gt_cls == cls) {
                cls_gts_per_img[img_id].push_back(gt_box);
            }
        }

        // 3. クラス c に属する予測を収集し、(confidence, image_id, box_ptr) のタプルで保存
        vector<tuple<float, int, array<float,6>>> cls_preds; 
        for (const auto& pred : preds) {
            int img_id = pred.first;
            const array<float,6>& pred_box = pred.second;  // float[6]
            int pred_cls = static_cast<int>(pred_box[0]);
            if (pred_cls == cls) {
                float confidence = pred_box[5];
                cls_preds.emplace_back(confidence, img_id, pred_box);
            }
        }

        // 4. confidence 降順でソート
        sort(cls_preds.begin(), cls_preds.end(),
             [](const auto& a, const auto& b) {
                 return get<0>(a) > get<0>(b);
             });

        // 5. GT のマッチ済みフラグを画像ID別に用意
        //    map<image_id, vector<bool>> matched_flags;
        map<int, vector<bool>> matched_flags;
        for (auto& kv : cls_gts_per_img) {
            int img_id = kv.first;
            size_t gt_count = kv.second.size();
            matched_flags[img_id] = vector<bool>(gt_count, false);
        }

        // 6. 各予測を TP or FP に分類する
        size_t N = cls_preds.size();
        vector<int> tp(N, 0), fp(N, 0);

        for (size_t i = 0; i < N; ++i) {
            float confidence;
            int img_id;
            array<float,6> pred_box;
            tie(confidence, img_id, pred_box) = cls_preds[i];

            // その画像 img_id に存在するクラス c の GT ボックス一覧を取得
            auto& gt_list = cls_gts_per_img[img_id];
            auto& flags   = matched_flags[img_id];

            float best_iou = 0.0f;
            int best_idx = -1;

            // 未マッチの GT を走査し、IoU 最大のインデックスを探す
            for (size_t j = 0; j < gt_list.size(); ++j) {
                if (flags[j]) continue;  // すでにマッチ済み GT はスキップ

                array<float,5> gt_box = gt_list[j];
                float iou = IoU(pred_box, gt_box);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_idx = static_cast<int>(j);
                }
            }

            // IoU ≥ 0.5 かつ対応する未マッチ GT があれば TP にする
            if (best_iou >= iou_thresh && best_idx != -1) {
                tp[i] = 1;
                flags[best_idx] = true;  // この GT は今後使えない
            } else {
                fp[i] = 1;
            }
        }

        // 7. Precision/Recall を計算
        //    total_gt: クラス c に属する GT の総数
        int total_gt = 0;
        for (auto& kv : cls_gts_per_img) {
            total_gt += static_cast<int>(kv.second.size());
        }

        vector<float> precisions, recalls;
        precisions.reserve(N);
        recalls.reserve(N);

        int cum_tp = 0, cum_fp = 0;
        for (size_t i = 0; i < N; ++i) {
            cum_tp += tp[i];
            cum_fp += fp[i];
            float prec = cum_tp / static_cast<float>(cum_tp + cum_fp + 1e-6f);
            float rec  = (total_gt > 0) 
                         ? (cum_tp / static_cast<float>(total_gt)) 
                         : 0.0f;
            precisions.push_back(prec);
            recalls.push_back(rec);
        }

        // 8. AP を計算（GT が 0 の時は 0 とする）
        float ap = 0.0f;
        if (total_gt > 0) {
            ap = compute_ap(precisions, recalls);
        }
        ap_list.push_back(ap);
    }
    for (size_t i = 0; i < ap_list.size(); ++i) {
        cout << "Class " << i << " AP: " << ap_list[i] << endl; // 各クラスの AP を表示
    }

    // 9. mAP@0.5 をクラス数で割って求める
    float sum_ap = accumulate(ap_list.begin(), ap_list.end(), 0.0f);
    float mAP = sum_ap / static_cast<float>(num_classes);
    return mAP;
}


int main(const int argc, const char **argv)
{
    program_start = chrono::high_resolution_clock::now();
    // set OpenCV thread = 1
    cv::setNumThreads(1); // OpenCV uses multiple threads by default, set to 1 for single thread
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

    auto attrs = xir::Attrs::create();
    auto runner =
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
    // cout << "Input Cnt: " << inputCnt << ", Output Cnt: " << outputCnt << endl;

    // get image path from file list text file
    string validation_images_path = argv[2];
    // cout << "validation_images_path = " << validation_images_path << endl;
    vector<string> imageFiles;
    vector<string> bboxFiles;
    if (validation_images_path.find(".txt") != string::npos) {
        // read image file list from text file
        ifstream infile(validation_images_path);
        string line;
        while (getline(infile, line)) {
            if (!line.empty()) {
                imageFiles.push_back(line);
                // *.jpg -> *.txt
                string bboxFile = line.substr(0, line.find_last_of('.')) + ".txt";
                bboxFiles.push_back(bboxFile);
            }
        }
        infile.close();
    } else {
        assert(validation_images_path.find(".txt") == string::npos);
    }
    // cout << "Number of images: " << imageFiles.size() << endl;
    concurrent_queue<imagePair> fr(30), shw(30);

    idxInputImage = 0; // reset frame index of input video
    VideoCapture video = VideoCapture(argv[2]);
    start_time = chrono::system_clock::now();

    static YoloContext yoloContext(runner.get(), shapes);
    // image_id, class, IoU, confidence
    vector<pair<int, array<float,6>>> preds; // for mAP calculation
    vector<pair<int, array<float,5>>> gts; // for mAP calculation

    for (const auto &fileName : imageFiles)
    {
        Mat img = imread(fileName);
        string bboxPath = bboxFiles[idxInputImage];
        if (img.empty())
        {
            cout << "Failed to read image: " << fileName << endl;
            continue;
        }
        auto bboxes = yoloContext.infer_bboxes(make_pair(idxInputImage, img));
        for (const auto &bbox : bboxes)
        {
            // bbox = {class, x_center, y_center, width, height, confidence} //もしかしたらconfidenceの定義違うかも
            array<float,6> pred = {bbox[4], bbox[0], bbox[1], bbox[2], bbox[3], bbox[6+bbox[4]]};
            // cout << "pred = {class, x_center, y_center, width, height, confidence} = "
            //      << pred[0] << ", " << pred[1] << ", " << pred[2] << ", "
            //      << pred[3] << ", " << pred[4] << ", " << pred[5] << endl;
            preds.emplace_back(idxInputImage, pred);
        }
        // calc iou between predicted bboxes and ground truth bboxes
        auto gt_bboxes = readGroundTruth(bboxPath);
        if (!gt_bboxes.empty())
        {
            for (const auto &bbox : gt_bboxes)
            {
                // bbox = {class, x_center, y_center, width, height, confidence}
                array<float,5> gt = {bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]};
                // cout << "GT bbox = {class, x_center, y_center, width, height} = "
                //      << gt[0] << ", " << gt[1] << ", " << gt[2] << ", "
                //      << gt[3] << ", " << gt[4] << endl;
                gts.emplace_back(idxInputImage, gt);
            }
        }
        idxInputImage++;
    }

    int num_classes = 3;
    float map50 = calculate_mAP50_multi_image(preds, gts, num_classes);
    cout << "mAP@0.5 = " << map50 << endl;

    return 0;
}
