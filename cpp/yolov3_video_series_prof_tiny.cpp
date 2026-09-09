// YOLOv3-tiny を 1本のループで回す（read → pre → dpu → post → disp を直列に）
// 計測は演習4の方法：実行中は足し込むだけ、表示は全部終わってから1回だけ

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>

#include <xir/graph/graph.hpp>
#include "vitis/ai/collection_helper.hpp"
#include "common.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

bool Lbox_on = false;                      // true: letterbox で 416x416 にする / false: resize
GraphInfo shapes;

// =====================================================================
//  計測の道具（演習4-2 の Profiler）
//    lap()       : 起点から今までをこのフレーム用の器に足し、起点を進める
//    frame_end() : フレームが最後まで行ったときだけ、器の中身を本体に移す
//    report()    : 全部終わってから1回だけ表を出す
// =====================================================================
using TP = steady_clock::time_point;

class Profiler {
public:
    int stage(const char* name, bool ext) {          // ext = 外部（DPU・画面）にまかせている段
        st_.push_back({name, ext, 0, 0});
        cur_.push_back(0);
        return (int)st_.size() - 1;
    }
    void frame_start() { t_ = steady_clock::now(); }
    void lap(int id) {
        TP n = steady_clock::now();
        cur_[id] += duration_cast<microseconds>(n - t_).count();
        t_ = n;
    }
    void frame_end() {                                // 'q' で途中で抜けたフレームはここを通らないので捨てられる
        TP n = steady_clock::now();
        if (!warm_) {
            for (size_t i = 0; i < st_.size(); i++) {
                st_[i].sum += cur_[i];
                if (cur_[i] > st_[i].max) st_[i].max = cur_[i];
            }
            frames_++;
            tend_ = n;                                // 集計の終点 ＝ 最後のフレームの終わり
        } else if (++skipped_ >= WARMUP) {            // 立ち上がりの数フレームは捨てる
            warm_ = false; t0_ = n;
        }
        for (auto& c : cur_) c = 0;
    }
    void report() const {
        if (frames_ == 0) { cout << "\n(フレームが " << WARMUP << " 枚以下なので集計なし)\n"; return; }
        double frame_ms = duration_cast<microseconds>(tend_ - t0_).count() / 1000.0 / frames_;  // = 1000 / FPS
        double sum_ms = 0;
        for (const auto& s : st_) sum_ms += s.sum / 1000.0 / frames_;
        cout << fixed << setprecision(1)
             << "\n" << frames_ << " フレーム（先頭 " << WARMUP << " 枚は捨てた）   "
             << (1000.0 / frame_ms) << " FPS\n\n"
             << "    平均     最大   種類  段\n";
        for (const auto& s : st_)
            cout << setw(8) << (s.sum / 1000.0 / frames_) << "ms"
                 << setw(7) << (s.max / 1000.0) << "ms"
                 << (s.ext ? "   外部  " : "   CPU   ") << s.name << "\n";
        cout << setw(8) << (frame_ms - sum_ms) << "ms"
             << setw(9) << "-" << "         計測外\n\n"
             << "    1フレーム " << frame_ms << "ms（= 1000 / FPS）− 各段の合計 "
             << sum_ms << "ms = 計測外 " << (frame_ms - sum_ms) << "ms\n";
    }
private:
    struct S { const char* name; bool ext; long long sum, max; };
    static const int WARMUP = 3;
    vector<S> st_;
    vector<long long> cur_;
    int frames_ = 0, skipped_ = 0;
    bool warm_ = true;
    TP t_ = steady_clock::now(), t0_ = steady_clock::now(), tend_ = steady_clock::now();
};

Profiler prof;
int P_READ, P_PRE, P_DPU, P_POST, P_DISP;              // 段の番号（main で登録）

// =====================================================================
//  YOLO 本体
// =====================================================================
Mat post_process(const Mat& frame, const vector<int8_t*>& out, const GraphInfo& shapes,
                 const float& scale, const int& sHeight, const int& sWidth) {
    auto img = frame.clone();
    vector<vector<float>> boxes;
    for (size_t i = 0; i < out.size(); i++) {
        int channel = shapes.outTensorList[i].channel;
        int width   = shapes.outTensorList[i].width;
        int height  = shapes.outTensorList[i].height;
        int sizeOut = shapes.outTensorList[i].size;
        boxes.reserve(sizeOut);
        detect_tiny(boxes, out[i], channel, height, width, i, sHeight, sWidth, scale);
    }
    if (Lbox_on) {
        correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight);
    }
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = img.rows;
    float w = img.cols;
    for (size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;
        if (res[i][res[i][4] + 6] > CONF) {           // res[i][4] = クラス番号、+6 でそのクラスのスコア
            int type = res[i][4];
            Scalar color = (type == 0) ? Scalar(0, 0, 255)
                         : (type == 1) ? Scalar(255, 0, 0)
                                       : Scalar(0, 255, 255);
            rectangle(img, Point(xmin, ymin), Point(xmax, ymax), color, 1, 1, 0);
        }
    }
    return img;
}

void setInputImageForYOLO(const Mat& frame, int8_t* data, float input_scale) {
    int width  = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size   = shapes.inTensorList[0].size;
    image img_new  = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for (int b = 0; b < height; ++b)
        for (int c = 0; c < width; ++c)
            for (int a = 0; a < 3; ++a)
                bb[b * width * 3 + c * 3 + a] = img_yolo.data[a * height * width + b * width + c];

    float scale = pow(2, 7);
    for (int i = 0; i < size; ++i) {
        data[i] = (int8_t)(bb.data()[i] * input_scale);
        if (data[i] < 0) data[i] = (int8_t)((float)(127 / scale) * input_scale);
    }
    free_image(img_new);
    free_image(img_yolo);
}

void setInputPointer(const Mat& frame, int8_t* data, const int& scale) {
    int width  = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size   = shapes.inTensorList[0].size;

    Mat img = frame.clone();
    cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat image2 = cv::Mat(height, width, CV_8SC3);
    cv::resize(img, image2, Size(width, height), 0, 0, cv::INTER_LINEAR);

    unsigned char* imdata = image2.data;
    for (int i = 0; i < size; ++i) {
        float dataf = static_cast<float>(imdata[i]);
        data[i] = static_cast<int>(dataf * static_cast<float>(scale) / 256.0);
        if (data[i] < 0) data[i] = 127;
    }
}

// 1フレームぶん： pre → dpu → post。段の切れ目で prof.lap() を呼ぶ
Mat runYOLO(vart::Runner* runner, const Mat& frame) {
    auto inputTensors  = cloneTensorBuffer(runner->get_input_tensors());
    auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());

    int inHeight  = shapes.inTensorList[0].height;
    int inWidth   = shapes.inTensorList[0].width;
    int inChannel = 3;
    int batchSize = 1;
    int inSize = inHeight * inWidth * inChannel;
    int8_t* imageInputs = new int8_t[inSize * batchSize];

    vector<int> output_mapping = shapes.output_mapping;
    auto conf_output_scale = get_output_scale(runner->get_output_tensors()[output_mapping[1]]);

    const int size0 = shapes.outTensorList[0].size;
    const int size1 = shapes.outTensorList[1].size;
    int8_t* result0 = new int8_t[size0 * batchSize];
    int8_t* result1 = new int8_t[size1 * batchSize];

    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

    // ---- pre: 416x416 にして int8 に量子化 ----
    if (Lbox_on) setInputImageForYOLO(frame, imageInputs, input_scale);
    else         setInputPointer(frame, imageInputs, input_scale);

    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs, inputTensors[0].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result0, outputTensors[output_mapping[0]].get()));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result1, outputTensors[output_mapping[1]].get()));
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());
    outputsPtr.push_back(outputs[1].get());
    prof.lap(P_PRE);

    // ---- dpu: 投げて、終わるのを待つ ----
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    prof.lap(P_DPU);

    // ---- post: 検出 → NMS → 枠を描く ----
    vector<int8_t*> results = {result0, result1};
    auto img = post_process(frame, results, shapes, conf_output_scale, inHeight, inWidth);
    prof.lap(P_POST);

    delete[] imageInputs;
    delete[] result0;
    delete[] result1;
    return img;
}

int main(const int argc, const char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " [model_file] [video_file]" << endl;
        return -1;
    }
    auto xmodel_file = std::string(argv[1]);

    auto graph = xir::Graph::deserialize(xmodel_file);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u) << "yolov3 should have one and only one dpu subgraph." << endl;
    cout << "create running for subgraph: " << subgraph[0]->get_name() << endl;

    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    auto inputTensors  = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    int inputCnt  = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList  = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    VideoCapture video;
    if (!video.open(argv[2])) {
        cout << "Fail to open specified video file:" << argv[2] << endl;
        return -1;
    }

    // 段の登録（表に出る順）
    P_READ = prof.stage("read (video.read)",           false);
    P_PRE  = prof.stage("pre  (resize / quantize)",    false);
    P_DPU  = prof.stage("dpu  (execute_async + wait)", true);
    P_POST = prof.stage("post (detect / NMS / draw)",  false);
    P_DISP = prof.stage("disp (imshow / waitKey)",     true);

    int index = 0;
    auto loop_start = steady_clock::now();
    while (true) {
        prof.frame_start();

        // ---- read ----
        Mat img;
        if (!video.read(img)) break;
        prof.lap(P_READ);

        // ---- pre / dpu / post（中で lap を呼ぶ）----
        Mat frame = runYOLO(runner.get(), img);

        // ---- disp: FPS を書き込んで表示 ----
        double sec = duration_cast<microseconds>(steady_clock::now() - loop_start).count() / 1e6;
        stringstream buffer;
        buffer << fixed << setprecision(1) << (index / sec) << " FPS";
        putText(frame, buffer.str(), cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
        imshow("YOLOv3 Detection@Xilinx DPU", frame);
        if (waitKey(1) == 'q') break;                 // ここで抜けたフレームは集計に入らない
        index++;
        prof.lap(P_DISP);

        prof.frame_end();
    }
    double loop_sec = duration_cast<microseconds>(steady_clock::now() - loop_start).count() / 1e6;

    // ---- 表示は全部終わってから1回だけ（release() の前に）----
    cout << fixed << setprecision(1)
         << "\n===== 全体: " << index << " フレーム / " << loop_sec << " 秒 = "
         << (index / loop_sec) << " FPS =====\n";
    prof.report();

    video.release();
    return 0;
}
