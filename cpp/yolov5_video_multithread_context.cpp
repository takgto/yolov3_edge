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
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

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
bool bInfer = true; // flag to infer input frame

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


typedef tuple<int, Mat, Mat> imageTuple; // tuple of index, resized image, original image
class tuplecomp
{
public:
    bool operator()(const imageTuple &n1, const imageTuple &n2) const
    {
        if (get<0>(n1) == get<0>(n2))
        {
            return (get<0>(n1) > get<0>(n2));
        }

        return get<0>(n1) > get<0>(n2);
    }
};

typedef tuple<int, Mat*, vector<int8_t>*> imageTupleInfer; // tuple of index, input image, output vector
class tupleInfercomp
{
public:
    bool operator()(const imageTupleInfer &n1, const imageTupleInfer &n2) const
    {
        if (get<0>(n1) == get<0>(n2))
        {
            return (get<0>(n1) > get<0>(n2));
        }

        return get<0>(n1) > get<0>(n2);
    }
};

queue<imagePair> queueInput; // queue of FIFO
queue<imageTuple> queueInputTuple; // queue of FIFO for tuple
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow; // priority queue by index comp.
// priority_queue<imageTuple, vector<imageTupleInfer>, tupleInfercomp> queueShowTuple; // priority queue by index comp. for tuple

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

        while (idxInputImage < maxFrame)
        {
            auto t0 = chrono::high_resolution_clock::now();
            Mat img;
            if (!video.read(img))
            {
                break;
            }
            auto pair = make_pair(idxInputImage, img);
            // end_time = chrono::system_clock::now();

            auto t1 = chrono::high_resolution_clock::now();
            auto read_dur = duration_cast<microseconds>(t1 - t0).count();
            auto read_start = (duration_cast<microseconds>(t1 - program_start)).count();
            logger.logRow("readFrame", {pair.first, read_start, read_dur});
            // avoid queue waiting
            out.push(pair);
            ++idxInputImage;
        }
        video.release();
    }
    out.push(make_pair(-1, Mat())); // push a sentinel value to indicate end of stream
    bReading = false; // set to false when reading is done
}

void quantize_u8c3_to_i8c3_neon_optimized(const Mat& in, Mat& out);

static int fromTo[] = {0,2 , 1, 1, 2, 0}; // BGR to RGB

void readAndPreprocess(const char *fileName, concurrent_queue<imageTuple> &out) {
    static int loop = 1; // video end of three times play

    start_time = chrono::system_clock::now();

    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    // 初期化
    avformat_network_init();
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, fileName, nullptr, nullptr) < 0) {
        std::cerr << "Failed to open input: " << fileName << std::endl;
        return;
    }
    avformat_find_stream_info(fmt_ctx, nullptr);
    int vid_stream = av_find_best_stream(fmt_ctx,
                                         AVMEDIA_TYPE_VIDEO,
                                         -1, -1, nullptr, 0);
    AVCodecParameters* p = fmt_ctx->streams[vid_stream]->codecpar;
    AVCodec* dec = avcodec_find_decoder(p->codec_id);
    AVCodecContext* dec_ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(dec_ctx, p);
    // マルチスレッドデコード
    dec_ctx->thread_count = 2;
    dec_ctx->thread_type  = FF_THREAD_FRAME;
    avcodec_open2(dec_ctx, dec, nullptr);

    // 変換コンテキスト
    SwsContext* sws_ctx = sws_getContext(
        dec_ctx->width, dec_ctx->height, dec_ctx->pix_fmt,
        dec_ctx->width, dec_ctx->height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    // 入出力フレーム／パケット
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgbf = av_frame_alloc();
    int rgb_buf_size = av_image_get_buffer_size(
        AV_PIX_FMT_BGR24,
        dec_ctx->width, dec_ctx->height, 1);
    std::vector<uint8_t> rgbbuf(rgb_buf_size);
    av_image_fill_arrays(rgbf->data, rgbf->linesize,
                         rgbbuf.data(), AV_PIX_FMT_BGR24,
                         dec_ctx->width, dec_ctx->height, 1);

    // OpenCV 出力 Mat
    cv::Mat image2(Size(width, height), CV_8UC3);
    cv::Mat rgb(Size(width, height), CV_8UC3);
    int idxInputImage = 0;


    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == vid_stream) {
            avcodec_send_packet(dec_ctx, pkt);
            while (avcodec_receive_frame(dec_ctx, frame) == 0) {
                auto t0 = chrono::high_resolution_clock::now();
                Mat image(Size(dec_ctx->width, dec_ctx->height), CV_8UC3);
                // YUV -> BGR24
                sws_scale(sws_ctx,
                          frame->data, frame->linesize, 0,
                          dec_ctx->height,
                          rgbf->data, rgbf->linesize);
                memcpy(image.data, rgbbuf.data(), rgb_buf_size);
                auto t1 = chrono::high_resolution_clock::now();
                // リサイズ・チャンネル変換・量子化
                cv::resize(image, image2,
                           cv::Size(width, height),
                           0,0, cv::INTER_NEAREST);
                auto t2 = chrono::high_resolution_clock::now();
                cv::mixChannels(&image2, 1, &rgb, 1, fromTo, 3);
                auto t3 = chrono::high_resolution_clock::now();
                Mat quantized(Size(width, height), CV_8SC3);
                quantize_u8c3_to_i8c3_neon_optimized(rgb, quantized);
                auto t4 = chrono::high_resolution_clock::now();

                // キューにプッシュ
                out.push(std::make_tuple(idxInputImage++,
                                         std::move(quantized),
                                         image));
                auto t5 = chrono::high_resolution_clock::now();

                auto read_dur = duration_cast<microseconds>(t1 - t0).count();
                // auto grab_dur = duration_cast<microseconds>(t0_1 - t0).count();
                // auto retrieve_dur = duration_cast<microseconds>(t1 - t0_1).count();
                auto resize_dur = duration_cast<microseconds>(t2 - t1).count();
                auto bgr2rgb_dur = duration_cast<microseconds>(t3 - t2).count();
                auto quantize_dur = duration_cast<microseconds>(t4 - t3).count();
                auto push_dur = duration_cast<microseconds>(t5 - t4).count();

                auto read_start = (duration_cast<microseconds>(t0 - program_start)).count();
                logger.logRow("0_readFrame", {idxInputImage, read_start, read_dur});
                logger.logRow("1_resize", {idxInputImage, read_start + read_dur, resize_dur});
                logger.logRow("2_bgr2rgb", {idxInputImage, read_start + read_dur + resize_dur, bgr2rgb_dur});
                logger.logRow("3_quantize", {idxInputImage, read_start + read_dur + resize_dur + bgr2rgb_dur, quantize_dur});
                logger.logRow("4_push", {idxInputImage, read_start + read_dur + resize_dur + bgr2rgb_dur + quantize_dur, push_dur});
            }
        }
        av_packet_unref(pkt);
    }
    // push a sentinel value to indicate end of stream
    out.push(make_tuple(-1, Mat(), Mat()));
    out.push(make_tuple(-1, Mat(), Mat()));
    bReading = false; // set to false when reading is done 
    
    // クリーンアップ
    av_frame_free(&frame);
    av_frame_free(&rgbf);
    av_packet_free(&pkt);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);
    avformat_network_deinit();
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

    while (bInfer || !in.empty())
    {
        auto pairIndexImg = in.pop();
        auto t0 = chrono::high_resolution_clock::now();
        frame = pairIndexImg.second;
        index = pairIndexImg.first;
        if (index == -1 || frame.rows <= 0 || frame.cols <= 0)
        {
            break;
        }

        auto show_time = chrono::system_clock::now();
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        double fps = index / (dura * 1e-6);
        std::snprintf(buf, sizeof(buf), "%.1f FPS", fps);
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
        auto t1 = chrono::high_resolution_clock::now();
        auto display_dur = duration_cast<microseconds>(t1 - t0).count();
        auto display_start = (duration_cast<microseconds>(t0 - program_start)).count();
        logger.logRow("11_displayFrame", {index, display_start, display_dur});
    }
    cv::destroyAllWindows();
}

void postProcess_OpenCV(int &idxInputImage, const Mat &frame, const vector<int8_t *> &out, 
                        const GraphInfo &shapes, const float &scale, const int &sHeight, const int &sWidth)
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
    logger.logRow("8_detect", {idxInputImage, post_start, post_detect});
    logger.logRow("9_nms", {idxInputImage, post_start + post_detect, nms_dur});
    logger.logRow("10_draw", {idxInputImage, post_start + post_detect + nms_dur, draw_dur});
}

void quantize_u8c3_to_i8c3_neon_optimized(const Mat& in, Mat& out)
{
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
        while (remaining >= 32)
        {

            uint8x16_t vin1 = vld1q_u8(in_ptr);
            uint8x16_t vin2 = vld1q_u8(in_ptr + 16);
            uint8x16_t vq1 = vshrq_n_u8(vin1, 1);
            uint8x16_t vq2 = vshrq_n_u8(vin2, 1);

            vst1q_s8(out_ptr,         vreinterpretq_s8_u8(vq1));
            vst1q_s8(out_ptr + 16,    vreinterpretq_s8_u8(vq2));

            // ポインタを 32 バイトだけ進める
            in_ptr  += 32;
            out_ptr += 32;
            remaining -= 32;
        }
        if (remaining >= 16)
        {
            uint8x16_t vin = vld1q_u8(in_ptr);
            uint8x16_t vq  = vshrq_n_u8(vin, 1);
            vst1q_s8(out_ptr, vreinterpretq_s8_u8(vq));

            in_ptr  += 16;
            out_ptr += 16;
            remaining -= 16;
        }
        for (int i = 0; i < remaining; i++)
        {
            // 1ビット右シフトして 0..127 に量子化
            *out_ptr++ = static_cast<int8_t>((*in_ptr++) >> 1);
        }
    }
}


// OpenCV version of setInputPointer function
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
    image2    .create(Size(width,height), CV_8UC3);
    rgb       .create(Size(width, height), CV_8UC3);
    quantized .create(Size(width,height), CV_8SC3);

    image = frame.clone();
    cv::resize(image, image2, Size(width, height), 0, 0, cv::INTER_NEAREST);
    mixChannels(&image2, 1, &rgb, 1, fromTo, 3); // BGR to RGB conversion

    // 3) scale を反映しつつ int8 へ量子化, padding
    // rgb.convertTo(quantized, CV_8S, input_scale / 128.0, 0); // 128 ? 256 ?
    quantize_u8c3_to_i8c3_neon_optimized(rgb, quantized);
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
        postProcess_OpenCV(index, frame, results, shapes, conf_output_scale,
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

    Mat inferNoPreProc(const imageTuple it)
    {
        auto t0 = chrono::high_resolution_clock::now();
        int index = get<0>(it);
        Mat processedImage = get<1>(it); // get the processed image
        Mat frame = get<2>(it); // get the original image
        std::memcpy(imageInputs, processedImage.data, inSize);
        auto t1 = chrono::high_resolution_clock::now();
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        auto t2 = chrono::high_resolution_clock::now();
        runner->wait(job_id.first, -1);
        auto t3 = chrono::high_resolution_clock::now();
        // ポスト処理へ
        std::vector<int8_t *> results = {result0, result1, result2};
        postProcess_OpenCV(index, frame, results, shapes, conf_output_scale,
                                            shapes.inTensorList[0].height, shapes.inTensorList[0].width);
        auto t4 = chrono::high_resolution_clock::now();

        auto memcpy_dur = duration_cast<microseconds>(t1 - t0).count();
        auto execute_dur = duration_cast<microseconds>(t2 - t1).count();
        auto wait_dur = duration_cast<microseconds>(t3 - t2).count();
        auto memcpy_start = (duration_cast<microseconds>(t0 - program_start)).count();
        auto execute_start = (duration_cast<microseconds>(t1 - program_start)).count();
        auto wait_start = (duration_cast<microseconds>(t2 - program_start)).count();
        logger.logRow("5_memcpy", {index, memcpy_start, memcpy_dur});
        logger.logRow("6_exec_async", {index, execute_start, execute_dur});
        logger.logRow("7_wait", {index, wait_start, wait_dur});
        
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

void runYOLO(vart::Runner *runner, concurrent_queue<imageTuple> &in, concurrent_queue<imagePair> &out)
{
    static YoloContext yoloContext(runner, shapes);
    while(bReading || !in.empty()){
        imageTuple tupleIndexImage = in.pop();
        int index = get<0>(tupleIndexImage);
        if (index == -1)
        {
            // cout << "End of video stream." << endl; // debug
            break;
        }
        out.push(make_pair(index, yoloContext.inferNoPreProc(tupleIndexImage)));     
    }
    out.push(make_pair(-1, Mat())); // push a dummy frame to signal end of processing
    bInfer = false; // set to false when reading is done
}

int main(const int argc, const char **argv)
{
    program_start = chrono::high_resolution_clock::now();
    cout << "concurrency = " << std::thread::hardware_concurrency() << std::endl;
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
    
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes; // get output size
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    concurrent_queue<imageTuple> fr(20);
    concurrent_queue<imagePair> shw(20); // display queue

    array<thread, 4> threadsList = {
        thread(readAndPreprocess, argv[2], ref(fr)),
        thread(displayFrame, ref(shw)),
        thread(runYOLO, runner.get(), ref(fr), ref(shw)),
        thread(runYOLO, runner1.get(), ref(fr), ref(shw)),
    };
    // imshow("YOLOv3 Detection@Xilinx DPU", Mat::zeros(416, 416, CV_8UC3));
    // waitKey(1); // wait for 1 ms to show the initial frame

    for (auto &t : threadsList)
    {
        if (t.joinable())
            t.join();
    }

    return 0;
}
