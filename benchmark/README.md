# ベンチマークの使い方
cppファイルを適切に編集して, 以下のコマンドを../cppで実行することで, ベンチマークを実行できます.

詳細な編集方法は下のセクションを参照してください.

```bash
./test_fps.sh <cppファイル> <xmodelファイル> <skip_compile>
```
## 例
```bash
./test_fps.sh yolov3_video_study_bench.cpp dpu_yolov3.xmodel 1
```
skip_compileを1にすると, コンパイルをスキップして実行します. 0にすると, ../cpp/make.shを実行してコンパイルします.
../cpp/video以下にある3つの動画に対して, 推論を行なってfpsや各処理の時間を計測します.

## 注意点
今は, 各処理の段階名をハードコーディングしています. そのため, cppファイルを編集してbenchmarkを実行できるようにいじる必要があります.
またbenchmark.hをインクルードする必要があります.

## 処理段階名とおおよその対応
- readFrame : readFrame関数の処理
- setInputPointer : setInput**関数の処理
- pre_process : exec_asyncの前の残りの処理: 
- exec_async : runner->exec_asyncの処理
- wait : runner->waitの処理
- post_process : postProcess関数の処理
- displayFrame : displayFrame関数の処理

今後は, 処理名をうまく取得して, ハードコーディングしないようにしたいです.

# ベンチマークの結果
ベンチマークの結果は, /test以下に保存されます. 保存名は, `<unix_time>_test/`です.
- mext.txt : cppファイル名やxmodelファイル名, 各動画に対するfps
- <動画名>_result.csv : 動画ごとの処理時間のcsvファイル
- statistics.csv : 各動画に対する処理時間のmean, std, median, max, minのcsvファイル
- gantt_chart_frame.png : traffic1.webmに対して最初の10フレームの処理時間をgantt chartにしたもの
- gantt_chart_thread.png : traffic1.webmに対して最初の10フレームの処理時間をスレッドごとにgantt chartにしたもの
- box_plot.png : 各動画に対する処理時間のboxplot

# ベンチマークのためのcppファイルの編集方法
cppファイルを編集して, ベンチマークを実行できるようにするための手順は以下の通りです.

まずglobal変数を定義します.
```cpp
# include "benchmark.h"
// Global Logger
// bench_tmp.csvは./test_fps.shで使うためなので, その必要がなければ適宜変更してください.
static CSVLogger logger("bench_tmp.csv", "tid,func,frame,start,latency");
static std::chrono::high_resolution_clock::time_point program_start;
int maxFrame = 100; // maximum number of frames to process, recommended to set to 100 or more
```
次に, main関数の最初に以下のコードを追加します.
```cpp
int main() {
    program_start = std::chrono::high_resolution_clock::now();
    ...
}
```
次に, 各処理段階の前後に以下のコードを追加します.また, frameIndexは, 処理するフレームのインデックスを表す変数で, 0から始まる整数です. 例えば, main関数の中でindexという変数を使っている場合は, indexをframeIndexとして使います.
multi threadingをしている場合は, frameIndexをtaskのidのように扱っていると思うので, そのidを使います.
```cpp
// 処理段階の前
auto start = std::chrono::high_resolution_clock::now();
// 処理段階の後
auto end = std::chrono::high_resolution_clock::now();

// 適当な場所で
long long func_dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
long long func_start = std::chrono::duration_cast<std::chrono::microseconds>(start - program_start).count();
logger.logRow("<function_name>", { <frameIndex>, func_start, func_dur });
```

また簡易的な処理速度の確認として,ScopTimerを使うこともできます. これは関数のスコープ内で自動的に時間を計測し, 終了時にログを標準出力に出力します.
```cpp
void some_function() {
    ScopeTimer timer("function_name");
    // 処理
}
```

最後にちょっと大変ですが, maxFrameの値になった時に適切に処理を終了するようにします.
yolov3_video_study_bench.cppやyolov3_video_subject1_bench.cppを参考にしてください.
```cpp
// subject1_bench.cppの例
int main() {
    program_start = std::chrono::high_resolution_clock::now();
    ...
    int index = 0;
    while (index < maxFrame) {
        ...
        index++;
    }
    return 0;
}
```

