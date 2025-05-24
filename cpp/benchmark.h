#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <thread>
#include <functional>
#include <sstream>

/**
 * @brief RAII タイマークラス
 * スコープ開始時に時刻を取得し、スコープ終了時に経過時間をマイクロ秒単位で出力します。
 * 出力にはスレッドIDを付加します。
 */
class ScopeTimer {
public:
    explicit ScopeTimer(const std::string& name)
        : name_(name),
          start_(std::chrono::high_resolution_clock::now()),
          tid_(std::this_thread::get_id())
    {}

    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::ostringstream oss;
        oss << "[thread " << tid_ << "] " << name_ << " took " << us << " µs";
        std::cout << oss.str() << std::endl;
    }

    // コピー・ムーブ禁止
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    std::thread::id tid_;
    static std::mutex output_mutex_;
};

std::mutex ScopeTimer::output_mutex_;

/**
 * @brief スレッドセーフな CSV ロガークラス
 * 各ログ行にスレッドIDとオプションで関数名を付与できます。
 */
class CSVLogger {
public:
    /**
     * @param path    出力ファイルパス
     * @param header  CSV のヘッダー行（カンマ区切り）
     */
    CSVLogger(const std::string& path, const std::string& header)
    {
        ofs_.open(path, std::ofstream::out | std::ofstream::trunc);
        if (ofs_.is_open()) {
            ofs_ << header << '\n';
        } else {
            std::lock_guard<std::mutex> lock(mtx_);
            std::cerr << "Failed to open CSV file: " << path << std::endl;
        }
    }

    /**
     * @brief 単純リストをカンマ区切りで出力 (スレッドIDなし)
     */
    void log(const std::initializer_list<long long>& values) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!ofs_.is_open()) return;
        bool first = true;
        for (auto v : values) {
            if (!first) ofs_ << ',';
            ofs_ << v;
            first = false;
        }
        ofs_ << '\n';
    }

    /**
     * @brief スレッドIDと関数名を先頭に付けて出力
     * @param funcName 関数名やラベル
     * @param values   ログする値のリスト
     */
    void logRow(const std::string& funcName, const std::initializer_list<long long>& values) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!ofs_.is_open()) return;
        // スレッドIDをハッシュ化して出力
        auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ofs_ << tid << ',' << funcName;
        for (auto v : values) {
            ofs_ << ',' << v;
        }
        ofs_ << '\n';
    }

private:
    std::ofstream ofs_;
    std::mutex mtx_;
};

#endif // BENCHMARK_H
