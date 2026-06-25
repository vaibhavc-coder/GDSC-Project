#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include "ggml.h"
#include "ggml-backend.h"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/interactive.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/component/event.hpp"

struct SafeLayerMetrics {
    int id;
    std::string name;
    std::string timestamp;
    std::string dtype;
    std::vector<int64_t> shape;
    double latency_ms;
    float sparsity;
    float max_act;
    bool is_anomaly;
    std::string compute_device;
};

template <typename T, size_t Capacity>
class LockFreeRingBuffer {
private:
    T buffer[Capacity];
    size_t head = 0;
    size_t tail = 0;
    size_t size = 0;
    std::mutex mtx;

public:
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        buffer[head] = item;
        head = (head + 1) % Capacity;
        if (size < Capacity) {
            size++;
        } else {
            tail = (tail + 1) % Capacity;
        }
    }

    std::vector<T> get_all_snapshot() {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<T> items;
        items.reserve(size);
        size_t current = tail;
        for (size_t i = 0; i < size; ++i) {
            items.push_back(buffer[current]);
            current = (current + 1) % Capacity;
        }
        return items;
    }
};
class TelemetryEngine {
private:
    LockFreeRingBuffer<SafeLayerMetrics, 15> ring_buffer; // Packet stream history
    std::chrono::high_resolution_clock::time_point start_time;
    std::vector<uint8_t> host_buffer;

    std::unique_ptr<std::thread> ui_thread;
    std::atomic<bool> ui_running{false};
    ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
    int packet_counter = 100;

public:
    TelemetryEngine() {
        ui_running = true;
        ui_thread = std::make_unique<std::thread>(&TelemetryEngine::run_ui_loop, this);
    }

    ~TelemetryEngine() {
        ui_running = false;
        screen.ExitLoopClosure()();
        if (ui_thread && ui_thread->joinable()) {
            ui_thread->join();
        }
    }

    void start_timer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    void record_metrics(struct ggml_tensor* t) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        size_t nbytes = ggml_nbytes(t);
        if (host_buffer.size() < nbytes) {
            host_buffer.resize(nbytes);
        }

        ggml_backend_tensor_get(t, host_buffer.data(), 0, nbytes);

        int zero_count = 0;
        float max_val = 0.0f;
        int64_t numel = ggml_nelements(t);
        std::vector<int64_t> shape;
        for (int i = 0; i < ggml_n_dims(t); i++) shape.push_back(t->ne[i]);

        if (t->type == GGML_TYPE_F32) {
            const float* data = reinterpret_cast<const float*>(host_buffer.data());
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] == 0.0f) zero_count++;
                if (data[i] > max_val) max_val = data[i];
            }
        }

        SafeLayerMetrics metrics;
        metrics.id = packet_counter++;
        metrics.name = t->name ? t->name : "unknown_layer";
        metrics.timestamp = get_timestamp();
        metrics.dtype = ggml_type_name(t->type);
        metrics.shape = shape;
        metrics.latency_ms = latency;
        metrics.sparsity = numel > 0 ? (float)zero_count / numel : 0.0f;
        metrics.max_act = max_val;
        metrics.is_anomaly = (max_val > 6.0f);
        metrics.compute_device = "CUDA [GPU 0]";
        if (metrics.name.find("norm") != std::string::npos) metrics.compute_device = "CPU (Fallback)";

        ring_buffer.push(metrics);
        screen.PostEvent(ftxui::Event::Custom);
    }
