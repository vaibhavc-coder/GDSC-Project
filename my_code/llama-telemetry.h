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
    std::chrono::high_resolution_clock::time_point start_time;
    std::vector<uint8_t> host_buffer;
    std::unique_ptr<std::thread> ui_thread;
    ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
    std::vector<SafeLayerMetrics> capture_history;
    std::vector<std::string> sequence_entries;
    std::vector<std::string> anomaly_entries;
public:
    TelemetryEngine() {
        ui_thread = std::make_unique<std::thread>(&TelemetryEngine::run_ui_loop, this);
    }
    ~TelemetryEngine() {
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
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    void record_metrics(struct ggml_tensor* t) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        size_t nbytes = ggml_nbytes(t);
        if (host_buffer.size() < nbytes) host_buffer.resize(nbytes);
        ggml_backend_tensor_get(t, host_buffer.data(), 0, nbytes);

        int zero_count = 0;
        float max_val = 0.0f;
        int64_t numel = ggml_nelements(t);
        
        std::vector<int64_t> shape;
        for (int i = 0; i < ggml_n_dims(t); i++) shape.push_back(t->ne[i]);

        std::vector<float> sample(64, 0.0f);

        if (t->type == GGML_TYPE_F32) {
            const float* data = reinterpret_cast<const float*>(host_buffer.data());
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] == 0.0f) zero_count++;
                if (std::abs(data[i]) > max_val) max_val = std::abs(data[i]);
                if (i < 64) sample[i] = std::abs(data[i]);
            }
        } 
        else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t* data = reinterpret_cast<const ggml_fp16_t*>(host_buffer.data());
            for (int64_t i = 0; i < numel; ++i) {
                float val = ggml_fp16_to_fp32(data[i]);
                if (val == 0.0f) zero_count++;
                if (std::abs(val) > max_val) max_val = std::abs(val);
                if (i < 64) sample[i] = std::abs(val);
            }
        }

        SafeLayerMetrics metrics = {
            t->name ? t->name : "unknown_layer",
            get_timestamp(),
            ggml_type_name(t->type),
            shape, latency,
            numel > 0 ? (float)zero_count / numel : 0.0f,
            max_val,
            (max_val > 10.0f || std::isnan(max_val)),
            sample
        };
        screen.Post([this, metrics]() {
            if (this->capture_history.size() >= MAX_HISTORY_SIZE) {
                this->capture_history.erase(this->capture_history.begin());
                this->sequence_entries.erase(this->sequence_entries.begin());
            }
            this->capture_history.push_back(metrics);
            std::stringstream entry;
            entry << "[" << std::fixed << std::setprecision(2) << metrics.latency_ms << "ms] " << metrics.name;
            this->sequence_entries.push_back(entry.str());

            if (metrics.is_anomaly) {
                this->anomaly_entries.push_back(metrics.timestamp + " | " + metrics.name + " (Max: " + std::to_string(metrics.max_act) + ")");
            }
        });
    }

private:
    ftxui::Element render_heatmap_block(float val, float max_val) {
        using namespace ftxui;
        if (max_val <= 0.0001f) return text("░░") | color(Color::GrayDark);
        float ratio = val / max_val;
        if (ratio > 0.8f) return text("██") | color(Color::Red);
        if (ratio > 0.5f) return text("▓▓") | color(Color::Orange1);
        if (ratio > 0.2f) return text("▒▒") | color(Color::Yellow);
        return text("░░") | color(Color::GrayDark);
    }

    void run_ui_loop() {
        using namespace ftxui;

        int selected_layer = 0;
        int selected_anomaly = 0;

        auto sequence_menu = Menu(&sequence_entries, &selected_layer);
        auto anomaly_menu = Menu(&anomaly_entries, &selected_anomaly);

        auto sequence_with_vim = CatchEvent(sequence_menu, [&](Event e) {
            if (e == Event::Character('j')) return sequence_menu->OnEvent(Event::ArrowDown);
            if (e == Event::Character('k')) return sequence_menu->OnEvent(Event::ArrowUp);
            return false;
        });

        auto main_container = Container::Horizontal({
            sequence_with_vim,
            anomaly_menu
        });

        auto dashboard = Renderer(main_container, [&] {
            auto list_win = window(text(" 1. PIPELINE SEQUENCE [j/k to scroll] ") | bold | color(Color::Cyan), 
                sequence_with_vim->Render() | vscroll_indicator | frame
            ) | size(WIDTH, PERCENT, 35);

            if (capture_history.empty()) {
                return hbox({list_win, center(text("Waiting for forward pass...")) | flex});
            }

            int idx = std::max(0, std::min(selected_layer, (int)capture_history.size() - 1));
            const auto& active = capture_history[idx];

            std::string shape_str = "[";
            for (size_t i = 0; i < active.shape.size(); i++) {
                shape_str += std::to_string(active.shape[i]) + (i < active.shape.size()-1 ? ", " : "]");
            }
            if (active.shape.empty()) shape_str = "[]";

            auto metrics_pane = window(text(" 2. RUNTIME METRICS ") | bold | color(Color::Cyan), vbox({
                text("Target    : " + active.name) | color(Color::White),
                text("Timestamp : " + active.timestamp) | color(Color::GrayLight),
                separator(),
                hbox({text("Shape     : "), text(shape_str) | color(Color::Green), text("  (" + active.dtype + ")")}),
                hbox({text("Latency   : "), text(std::to_string(active.latency_ms) + " ms") | color(active.latency_ms > 50.0 ? Color::Red : Color::Green)}),
                hbox({text("Sparsity  : "), gauge(active.sparsity) | color(Color::Blue) | size(WIDTH, EQUAL, 20), text(" " + std::to_string((int)(active.sparsity * 100)) + "%")}),
                text("Max Activ : " + std::to_string(active.max_act))
            }));

            Elements matrix_rows;
            for (int r = 0; r < 8; r++) {
                Elements col;
                for (int c = 0; c < 8; c++) {
                    int s_idx = r * 8 + c;
                    col.push_back(render_heatmap_block(active.tensor_sample[s_idx], active.max_act));
                }
                matrix_rows.push_back(hbox(col));
            }
            auto matrix_pane = window(text(" 3. 8x8 TENSOR HEATMAP ") | bold | color(Color::Cyan), 
                hbox({ vbox(matrix_rows) | flex, separator(), text("██ High\n▓▓ Mid\n▒▒ Low\n░░ Zero") | color(Color::GrayLight) })
            );

            auto anomaly_win = window(text(" 4. ANOMALY LEDGER ") | bold | color(Color::Cyan),
                anomaly_entries.empty() ? text("No anomalies detected.") | color(Color::GrayDark) 
                                        : anomaly_menu->Render() | vscroll_indicator | frame
            );

            auto right_pane = vbox({
                metrics_pane,
                matrix_pane,
                anomaly_win | flex
            });

            return vbox({
                text(" [Tab]: Switch Panels  |  [j/k]: Navigate  |  [q]: Quit ") | inverted,
                hbox({list_win, right_pane | flex})
            });
        });

        auto global_interceptor = CatchEvent(dashboard, [&](Event event) {
            if (event == Event::Character('q') || event == Event::Character('Q')) {
                screen.ExitLoopClosure()();
                return true;
            }
            return false;
        });

        screen.Loop(global_interceptor);
    }
};
