#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/component/event.hpp"

struct SafeLayerMetrics {
    std::string name;
    std::string timestamp;
    std::string dtype;
    std::vector<int64_t> shape;
    double latency_ms;
    float sparsity;
    float max_act;
    bool is_anomaly;
    std::array<float, 64> tensor_sample;
};

struct PendingCapture {
    std::string name;
    std::string timestamp;
    std::string dtype;
    std::vector<int64_t> shape;
    double latency_ms;
    ggml_type type;
    int64_t numel;
    std::vector<uint8_t> raw;
};

class TelemetryEngine {
private:
    static constexpr size_t MAX_HISTORY_SIZE = 256;
    static constexpr int64_t MAX_SCAN_ELEMENTS = 1 << 16;

    std::chrono::high_resolution_clock::time_point start_time;
    std::unique_ptr<std::thread> ui_thread;
    std::unique_ptr<std::thread> worker_thread;
    ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();

    std::vector<SafeLayerMetrics> capture_history;
    std::vector<std::string> sequence_entries;
    std::vector<std::string> anomaly_entries;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<PendingCapture> pending_queue;
    std::atomic<bool> shutdown_requested{false};

public:
    TelemetryEngine() {
        worker_thread = std::make_unique<std::thread>(&TelemetryEngine::run_worker_loop, this);
        ui_thread = std::make_unique<std::thread>(&TelemetryEngine::run_ui_loop, this);
    }

    ~TelemetryEngine() {
        shutdown_requested.store(true);
        queue_cv.notify_all();
        if (worker_thread && worker_thread->joinable()) {
            worker_thread->join();
        }
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

        std::string layer_name = t->name ? t->name : "unknown_layer";

        std::vector<int64_t> shape;
        for (int i = 0; i < ggml_n_dims(t); i++) shape.push_back(t->ne[i]);
        int64_t numel = ggml_nelements(t);

        bool is_heavy_layer = (layer_name.find("attn") != std::string::npos ||
                               layer_name.find("ffn") != std::string::npos ||
                               layer_name.find("wq") != std::string::npos);

        PendingCapture cap;
        cap.name = layer_name;
        cap.timestamp = get_timestamp();
        cap.dtype = ggml_type_name(t->type);
        cap.shape = shape;
        cap.latency_ms = latency;
        cap.type = t->type;
        cap.numel = numel;

        if (is_heavy_layer && numel > 0 && t->data != nullptr) {
            int64_t scan_n = std::min(numel, MAX_SCAN_ELEMENTS);
            size_t type_size = ggml_type_size(t->type);
            size_t bytes_to_copy = static_cast<size_t>(scan_n) * type_size;
            cap.raw.resize(bytes_to_copy);
            std::memcpy(cap.raw.data(), t->data, bytes_to_copy);
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            pending_queue.push_back(std::move(cap));
        }
        queue_cv.notify_one();
    }

private:
    void run_worker_loop() {
        while (true) {
            PendingCapture cap;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] {
                    return shutdown_requested.load() || !pending_queue.empty();
                });
                if (shutdown_requested.load() && pending_queue.empty()) return;
                cap = std::move(pending_queue.front());
                pending_queue.erase(pending_queue.begin());
            }

            SafeLayerMetrics metrics;
            metrics.name = cap.name;
            metrics.timestamp = cap.timestamp;
            metrics.dtype = cap.dtype;
            metrics.shape = cap.shape;
            metrics.latency_ms = cap.latency_ms;
            metrics.tensor_sample.fill(0.0f);

            int64_t zero_count = 0;
            float max_val = 0.0f;
            bool has_nan = false;
            int64_t scan_n = static_cast<int64_t>(
                cap.raw.empty() ? 0 : cap.raw.size() / ggml_type_size(cap.type));

            if (cap.type == GGML_TYPE_F32 && scan_n > 0) {
                const float* data = reinterpret_cast<const float*>(cap.raw.data());
                for (int64_t i = 0; i < scan_n; ++i) {
                    float v = data[i];
                    if (std::isnan(v)) has_nan = true;
                    if (v == 0.0f) zero_count++;
                    float av = std::abs(v);
                    if (av > max_val) max_val = av;
                    if (i < 64) metrics.tensor_sample[i] = av;
                }
            } else if (cap.type == GGML_TYPE_F16 && scan_n > 0) {
                const ggml_fp16_t* data = reinterpret_cast<const ggml_fp16_t*>(cap.raw.data());
                for (int64_t i = 0; i < scan_n; ++i) {
                    float v = ggml_fp16_to_fp32(data[i]);
                    if (std::isnan(v)) has_nan = true;
                    if (v == 0.0f) zero_count++;
                    float av = std::abs(v);
                    if (av > max_val) max_val = av;
                    if (i < 64) metrics.tensor_sample[i] = av;
                }
            }

            metrics.sparsity = scan_n > 0 ? (float)zero_count / (float)scan_n : 0.0f;
            metrics.max_act = max_val;
            metrics.is_anomaly = (max_val > 15.0f || has_nan);

            screen.Post([this, metrics]() {
                if (this->capture_history.size() >= MAX_HISTORY_SIZE) {
                    this->capture_history.erase(this->capture_history.begin());
                    this->sequence_entries.erase(this->sequence_entries.begin());
                }
                this->capture_history.push_back(metrics);

                std::stringstream entry;
                entry << "[" << std::fixed << std::setprecision(2) << metrics.latency_ms << "ms] " << metrics.name;
                this->sequence_entries.push_back(entry.str());

                if (metrics.is_anomaly && metrics.max_act > 0.0f) {
                    std::stringstream anomaly;
                    anomaly << metrics.timestamp << " | " << metrics.name
                            << " (Max: " << std::fixed << std::setprecision(3) << metrics.max_act << ")";
                    this->anomaly_entries.push_back(anomaly.str());
                }
            });
        }
    }

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
        int focused_panel = 0;
        int locked_layer = -1;

        auto sequence_menu = Menu(&sequence_entries, &selected_layer);
        auto anomaly_menu = Menu(&anomaly_entries, &selected_anomaly);

        auto sequence_with_keys = CatchEvent(sequence_menu, [&](Event e) {
            if (e == Event::Character('j')) return sequence_menu->OnEvent(Event::ArrowDown);
            if (e == Event::Character('k')) return sequence_menu->OnEvent(Event::ArrowUp);
            if (e == Event::Character(' ')) {
                if (!capture_history.empty()) {
                    locked_layer = (locked_layer == selected_layer) ? -1 : selected_layer;
                }
                return true;
            }
            return false;
        });

        auto anomaly_with_keys = CatchEvent(anomaly_menu, [&](Event e) {
            if (e == Event::Character('j')) return anomaly_menu->OnEvent(Event::ArrowDown);
            if (e == Event::Character('k')) return anomaly_menu->OnEvent(Event::ArrowUp);
            return false;
        });

        auto main_container = Container::Horizontal({
            sequence_with_keys,
            anomaly_with_keys
        });

        auto dashboard = Renderer(main_container, [&] {
            bool seq_focused = (focused_panel == 0);

            auto list_title = " 1. PIPELINE SEQUENCE [j/k to scroll, space to lock] ";
            auto list_win = window(
                text(list_title) | bold | color(seq_focused ? Color::Cyan : Color::GrayLight),
                sequence_with_keys->Render() | vscroll_indicator | frame
            ) | size(WIDTH, EQUAL, 38);

            if (capture_history.empty()) {
                return hbox({list_win, center(text("Waiting for token generation...")) | flex});
            }

            int view_idx = (locked_layer >= 0) ? locked_layer : selected_layer;
            int safe_idx = std::max(0, std::min(view_idx, (int)capture_history.size() - 1));
            const auto& active = capture_history[safe_idx];

            std::string shape_str = "[";
            for (size_t i = 0; i < active.shape.size(); i++) {
                shape_str += std::to_string(active.shape[i]) + (i < active.shape.size()-1 ? ", " : "]");
            }
            if (active.shape.empty()) shape_str = "[]";

            std::string lock_indicator = (locked_layer >= 0) ? " [LOCKED]" : " [LIVE]";

            auto metrics_pane = window(text(" 2. RUNTIME METRICS" + lock_indicator + " ") | bold | color(Color::Cyan), vbox({
                text("Target    : " + active.name) | color(Color::White),
                text("Timestamp : " + active.timestamp) | color(Color::GrayLight),
                separator(),
                hbox({text("Shape     : "), text(shape_str) | color(Color::Green), text("  (" + active.dtype + ")")}),
                hbox({text("Latency   : "), text(std::to_string(active.latency_ms) + " ms") | color(active.latency_ms > 10.0 ? Color::Red : Color::Green)}),
                hbox({text("Sparsity  : "), gauge(active.sparsity) | color(Color::Blue) | size(WIDTH, EQUAL, 20), text(" " + std::to_string((int)(active.sparsity * 100)) + "%")}),
                hbox({text("Max Activ : "), text([&]{ std::stringstream s; s << std::fixed << std::setprecision(4) << active.max_act; return s.str(); }())})
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

            auto matrix_pane = window(text(" 3. 8x8 TENSOR HEATMAP (Attn/FFN) ") | bold | color(Color::Cyan),
                hbox({ vbox(matrix_rows) | flex, separator(), text("██ High\n▓▓ Mid\n▒▒ Low\n░░ Zero") | color(Color::GrayLight) })
            );

            auto anomaly_title = " 4. ANOMALY LEDGER (>15.0f) [j/k to scroll] ";
            auto anomaly_win = window(
                text(anomaly_title) | bold | color(seq_focused ? Color::GrayLight : Color::Cyan),
                anomaly_entries.empty() ? text("No clipping risks detected.") | color(Color::GrayDark)
                                        : anomaly_with_keys->Render() | vscroll_indicator | frame
            );

            auto right_pane = vbox({
                metrics_pane,
                matrix_pane,
                anomaly_win | flex
            });

            return vbox({
                text(" [Tab]: Switch Panels  |  [j/k]: Navigate  |  [space]: Lock layer  |  [q]: Quit ") | inverted,
                hbox({list_win, right_pane | flex})
            });
        });

        auto global_interceptor = CatchEvent(dashboard, [&](Event event) {
            if (event == Event::Character('q') || event == Event::Character('Q')) {
                screen.ExitLoopClosure()();
                return true;
            }
            if (event == Event::Tab) {
                focused_panel = (focused_panel + 1) % 2;
                main_container->SetActiveChild(focused_panel == 0 ? sequence_with_keys : anomaly_with_keys);
                return true;
            }
            if (event == Event::TabReverse) {
                focused_panel = (focused_panel + 1) % 2;
                main_container->SetActiveChild(focused_panel == 0 ? sequence_with_keys : anomaly_with_keys);
                return true;
            }
            return false;
        });

        screen.Loop(global_interceptor);
    }
};

inline bool ggml_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* engine = static_cast<TelemetryEngine*>(user_data);

    if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
        t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE) {
        return true;
    }

    if (ask) {
        engine->start_timer();
    } else {
        engine->record_metrics(t);
    }
    return true;
}
