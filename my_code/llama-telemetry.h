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

struct LayerSnapshot {
    std::string name;
    std::string timestamp;
    std::string dtype;
    std::vector<int64_t> shape;
    double latency_ms;
    float sparsity;
    float max_act;
    bool is_anomaly;
    std::array<float, 64> sample;
};
struct RawCapture {
    std::string name;
    std::string timestamp;
    std::string dtype;
    std::vector<int64_t> shape;
    double latency_ms;
    ggml_type type;
    int64_t numel;
    std::vector<uint8_t> bytes;
};

class TelemetryEngine {
private:
    static constexpr size_t HISTORY_CAP = 256;
    static constexpr int64_t SCAN_CAP = 1 << 16;

    std::chrono::high_resolution_clock::time_point t_start;
    std::unique_ptr<std::thread> ui_thread;
    std::unique_ptr<std::thread> crunch_thread;
    ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();

    std::vector<LayerSnapshot> history;
    std::vector<std::string> layer_list;
    std::vector<std::string> anomaly_list;

    std::mutex inbox_mutex;
    std::condition_variable inbox_cv;
    std::vector<RawCapture> inbox;
    std::atomic<bool> stopping{false};

public:
    TelemetryEngine() {
        crunch_thread = std::make_unique<std::thread>(&TelemetryEngine::crunch_loop, this);
        ui_thread = std::make_unique<std::thread>(&TelemetryEngine::ui_loop, this);
    }

    ~TelemetryEngine() {
        stopping.store(true);
        inbox_cv.notify_all();
        if (crunch_thread && crunch_thread->joinable()) crunch_thread->join();

        screen.ExitLoopClosure()();
        if (ui_thread && ui_thread->joinable()) ui_thread->join();
    }

    void start_timer() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    std::string now_str() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        auto t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    void capture(struct ggml_tensor* t) {
        double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();

        std::string name = t->name ? t->name : "unnamed_layer";

        std::vector<int64_t> shape;
        for (int i = 0; i < ggml_n_dims(t); i++) shape.push_back(t->ne[i]);
        int64_t numel = ggml_nelements(t);

        bool worth_watching = name.find("attn") != std::string::npos ||
                              name.find("ffn") != std::string::npos ||
                              name.find("wq") != std::string::npos;

        RawCapture raw;
        raw.name = name;
        raw.timestamp = now_str();
        raw.dtype = ggml_type_name(t->type);
        raw.shape = shape;
        raw.latency_ms = elapsed;
        raw.type = t->type;
        raw.numel = numel;

        if (worth_watching && numel > 0 && t->data != nullptr) {
            int64_t n = std::min(numel, SCAN_CAP);
            size_t nbytes = static_cast<size_t>(n) * ggml_type_size(t->type);
            raw.bytes.resize(nbytes);
            std::memcpy(raw.bytes.data(), t->data, nbytes);
        }

        {
            std::lock_guard<std::mutex> lock(inbox_mutex);
            inbox.push_back(std::move(raw));
        }
        inbox_cv.notify_one();
    }

private:
    static float read_as_float(const uint8_t* bytes, int64_t i, ggml_type type) {
        if (type == GGML_TYPE_F32) {
            return reinterpret_cast<const float*>(bytes)[i];
        }
        return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t*>(bytes)[i]);
    }
    void crunch_loop() {
        while (true) {
            RawCapture raw;
            {
                std::unique_lock<std::mutex> lock(inbox_mutex);
                inbox_cv.wait(lock, [this] { return stopping.load() || !inbox.empty(); });
                if (stopping.load() && inbox.empty()) return;
                raw = std::move(inbox.front());
                inbox.erase(inbox.begin());
            }

            LayerSnapshot snap;
            snap.name = raw.name;
            snap.timestamp = raw.timestamp;
            snap.dtype = raw.dtype;
            snap.shape = raw.shape;
            snap.latency_ms = raw.latency_ms;
            snap.sample.fill(0.0f);

            int64_t zeros = 0;
            float peak = 0.0f;
            bool saw_nan = false;

            bool scannable = (raw.type == GGML_TYPE_F32 || raw.type == GGML_TYPE_F16) && !raw.bytes.empty();
            int64_t n = scannable ? static_cast<int64_t>(raw.bytes.size() / ggml_type_size(raw.type)) : 0;

            for (int64_t i = 0; i < n; ++i) {
                float v = read_as_float(raw.bytes.data(), i, raw.type);
                if (std::isnan(v)) { saw_nan = true; continue; }
                if (v == 0.0f) zeros++;
                float av = std::abs(v);
                if (av > peak) peak = av;
                if (i < 64) snap.sample[i] = av;
            }

            snap.sparsity = n > 0 ? (float)zeros / (float)n : 0.0f;
            snap.max_act = peak;
            snap.is_anomaly = (peak > 15.0f || saw_nan);

            screen.Post([this, snap]() {
                if (history.size() >= HISTORY_CAP) {
                    history.erase(history.begin());
                    layer_list.erase(layer_list.begin());
                }
                history.push_back(snap);

                std::stringstream row;
                row << "[" << std::fixed << std::setprecision(2) << snap.latency_ms << "ms] " << snap.name;
                layer_list.push_back(row.str());

                if (snap.is_anomaly && snap.max_act > 0.0f) {
                    std::stringstream alert;
                    alert << snap.timestamp << " | " << snap.name
                          << " (Max: " << std::fixed << std::setprecision(3) << snap.max_act << ")";
                    anomaly_list.push_back(alert.str());
                }
            });
        }
    }

    ftxui::Element heat_cell(float val, float peak) {
        using namespace ftxui;
        if (peak <= 0.0001f) return text("░░") | color(Color::GrayDark);
        float ratio = val / peak;
        if (ratio > 0.8f) return text("██") | color(Color::Red);
        if (ratio > 0.5f) return text("▓▓") | color(Color::Orange1);
        if (ratio > 0.2f) return text("▒▒") | color(Color::Yellow);
        return text("░░") | color(Color::GrayDark);
    }

    void ui_loop() {
        using namespace ftxui;

        int cursor = 0;
        int anomaly_cursor = 0;
        int focused_panel = 0;  
        int pinned_layer = -1;   

        auto layer_menu = Menu(&layer_list, &cursor);
        auto anomaly_menu = Menu(&anomaly_list, &anomaly_cursor);

        auto layer_panel = CatchEvent(layer_menu, [&](Event e) {
            if (e == Event::Character('j')) return layer_menu->OnEvent(Event::ArrowDown);
            if (e == Event::Character('k')) return layer_menu->OnEvent(Event::ArrowUp);
            if (e == Event::Character(' ')) {
                if (!history.empty()) {
                    pinned_layer = (pinned_layer == cursor) ? -1 : cursor;
                }
                return true;
            }
            return false;
        });

        auto anomaly_panel = CatchEvent(anomaly_menu, [&](Event e) {
            if (e == Event::Character('j')) return anomaly_menu->OnEvent(Event::ArrowDown);
            if (e == Event::Character('k')) return anomaly_menu->OnEvent(Event::ArrowUp);
            return false;
        });

        auto root = Container::Horizontal({ layer_panel, anomaly_panel });

        auto dashboard = Renderer(root, [&] {
            bool on_layers = (focused_panel == 0);

            auto list_win = window(
                text(" 1. PIPELINE SEQUENCE [j/k to scroll, space to lock] ") | bold | color(on_layers ? Color::Cyan : Color::GrayLight),
                layer_panel->Render() | vscroll_indicator | frame
            ) | size(WIDTH, EQUAL, 38);

            if (history.empty()) {
                return hbox({list_win, center(text("Waiting for token generation...")) | flex});
            }

            int idx = (pinned_layer >= 0) ? pinned_layer : cursor;
            idx = std::max(0, std::min(idx, (int)history.size() - 1));
            const auto& view = history[idx];

            std::string shape_str = "[";
            for (size_t i = 0; i < view.shape.size(); i++) {
                shape_str += std::to_string(view.shape[i]) + (i < view.shape.size() - 1 ? ", " : "]");
            }
            if (view.shape.empty()) shape_str = "[]";

            std::string pin_tag = (pinned_layer >= 0) ? " [LOCKED]" : " [LIVE]";

            auto metrics_win = window(text(" 2. RUNTIME METRICS" + pin_tag + " ") | bold | color(Color::Cyan), vbox({
                text("Target    : " + view.name) | color(Color::White),
                text("Timestamp : " + view.timestamp) | color(Color::GrayLight),
                separator(),
                hbox({text("Shape     : "), text(shape_str) | color(Color::Green), text("  (" + view.dtype + ")")}),
                hbox({text("Latency   : "), text(std::to_string(view.latency_ms) + " ms") | color(view.latency_ms > 10.0 ? Color::Red : Color::Green)}),
                hbox({text("Sparsity  : "), gauge(view.sparsity) | color(Color::Blue) | size(WIDTH, EQUAL, 20), text(" " + std::to_string((int)(view.sparsity * 100)) + "%")}),
                hbox({text("Max Activ : "), text([&]{ std::stringstream s; s << std::fixed << std::setprecision(4) << view.max_act; return s.str(); }())})
            }));

            Elements rows;
            for (int r = 0; r < 8; r++) {
                Elements row;
                for (int c = 0; c < 8; c++) {
                    row.push_back(heat_cell(view.sample[r * 8 + c], view.max_act));
                }
                rows.push_back(hbox(row));
            }

            auto heatmap_win = window(text(" 3. 8x8 TENSOR HEATMAP (Attn/FFN) ") | bold | color(Color::Cyan),
                hbox({ vbox(rows) | flex, separator(), text("██ High\n▓▓ Mid\n▒▒ Low\n░░ Zero") | color(Color::GrayLight) })
            );

            auto anomaly_win = window(
                text(" 4. ANOMALY LEDGER (>15.0f) [j/k to scroll] ") | bold | color(on_layers ? Color::GrayLight : Color::Cyan),
                anomaly_list.empty() ? text("No clipping risks detected.") | color(Color::GrayDark)
                                     : anomaly_panel->Render() | vscroll_indicator | frame
            );

            auto right_side = vbox({ metrics_win, heatmap_win, anomaly_win | flex });

            return vbox({
                text(" [Tab]: Switch Panels  |  [j/k]: Navigate  |  [space]: Lock layer  |  [q]: Quit ") | inverted,
                hbox({list_win, right_side | flex})
            });
        });

        auto with_global_keys = CatchEvent(dashboard, [&](Event event) {
            if (event == Event::Character('q') || event == Event::Character('Q')) {
                screen.ExitLoopClosure()();
                return true;
            }
            if (event == Event::Tab || event == Event::TabReverse) {
                focused_panel = (focused_panel + 1) % 2;
                root->SetActiveChild(focused_panel == 0 ? layer_panel : anomaly_panel);
                return true;
            }
            return false;
        });

        screen.Loop(with_global_keys);
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
        engine->capture(t);
    }
    return true;
}
