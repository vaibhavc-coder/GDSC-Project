#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <queue>
#include <string>
#include <chrono>
#include <mutex>
#include <cmath>

#include "ggml.h"
#include "ggml-backend.h"

struct LayerMetrics {
    std::string layer_name;
    std::vector<int64_t> shape;
    double latency_ms;
    float sparsity;
    float mean_act;
    float max_act;
    int64_t timestamp;

    bool operator>(const LayerMetrics& other) const {
        return latency_ms > other.latency_ms;
    }
};

class TelemetryEngine {
private:
    std::deque<LayerMetrics> ring_buffer;
    size_t max_buffer_size;
    std::priority_queue<LayerMetrics, std::vector<LayerMetrics>, std::greater<LayerMetrics>> top_k_latencies;
    size_t k_dominators = 5;

    std::mutex mtx;
    std::chrono::high_resolution_clock::time_point start_time;
    std::vector<uint8_t> host_buffer;

public:
    TelemetryEngine(size_t buffer_size = 1000) : max_buffer_size(buffer_size) {}

    void start_timer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void record_metrics(struct ggml_tensor* t) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::vector<int64_t> shape;
        for (int i = 0; i < ggml_n_dims(t); ++i) {
            shape.push_back(t->ne[i]);
        }

        size_t nbytes = ggml_nbytes(t);
        if (host_buffer.size() < nbytes) {
            host_buffer.resize(nbytes);
        }
        
        ggml_backend_tensor_get(t, host_buffer.data(), 0, nbytes);

        int zero_count = 0;
        float max_val = -INFINITY;
        double sum = 0.0;
        int64_t numel = ggml_nelements(t);

        if (t->type == GGML_TYPE_F32) {
            const float* data = reinterpret_cast<const float*>(host_buffer.data());
            for (int64_t i = 0; i < numel; ++i) {
                float val = data[i];
                if (val == 0.0f) zero_count++;
                if (val > max_val) max_val = val;
                sum += val;
            }
        }

        LayerMetrics metrics = {
            t->name ? t->name : "unknown_layer",
            shape,
            latency,
            numel > 0 ? (float)zero_count / numel : 0.0f,
            numel > 0 ? (float)(sum / numel) : 0.0f,
            max_val,
            std::chrono::system_clock::now().time_since_epoch().count()
        };

        std::lock_guard<std::mutex> lock(mtx);
        
        ring_buffer.push_back(metrics);
        if (ring_buffer.size() > max_buffer_size) {
            ring_buffer.pop_front();
        }

        top_k_latencies.push(metrics);
        if (top_k_latencies.size() > k_dominators) {
            top_k_latencies.pop(); 
        }
    }

    // Prints a formatted TUI-style report to the console
    void print_report() {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "\n=========================================================\n";
        std::cout << "      LLM TELEMETRY REPORT: TOP " << k_dominators << " SLOWEST LAYERS\n";
        std::cout << "=========================================================\n";
        
        // The priority queue doesn't allow standard iteration, so we copy and pop
        auto temp_pq = top_k_latencies;
        std::vector<LayerMetrics> sorted_metrics;
        while (!temp_pq.empty()) {
            sorted_metrics.push_back(temp_pq.top());
            temp_pq.pop();
        }
        
        // Print in descending order (Slowest at the top)
        for (auto it = sorted_metrics.rbegin(); it != sorted_metrics.rend(); ++it) {
            std::cout << "\033[1;33m[Layer]\033[0m " << it->layer_name << "\n";
            std::cout << "  -> Latency   : " << it->latency_ms << " ms\n";
            std::cout << "  -> Sparsity  : " << (it->sparsity * 100.0f) << " %\n";
            std::cout << "  -> Mean Act  : " << it->mean_act << "\n";
            std::cout << "  -> Max Act   : " << it->max_act << "\n";
            std::cout << "---------------------------------------------------------\n";
        }
    }

    // The Destructor: Automatically triggers when the program closes
    ~TelemetryEngine() {
        print_report();
    }
};

static bool ggml_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* engine = static_cast<TelemetryEngine*>(user_data);
    
    if (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE) {
        return true; 
    }

    if (ask) {
        engine->start_timer();
        return true; 
    } else {
        engine->record_metrics(t);
        return true;
    }
}