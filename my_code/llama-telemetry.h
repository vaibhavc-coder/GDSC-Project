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
