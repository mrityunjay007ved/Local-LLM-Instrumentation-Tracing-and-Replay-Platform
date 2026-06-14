#pragma once
#include <mutex>
#include <string>
#include <algorithm>

struct LayerCapture {
    int   layer_idx  = -1;
    float latency_ms = 0.0f;
    float sparsity   = 0.0f;
    float logit_max  = 0.0f;
    float logit_min  = 0.0f;
    char  name[64]   = {};
    float attn[8][8] = {};
    bool  has_attn   = false;
    int   attn_n     = 0;
};

struct RingBuffer {
    static constexpr int CAP = 512;

    LayerCapture buf[CAP];
    int write_idx = 0;
    int count     = 0;
    std::mutex mtx;

    void push(const LayerCapture& c) {
        std::lock_guard<std::mutex> lock(mtx);
        buf[write_idx % CAP] = c;
        write_idx++;
        count = std::min(count + 1, CAP);
    }

    LayerCapture get(int i) {
        std::lock_guard<std::mutex> lock(mtx);
        return buf[i % CAP];
    }

    int size() {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
};