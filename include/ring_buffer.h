#pragma once
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

struct LayerCapture {
    int   layer_idx  = -1;
    int   token_idx  = -1;
    float latency_ms = 0.0f;
    float sparsity   = 0.0f;
    float logit_max  = 0.0f;
    float logit_min  = 0.0f;
    char  name[64]   = {};
    float attn[8][8] = {};
    bool  has_attn   = false;
    int   attn_n     = 0;
};

struct TokenBatch {
    int   token_idx     = -1;
    char  token_str[32] = {};
    std::vector<LayerCapture> layers;
    bool  complete = false;
};

struct TokenHistory {
    static constexpr int CAP = 64;

    TokenBatch batches[CAP];
    int count     = 0;
    int write_idx = 0;
    std::mutex mtx;

    void begin_token(int token_idx, const char* token_str) {
        std::lock_guard<std::mutex> lock(mtx);
        int slot = write_idx % CAP;
        batches[slot] = TokenBatch{};
        batches[slot].token_idx = token_idx;
        snprintf(batches[slot].token_str,
                 sizeof(batches[slot].token_str), "%s", token_str);
        batches[slot].complete = false;
        write_idx++;
        count = std::min(count + 1, CAP);
    }

    void push_layer(const LayerCapture& c) {
        std::lock_guard<std::mutex> lock(mtx);
        if (count == 0) return;
        int slot = (write_idx - 1) % CAP;
        batches[slot].layers.push_back(c);
    }

    void end_token() {
        std::lock_guard<std::mutex> lock(mtx);
        if (count == 0) return;
        int slot = (write_idx - 1) % CAP;
        batches[slot].complete = true;
    }

    TokenBatch get(int i) {
        std::lock_guard<std::mutex> lock(mtx);
        return batches[i % CAP];
    }

    int size() {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
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