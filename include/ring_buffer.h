#pragma once
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

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
    int64_t shape[4] = {0, 0, 0, 0};
    int     n_dims   = 0;

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

struct LayerStats {
    int   layer_idx    = -1;
    char  name[64]     = {};
    float avg_latency  = 0.0f;
    float avg_sparsity = 0.0f;
    float max_latency  = 0.0f;
    float min_latency  = 0.0f;
    int   sample_count = 0;
    bool  is_slow      = false;
    bool  is_sparse    = false;

};

struct AnomalyReport {
    std::vector<LayerStats> stats;
    float mean_latency  = 0.0f;
    float mean_sparsity = 0.0f;
    bool  ready         = false;

    void compute(TokenHistory& history) {
        std::map<int, LayerStats> acc;

        int n_tokens = history.size();
        for (int t = 0; t < n_tokens; t++) {
            TokenBatch tb = history.get(t);
            for (auto& c : tb.layers) {
                int idx = c.layer_idx;
                if (idx < 0) continue;
                auto& s = acc[idx];
                s.layer_idx = idx;
                if (s.sample_count == 0)  // only set name on first capture
                snprintf(s.name, sizeof(s.name), "%s", c.name);
                    s.avg_latency  += c.latency_ms;
                s.avg_sparsity += c.sparsity;
                s.max_latency   = std::max(s.max_latency, c.latency_ms);
                if (s.min_latency == 0.0f)
                    s.min_latency = c.latency_ms;
                else
                    s.min_latency = std::min(s.min_latency, c.latency_ms);
                s.sample_count++;
            }
        }

        for (auto& [idx, s] : acc) {
            if (s.sample_count > 0) {
                s.avg_latency  /= s.sample_count;
                s.avg_sparsity /= s.sample_count;
            }
            stats.push_back(s);
        }

        std::sort(stats.begin(), stats.end(),
            [](const LayerStats& a, const LayerStats& b) {
                return a.layer_idx < b.layer_idx;
            });

        if (!stats.empty()) {
            for (auto& s : stats) {
                mean_latency  += s.avg_latency;
                mean_sparsity += s.avg_sparsity;
            }
            mean_latency  /= stats.size();
            mean_sparsity /= stats.size();
        }

        for (auto& s : stats) {
            s.is_slow   = s.avg_latency  > mean_latency  * 2.0f;
            s.is_sparse = mean_sparsity > 0.01f && s.avg_sparsity > mean_sparsity * 2.0f;
        }

        ready = true;
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