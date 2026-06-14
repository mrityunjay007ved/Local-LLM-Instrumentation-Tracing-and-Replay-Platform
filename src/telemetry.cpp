#include "llama.h"
#include "ring_buffer.h"
#include "ggml.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cmath>

RingBuffer    g_rb;
TokenHistory  g_history;
AnomalyReport g_anomaly;
std::atomic<bool> g_running     = true;
std::atomic<int>  g_token_count = 0;

struct EvalCallbackData {
    RingBuffer*   rb;
    TokenHistory* history;
    std::chrono::time_point<std::chrono::high_resolution_clock> layer_start;
    int last_layer_idx = -99;
};

static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (!t || !t->name || t->name[0] == '\0') return true;

    std::string name(t->name);
    if (name.find("(view)")     != std::string::npos) return true;
    if (name.find("(permuted)") != std::string::npos) return true;
    if (name.find("(cont)")     != std::string::npos) return true;
    if (name.find("(reshaped)") != std::string::npos) return true;

    bool is_layer = name.find("l_out")       != std::string::npos;
    bool is_kqsm  = name.find("kq_soft_max") != std::string::npos;
    if (!is_layer && !is_kqsm) return true;

    auto* data = (EvalCallbackData*)user_data;
    auto  now  = std::chrono::high_resolution_clock::now();
    float ms   = std::chrono::duration<float, std::milli>(
                     now - data->layer_start).count();

    int layer_idx = -1;
    sscanf(t->name, "%*[^-]-%d", &layer_idx);

    // deduplicate
    if (is_layer) {
        if (layer_idx == data->last_layer_idx) {
            data->layer_start = now;
            return true;
        }
        data->last_layer_idx = layer_idx;
    }

    if (is_kqsm) {
        LayerCapture c;
        c.layer_idx  = layer_idx;
        c.latency_ms = ms;
        c.sparsity   = 0.0f;
        c.logit_max  = 0.0f;
        c.logit_min  = 0.0f;
        c.has_attn   = false;
        c.attn_n     = 0;
        snprintf(c.name, sizeof(c.name), "kq_soft_max-%d", layer_idx);
        if (t->data && t->buffer &&
            ggml_backend_buffer_is_host(t->buffer) &&
            t->type == GGML_TYPE_F32) {
            int seq    = (int)t->ne[0];
            int n      = std::min(seq, 8);
            c.attn_n   = n;
            c.has_attn = true;
            float* fdata = (float*)t->data;
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                    c.attn[row][col] = fdata[row * seq + col];
        }
        data->rb->push(c);
        data->layer_start = now;
        return true;
    }

    float sparsity = 0.0f;
    if (t->data && t->buffer &&
        ggml_backend_buffer_is_host(t->buffer) &&
        t->type == GGML_TYPE_F32) {
        float*  fdata  = (float*)t->data;
        int64_t total  = ggml_nelements(t);
        int64_t sample = std::min(total, (int64_t)1024);
        int     zeros  = 0;
        for (int64_t i = 0; i < sample; i++)
            if (fabsf(fdata[i]) < 1e-6f) zeros++;
        sparsity = (float)zeros / (float)sample;
    }

    LayerCapture c;
    c.layer_idx  = layer_idx;
    c.latency_ms = ms;
    c.sparsity   = sparsity;
    c.logit_max  = 0.0f;
    c.logit_min  = 0.0f;
    c.has_attn   = false;
    c.attn_n     = 0;
    snprintf(c.name, sizeof(c.name), "%s", t->name);
    data->rb->push(c);
    data->history->push_layer(c);
    data->layer_start = now;
    return true;
}

static void run_inference(const std::string& model_path,
                          const std::string& prompt) {
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model* model = llama_model_load_from_file(
                             model_path.c_str(), model_params);
    if (!model) { g_running = false; return; }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(
        vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                   prompt_tokens.data(), n_prompt, true, true);

    EvalCallbackData cb_data;
    cb_data.rb          = &g_rb;
    cb_data.history     = &g_history;
    cb_data.layer_start = std::chrono::high_resolution_clock::now();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = n_prompt + 128;
    ctx_params.n_batch           = n_prompt;
    ctx_params.no_perf           = false;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.cb_eval           = eval_callback;
    ctx_params.cb_eval_user_data = &cb_data;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { g_running = false; return; }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(
                            prompt_tokens.data(), prompt_tokens.size());
    int n_vocab = llama_vocab_n_tokens(vocab);
    char token_buf[32];

    for (int n_pos = 0;
         n_pos + batch.n_tokens < n_prompt + 64 && g_running; ) {

        int current_token_idx = g_token_count.load();
        llama_token cur_tok = batch.token[0];
        int n = llama_token_to_piece(vocab, cur_tok,
                                     token_buf, sizeof(token_buf), 0, true);
        if (n < 0) snprintf(token_buf, sizeof(token_buf), "?");
        else token_buf[n] = '\0';

        g_history.begin_token(current_token_idx, token_buf);
        cb_data.layer_start = std::chrono::high_resolution_clock::now();
        cb_data.last_layer_idx = -99;  // reset per token

        if (llama_decode(ctx, batch)) break;

        g_history.end_token();

        float* logits   = llama_get_logits(ctx);
        float logit_max = *std::max_element(logits, logits + n_vocab);
        float logit_min = *std::min_element(logits, logits + n_vocab);
        (void)logit_max; (void)logit_min;

        n_pos += batch.n_tokens;
        g_token_count++;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    g_anomaly.compute(g_history);
    g_running = false;
}

static void run_tui() {
    using namespace ftxui;
    auto screen = ScreenInteractive::Fullscreen();

    int  selected_token = 0;
    int  selected_layer = 0;
    bool token_view     = true;

    std::thread refresher([&] {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            screen.PostEvent(Event::Custom);
        }
        screen.PostEvent(Event::Custom);
    });

    auto renderer = Renderer([&] {
        int n_tokens = g_history.size();
        if (n_tokens > 0 && selected_token >= n_tokens)
            selected_token = n_tokens - 1;

        // panel 1: token list with scrolling window
        Elements token_rows;
        int tok_window = 10;
        int tok_start  = std::max(0, selected_token - tok_window / 2);
        if (tok_start + tok_window > n_tokens)
            tok_start = std::max(0, n_tokens - tok_window);
        for (int i = tok_start;
             i < std::min(n_tokens, tok_start + tok_window); i++) {
            TokenBatch tb = g_history.get(i);
            std::string line =
                "tok." + std::to_string(tb.token_idx) +
                " [" + std::string(tb.token_str) + "]" +
                " → " + std::to_string(tb.layers.size()) + " layers" +
                (tb.complete ? "" : " ...");
            if (token_view && i == selected_token)
                token_rows.push_back(text(line) | inverted);
            else
                token_rows.push_back(text(line));
        }
        if (token_rows.empty())
            token_rows.push_back(text("waiting for tokens..."));
        auto token_panel = window(
            text(" 1. TOKEN HISTORY  [j/k] select  [Enter] drill in "),
            vbox(token_rows)
        );

        // panel 2: layers for selected token
        Elements layer_rows;
        if (n_tokens > 0 && selected_token < n_tokens) {
            TokenBatch tb = g_history.get(selected_token);
            int n_layers = (int)tb.layers.size();
            if (selected_layer >= n_layers && n_layers > 0)
                selected_layer = n_layers - 1;
            for (int i = 0; i < n_layers; i++) {
                LayerCapture& c = tb.layers[i];
                std::string cname(c.name);
                if (cname.find("kq_soft_max") != std::string::npos) continue;
                std::string line =
                    cname +
                    " | " + std::to_string((int)c.latency_ms) + "ms" +
                    " | " + std::to_string((int)(c.sparsity * 100)) + "%";
                if (!token_view && i == selected_layer)
                    layer_rows.push_back(text(line) | inverted);
                else
                    layer_rows.push_back(text(line));
            }
        }
        if (layer_rows.empty())
            layer_rows.push_back(text("select a token first"));
        auto layer_panel = window(
            text(" 2. LAYERS  [j/k] select  [Esc] back "),
            vbox(layer_rows)
        );

        // panel 3: metrics
        Elements metric_rows;
        if (n_tokens > 0 && selected_token < n_tokens) {
            TokenBatch tb = g_history.get(selected_token);
            if (!tb.layers.empty() &&
                selected_layer < (int)tb.layers.size()) {
                LayerCapture& sel = tb.layers[selected_layer];
                metric_rows = {
                    text("Token   : [" + std::string(tb.token_str) + "]"),
                    text("Layer   : " + std::string(sel.name)),
                    text("Latency : " + std::to_string(sel.latency_ms) + " ms"),
                    hbox({
                        text("Sparsity: "),
                        gauge(sel.sparsity) | flex,
                        text(" " + std::to_string((int)(sel.sparsity*100)) + "%"),
                    }),
                };
            }
        }
        if (metric_rows.empty()) metric_rows = { text("waiting...") };
        auto metrics = window(text(" 3. RUNTIME METRICS "), vbox(metric_rows));

        // panel 4: anomaly report
        Elements anomaly_rows;
        if (g_anomaly.ready) {
            anomaly_rows.push_back(
                text("mean: " + std::to_string((int)g_anomaly.mean_latency) + "ms")
            );
            anomaly_rows.push_back(separator());
            for (auto& s : g_anomaly.stats) {
                std::string flag;
                if (s.is_slow)   flag += " ⚠SLOW";
                if (s.is_sparse) flag += " ⚠SPARSE";
                std::string line =
                    std::string(s.name) +
                    " | " + std::to_string((int)s.avg_latency) + "ms" + flag;
                if (s.is_slow || s.is_sparse)
                    anomaly_rows.push_back(text(line) | bold);
                else
                    anomaly_rows.push_back(text(line));
            }
        } else {
            anomaly_rows.push_back(text("computing after"));
            anomaly_rows.push_back(text("inference ends..."));
        }
        auto anomaly_panel = window(
            text(" 4. ANOMALY REPORT "),
            vbox(anomaly_rows) | frame | size(HEIGHT, LESS_THAN, 15)
        );

        return vbox({
            text(" TRANSFORMER TELEMETRY  |  [j/k] nav  |  [Enter] drill  |  [Esc] back  |  [Q] quit ")
                | center,
            separator(),
            hbox({
                vbox({
                    token_panel | size(HEIGHT, LESS_THAN, 12),
                    layer_panel | size(HEIGHT, LESS_THAN, 12),
                    metrics,
                }) | flex,
                vbox({
                    window(
                        text(" STATUS "),
                        vbox({
                            text("Tokens : " + std::to_string(g_token_count.load())),
                            text("Hist   : " + std::to_string(n_tokens) + "/" +
                                 std::to_string(TokenHistory::CAP)),
                            text(g_running.load() ? "● running" : "■ done"),
                            separator(),
                            text(token_view ? "token nav" : "layer nav"),
                            text("sel_tok: " + std::to_string(selected_token)),
                            text("sel_lyr: " + std::to_string(selected_layer)),
                        })
                    ) | size(HEIGHT, LESS_THAN, 10),
                    anomaly_panel | flex,
                }) | size(WIDTH, LESS_THAN, 30),
            }),
        });
    });

    auto component = CatchEvent(renderer, [&](Event event) {
        int n_tokens = g_history.size();
        if (event == Event::Character('j')) {
            if (token_view) {
                selected_token = std::min(selected_token + 1,
                                          std::max(0, n_tokens - 1));
            } else {
                if (n_tokens > 0 && selected_token < n_tokens) {
                    TokenBatch tb = g_history.get(selected_token);
                    selected_layer = std::min(selected_layer + 1,
                        std::max(0, (int)tb.layers.size() - 1));
                }
            }
            return true;
        }
        if (event == Event::Character('k')) {
            if (token_view)
                selected_token = std::max(selected_token - 1, 0);
            else
                selected_layer = std::max(selected_layer - 1, 0);
            return true;
        }
        if (event == Event::Return) {
            token_view     = false;
            selected_layer = 0;
            return true;
        }
        if (event == Event::Escape) {
            token_view = true;
            return true;
        }
        if (event == Event::Character('q') ||
            event == Event::Character('Q')) {
            g_running = false;
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    screen.Loop(component);
    refresher.join();
}

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 2) {
        printf("usage: %s model.gguf [prompt]\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    std::string prompt     = argc > 2 ? argv[2] : "Hello my name is";
    std::thread inference_thread(run_inference, model_path, prompt);
    run_tui();
    inference_thread.join();
    return 0;
}