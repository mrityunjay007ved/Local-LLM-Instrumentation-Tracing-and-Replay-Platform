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
        c.n_dims     = 0;
        snprintf(c.name, sizeof(c.name), "kq_soft_max-%d", layer_idx);
        if (t->data && t->buffer &&
            ggml_backend_buffer_is_host(t->buffer) &&
            t->type == GGML_TYPE_F32) {
            int seq  = (int)t->ne[0];
            int n    = std::min(seq, 8);
            c.attn_n   = n;
            c.has_attn = true;
            float* fdata = (float*)t->data;
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                    c.attn[row][col] = fdata[row * seq + col];
        }
        data->rb->push(c);
        data->history->push_layer(c);
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

    // capture shape
    c.n_dims = 0;
    for (int d = 0; d < 4; d++) {
        c.shape[d] = t->ne[d];
        if (t->ne[d] > 1) c.n_dims = d + 1;
    }
    if (c.n_dims == 0) c.n_dims = 1;

    data->rb->push(c);
    data->history->push_layer(c);
    data->layer_start = now;
    return true;
}

static void run_inference(const std::string& model_path,
                          const std::string& prompt) {
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
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
        cb_data.last_layer_idx = -99;

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

    int selected_token = 0;
    int selected_layer = 0;
    int focus          = 0;

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

        // ── panel 1: token list ───────────────────────
        Elements token_rows;
        int tok_window = 10;
        int tok_start  = std::max(0, selected_token - tok_window / 2);
        if (tok_start + tok_window > n_tokens)
            tok_start = std::max(0, n_tokens - tok_window);
        for (int i = tok_start;
             i < std::min(n_tokens, tok_start + tok_window); i++) {
            TokenBatch tb = g_history.get(i);
            int real_count = 0;
            for (auto& cap : tb.layers) {
                std::string cn(cap.name);
                if (cn.find("kq_soft_max") == std::string::npos)
                    real_count++;
            }
            std::string line =
                "tok." + std::to_string(tb.token_idx) +
                " [" + std::string(tb.token_str) + "]" +
                " → " + std::to_string(real_count) + " layers" +
                (tb.complete ? "" : " ...");
            if (focus == 0 && i == selected_token)
                token_rows.push_back(text(line) | inverted);
            else
                token_rows.push_back(text(line));
        }
        if (token_rows.empty())
            token_rows.push_back(text("waiting for tokens..."));
        auto token_panel = window(
            text(focus == 0 ? " 1. TOKEN HISTORY [active] "
                            : " 1. TOKEN HISTORY "),
            vbox(token_rows)
        );

        // ── panel 2: layer list ───────────────────────
        Elements layer_rows;
        std::vector<int> visible_indices;
        if (n_tokens > 0 && selected_token < n_tokens) {
            TokenBatch tb = g_history.get(selected_token);
            for (int i = 0; i < (int)tb.layers.size(); i++) {
                std::string cname(tb.layers[i].name);
                if (cname.find("kq_soft_max") == std::string::npos)
                    visible_indices.push_back(i);
            }
            int n_visible = (int)visible_indices.size();
            if (selected_layer >= n_visible && n_visible > 0)
                selected_layer = n_visible - 1;

            int lyr_window = 10;
            int lyr_start  = std::max(0, selected_layer - lyr_window / 2);
            if (lyr_start + lyr_window > n_visible)
                lyr_start = std::max(0, n_visible - lyr_window);

            for (int vi = lyr_start;
                 vi < std::min(n_visible, lyr_start + lyr_window); vi++) {
                int i = visible_indices[vi];
                LayerCapture& c = tb.layers[i];
                std::string line =
                    std::string(c.name) +
                    " | " + std::to_string((int)c.latency_ms) + "ms" +
                    " | " + std::to_string((int)(c.sparsity * 100)) + "%";
                if (focus == 1 && vi == selected_layer)
                    layer_rows.push_back(text(line) | inverted);
                else
                    layer_rows.push_back(text(line));
            }
        }
        if (layer_rows.empty())
            layer_rows.push_back(text("select a token first"));
        auto layer_panel = window(
            text(focus == 1 ? " 2. LAYERS [active] " : " 2. LAYERS "),
            vbox(layer_rows)
        );

        // ── panel 3: metrics ──────────────────────────
        Elements metric_rows;
        if (n_tokens > 0 && selected_token < n_tokens) {
            TokenBatch tb = g_history.get(selected_token);
            if (!visible_indices.empty() &&
                selected_layer < (int)visible_indices.size()) {
                LayerCapture& sel =
                    tb.layers[visible_indices[selected_layer]];

                std::string shape_str = "[";
                for (int d = 0; d < sel.n_dims; d++) {
                    if (d > 0) shape_str += ", ";
                    shape_str += std::to_string(sel.shape[d]);
                }
                shape_str += "]";

                metric_rows = {
                    text("Token   : [" + std::string(tb.token_str) + "]"),
                    text("Layer   : " + std::string(sel.name)),
                    text("Shape   : " + shape_str),
                    text("Latency : " +
                         std::to_string(sel.latency_ms) + " ms"),
                    hbox({
                        text("Sparsity: "),
                        gauge(sel.sparsity) | flex,
                        text(" " +
                             std::to_string((int)(sel.sparsity*100)) + "%"),
                    }),
                };
            }
        }
        if (metric_rows.empty()) metric_rows = { text("waiting...") };
        auto metrics = window(
            text(focus == 2 ? " 3. RUNTIME METRICS [active] "
                            : " 3. RUNTIME METRICS "),
            vbox(metric_rows)
        );

        // ── panel 4: attention matrix ─────────────────
        Elements attn_rows;
        if (n_tokens > 0 && selected_token < n_tokens) {
            TokenBatch tb = g_history.get(selected_token);
            int target_layer_idx = -1;
            if (!visible_indices.empty() &&
                selected_layer < (int)visible_indices.size()) {
                target_layer_idx =
                    tb.layers[visible_indices[selected_layer]].layer_idx;
            }
            LayerCapture attn_cap;
            bool found = false;
            for (auto& cap : tb.layers) {
                std::string cname(cap.name);
                if (cname.find("kq_soft_max") != std::string::npos &&
                    cap.layer_idx == target_layer_idx &&
                    cap.has_attn) {
                    attn_cap = cap;
                    found    = true;
                    break;
                }
            }
            if (found && attn_cap.attn_n > 0) {
                attn_rows.push_back(
                    text("layer " + std::to_string(target_layer_idx) +
                         " | n=" + std::to_string(attn_cap.attn_n))
                );
                attn_rows.push_back(separator());
                int n = attn_cap.attn_n;
                for (int row = 0; row < n; row++) {
                    std::string line;
                    for (int col = 0; col < n; col++) {
                        float v = attn_cap.attn[row][col];
                        if      (v > 0.5f)  line += "█";
                        else if (v > 0.25f) line += "▓";
                        else if (v > 0.1f)  line += "▒";
                        else if (v > 0.01f) line += "░";
                        else                line += " ";
                    }
                    attn_rows.push_back(text(line));
                }
            } else {
                attn_rows.push_back(text("no attn data"));
                attn_rows.push_back(text("select a layer"));
            }
        }
        if (attn_rows.empty()) attn_rows.push_back(text("waiting..."));
        auto attn_panel = window(
            text(" 4. ATTENTION MATRIX "),
            vbox(attn_rows)
        );

        // ── panel 5: anomaly report ───────────────────
        Elements anomaly_rows;
        if (g_anomaly.ready) {
            anomaly_rows.push_back(
                text("mean: " +
                     std::to_string((int)g_anomaly.mean_latency) + "ms")
            );
            anomaly_rows.push_back(separator());
            for (auto& s : g_anomaly.stats) {
                std::string flag;
                if (s.is_slow)   flag += " ⚠SLOW";
                if (s.is_sparse) flag += " ⚠SPARSE";
                // use short name to fit in panel
                std::string short_name = "L" +
                    std::to_string(s.layer_idx);
                std::string line =
                    short_name +
                    " | " + std::to_string((int)s.avg_latency) + "ms" +
                    flag;
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
            text(" 5. ANOMALY REPORT "),
            vbox(anomaly_rows) | frame | size(HEIGHT, LESS_THAN, 20)
        );

        return vbox({
            text(" TRANSFORMER TELEMETRY  |  [Tab] focus  |  [j/k] nav  |  [Enter] drill  |  [Esc] back  |  [Q] quit ")
                | center,
            separator(),
            hbox({
                vbox({
                    token_panel | size(HEIGHT, LESS_THAN, 12),
                    layer_panel | size(HEIGHT, LESS_THAN, 12),
                    hbox({
                        metrics    | flex,
                        attn_panel | flex,
                    }),
                }) | flex,
                vbox({
                    window(
                        text(" STATUS "),
                        vbox({
                            text("Tokens : " +
                                 std::to_string(g_token_count.load())),
                            text("Hist   : " +
                                 std::to_string(n_tokens) + "/" +
                                 std::to_string(TokenHistory::CAP)),
                            text(g_running.load() ? "● running" : "■ done"),
                            separator(),
                            text(focus == 0 ? "focus: tokens" :
                                 focus == 1 ? "focus: layers" :
                                              "focus: metrics"),
                            text("sel_tok: " +
                                 std::to_string(selected_token)),
                            text("sel_lyr: " +
                                 std::to_string(selected_layer)),
                        })
                    ) | size(HEIGHT, LESS_THAN, 10),
                    anomaly_panel | flex,
                }) | size(WIDTH, LESS_THAN, 22),
            }),
        });
    });

    auto component = CatchEvent(renderer, [&](Event event) {
        int n_tokens = g_history.size();

        if (event == Event::Tab) {
            focus = (focus + 1) % 3;
            return true;
        }
        if (event == Event::Character('j')) {
            if (focus == 0) {
                selected_token = std::min(selected_token + 1,
                                          std::max(0, n_tokens - 1));
            } else if (focus == 1) {
                if (n_tokens > 0 && selected_token < n_tokens) {
                    TokenBatch tb = g_history.get(selected_token);
                    int vis_count = 0;
                    for (auto& c : tb.layers) {
                        std::string cn(c.name);
                        if (cn.find("kq_soft_max") == std::string::npos)
                            vis_count++;
                    }
                    selected_layer = std::min(selected_layer + 1,
                        std::max(0, vis_count - 1));
                }
            }
            return true;
        }
        if (event == Event::Character('k')) {
            if (focus == 0)
                selected_token = std::max(selected_token - 1, 0);
            else if (focus == 1)
                selected_layer = std::max(selected_layer - 1, 0);
            return true;
        }
        if (event == Event::Return) {
            if (focus == 0) {
                focus          = 1;
                selected_layer = 0;
            } else if (focus == 1) {
                focus = 2;
            }
            return true;
        }
        if (event == Event::Escape) {
            if (focus > 0) focus--;
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