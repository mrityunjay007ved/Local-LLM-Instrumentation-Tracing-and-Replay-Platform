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

// ── shared state ──────────────────────────────────────────
RingBuffer g_rb;
std::atomic<bool> g_running     = true;
std::atomic<int>  g_token_count = 0;

// ── eval callback data ────────────────────────────────────
struct EvalCallbackData {
    RingBuffer* rb;
    std::chrono::time_point<std::chrono::high_resolution_clock> layer_start;
};

// fires after every tensor op inside llama_decode
static bool eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (!t || !t->name || t->name[0] == '\0') return true;

    std::string name(t->name);

    // skip views and permutations — only capture base tensors
    if (name.find("(view)")      != std::string::npos) return true;
    if (name.find("(permuted)")  != std::string::npos) return true;
    if (name.find("(cont)")      != std::string::npos) return true;
    if (name.find("(reshaped)")  != std::string::npos) return true;

    // capture these three:
    bool is_layer  = name.find("l_out")       != std::string::npos;
    bool is_mlp    = name.find("ffn_out")     != std::string::npos;
    bool is_kqsm   = name.find("kq_soft_max") != std::string::npos;
    if (!is_layer && !is_mlp && !is_kqsm) return true;

    auto* data = (EvalCallbackData*)user_data;
    auto  now  = std::chrono::high_resolution_clock::now();
    float ms   = std::chrono::duration<float, std::milli>(
                     now - data->layer_start).count();

    int layer_idx = -1;
    sscanf(t->name, "%*[^-]-%d", &layer_idx);

    // ── attention matrix ──────────────────────────────
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

            // shape: [n_kv, n_tokens, n_heads]
            int seq    = (int)t->ne[0];
            int n      = std::min(seq, 8);
            c.attn_n   = n;
            c.has_attn = true;

            float* fdata = (float*)t->data;
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    c.attn[row][col] = fdata[row * seq + col];
                }
            }
        }
        data->rb->push(c);
        data->layer_start = now;
        return true;
    }

    // ── layer output / mlp output ─────────────────────
    float sparsity = 0.0f;
    if (t->data && t->buffer &&
        ggml_backend_buffer_is_host(t->buffer) &&
        t->type == GGML_TYPE_F32) {
        float*  fdata  = (float*)t->data;
        int64_t total  = ggml_nelements(t);
        int64_t sample = std::min(total, (int64_t)1024);
        int     zeros  = 0;
        for (int64_t i = 0; i < sample; i++) {
            if (fabsf(fdata[i]) < 1e-6f) zeros++;
        }
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

    data->layer_start = now;
    return true;
}
// ── inference thread ──────────────────────────────────────
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
    cb_data.layer_start = std::chrono::high_resolution_clock::now();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = n_prompt + 128;
    ctx_params.n_batch           = n_prompt;
    ctx_params.no_perf           = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
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

    for (int n_pos = 0;
         n_pos + batch.n_tokens < n_prompt + 64 && g_running; ) {

        cb_data.layer_start = std::chrono::high_resolution_clock::now();
        if (llama_decode(ctx, batch)) break;

        float* logits   = llama_get_logits(ctx);
        float logit_max = *std::max_element(logits, logits + n_vocab);
        float logit_min = *std::min_element(logits, logits + n_vocab);
        (void)logit_max;
        (void)logit_min;

        n_pos += batch.n_tokens;
        g_token_count++;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    g_running = false;
}

// ── TUI thread ────────────────────────────────────────────
static void run_tui() {
    using namespace ftxui;
    auto screen = ScreenInteractive::Fullscreen();

    int selected = 0;

    std::thread refresher([&] {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            screen.PostEvent(Event::Custom);
        }
        screen.PostEvent(Event::Custom);
    });

    auto renderer = Renderer([&] {
        int n = g_rb.size();
        if (n > 0 && selected >= n) selected = n - 1;

        // ── panel 1: layer list, j/k navigable ───────
        Elements rows;
        int window_size = 18;
        int start = std::max(0, selected - window_size / 2);
        if (start + window_size > n)
            start = std::max(0, n - window_size);

        for (int i = start; i < std::min(n, start + window_size); i++) {
            LayerCapture c = g_rb.get(i);
            std::string cname(c.name);

            // skip KQsoft entries from the layer list
            if (cname.find("kq_soft_max") != std::string::npos) continue;

            std::string line =
                cname +
                " | " + std::to_string((int)c.latency_ms) + "ms" +
                " | " + std::to_string((int)(c.sparsity * 100)) + "%";
            if (i == selected)
                rows.push_back(text(line) | inverted);
            else
                rows.push_back(text(line));
        }
        if (rows.empty()) rows.push_back(text("waiting for inference..."));

        auto topology = window(
            text(" 1. LAYER CAPTURES  [j/k navigate] "),
            vbox(rows)
        );

        // ── panel 2: live counters ────────────────────
        auto live = window(
            text(" 2. LIVE PACKETS "),
            vbox({
                text("Tokens : " +
                     std::to_string(g_token_count.load())),
                text("Buffer : " +
                     std::to_string(n) + "/" +
                     std::to_string(RingBuffer::CAP)),
                text(g_running.load() ? "● running" : "■ done"),
                separator(),
                text("sel: " + std::to_string(selected)),
            })
        );

        // ── panel 3: selected layer metrics ──────────
        Elements metric_rows;
        if (n > 0 && selected < n) {
            LayerCapture sel = g_rb.get(selected);
            metric_rows = {
                text("Layer   : " + std::string(sel.name)),
                text("Latency : " +
                     std::to_string(sel.latency_ms) + " ms"),
                hbox({
                    text("Sparsity: "),
                    gauge(sel.sparsity) | flex,
                    text(" " +
                         std::to_string((int)(sel.sparsity * 100)) + "%"),
                }),
                text("idx     : " + std::to_string(sel.layer_idx)),
            };
        } else {
            metric_rows = { text("waiting...") };
        }
        auto metrics = window(
            text(" 3. RUNTIME METRICS "),
            vbox(metric_rows)
        );

        // ── panel 4: attention matrix ─────────────────
        // find KQsoft entry matching selected layer's idx
        LayerCapture attn_data;
        bool found_attn = false;
        if (n > 0 && selected < n) {
            LayerCapture sel = g_rb.get(selected);
            for (int i = 0; i < n; i++) {
                LayerCapture c = g_rb.get(i);
                std::string cname(c.name);
                if (cname.find("kq_soft_max") != std::string::npos &&
                    c.layer_idx == sel.layer_idx &&
                    c.has_attn) {
                    attn_data  = c;
                    found_attn = true;
                    break;
                }
            }
        }

        Elements attn_rows;
        if (found_attn && attn_data.has_attn && attn_data.attn_n > 0) {
            int n_tok = attn_data.attn_n;
            for (int row = 0; row < n_tok; row++) {
                std::string line;
                for (int col = 0; col < n_tok; col++) {
                    float v = attn_data.attn[row][col];
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
            attn_rows.push_back(text("navigate to a"));
            attn_rows.push_back(text("layer to load"));
        }

        auto attn = window(
            text(" 4. ATTENTION MATRIX "),
            vbox(attn_rows)
        );

        return vbox({
            text(" TRANSFORMER TELEMETRY  |  [j/k] navigate  |  [Q] quit ")
                | center,
            separator(),
            hbox({
                vbox({
                    topology | size(HEIGHT, LESS_THAN, 20),
                    hbox({
                        metrics | flex,
                        attn | flex,
                    }),
                }) | flex,
                live | size(WIDTH, LESS_THAN, 25),
            }),
        });
    });

    auto component = CatchEvent(renderer, [&](Event event) {
        int n = g_rb.size();
        if (event == Event::Character('j')) {
            selected = std::min(selected + 1, std::max(0, n - 1));
            return true;
        }
        if (event == Event::Character('k')) {
            selected = std::max(selected - 1, 0);
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

// ── main ──────────────────────────────────────────────────
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