#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Synthetic end-to-end check for GGML_TYPE_TQ1_0_128.
// Writes a small llama model whose 2-D weights are all quantized to tq1_0_128, then compares the
// CPU reference against the GPU MMVQ path and against the dequantize-to-f16 fallback.
// The fallback is selected with GGML_TQ1_128_NO_MMVQ=1, so a non-zero difference between the two
// GPU runs also proves that the native kernel was used for the first one.
// A second model with the same weights stored as f16 gives the plain CPU-vs-GPU drift to compare
// against, so the codec is not blamed for differences that every weight type shows.

struct model_config {
    int n_vocab = 64;
    int n_embd  = 256;
    int n_layer = 4;
    int n_head  = 4;
    int n_ff    = 512;
};

// tq1_0_128: the type under test
// f16:        the same weights with the codec error removed (dequantized, stored f16)
// q8_0:       a stock type, the control for the drift every 8-bit activation path shows
enum class model_kind { tq1_0_128, f16, q8_0 };

static ggml_type model_quant_type(model_kind kind) {
    return kind == model_kind::q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_TQ1_0_128;
}

struct tensor_spec {
    std::string name;
    int64_t ne0;
    int64_t ne1; // 1 for a 1-D tensor
};

static const int N_TOKENS  = 8;
static const int N_PREFILL = 4;

static std::vector<tensor_spec> model_tensors(const model_config & c) {
    std::vector<tensor_spec> specs = {
        { "token_embd.weight",  c.n_embd, c.n_vocab },
        { "output_norm.weight", c.n_embd, 1         },
        { "output.weight",      c.n_embd, c.n_vocab },
    };

    for (int i = 0; i < c.n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        specs.push_back({ p + "attn_norm.weight",   c.n_embd, 1        });
        specs.push_back({ p + "attn_q.weight",      c.n_embd, c.n_embd });
        specs.push_back({ p + "attn_k.weight",      c.n_embd, c.n_embd });
        specs.push_back({ p + "attn_v.weight",      c.n_embd, c.n_embd });
        specs.push_back({ p + "attn_output.weight", c.n_embd, c.n_embd });
        specs.push_back({ p + "ffn_norm.weight",    c.n_embd, 1        });
        specs.push_back({ p + "ffn_gate.weight",    c.n_embd, c.n_ff   });
        specs.push_back({ p + "ffn_up.weight",      c.n_embd, c.n_ff   });
        specs.push_back({ p + "ffn_down.weight",    c.n_ff,   c.n_embd });
    }
    return specs;
}

static bool write_model(const std::string & path, const model_config & c, model_kind kind) {
    // the tensors are allocated in the context, so the pool has to fit the model
    // the f16 control stores dequantized f16 data, so its stored type size differs from the staging type
    size_t pool = 1024*1024;
    for (const tensor_spec & spec : model_tensors(c)) {
        const ggml_type stored = spec.ne1 == 1 ? GGML_TYPE_F32 :
            (kind == model_kind::f16 ? GGML_TYPE_F16 : model_quant_type(kind));
        pool += ggml_row_size(stored, spec.ne0) * (size_t) spec.ne1;
    }

    ggml_init_params gparams = {
        /*.mem_size   =*/ pool,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * gctx = ggml_init(gparams);
    if (gctx == nullptr) {
        printf("%s: could not create the ggml context\n", __func__);
        return false;
    }

    gguf_context * gugf = gguf_init_empty();

    gguf_set_val_str (gugf, "general.architecture", "llama");
    gguf_set_val_str (gugf, "general.name",         "synthetic-tq1-0-128");

    gguf_set_val_u32 (gugf, "llama.context_length",                   4096);
    gguf_set_val_u32 (gugf, "llama.embedding_length",                 c.n_embd);
    gguf_set_val_u32 (gugf, "llama.block_count",                      c.n_layer);
    gguf_set_val_u32 (gugf, "llama.feed_forward_length",              c.n_ff);
    gguf_set_val_u32 (gugf, "llama.attention.head_count",             c.n_head);
    gguf_set_val_u32 (gugf, "llama.attention.head_count_kv",          c.n_head);
    gguf_set_val_f32 (gugf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_bool(gugf, "llama.attention.causal",                 true);
    gguf_set_val_u32 (gugf, "llama.rope.dimension_count",             c.n_embd/c.n_head);
    gguf_set_val_f32 (gugf, "llama.rope.freq_base",                   10000.0f);

    // a minimal SPM vocab: the test feeds token ids directly, the tokens only have to load
    std::vector<std::string> tokens = { "<unk>", "<s>", "</s>" };
    for (int i = (int) tokens.size(); i < c.n_vocab; ++i) {
        tokens.push_back("<t" + std::to_string(i) + ">");
    }

    std::vector<const char *> token_ptrs;
    for (const std::string & t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    std::vector<float>   scores(c.n_vocab, 0.0f);
    std::vector<int32_t> types (c.n_vocab, 1);
    types[0] = 2; // LLAMA_TOKEN_TYPE_UNKNOWN
    types[1] = 3; // LLAMA_TOKEN_TYPE_CONTROL
    types[2] = 3;

    gguf_set_val_str (gugf, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str (gugf, "tokenizer.ggml.tokens",     token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(gugf, "tokenizer.ggml.scores",     GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gugf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,   types.data(),  types.size());
    gguf_set_val_u32 (gugf, "tokenizer.ggml.bos_token_id",     1);
    gguf_set_val_u32 (gugf, "tokenizer.ggml.eos_token_id",     2);
    gguf_set_val_u32 (gugf, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_bool(gugf, "tokenizer.ggml.add_bos_token",    true);
    gguf_set_val_bool(gugf, "tokenizer.ggml.add_eos_token",    false);

    const std::vector<tensor_spec> specs = model_tensors(c);

    for (size_t i = 0; i < specs.size(); ++i) {
        const tensor_spec & spec = specs[i];

        std::mt19937 rng(0x7a1 + (uint32_t) i);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        const size_t n = (size_t) (spec.ne0 * spec.ne1);
        std::vector<float> data(n);
        for (size_t k = 0; k < n; ++k) {
            data[k] = dist(rng) / std::sqrt((float) spec.ne0);
        }

        ggml_tensor * t = nullptr;

        if (spec.ne1 == 1) {
            t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, spec.ne0);
            memcpy(t->data, data.data(), n*sizeof(float));
        } else {
            const ggml_type qtype = model_quant_type(kind);

            // every quantized row must be a whole number of blocks
            if (spec.ne0 % ggml_blck_size(qtype) != 0) {
                printf("%s: row size %" PRId64 " of %s is not a multiple of %d\n",
                        __func__, spec.ne0, spec.name.c_str(), (int) ggml_blck_size(qtype));
                gguf_free(gugf);
                ggml_free(gctx);
                return false;
            }

            const size_t row_size = ggml_row_size(qtype, spec.ne0);
            std::vector<uint8_t> qdata((size_t) spec.ne1 * row_size);

            ggml_quantize_chunk(qtype, data.data(), qdata.data(), 0, spec.ne1, spec.ne0, nullptr);

            if (kind == model_kind::f16) {
                // the control model: the same weights with the codec error removed
                std::vector<float> ref(n);
                ggml_get_type_traits(qtype)->to_float(qdata.data(), ref.data(), (int64_t) n);

                t = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, spec.ne0, spec.ne1);
                ggml_fp32_to_fp16_row(ref.data(), (ggml_fp16_t *) t->data, (int64_t) n);
            } else {
                t = ggml_new_tensor_2d(gctx, qtype, spec.ne0, spec.ne1);
                memcpy(t->data, qdata.data(), qdata.size());
            }
        }

        ggml_set_name(t, spec.name.c_str());
        gguf_add_tensor(gugf, t);
    }

    const bool ok = gguf_write_to_file(gugf, path.c_str(), false);
    if (!ok) {
        printf("%s: could not write %s\n", __func__, path.c_str());
    }

    gguf_free(gugf);
    ggml_free(gctx);

    return ok;
}

static bool run_model(const std::string & path, int n_gpu_layers, std::vector<float> & logits) {
    printf("run: %s (n_gpu_layers = %d)\n", path.c_str(), n_gpu_layers);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        printf("could not load %s\n", path.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 64;
    cparams.n_batch  = N_TOKENS;
    cparams.n_ubatch = N_TOKENS;
    cparams.no_perf  = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        printf("could not create a context for %s\n", path.c_str());
        llama_model_free(model);
        return false;
    }

    const int32_t ids[N_TOKENS] = { 3, 5, 7, 11, 13, 4, 6, 8 };

    int n = 0;
    while (n < N_TOKENS) {
        // a prefill of several tokens, then single-token decodes
        const int n_step = n == 0 ? N_PREFILL : 1;

        llama_batch batch = llama_batch_init(n_step, 0, 1);
        for (int j = 0; j < n_step; ++j) {
            batch.token[j]     = ids[n + j];
            batch.pos[j]       = n + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = j == n_step - 1;
        }
        batch.n_tokens = n_step;

        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            printf("llama_decode failed with %d on %s\n", ret, path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            return false;
        }

        const float * op = llama_get_logits(ctx);
        logits.insert(logits.end(), op, op + (size_t) llama_vocab_n_tokens(llama_model_get_vocab(model)));

        n += n_step;
    }

    llama_free(ctx);
    llama_model_free(model);

    return true;
}

static double compare(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        printf("%s: logit count mismatch: %zu != %zu\n", label, a.size(), b.size());
        return 1e30;
    }

    double max_diff = 0.0;
    double max_ref  = 0.0;
    double sum_diff = 0.0;
    double sum_ref  = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        max_diff = std::max(max_diff, std::fabs(d));
        max_ref  = std::max(max_ref,  std::fabs((double) a[i]));
        sum_diff += d*d;
        sum_ref  += (double) a[i]*(double) a[i];
    }

    const double rel  = max_ref > 0.0 ? max_diff/max_ref : 0.0;
    const double nmse = sum_ref > 0.0 ? sum_diff/sum_ref : 0.0;

    printf("%-28s: max |logit| = %8.3f, max diff = %10.3e, relative = %10.3e, nmse = %10.3e\n",
            label, max_ref, max_diff, rel, nmse);

    return rel;
}

static void set_no_mmvq(void) {
#if defined(_WIN32)
    _putenv_s("GGML_TQ1_128_NO_MMVQ", "1");
#else
    setenv("GGML_TQ1_128_NO_MMVQ", "1", 1);
#endif
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string dir = argc > 1 ? argv[1] : ".";

    model_config c;
    if (argc > 5) {
        c.n_embd  = atoi(argv[2]);
        c.n_ff    = atoi(argv[3]);
        c.n_layer = atoi(argv[4]);
        c.n_vocab = atoi(argv[5]);
        c.n_head  = std::max(1, c.n_embd/64);
    }

    const std::string path     = dir + "/tq1-0-128.gguf";
    const std::string path_ref = dir + "/tq1-0-128-ref.gguf";
    const std::string path_q8  = dir + "/tq1-0-128-q8.gguf";

    llama_backend_init();
    if (getenv("TQ1_E2E_LOG") == nullptr) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    printf("writing %s (n_embd = %d, n_ff = %d, n_layer = %d, n_vocab = %d)\n",
            path.c_str(), c.n_embd, c.n_ff, c.n_layer, c.n_vocab);
    if (!write_model(path, c, model_kind::tq1_0_128)) {
        printf("FAIL: could not write %s\n", path.c_str());
        return 1;
    }
    if (!write_model(path_ref, c, model_kind::f16)) {
        printf("FAIL: could not write %s\n", path_ref.c_str());
        return 1;
    }
    if (!write_model(path_q8, c, model_kind::q8_0)) {
        printf("FAIL: could not write %s\n", path_q8.c_str());
        return 1;
    }

    if (getenv("TQ1_E2E_WRITE_ONLY") != nullptr) {
        printf("OK (write only)\n");
        return 0;
    }

    std::vector<float> l_cpu;
    std::vector<float> l_mmvq;
    std::vector<float> l_fallback;
    std::vector<float> r_cpu;
    std::vector<float> r_gpu;
    std::vector<float> q_cpu;
    std::vector<float> q_gpu;

    bool ok = true;

    ok = run_model(path,      0, l_cpu)      && ok; // CPU: ggml_vec_dot_tq1_0_128_q8_0
    ok = run_model(path,     99, l_mmvq)     && ok; // GPU: mul_mat_vec_q
    ok = run_model(path_ref,  0, r_cpu)      && ok; // CPU: f16 control
    ok = run_model(path_ref, 99, r_gpu)      && ok; // GPU: f16 control
    ok = run_model(path_q8,   0, q_cpu)      && ok; // CPU: stock q8_0
    ok = run_model(path_q8,  99, q_gpu)      && ok; // GPU: stock q8_0

    // the fallback must run the same model with a different kernel, so the two GPU runs differ
    set_no_mmvq();
    ok = run_model(path, 99, l_fallback) && ok;

    ok = !l_cpu.empty() && !l_mmvq.empty() && !l_fallback.empty() && !r_cpu.empty() && !r_gpu.empty() &&
         !q_cpu.empty() && !q_gpu.empty() && ok;
    if (!ok) {
        printf("FAIL: could not run the model\n");
        return 1;
    }

    const double rel_cpu_gpu   = compare("tq1_0_128: cpu vs gpu mmvq",      l_cpu,      l_mmvq);
    const double rel_cpu_fb    = compare("tq1_0_128: cpu vs gpu fallback",  l_cpu,      l_fallback);
    const double rel_mmvq_fb   = compare("tq1_0_128: gpu mmvq vs fallback", l_mmvq,     l_fallback);
    const double rel_control   = compare("f16 control: cpu vs gpu",         r_cpu,      r_gpu);
    const double rel_q8        = compare("q8_0 control: cpu vs gpu",        q_cpu,      q_gpu);

    // The kernel itself is checked per op by test-backend-ops (nmse <= 1e-7 against the CPU
    // reference). At the model level the differences below are dominated by the 8-bit activation
    // quantization both q8 paths share: the grid is block_amax/127, so a ~1e-4 backend difference
    // in the residual stream flips the rounding of ~1% of the activations, and those flips are
    // not common mode. rel_cpu_fb is that term alone (same kernel family, f16 activations on one
    // side) and rel_q8 is it for a stock type.

    if (rel_cpu_gpu > 5e-2) {
        printf("FAIL: the MMVQ path does not agree with the CPU reference\n");
        return 1;
    }
    if (rel_mmvq_fb > 5e-2) {
        printf("FAIL: the MMVQ path does not agree with the f16 fallback\n");
        return 1;
    }
    if (rel_mmvq_fb == 0.0) {
        printf("FAIL: the fallback run is bit-identical to the MMVQ run, so no native kernel was used\n");
        return 1;
    }

    const double drift_ref = std::max(rel_cpu_fb, rel_q8);
    if (rel_cpu_gpu > 2.0*drift_ref + 1e-3) {
        printf("FAIL: the tq1_0_128 model drifts more between backends (%.3e) than the 8-bit "
               "activation paths it is compared against (%.3e, q8_0 control %.3e)\n",
                rel_cpu_gpu, rel_cpu_fb, rel_q8);
        return 1;
    }

    // Both q8 paths are measured against an f16-activation run of the same weights on the same
    // backend, so MMVQ must not sit further from it than the CPU reference does.
    if (rel_mmvq_fb > 2.0*rel_cpu_fb + 1e-3) {
        printf("FAIL: MMVQ is further from the f16 activation path (%.3e) than the CPU reference "
               "is (%.3e), so the kernel carries error the CPU vec_dot does not\n",
                rel_mmvq_fb, rel_cpu_fb);
        return 1;
    }

    printf("OK (f16 control drift %.3e, 8-bit activation drift %.3e)\n", rel_control, drift_ref);

    return 0;
}
