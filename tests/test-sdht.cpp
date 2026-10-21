#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// the fold contract: a weight W with a per-input-channel diagonal diag(s) and a block Hartley rotation H folded in
// is stored as W' = W*diag(s)*H, and the model scales the matmul input by diag(1/s) before applying H
// this test writes both models, loads them with llama and compares the logits

static const int N_VOCAB    = 32;
static const int N_EMBD     = 64;
static const int N_LAYER    = 2;
static const int N_HEAD     = 4;
static const int N_HEAD_DIM = N_EMBD / N_HEAD;
static const int N_FF       = 128;
static const int DHT_BLOCK  = 64;
static const int N_TOKENS   = 8;
static const int N_PREFILL  = 4;

struct tensor_spec {
    std::string name;
    int64_t ne0;
    int64_t ne1; // 1 for a 1-D tensor
    bool fold;
};

static std::vector<tensor_spec> model_tensors() {
    std::vector<tensor_spec> specs = {
        { "token_embd.weight",  N_EMBD,     N_VOCAB, false },
        { "output_norm.weight", N_EMBD,     1,       false },
        { "output.weight",      N_EMBD,     N_VOCAB, true  },
    };

    for (int i = 0; i < N_LAYER; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        specs.push_back({ p + "attn_norm.weight",   N_EMBD,          1,    false });
        specs.push_back({ p + "attn_q.weight",      N_EMBD,          N_EMBD, true });
        specs.push_back({ p + "attn_k.weight",      N_EMBD,          N_EMBD, true });
        specs.push_back({ p + "attn_v.weight",      N_EMBD,          N_EMBD, true });
        specs.push_back({ p + "attn_output.weight", N_EMBD,          N_EMBD, true });
        specs.push_back({ p + "ffn_norm.weight",    N_EMBD,          1,    false });
        specs.push_back({ p + "ffn_gate.weight",    N_EMBD,          N_FF,   true });
        specs.push_back({ p + "ffn_up.weight",      N_EMBD,          N_FF,   true });
        specs.push_back({ p + "ffn_down.weight",    N_FF,            N_EMBD, true });
    }
    return specs;
}

// H[c][r] = cas(2*pi*c*r/B)/sqrt(B), the same matrix load_tensors_hadamard builds for the DHT rotation
static std::vector<double> dht_matrix() {
    std::vector<double> h((size_t) DHT_BLOCK * DHT_BLOCK);
    for (int64_t c = 0; c < DHT_BLOCK; ++c) {
        for (int64_t r = 0; r < DHT_BLOCK; ++r) {
            const double phi = 6.283185307179586476925286766559 * (double) c * (double) r / (double) DHT_BLOCK;
            h[(size_t) c * DHT_BLOCK + r] = (std::cos(phi) + std::sin(phi)) / std::sqrt((double) DHT_BLOCK);
        }
    }
    return h;
}

// a is [ne1][ne0] (the layout of a ggml tensor with ne0 fastest), folded in place
static void fold_in_place(std::vector<float> & a, int64_t ne0, int64_t ne1, const std::vector<float> & s) {
    const std::vector<double> h = dht_matrix();
    std::vector<float> out(a.size());

    for (int64_t n = 0; n < ne1; ++n) {
        for (int64_t b = 0; b < ne0 / DHT_BLOCK; ++b) {
            for (int64_t r = 0; r < DHT_BLOCK; ++r) {
                double acc = 0.0;
                for (int64_t c = 0; c < DHT_BLOCK; ++c) {
                    acc += (double) a[n*ne0 + b*DHT_BLOCK + c] * (double) s[b*DHT_BLOCK + c] * h[(size_t) c*DHT_BLOCK + r];
                }
                out[n*ne0 + b*DHT_BLOCK + r] = (float) acc;
            }
        }
    }
    a = out;
}

static bool write_model(const std::string & path, bool folded, bool legacy = false, bool mixed = false, bool missing_version = false) {
    const std::vector<tensor_spec> specs = model_tensors();

    ggml_init_params gparams = {
        /*.mem_size   =*/ 8*1024*1024,
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
    gguf_set_val_str (gugf, "general.name",         "tiny-sdht");

    gguf_set_val_u32 (gugf, "llama.context_length",                     64);
    gguf_set_val_u32 (gugf, "llama.embedding_length",                   N_EMBD);
    gguf_set_val_u32 (gugf, "llama.block_count",                        N_LAYER);
    gguf_set_val_u32 (gugf, "llama.feed_forward_length",                N_FF);
    gguf_set_val_u32 (gugf, "llama.attention.head_count",               N_HEAD);
    gguf_set_val_u32 (gugf, "llama.attention.head_count_kv",            N_HEAD);
    gguf_set_val_f32 (gugf, "llama.attention.layer_norm_rms_epsilon",   1e-5f);
    gguf_set_val_bool(gugf, "llama.attention.causal",                   true);
    gguf_set_val_u32 (gugf, "llama.rope.dimension_count",               N_HEAD_DIM);
    gguf_set_val_f32 (gugf, "llama.rope.freq_base",                     10000.0f);

    // a minimal SPM vocab: the test feeds token ids directly, the tokens only have to load
    std::vector<std::string> tokens = { "<unk>", "<s>", "</s>" };
    for (int i = (int) tokens.size(); i < N_VOCAB; ++i) {
        tokens.push_back("<t" + std::to_string(i) + ">");
    }
    std::vector<const char *> token_ptrs;
    for (const std::string & t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    std::vector<float>   scores(N_VOCAB, 0.0f);
    std::vector<int32_t> types (N_VOCAB, 1);
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

    std::vector<std::string> folded_names;
    std::vector<const char *> folded_ptrs;

    for (size_t i = 0; i < specs.size(); ++i) {
        const tensor_spec & spec = specs[i];

        ggml_tensor * t = spec.ne1 == 1
            ? ggml_new_tensor_1d(gctx, GGML_TYPE_F32, spec.ne0)
            : ggml_new_tensor_2d(gctx, GGML_TYPE_F32, spec.ne0, spec.ne1);
        ggml_set_name(t, spec.name.c_str());

        std::mt19937 rng(0x5eed + (uint32_t) i);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        const size_t n = (size_t) (spec.ne0 * spec.ne1);
        std::vector<float> data(n);
        for (size_t k = 0; k < n; ++k) {
            data[k] = dist(rng) / std::sqrt((float) spec.ne0);
        }

        if (folded && spec.fold) {
            // s is recovered from the stored diagonal, so the fold and the runtime use the same value
            std::normal_distribution<float> ldist(0.0f, 0.5f);
            std::vector<float> s(spec.ne0);
            std::vector<ggml_fp16_t> d(spec.ne0);
            for (int64_t k = 0; k < spec.ne0; ++k) {
                d[k] = ggml_fp32_to_fp16(1.0f/std::exp(ldist(rng)));
                s[k] = 1.0f/ggml_fp16_to_fp32(d[k]);
            }

            fold_in_place(data, spec.ne0, spec.ne1, s);

            ggml_tensor * diag = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, spec.ne0);
            const std::string dname = spec.name + ".input_diag";
            ggml_set_name(diag, dname.c_str());
            memcpy(diag->data, d.data(), d.size()*sizeof(ggml_fp16_t));
            gguf_add_tensor(gugf, diag);

            folded_names.push_back(spec.name);
        }

        memcpy(t->data, data.data(), n*sizeof(float));
        gguf_add_tensor(gugf, t);
    }

    if (folded) {
        for (const std::string & name : folded_names) {
            folded_ptrs.push_back(name.c_str());
        }

        const std::string prefix = legacy ? "prism.hadamard." : "quantization.rotation.";
        if (!missing_version) {
            gguf_set_val_u32(gugf, (prefix + "version").c_str(), 1);
        }
        gguf_set_val_u32(gugf, (prefix + "block_size").c_str(), DHT_BLOCK);
        gguf_set_val_str(gugf, (prefix + "transform").c_str(), "normalized-smooth-discrete-hartley");
        gguf_set_val_str(gugf, (prefix + "axis").c_str(), "input-last-dimension");
        gguf_set_val_str(gugf, (prefix + "sign_mode").c_str(), "identity");
        gguf_set_arr_str(gugf, (prefix + "weight_names").c_str(), folded_ptrs.data(), folded_ptrs.size());
        gguf_set_arr_str(gugf, (prefix + "diagonal_weight_names").c_str(), folded_ptrs.data(), folded_ptrs.size());
        if (mixed) {
            gguf_set_val_u32(gugf, "prism.hadamard.version", 1);
        }
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
    printf("run: model loaded\n");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 64;
    cparams.n_batch   = N_TOKENS;
    cparams.n_ubatch  = N_TOKENS;
    cparams.no_perf   = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        printf("could not create a context for %s\n", path.c_str());
        llama_model_free(model);
        return false;
    }
    printf("run: context created\n");

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
        logits.insert(logits.end(), op, op + N_VOCAB);

        n += n_step;
    }

    llama_free(ctx);
    llama_model_free(model);

    return true;
}

static bool compare(const std::string & a, const std::string & b, int n_gpu_layers, bool exact = false) {
    std::vector<float> la;
    std::vector<float> lb;

    if (!run_model(a, n_gpu_layers, la) || !run_model(b, n_gpu_layers, lb)) {
        return false;
    }
    if (la.size() != lb.size()) {
        printf("logit count mismatch: %zu != %zu\n", la.size(), lb.size());
        return false;
    }

    double max_diff = 0.0;
    double max_ref  = 0.0;
    double sum_diff = 0.0;
    double sum_ref  = 0.0;
    for (size_t i = 0; i < la.size(); ++i) {
        const double d = la[i] - lb[i];
        max_diff = std::max(max_diff, std::fabs(d));
        max_ref  = std::max(max_ref,  std::fabs((double) la[i]));
        sum_diff += d*d;
        sum_ref  += (double) la[i]*(double) la[i];
    }

    const double rel = max_diff/max_ref;
    const double nmse = sum_diff/sum_ref;

    printf("n_gpu_layers = %2d: max |logit| = %8.3f, max diff = %10.3e, relative = %10.3e, nmse = %10.3e\n",
            n_gpu_layers, max_ref, max_diff, rel, nmse);

    if ((exact && la != lb) || rel > 1e-3 || nmse > 1e-6) {
        printf("FAIL: the folded model does not reproduce the plain model\n");
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string dir = argc > 1 ? argv[1] : ".";

    const std::string plain_path  = dir + "/sdht-plain.gguf";
    const std::string folded_path = dir + "/sdht-folded.gguf";

    llama_backend_init();
    if (getenv("SDHT_LOG") == nullptr) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    printf("writing %s\n", plain_path.c_str());
    if (!write_model(plain_path, false)) {
        printf("FAIL: could not write %s\n", plain_path.c_str());
        return 1;
    }
    printf("writing %s\n", folded_path.c_str());
    if (!write_model(folded_path, true)) {
        printf("FAIL: could not write %s\n", folded_path.c_str());
        return 1;
    }

    printf("plain:  %s\n", plain_path.c_str());
    printf("folded: %s\n", folded_path.c_str());

    bool ok = true;
    ok = compare(plain_path, folded_path,  0) && ok; // CPU: the rotation is a plain matmul
    ok = compare(plain_path, folded_path, 99) && ok; // GPU: the rotation goes through the DHT op

    const std::string legacy_path = dir + "/sdht-legacy.gguf";
    const std::string mixed_path = dir + "/sdht-mixed.gguf";
    const std::string incomplete_path = dir + "/sdht-incomplete.gguf";
    ok = write_model(legacy_path, true, true) && ok;
    ok = compare(folded_path, legacy_path, 0, true) && ok;
    ok = compare(folded_path, legacy_path, 99, true) && ok;
    for (bool incomplete : { false, true }) {
        const std::string & path = incomplete ? incomplete_path : mixed_path;
        ok = write_model(path, true, false, !incomplete, incomplete) && ok;
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(path.c_str(), params);
        if (model != nullptr) {
            printf("FAIL: invalid rotation metadata was accepted: %s\n", path.c_str());
            llama_model_free(model);
            ok = false;
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
