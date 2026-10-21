#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_dht(const struct ggml_tensor * op);

// Returns whether the discrete Hartley transform could be used.
bool ggml_cuda_op_dht(ggml_backend_cuda_context & ctx, const struct ggml_tensor * src, struct ggml_tensor * dst);
