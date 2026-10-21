# SDHT and TQ1_0_128 implementation notes

## Scope

This source snapshot adds inference support for a model-specific quantization representation. The complete weight calibration, allocation, and QAD recovery process is outside this runtime repository. In particular, fitted quantization inside the runtime and the Python reference codec are different encoding routes; neither alone recreates the released model's calibrated weights.

## Ternary block layout

`GGML_TYPE_TQ1_0_128` uses local type ID `43` and a 128-weight block:

| Field | Bytes | Meaning |
| --- | ---: | --- |
| `qs` | 24 | Base-3 packed codes for the first 120 weights |
| `qh` | 2 | Packed codes for the last 8 weights |
| `d` | 2 | FP16 block scale, at byte offset 26 |
| Total | 28 | 1.75 bits per weight |

Decoded values are `d * q`, where `q` is one of `-1`, `0`, and `1`. The byte order and symbol mapping are defined in `ggml/src/ggml-quants.c` and `gguf-py/gguf/quants.py`. Do not interpret this block as the upstream 256-weight `TQ1_0` format or move the scale to the start of the block. Custom numerical enum values are not a guarantee of compatibility with future upstream formats.

The native fitted encoder searches ternary assignments and scales with optional importance weights and includes stored FP16 rounding in its selection. The reference encoder remains useful for deterministic codec checks. A runtime rebuild does not re-encode an existing GGUF.

## Folded smoothing and rotation

For a weight matrix `W`, a positive diagonal `D`, and a symmetric orthonormal Hartley matrix `H`, the folded representation is:

```text
W_folded = W * D * H
x_runtime = H * D^-1 * x
W_folded * x_runtime = W * x
```

The equality holds before quantization and floating-point error. Applying the transform to the input is required when the weights were exported in that basis. Changing quantized weight bits cannot substitute for this graph operation.

New GGUF files use the neutral `quantization.rotation.*` namespace. Accepted Hartley transform strings are `normalized-smooth-discrete-hartley` and `normalized-discrete-hartley`, with `axis=input-last-dimension` and `sign_mode=identity`. The loader also retains the separately supported Walsh-Hadamard transform route. Keep the declared transform and packed weights consistent.

Weights listed in `quantization.rotation.diagonal_weight_names` require a companion tensor named `<weight-name>.input_diag`: an FP16 vector with one value per input channel. It stores the **inverse** diagonal used at runtime. The loader validates the tensor's shape and type, expands it for execution, and attaches it to the transform. Model metadata names, version checks, tied embedding/output handling, and allowed graph paths are validated in `src/llama-model.cpp`.

`src/llama-graph.cpp` applies the inverse diagonal and then the rotation before the relevant matrix multiplication. Its transformed-input cache includes the input tensor, rotation, and diagonal; a cache that ignores the diagonal can reuse the wrong activation.


### Metadata compatibility and migration

The key suffixes retain their existing meanings: `version`, `tied_output`, `block_size`, `transform`, `axis`, `sign_mode`, `sign_widths`, `sign_values`, `weight_names`, `inverse_weight_names`, `diagonal_weight_names`, and `gdn_v_grouped`. Versions 1 and 2 keep their existing semantics. The Python writer exposes `Keys.Rotation` and `add_rotation_*` methods. New converters write only the canonical namespace.

The loader accepts the previous `prism.hadamard.*` namespace as a complete legacy contract. It rejects files containing keys from both namespaces, even if some values agree, and rejects rotation metadata without a version. This avoids silently mixing transforms or ignoring partial metadata. The legacy spelling exists only for reading and testing old files, rather than as the name of this format.

Existing models do not need to be rewritten to run with this runtime. To create a new file with clean metadata names, install this repository's Python package and run:

```sh
python -m pip install ./gguf-py
python -m gguf.scripts.gguf_migrate_rotation input.gguf output.gguf
```

The migration command requires a new output path, refuses to overwrite files, and copies the tensor-info table and entire tensor payload without re-encoding. Metadata value bytes and tensor names are preserved; only metadata key names and resulting header alignment change. It prints the payload SHA-256. For a large model it reads and writes the whole file, so allow enough disk space. Older runtime builds that recognize only the legacy namespace must be upgraded before loading the renamed file.

## GPU implementation

`ggml/src/ggml-cuda/dht.cu` implements the normalized Hartley transform through a small FFT, using `Re(FFT) - Im(FFT)`. Dedicated widths are 64, 128, 256, and 512. The optimized path requires compatible tensor shapes, types, and contiguity; other graph paths retain matrix multiplication behavior. HIP compiles the shared CUDA-style backend source through its compatibility layer.

The custom ternary GPU matrix-vector path and dequantized fallback are implemented in the shared backend files `vecdotq.cuh`, `mmvq.cu`, `dequantize.cuh`, and `convert.cu`. The environment switch `GGML_TQ1_128_NO_MMVQ` selects the fallback for diagnostic comparisons. It is not needed for normal inference.

## Key source locations

| Area | Files |
| --- | --- |
| Type and packed layout | `ggml/include/ggml.h`, `ggml/src/ggml-common.h` |
| Native codec | `ggml/src/ggml-quants.c`, `ggml/src/ggml-quants.h` |
| CPU dot product | `ggml/src/ggml-cpu/quants.c` |
| GPU DHT | `ggml/src/ggml-cuda/dht.cu`, `dht.cuh` |
| GPU ternary kernels | `ggml/src/ggml-cuda/vecdotq.cuh`, `mmvq.cu`, `dequantize.cuh`, `convert.cu` |
| GGUF Python definitions | `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py` |
| Metadata loading | `src/llama-model.cpp`, `src/llama-model-loader.cpp`, `src/llama-arch.cpp` |
| Input graph transform | `src/llama-graph.cpp` |
| Synthetic tests | `tests/test-sdht.cpp`, `tests/test-tq1-0-128.cpp` |

See `sdht-runtime.patch` for the complete runtime change relative to the pinned upstream base. That patch excludes publication documentation and workflow suffix changes.
