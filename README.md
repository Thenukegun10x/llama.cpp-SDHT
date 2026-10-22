# llama.cpp-SDHT

Custom llama.cpp runtime for GGUF models with **SDHT smoothing and rotation** and the **TQ1_0_128 ternary weight format**. This is the runtime used for the experimental LFM2.5-24B-A2B ternary QAD model.

Derived from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The upstream MIT license, copyright notices, third-party licenses, and source tree are preserved. This is a separately published derivative, maintained independently from upstream. See [upstream provenance](UPSTREAM.md) and the [original README](README-upstream.md).

## What changes

- `TQ1_0_128`: 128 ternary weights in 28 bytes, including the FP16 block scale, for **1.75 bits per weight at the block level**. A mixed-precision model has a different whole-model BPW.
- CPU quantization, dequantization, and dot products, plus CUDA/HIP dequantization and matrix-vector kernels for the custom type.
- A normalized discrete Hartley transform for folded weight rotations, with a dedicated GPU transform for widths 64, 128, 256, and 512.
- Loading and applying existing FP16 input smoothing diagonals stored alongside the quantized weights.
- GGUF type definitions and Python codec support, model-loader integration, and synthetic end-to-end tests.

The custom runtime type ID and `quantization.rotation.*` metadata are extensions used by these model files. The `transform` field explicitly identifies **Hartley** or Walsh-Hadamard rotation. Existing files with the legacy metadata keys still load. Use this runtime for these models; a stock llama.cpp build may reject the custom type or lack the required transform behavior. See the [format and implementation notes](docs/SDHT.md).

This repository contains inference and codec source. Model checkpoints, teacher caches, Hessian collection, and the complete model-specific PTQ/QAD training pipeline are distributed separately. Running the standard quantizer alone does not reproduce the trained model.

## Build on Windows with AMD HIP

The existing local runtime was built with ROCm 7.1, Visual Studio 2022 x64 Build Tools, CMake, and Ninja, and used on an RX 9070 XT (`gfx1201`). Open an **x64 Native Tools Command Prompt** so the Windows SDK and linker are available. Set the ROCm location to your installation:

```bat
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "ROCM_PATH=%HIP_PATH%"
set "PATH=%HIP_PATH%\bin;%PATH%"
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-hip --config Release --target llama-server llama-cli llama-perplexity test-sdht test-tq1-0-128 --parallel 4
```

Choose `GPU_TARGETS` for your GPU. This example selects only `gfx1201`; it does not promise support for every AMD architecture. Ensure the matching ROCm runtime libraries are on `PATH` when launching the executables.

### Server web UI

The default build tries to obtain upstream prebuilt web assets. Their version lookup can fail for a standalone repository with a short, unrelated commit history. For a source build of the web UI, install Node.js/npm, configure with `-DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF`, and build `llama-server` again. This requires downloading npm dependencies.

For an API-only build with no UI download, configure with `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`. The HTTP API still works, but the embedded chat page is unavailable. See [upstream UI build documentation](tools/ui/README.md).

## CPU build

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cpu --config Release --target llama-server llama-cli llama-perplexity test-sdht test-tq1-0-128 --parallel 4
```

On Windows, use an x64 developer shell and add `-G Ninja` for the same output layout used below. Multi-configuration generators can place executables under `bin/Release` instead of `bin`.

## Run a model

In PowerShell, after building the HIP runtime:

```powershell
# Set this to the device index of your discrete GPU, if needed.
$env:HIP_VISIBLE_DEVICES = '1'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
.\build-hip\bin\llama-server.exe -m 'C:\models\your-model.gguf' --host 127.0.0.1 --port 12345 --ctx-size 131072 --parallel 1 --n-gpu-layers 99 --jinja
```

Device indices depend on your installation; `1` was the discrete GPU on the original native HIP setup. A 131072-token context allocates substantial KV-cache memory. Use a smaller context if needed. The context allocation setting does not establish model quality across that full length.

For other operating systems, use the corresponding executable under `build-hip/bin`. See [server usage](tools/server/README.md) and the [general build guide](docs/build.md) for the inherited runtime options.

## Tests and validation scope

```sh
ctest --test-dir build-hip -C Release -R "^test-(sdht|tq1-0-128)$" --output-on-failure
```

`test-sdht` compares synthetic plain and folded models, checks identical logits for canonical and legacy metadata, and rejects mixed or incomplete namespaces. `test-tq1-0-128` compares CPU, native GPU matrix-vector, and dequantized fallback paths against F16/Q8 controls. The latter test requires a working CUDA/HIP backend to validate distinct GPU paths; it is not a CPU-only pass criterion. Both tests write small synthetic GGUF files in the test working directory.

The runtime builds on source commit `3a75675bffeeaa3c8a35c192d7324fed7bcd8939`. The AMD HIP backend works on the tested Windows setup. The NVIDIA CUDA backend has also been tested and is in working order. The additional custom format is not implemented for every backend inherited from upstream.

Upstream GitHub Actions are retained with `.disabled` suffixes so publication does not automatically launch the upstream build/release matrix. Enable selected workflows after adapting them for this repository.

## License and credit

[MIT](LICENSE), with upstream and bundled third-party notices preserved. Credit goes to the ggml/llama.cpp authors for the base runtime. Model weights have their own license; this runtime's MIT license does not change it.
