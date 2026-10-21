# Upstream provenance

This is an independently published derivative of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), rather than a separate implementation of the base runtime.

| Item | Identity |
| --- | --- |
| Upstream base | `6afa9a3134dc4833ab58e625464aa67e5fc7eea5` |
| Custom runtime source | `3a75675bffeeaa3c8a35c192d7324fed7bcd8939` |
| Custom runtime delta | 38 files; see `sdht-runtime.patch` |
| Runtime license | MIT; see `LICENSE` |

The source was exported from the custom runtime commit, without its local `.git` directory or compiled build trees. Upstream tokenizer vocabulary GGUF fixtures remain under `models/` because they are part of the upstream source/test distribution. They are not the LFM model checkpoint.

The complete upstream README is retained as `README-upstream.md`, and bundled licenses and copyright notices are preserved. The upstream source snapshot already contains other upstream features, including the earlier folded-basis infrastructure; the SDHT patch does not claim authorship of them.

Publication adds a project README, these provenance notes, `docs/SDHT.md`, a machine-readable `SOURCE_PROVENANCE.json`, the runtime patch, and local credential/checkpoint ignore patterns. Upstream Actions workflow filenames gain a `.disabled` suffix; their contents are unchanged. No inference source is changed by publication preparation.

The original local checkout and its upstream remote remain separate. This source tree can be initialized in the new `llama.cpp-SDHT` repository without connecting that working runtime checkout to a new remote.

Report issues specific to this derivative in its repository. Upstream contribution policies and upstream model/backend support should not be interpreted as support commitments for these custom extensions.
