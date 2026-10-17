# Credits and provenance

This repository is a derivative of existing projects, not an original inference engine by its maintainer.

| Work | Original author / source | Relationship to this fork |
|---|---|---|
| TensorFold engine | [ashhart / TensorFold contributors](https://github.com/ashhart/TensorFold) | Apache-2.0 engine; existing notices and prior MIT notices retained. The DSV41 fork identifies its base as TensorFold 1.0.2, `f8fe17d`. |
| Native Zig DeepSeek-V4.1 family, TP2 / GB10 implementation | **Bertholomus AI (Albert Lee)**, [bertholomus/TensorFold, deepseek-v41-zig](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) | Direct upstream. Our public baseline is `dfc8fe50af303fcbdda0e839cae3a18eb53be2e1`; its history is retained. |
| Deployment recipe and kernel-kit preparation | [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | Original deployment instructions. Runtime assets remain separate and retain their own licenses. |
| DeepSeek-V4.1-Flash model | [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Model and Engram inputs; not distributed here. |
| EXL3 quantization used in local tests | [Mia-AiLab](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | Runtime weights; not distributed here. |
| EXL3 format and ecosystem | [Turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | See upstream third-party notices. |
| Earlier vLLM deployment reference supplied by the user | [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | Background deployment reference, not the source of our scheduler patch. No code from this kit was incorporated into our added changes. Its license is not replaced by this repository's license. |
| Local changes and measurements | [ywytrew](https://github.com/ywytrew), with Codex assistance | Image-content cache key, metrics compatibility, cache-conflict repair, cooperative prefill scheduling, tests and documentation. These additions are Apache-2.0. |

The existing [DSV41 ATTRIBUTION](tools/dsv41-zig/ATTRIBUTION.md) describes Bertholomus AI's implementation sources and clean-room statement. We preserve that statement as upstream attribution; it is not a claim that we authored their model implementation.

`LICENSE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, and `tools/dsv41-zig/ATTRIBUTION.md` are retained byte-for-byte from the qualified source. `NOTICE` keeps its previous content and appends a section describing this fork. `README.upstream.md` preserves the previous README. No endorsement by the original authors is implied.
