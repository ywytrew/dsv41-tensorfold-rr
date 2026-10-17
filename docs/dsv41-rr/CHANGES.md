# Changes from the public DSV41 baseline

Public upstream: `bertholomus/TensorFold`, `deepseek-v41-zig`,
`dfc8fe50af303fcbdda0e839cae3a18eb53be2e1`.

The retained local commit `c05f1364c9c3c2cedfe64ed8defc27514f8c1d01` adds image-content
cache keys. This publication also retains the qualified local cached-token / vLLM-name
metrics compatibility and model-context field. Those are pre-existing local changes,
not benefits attributed to the new round-robin scheduler.

The cache-conflict fix protects retained snapshot extents during resume and tail writes,
relocates when required, and pins the source while allocating. It does not eliminate
finite-pool allocation failures or introduce the separate pool/GC experiment.

The cooperative scheduler adds an optional backend `prefill_step` callback, partial
stream readiness, round-robin host progression, and forwarding through the CUDA host.
The DSV41 family keeps per-stream PP cursors and safely resumes Engram work, including
the TP `fill_resume` operation. Text yields at 2048-token boundaries with up to eight
decode rounds between turns. Images retain the synchronous path. An opt-in single-pending
PP admission policy remains for comparison; the launch examples select round robin.

The new lane-host tests cover scheduling, decode progress, cancellation, failure and
ContextFull requeue. CUDA wrapper tests cover forwarding of cooperative-prefill options.
The exact qualified logic was already tested; publication only prepends the modification
notice to each changed source file. `qualified-source-sha256.json` records all qualified
source bytes before these notices. No kernels, model weights, quantization or pool size
were changed by the scheduler patch.

## Source files changed from the public baseline

- `zig/src/core/lane_host.zig`
- `zig/src/core/lane_host_prefill_test.zig`
- `zig/src/core/lanes/backend.zig`
- `zig/src/core/lanes/engine.zig`
- `zig/src/core/lanes/stream.zig`
- `zig/src/core/prompt_cache.zig`
- `zig/src/core/prompt_cache_test.zig`
- `zig/src/families/dsv41/lanes.zig`
- `zig/src/families/dsv41/native.zig`
- `zig/src/native/cuda.zig`
- `zig/src/native/flashnext_host.zig`
- `zig/src/server/chat.zig`
- `zig/src/server/metrics.zig`
- `zig/src/server/prompt.zig`
- `zig/src/server/server.zig`
- `zig/src/server/status_routes.zig`

## Publication-only additions

- README / CREDITS / NOTICE changes distinguish original authors and this fork.
- `tools/dsv41-rr/` supplies build and configurable single-/dual-cable launch examples.
- `docs/dsv41-rr/` records deployment, numerical results, correctness summary and provenance.
- Source comments identify modified files; executable engine logic is unchanged from qualification.

The new examples do not package the maintainer's gateway, credentials, model files,
machine-specific entrypoints or private request logs. Benchmark summaries describe
bounded local observations, not a general speed or correctness guarantee.
