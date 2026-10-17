# Build and deployment notes

Use a Linux checkout: upstream includes golden-test filenames that Windows filesystems cannot represent. Build and run on Linux aarch64. The tested toolchain was Zig 0.17.0 and the original recipe's CUDA environment based on `nvcr.io/nvidia/pytorch:26.07-py3`; our prepared local image was named `dsv41-tp2-engine:local`. That local image name is not a public downloadable image. Prepare the environment using [the original recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) and set `IMAGE` to your image.

`tools/dsv41-rr/build.sh` runs:

```bash
zig build test -j1 --summary all
zig build install native -j1 -Dtarget=aarch64-linux-gnu \
  -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
  --prefix /output --summary all
```

It mounts explicit cache and output directories and does not load a model or request a GPU. Source notices added for publication do not alter engine logic; build hashes can change because debug/source metadata changes. The recorded qualification hashes identify the binaries that were actually tested, not a promise that every rebuild is byte-identical.

## Runtime prerequisites

Use the upstream recipe to prepare the model, kit and rank-specific cache on **both nodes**. Defaults under `DSV41_HOME=$HOME/dsv41`:

| Variable | Default path | Requirement |
|---|---|---|
| `MODEL_DIR` | `ckpt` | EXL3 model directory |
| `ENGRAM_DIR` | `engram` | Original Engram inputs |
| `CACHE_DIR` | `cache` | Includes `tf-ds-rank*` and `dsv41_token_map.json` |
| `KIT_DIR` | `kit-from-hf/kit` | Original kernel kit with generated inputs and its license notices |
| `OUT_DIR` | repository `build-out-rr` | Same release output on both nodes |
| `HF_HUB` | `$HOME/.cache/huggingface/hub` | If Engram is a snapshot inside this tree, the whole hub is mounted so blob symlinks resolve |

Other model/kit symlinks must resolve inside their mounted directories. Verify the kit using the original recipe's manifest before serving.

`HEAD` and `RDMA_GID_INDEX` are required. HCA names and `IFACE` are examples, not automatic hardware detection. The two nodes must agree on TP port, context/pool size, cable selection, and engine build. `CTX=1048576` controls both rank 0 context and rank 1 pool; `PCACHE=2` is GiB. No pool/GC policy changes are included.

Launch the worker first, then the head, using `start-single.sh` or `start-dual.sh`. The scripts do not delete an existing container, restart gateways, or allocate ports on other services. They fail if their container name already exists. `DRY_RUN=1` validates paths and prints the command without running Docker. `API_KEY_FILE=/absolute/file` adds the server's existing key-file authentication; otherwise the example binds loopback without authentication. Keep access behind an appropriately configured gateway or tunnel.

## Modes and rollback

Set these on **rank 0** before launch (the wrappers supply these defaults):

```bash
export TF_DS_PREFILL_INTERLEAVE=1
export TF_DS_PREFILL_SERIAL=0
export TF_DS_PREFILL_TRACE=0
```

Both binaries must be upgraded together: this change adds the TP `fill_resume` operation. Do not mix workers built from a different patch set. In particular, a separate allocator experiment may use a conflicting operation number; it was not merged or qualified here.

For the synchronous scheduling comparison, use `TF_DS_PREFILL_INTERLEAVE=0`. This retains the cache and metrics fixes and is not an exact checkout of public upstream. `INTERLEAVE=1,SERIAL=1` selects the optional single-pending-PP experiment.

Before your own deployment, retain the prior source, binaries and launch scripts. Drain requests before manually replacing a running model. Keep old output directories so you can switch both ranks back to the same old release. The sample scripts deliberately contain no local production routing or site-specific rollback command.

Future experiments should use a dedicated endpoint, distinct container names (`NAME`), an unused `TP_PORT`, and an idle pair of GPUs. The samples reject `PORT=8001`; they do not modify an existing 8001 service. Sharing already-full GPUs is not isolated just because the HTTP port differs.

## Qualification status

The actual production single- and dual-cable entrypoints were each smoke-tested with two concurrent ~13K prompts, 128 outputs each: request/output/token hashes matched and restore failures stayed zero. The long 190K qualification used dual RoCE. Public scripts are generalized from those commands and were checked with shell syntax and dry runs; the local production gateway is not packaged here. See [BENCHMARKS.md](BENCHMARKS.md) for measured scope and limits.
