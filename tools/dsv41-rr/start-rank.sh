#!/usr/bin/env bash
# Copyright 2026 ywytrew. SPDX-License-Identifier: Apache-2.0
# Generalized from the qualified local deployment, using Bertholomus AI's runtime recipe.
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[01]$ ]]; then
  echo 'Usage: start-single.sh {0|1} or start-dual.sh {0|1}; start rank 1 first.' >&2
  exit 2
fi
rank=$1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${HEAD:?Set HEAD to the head node interconnect address}"
: "${RDMA_GID_INDEX:?Set RDMA_GID_INDEX after checking both nodes}"
: "${HCA:?Use a cable wrapper or set HCA}"
[[ $RDMA_GID_INDEX =~ ^[0-9]+$ ]]
IFACE=${IFACE:-enp1s0f1np1}
DSV41_HOME=${DSV41_HOME:-$HOME/dsv41}
MODEL_DIR=$(realpath -e -- "${MODEL_DIR:-$DSV41_HOME/ckpt}")
ENGRAM_DIR=$(realpath -e -- "${ENGRAM_DIR:-$DSV41_HOME/engram}")
CACHE_DIR=$(realpath -e -- "${CACHE_DIR:-$DSV41_HOME/cache}")
KIT_DIR=$(realpath -e -- "${KIT_DIR:-$DSV41_HOME/kit-from-hf/kit}")
OUT_DIR=$(realpath -e -- "${OUT_DIR:-$ROOT/build-out-rr}")
IMAGE=${IMAGE:-dsv41-tp2-engine:local}
NAME=${NAME:-dsv41-rr-rank$rank}
CTX=${CTX:-1048576}; PCACHE=${PCACHE:-2}; PARALLEL=${PARALLEL:-4}
TP_PORT=${TP_PORT:-29620}; PORT=${PORT:-18891}
[[ $CTX =~ ^[1-9][0-9]*$ && $PARALLEL =~ ^[1-9][0-9]*$ ]]
[[ $TP_PORT =~ ^[0-9]+$ && $PORT =~ ^[0-9]+$ && $PORT != 8001 ]]
for dir in "$MODEL_DIR" "$ENGRAM_DIR" "$CACHE_DIR" "$KIT_DIR" "$OUT_DIR"; do test -d "$dir"; done
test -f "$CACHE_DIR/dsv41_token_map.json"
test -x "$OUT_DIR/native/bin/tensorfold-native"
test -x "$OUT_DIR/bin/tf-dsv41-lanes"
args=(docker run -d --name "$NAME" --gpus all --network host --ipc host
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/cache:ro" -v "$KIT_DIR:/kit:ro" -v "$OUT_DIR:/out:ro"
  -e TF_DS_REPLAY_FLOOR=1 -e "NCCL_IB_HCA=$HCA" -e "NCCL_IB_GID_INDEX=$RDMA_GID_INDEX"
  -e "NCCL_SOCKET_IFNAME=$IFACE" -e "TF_RDMA_GID_INDEX=$RDMA_GID_INDEX")
HF_HUB=${HF_HUB:-$HOME/.cache/huggingface/hub}
EARG=/engram
if [[ -d $HF_HUB ]]; then
  HF_HUB=$(realpath -e -- "$HF_HUB")
  if [[ $ENGRAM_DIR == "$HF_HUB/"* ]]; then
    args+=(-v "$HF_HUB:/hub:ro")
    EARG=/hub/${ENGRAM_DIR#"$HF_HUB/"}
  fi
fi
if [[ $EARG == /engram ]]; then args+=(-v "$ENGRAM_DIR:/engram:ro"); fi
if [[ $rank == 1 ]]; then
  args+=("$IMAGE" /out/bin/tf-dsv41-lanes /model /cache/tf-ds-rank 1 2 "$HEAD" "$TP_PORT" /kit
    --engram "$EARG" --token-map /cache/dsv41_token_map.json --pool "$CTX" --rdma "$HCA"
    --graphs 1 --side 1 --prefetch 1 --aio 1)
else
  args+=(-e "TF_DS_PREFILL_INTERLEAVE=${TF_DS_PREFILL_INTERLEAVE:-1}"
    -e "TF_DS_PREFILL_SERIAL=${TF_DS_PREFILL_SERIAL:-0}" -e "TF_DS_PREFILL_TRACE=${TF_DS_PREFILL_TRACE:-0}"
    -e TF_TP_WORLD=2 -e "TF_TP_MASTER=$HEAD" -e "TF_TP_PORT=$TP_PORT"
    -e TF_DS_RANK_CACHE=/cache/tf-ds-rank -e TF_DS_KIT=/kit -e TENSORFOLD_CUDA_KERNELS=/kit/aot
    -e "TF_DS_ENGRAM=$EARG" -e TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json
    -e "TF_RDMA_DEVICES=$HCA" -e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=1
    -e TF_DS_L2_PREFETCH=1 -e TF_DS_ENGRAM_AIO=1)
  auth=()
  if [[ -n ${API_KEY_FILE:-} ]]; then
    API_KEY_FILE=$(realpath -e -- "$API_KEY_FILE"); test -f "$API_KEY_FILE"
    args+=(-v "$API_KEY_FILE:/api-key:ro"); auth=(--api-key-file /api-key)
  fi
  args+=("$IMAGE" /out/native/bin/tensorfold-native serve /model --host 127.0.0.1 --port "$PORT"
    --name DeepSeek-V4.1-Flash-TF --alias dsv41-rr "${auth[@]}" --context "$CTX" --parallel "$PARALLEL"
    --thinking --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --backend cuda --prompt-cache-gib "$PCACHE")
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${args[@]}"; printf '\n'; else exec "${args[@]}"; fi
