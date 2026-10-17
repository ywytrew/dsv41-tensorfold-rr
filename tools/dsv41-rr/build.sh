#!/usr/bin/env bash
# Copyright 2026 ywytrew. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${ZIG_DIR:?Set ZIG_DIR to the Linux aarch64 Zig 0.17.0 directory}"
IMAGE=${IMAGE:-dsv41-tp2-engine:local}
OUT_DIR=${OUT_DIR:-$ROOT/build-out-rr}
BUILD_CACHE=${BUILD_CACHE:-$ROOT/.zig-cache/rr-build}
test -x "$ZIG_DIR/zig"
test "$("$ZIG_DIR/zig" version)" = 0.17.0
docker image inspect "$IMAGE" >/dev/null
mkdir -p -- "$OUT_DIR" "$BUILD_CACHE"
ZIG_DIR=$(realpath -e -- "$ZIG_DIR")
OUT_DIR=$(realpath -e -- "$OUT_DIR")
BUILD_CACHE=$(realpath -e -- "$BUILD_CACHE")
run() {
  docker run --rm --network none --cpus 2 --memory 4g --memory-swap 4g \
    -e ZIG_GLOBAL_CACHE_DIR=/build-cache/global \
    -v "$ROOT:/work" -v "$ZIG_DIR:/zig:ro" -v "$BUILD_CACHE:/build-cache" \
    -v "$OUT_DIR:/output" -w /work --entrypoint /bin/bash "$IMAGE" -lc "$1"
}
run '/zig/zig build test -j1 --cache-dir /build-cache/cache --summary all'
run '/zig/zig build install native -j1 -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix /output --cache-dir /build-cache/cache --summary all'
sha256sum "$OUT_DIR/native/bin/tensorfold-native" "$OUT_DIR/bin/tf-dsv41-lanes"
