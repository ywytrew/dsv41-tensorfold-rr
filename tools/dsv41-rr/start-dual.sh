#!/usr/bin/env bash
# Copyright 2026 ywytrew. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export HCA=${HCA:-rocep1s0f0,rocep1s0f1}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/start-rank.sh" "$@"
