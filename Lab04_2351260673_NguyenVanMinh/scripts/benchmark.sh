#!/usr/bin/env bash
set -e

# ==============================================================================
# Script: benchmark.sh
# Purpose: Đo kiểm hiệu năng (Latency, RTF, Discontinuity, Footprint) giữa 3 phương pháp
# ==============================================================================

# Kích hoạt .venv nếu có
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p audio_output figures

echo "=== Bắt đầu chạy Benchmark Lab 04 TTS ==="
python3 -m src.pipeline --benchmark "$@"

echo "=== Hoàn thành Benchmark! ==="
