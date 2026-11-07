#!/usr/bin/env bash
set -e

# ==============================================================================
# Script: run_pipeline.sh
# Purpose: Thực thi toàn bộ pipeline Lab 04 TTS (A -> G) và xuất kết quả
# ==============================================================================

# Kích hoạt .venv nếu có
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Đảm bảo các thư mục đầu ra tồn tại
mkdir -p audio_output figures

echo "=== [1/2] Kiểm tra thư mục dữ liệu và đầu ra ==="
echo "Output directories: audio_output/, figures/"

echo "=== [2/2] Chạy pipeline tổng hợp tiếng nói ==="
python3 -m src.pipeline "$@"

echo "=== Hoàn thành pipeline! Kiểm tra kết quả trong audio_output/ và figures/ ==="
