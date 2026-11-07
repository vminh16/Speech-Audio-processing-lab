#!/usr/bin/env bash
set -e

# ==============================================================================
# Script: setup_env.sh
# Purpose: Khởi tạo và thiết lập môi trường ảo .venv cho Lab 04 TTS
# ==============================================================================

echo "=== [1/4] Kiểm tra phiên bản Python ==="
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Lỗi: Không tìm thấy Python trong hệ thống. Vui lòng cài đặt Python 3.8+!" >&2
    exit 1
fi

$PYTHON_CMD --version

echo "=== [2/4] Khởi tạo môi trường ảo .venv ==="
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "Đã tạo môi trường ảo tại .venv/"
else
    echo "Môi trường ảo .venv/ đã tồn tại."
fi

echo "=== [3/4] Kích hoạt .venv và nâng cấp pip ==="
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "=== [4/4] Cài đặt dependencies từ requirements.txt ==="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Cài đặt hoàn tất thành công!"
else
    echo "Cảnh báo: Không tìm thấy requirements.txt" >&2
fi

echo "=== Hoàn thành! Kích hoạt môi trường bằng lệnh: source .venv/bin/activate ==="
