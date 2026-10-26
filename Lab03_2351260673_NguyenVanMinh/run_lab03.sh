#!/usr/bin/env bash
# Chạy pipeline Lab 3 (nhận dạng từ lệnh bằng Gaussian HMM) từ đầu đến cuối.
#
#   ./run_lab03.sh [bước ...]
#
# Bước: setup | check | train | eval | dtw | lm | exp | report | test | notebook | clean
#       all = check train eval dtw lm exp report   (mặc định khi không truyền bước)
#
# Biến môi trường:
#   PYTHON=python3.11   interpreter dùng để tạo .venv
#   SKIP_SETUP=1        không tạo/cài .venv (dùng môi trường đang kích hoạt)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
VENV=".venv"
LOG_DIR="results/logs"
export MPLBACKEND=Agg PYTHONHASHSEED=0

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }

setup() {
  if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
    echo "[setup] SKIP_SETUP=1 -> dùng môi trường hiện tại"; return
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[setup] tạo $VENV bằng $PYTHON"
    "$PYTHON" -m venv "$VENV"
  fi
  # Chỉ cài lại khi requirements.txt thay đổi.
  local stamp="$VENV/.requirements.sha"
  local sum; sum="$(sha1sum requirements.txt | cut -d' ' -f1)"
  if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$sum" ]]; then
    echo "[setup] cài thư viện từ requirements.txt"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
    echo "$sum" > "$stamp"
  fi
}

py() {
  if [[ "${SKIP_SETUP:-0}" == "1" ]]; then python "$@"; else "$VENV/bin/python" "$@"; fi
}

run_module() {  # run_module <module> [args...]
  local mod="$1"; shift
  echo; echo "==================== $mod ===================="
  py -m "scripts.$mod" "$@" 2>&1 | tee -a "$LOG_DIR/pipeline.log"
  return "${PIPESTATUS[0]}"
}

run_step() {
  case "$1" in
    setup)    setup ;;
    check)    run_module check_data ;;
    train)    run_module train ;;
    eval)     run_module evaluate_hmm ;;
    dtw)      run_module evaluate_dtw ;;
    lm)       run_module lm_rerank ;;
    exp)      run_module experiments ;;
    report)   run_module report ;;
    test)     py -m pytest -q tests ;;
    notebook) py -m jupyter nbconvert --to notebook --execute --inplace Lab03.ipynb ;;
    clean)    rm -rf models/*.pkl models/meta.json results/*.png results/*.csv results/*.json results/*.md "$LOG_DIR" ;;
    *) echo "Bước không hợp lệ: $1" >&2; usage >&2; exit 2 ;;
  esac
}

steps=("$@")
if [[ "${steps[0]:-}" =~ ^(-h|--help|help)$ ]]; then usage; exit 0; fi
ALL_STEPS=(check train eval dtw lm exp report)
[[ ${#steps[@]} -eq 0 ]] && steps=(all)
expanded=()
for s in "${steps[@]}"; do
  if [[ "$s" == "all" ]]; then expanded+=("${ALL_STEPS[@]}"); else expanded+=("$s"); fi
done
steps=("${expanded[@]}")

mkdir -p "$LOG_DIR"
if [[ "${steps[0]}" != "setup" && "${steps[0]}" != "clean" ]]; then
  setup
fi

start=$SECONDS
for s in "${steps[@]}"; do run_step "$s"; done
echo; echo "Hoàn tất (${steps[*]}) sau $((SECONDS - start)) s. Kết quả: results/, mô hình: models/"
