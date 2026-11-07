"""
Execution Script for Phase 2: Baseline Concatenation and Cross-fade Smoothing.
CSE457 - Lab 04: Text-to-Speech
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from tabulate import tabulate

from src.concatenative import (
    TOKEN_TO_FILENAME,
    build_baseline_sequence,
    synthesize_baseline,
)
from src.smoothing import (
    crossfade_sequence,
    evaluate_crossfade_experiment,
)
from src.text_norm import normalize_queue_text
from src.utils import plot_boundary_spec


def run_phase2(
    text: str = "Mời khách hàng số 23 đến quầy 5.",
    units_dir: str = "data/units",
    output_audio_dir: str = "audio_output",
    output_figure: str = "figures/02_boundary_spec.png",
    sr: int = 16000,
) -> dict:
    """
    Thực thi toàn bộ Giai đoạn 2:
    1. Chuẩn hóa văn bản đầu vào.
    2. Nạp các unit candidate _01.
    3. Ghép nối baseline xfade = 0ms -> audio_output/raw_concat.wav.
    4. Ghép nối làm mượt xfade = 15ms -> audio_output/best_concat.wav.
    5. Thực hiện Thí nghiệm 1 (khảo sát xfade 0, 5, 15, 30 ms).
    6. Vẽ và xuất Hình 02 -> figures/02_boundary_spec.png.
    """
    print(f"=== [Giai đoạn 2] Bắt đầu xử lý câu: '{text}' ===")
    tokens = normalize_queue_text(text)
    print(f"Chuỗi token đã chuẩn hóa: {tokens}")

    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_figure)), exist_ok=True)

    # Kiểm tra xem các file unit đã có trong data/units/ chưa
    missing_tokens = []
    for tok in tokens:
        fname = f"{TOKEN_TO_FILENAME.get(tok, tok)}_01.wav"
        if not (Path(units_dir) / fname).exists():
            missing_tokens.append(fname)

    if missing_tokens:
        print(f"\n[LƯU Ý] Chưa tìm thấy {len(missing_tokens)} file âm thanh trong {units_dir}:")
        print(f"  {missing_tokens[:5]} ... (tổng {len(missing_tokens)} file)")
        print("  -> Hệ thống đang chờ bạn đặt các file WAV tự thu âm vào thư mục 'data/units/'.")
        return {"status": "waiting_for_audio", "missing": missing_tokens}

    # 1. Nạp units
    units = build_baseline_sequence(tokens, units_dir=units_dir, candidate_id="01")

    # 2. Sinh Baseline (xfade = 0ms)
    y_raw, boundaries_raw = synthesize_baseline(tokens, units_dir=units_dir, sr=sr)
    raw_path = os.path.join(output_audio_dir, "raw_concat.wav")
    sf.write(raw_path, y_raw, sr)
    print(f"-> Đã xuất Baseline Audio (xfade = 0 ms): {raw_path} ({len(y_raw)/sr:.2f}s)")

    # 3. Sinh Làm mượt (xfade = 15ms)
    y_smooth, boundaries_smooth = crossfade_sequence(units, sr=sr, xfade_ms=15.0)
    smooth_path = os.path.join(output_audio_dir, "best_concat.wav")
    sf.write(smooth_path, y_smooth, sr)
    print(f"-> Đã xuất Smoothed Audio (xfade = 15 ms): {smooth_path} ({len(y_smooth)/sr:.2f}s)")

    # 4. Thực hiện Thí nghiệm 1
    print("\n--- KẾT QUẢ THÍ NGHIỆM 1: KHẢO SÁT ĐỘ DÀI CROSS-FADE ---")
    exp_results = evaluate_crossfade_experiment(units, sr=sr, xfade_levels=[0.0, 5.0, 15.0, 30.0])
    table_data = [
        [r["xfade_ms"], r["mean_d_spec"], r["num_boundaries"], r["audio_length_sec"]]
        for r in exp_results
    ]
    print(tabulate(table_data, headers=["XFade (ms)", "Mean D_spec (dB^2)", "Số biên nối", "Độ dài audio (s)"], tablefmt="github"))

    # 5. Vẽ Hình 02
    plot_boundary_spec(
        y_raw=y_raw,
        y_smooth=y_smooth,
        boundaries_raw=boundaries_raw,
        boundaries_smooth=boundaries_smooth,
        sr=sr,
        output_fig=output_figure,
        tokens=tokens,
    )
    print(f"-> Đã lưu Hình 02 phân tích biên nối: {output_figure}")

    return {
        "status": "success",
        "raw_audio": raw_path,
        "smooth_audio": smooth_path,
        "figure": output_figure,
        "experiment_results": exp_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Giai đoạn 2 - Lab 4 TTS")
    parser.add_argument("--text", default="Mời khách hàng số 23 đến quầy 5.", help="Văn bản thông báo kiểm thử")
    parser.add_argument("--units-dir", default="data/units", help="Thư mục chứa units")
    parser.add_argument("--out-dir", default="audio_output", help="Thư mục xuất file âm thanh")
    parser.add_argument("--out-fig", default="figures/02_boundary_spec.png", help="File ảnh đồ thị Hình 02")
    args = parser.parse_args()

    run_phase2(
        text=args.text,
        units_dir=args.units_dir,
        output_audio_dir=args.out_dir,
        output_figure=args.out_fig,
    )


if __name__ == "__main__":
    main()
