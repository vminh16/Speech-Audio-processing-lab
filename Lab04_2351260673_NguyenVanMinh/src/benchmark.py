"""
TTS Performance and Metrics Benchmark Module.
CSE457 - Lab 04: Text-to-Speech
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from tabulate import tabulate

from src.concatenative import build_baseline_sequence
from src.smoothing import crossfade_sequence, compute_spectral_discontinuity
from src.text_norm import normalize_queue_text


DEFAULT_BENCHMARK_SENTENCES = [
    "Mời khách hàng số 23 đến quầy 5.",
    "Mời khách hàng số 08 đến quầy 10.",
    "Mời khách hàng số 14 đến quầy 2.",
]


def get_directory_size_mb(directory: str) -> float:
    """Tính tổng dung lượng thư mục theo đơn vị Megabyte (MB)."""
    p = Path(directory)
    if not p.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def run_tts_benchmark(
    sentences: Optional[List[str]] = None,
    units_dir: str = "data/units",
    sr: int = 16000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Thực hiện đo đạc Benchmark toàn diện cho hệ thống TTS:
    1. Độ trễ (Latency): Thời gian chuẩn hóa text + nạp unit + ghép nối/làm mượt.
    2. Real-Time Factor (RTF): Tỷ số giữa thời gian xử lý và thời lượng âm thanh phát sinh.
    3. Spectral Discontinuity (D_spec): Độ gián đoạn phổ trước và sau khi làm mượt.
    4. Memory & Disk Footprint: Dung lượng lưu trữ của kho inventory.
    """
    if sentences is None:
        sentences = DEFAULT_BENCHMARK_SENTENCES

    results = []
    window_samples = int(sr * 0.025)  # 25ms window

    if verbose:
        print("\n" + "=" * 78)
        print("    CSE457 LAB 04 - BENCHMARK HIỆU NĂNG TỔNG HỢP TIẾNG NÓI (TTS)")
        print("=" * 78)

    for idx, sentence in enumerate(sentences, 1):
        # 1. Đo thời gian End-to-End
        t_start = time.perf_counter()

        # Step A: Text Normalization
        t_norm_start = time.perf_counter()
        tokens = normalize_queue_text(sentence)
        t_norm_ms = (time.perf_counter() - t_norm_start) * 1000.0

        # Step B: Load Units
        t_load_start = time.perf_counter()
        units = build_baseline_sequence(tokens, units_dir=units_dir, candidate_id="01")
        t_load_ms = (time.perf_counter() - t_load_start) * 1000.0

        # Step C: Synthesis (Raw baseline & Smoothed)
        t_synth_start = time.perf_counter()
        y_raw, boundaries_raw = crossfade_sequence(units, sr=sr, xfade_ms=0.0)
        y_smooth, boundaries_smooth = crossfade_sequence(units, sr=sr, xfade_ms=15.0)
        t_synth_ms = (time.perf_counter() - t_synth_start) * 1000.0

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 2. Đo các chỉ số âm học
        audio_dur_sec = len(y_smooth) / sr
        rtf = (total_latency_ms / 1000.0) / audio_dur_sec if audio_dur_sec > 0 else 0.0

        # Đo D_spec trên raw
        d_raw_list = []
        for b_time in boundaries_raw:
            b_idx = int(b_time * sr)
            if b_idx >= window_samples and (b_idx + window_samples) <= len(y_raw):
                d_val = compute_spectral_discontinuity(
                    y_raw[b_idx - window_samples : b_idx],
                    y_raw[b_idx : b_idx + window_samples],
                    sr=sr,
                )
                d_raw_list.append(d_val)

        # Đo D_spec trên smooth (15ms)
        d_smooth_list = []
        for b_time in boundaries_smooth:
            b_idx = int(b_time * sr)
            if b_idx >= window_samples and (b_idx + window_samples) <= len(y_smooth):
                d_val = compute_spectral_discontinuity(
                    y_smooth[b_idx - window_samples : b_idx],
                    y_smooth[b_idx : b_idx + window_samples],
                    sr=sr,
                )
                d_smooth_list.append(d_val)

        avg_d_raw = float(np.mean(d_raw_list)) if d_raw_list else 0.0
        avg_d_smooth = float(np.mean(d_smooth_list)) if d_smooth_list else 0.0
        d_reduction_pct = ((avg_d_raw - avg_d_smooth) / avg_d_raw * 100.0) if avg_d_raw > 0 else 0.0

        results.append({
            "id": idx,
            "sentence": sentence,
            "token_count": len(tokens),
            "norm_ms": round(t_norm_ms, 2),
            "load_ms": round(t_load_ms, 2),
            "synth_ms": round(t_synth_ms, 2),
            "latency_ms": round(total_latency_ms, 2),
            "audio_duration_sec": round(audio_dur_sec, 2),
            "rtf": round(rtf, 4),
            "d_spec_raw": round(avg_d_raw, 2),
            "d_spec_smooth": round(avg_d_smooth, 2),
            "d_reduction_pct": round(d_reduction_pct, 1),
        })

    # Đo footprint của inventory
    footprint_mb = get_directory_size_mb(units_dir)
    total_files = len(list(Path(units_dir).glob("*.wav")))

    if verbose:
        # Bảng 1: Hiệu năng thời gian thực thi (Latency & RTF)
        table_perf = [
            [
                r["id"],
                r["sentence"][:32] + "...",
                r["token_count"],
                f"{r['latency_ms']} ms",
                f"{r['audio_duration_sec']} s",
                f"{r['rtf']}x",
                "Cực nhanh (RTF << 1.0)" if r["rtf"] < 0.1 else "Đạt chuẩn",
            ]
            for r in results
        ]
        print("\n1. BẢNG ĐO THỜI GIAN ĐÁP ỨNG (LATENCY) & REAL-TIME FACTOR (RTF):")
        print(tabulate(
            table_perf,
            headers=["#", "Câu kiểm thử", "Tokens", "Latency (ms)", "Thời lượng (s)", "RTF", "Đánh giá"],
            tablefmt="github",
        ))

        # Bảng 2: Phân tích độ mượt biên nối (D_spec Reduction)
        table_qual = [
            [
                r["id"],
                f"{r['d_spec_raw']} dB²",
                f"{r['d_spec_smooth']} dB²",
                f"-{r['d_reduction_pct']}%",
                "Triệt tiêu tiếng click, biên nối trơn",
            ]
            for r in results
        ]
        print("\n2. BẢNG ĐO ĐỘ GIẢM GIÁN ĐOẠN PHỔ (SPECTRAL DISCONTINUITY - D_spec):")
        print(tabulate(
            table_qual,
            headers=["#", "D_spec Thô (0ms)", "D_spec Mượt (15ms)", "Mức cải thiện (%)", "Hiệu ứng âm học"],
            tablefmt="github",
        ))

        # Bảng 3: Tài nguyên hệ thống (Footprint)
        print("\n3. DUNG LƯỢNG VÀ TÀI NGUYÊN HỆ THỐNG (RESOURCE FOOTPRINT):")
        print(f"  - Thư mục dữ liệu: {units_dir}")
        print(f"  - Tổng số file WAV: {total_files} files (Mono, 16 kHz, 16-bit PCM)")
        print(f"  - Kích thước lưu trữ trên ổ đĩa: {footprint_mb} MB")
        print("=" * 78 + "\n")

    return {
        "sentences": results,
        "units_dir": units_dir,
        "total_files": total_files,
        "footprint_mb": footprint_mb,
    }


if __name__ == "__main__":
    run_tts_benchmark()
