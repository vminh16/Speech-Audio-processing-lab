"""
Script thực thi Giai đoạn 6: TTS Học sâu với Mô hình Pretrained Offline (Neural TTS).
CSE457 - Lab 04: Text-to-Speech

Nhiệm vụ:
1. Tổng hợp âm thanh cho 3 câu kiểm thử chuẩn:
   - Câu 1: "Mời khách hàng số 23 đến quầy 5."
   - Câu 2: "Mời khách hàng số 10 đến quầy 2."
   - Câu 3: "Khách hàng số 49 đến quầy 8."
2. Đo đạc chính xác thời gian trễ (Latency, ms) và Real-Time Factor (RTF).
3. Xuất file âm thanh chính: audio_output/neural_tts.wav.
4. Xuất biểu đồ Mel-spectrogram trung gian (figures/05_neural_melspec.png).
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from tabulate import tabulate

from src.neural_tts import plot_neural_melspec, synthesize

TEST_SENTENCES = [
    "Mời khách hàng số 23 đến quầy 5.",
    "Mời khách hàng số 10 đến quầy 2.",
    "Khách hàng số 49 đến quầy 8.",
]


def run_phase6(
    test_sentences: List[str] = None,
    output_audio: str = "audio_output/neural_tts.wav",
    output_figure: str = "figures/05_neural_melspec.png",
    sr: int = 16000,
) -> Dict[str, Any]:
    """
    Chạy toàn bộ pipeline Giai đoạn 6 và tổng hợp bảng đo kiểm latency.
    """
    if test_sentences is None:
        test_sentences = TEST_SENTENCES

    os.makedirs(os.path.dirname(os.path.abspath(output_audio)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_figure)), exist_ok=True)

    print("=" * 70)
    print("🚀 GIAI ĐOẠN 6: TTS HỌC SÂU VỚI MÔ HÌNH PRETRAINED OFFLINE (NEURAL TTS)")
    print("=" * 70)

    benchmarks = []
    main_y = None
    main_extra = None

    for idx, sentence in enumerate(test_sentences):
        print(f"[*] [{idx+1}/{len(test_sentences)}] Tổng hợp: \"{sentence}\"")
        y, out_sr, extra = synthesize(sentence, sr=sr)

        benchmarks.append({
            "id": idx + 1,
            "text": sentence,
            "duration_sec": extra["duration_sec"],
            "acoustic_ms": extra["acoustic_latency_ms"],
            "vocoder_ms": extra["vocoder_latency_ms"],
            "latency_ms": extra["latency_ms"],
            "rtf": extra["rtf"],
        })

        if idx == 0:
            main_y = y
            main_extra = extra
            sf.write(output_audio, y, sr, subtype="PCM_16")
            print(f"    -> Đã lưu file âm thanh câu 1: {output_audio}")

    # Vẽ biểu đồ Mel-spectrogram của câu 1
    print(f"\n[*] Vẽ biểu đồ Mel-spectrogram trung gian: {output_figure}...")
    plot_neural_melspec(
        mel_db=main_extra["mel"],
        y_audio=main_y,
        text=test_sentences[0],
        sr=sr,
        output_fig=output_figure,
    )
    print(f"    -> Đã lưu biểu đồ thành công: {output_figure}")

    # Bảng số liệu đo kiểm Latency & RTF
    table_rows = []
    for b in benchmarks:
        table_rows.append([
            f"Câu {b['id']}",
            f"{b['duration_sec']:.2f}s",
            "Integrated VITS",
            "Integrated VITS",
            f"{b['latency_ms']:.1f} ms",
            f"{b['rtf']:.4f}x",
        ])

    print("\n--- BẢNG ĐO KIỂM HIỆU NĂNG NEURAL TTS TRÊN 3 CÂU THỬ NGHIỆM ---")
    headers = ["Câu kiểm thử", "Thời lượng", "Acoustic Model", "Neural Vocoder", "Tổng Latency", "RTF"]
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))

    avg_latency = float(np.mean([b["latency_ms"] for b in benchmarks]))
    avg_rtf = float(np.mean([b["rtf"] for b in benchmarks]))
    print(f"\n[+] Latency trung bình: {avg_latency:.2f} ms | RTF trung bình: {avg_rtf:.4f}x")
    print("\n✅ Hoàn thành Giai đoạn 6 thành công!")

    return {
        "status": "success",
        "output_audio": output_audio,
        "output_figure": output_figure,
        "benchmarks": benchmarks,
        "avg_latency_ms": avg_latency,
        "avg_rtf": avg_rtf,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Giai đoạn 6: Offline Neural TTS")
    parser.add_argument("--output-audio", default="audio_output/neural_tts.wav", help="File âm thanh xuất ra")
    parser.add_argument("--output-fig", default="figures/05_neural_melspec.png", help="File ảnh Mel-spectrogram")
    args = parser.parse_args()

    run_phase6(
        output_audio=args.output_audio,
        output_figure=args.output_fig,
    )
