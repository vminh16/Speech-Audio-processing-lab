"""
Script thực thi Giai đoạn 5: Tổng hợp Tham số dựa trên HMM (HMM-based Parametric Synthesis).
CSE457 - Lab 04: Text-to-Speech

Nhiệm vụ:
1. Sinh chuỗi trạng thái HMM từ văn bản ("Mời khách hàng số 23 đến quầy 5.").
2. Thực hiện làm mượt quỹ đạo tham số qua thuật toán MLPG (Maximum Likelihood Parameter Generation).
3. Vẽ biểu đồ quỹ đạo tham số âm học (figures/04_hmm_trajectories.png).
4. Khảo sát can thiệp tham số (Tốc độ 1.2x, Dịch Pitch 1.1x trực tiếp trong miền tham số).
5. Xuất file âm thanh tổng hợp: audio_output/hmm_tts.wav.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import soundfile as sf
from tabulate import tabulate

from src.hmm_tts import (
    generate_hmm_trajectories,
    parametric_vocoder,
    plot_hmm_trajectories,
    synthesize_hmm,
)


def run_phase5(
    text: str = "Mời khách hàng số 23 đến quầy 5.",
    output_audio: str = "audio_output/hmm_tts.wav",
    output_figure: str = "figures/04_hmm_trajectories.png",
    sr: int = 16000,
) -> Dict[str, Any]:
    """
    Thực thi Giai đoạn 5 HMM Synthesis và xuất kết quả thực nghiệm.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_audio)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_figure)), exist_ok=True)

    print("=" * 70)
    print("🚀 GIAI ĐOẠN 5: TỔNG HỢP THAM SỐ DỰA TRÊN HMM (HMM PARAMETRIC SYNTHESIS)")
    print("=" * 70)
    print(f"[*] Câu văn bản đầu vào: \"{text}\"")

    t_start = time.perf_counter()
    # 1. Sinh âm thanh bằng HMM synthesize
    y_hmm, out_sr, tracks = synthesize_hmm(text, sr=sr, speed=1.0, pitch_scale=1.0)
    t_synth = (time.perf_counter() - t_start) * 1000.0  # ms

    # 2. Phân tích tác dụng của ràng buộc MLPG (Delta c_t)
    # Đo độ giảm bước nhảy đột ngột (step jump discontinuity) trên Formant F2 và F0
    f2_step_jump = np.max(np.abs(np.diff(tracks["f2_step"])))
    f2_smooth_jump = np.max(np.abs(np.diff(tracks["f2"])))
    f2_jump_reduction = ((f2_step_jump - f2_smooth_jump) / f2_step_jump * 100.0) if f2_step_jump > 0 else 0.0

    print(f"[*] Đã sinh xong âm thanh HMM trong {t_synth:.2f} ms")
    print(f"    - Thời lượng âm thanh: {len(y_hmm)/sr:.2f} giây")
    print(f"    - Bước nhảy lớn nhất F2 (State Step Jump): {f2_step_jump:.1f} Hz/frame")
    print(f"    - Bước nhảy lớn nhất F2 sau MLPG (Delta c_t): {f2_smooth_jump:.1f} Hz/frame")
    print(f"    - Độ triệt tiêu bước nhảy đột ngột F2: {f2_jump_reduction:.1f}%")

    # 3. Lưu file âm thanh chuẩn HMM
    sf.write(output_audio, y_hmm, sr, subtype="PCM_16")
    print(f"[*] Đã lưu file âm thanh tổng hợp: {output_audio}")

    # 4. Thử nghiệm can thiệp trực tiếp tham số (Speed 1.2x, Pitch 1.1x)
    y_fast, _, _ = synthesize_hmm(text, sr=sr, speed=1.2, pitch_scale=1.0)
    y_high, _, _ = synthesize_hmm(text, sr=sr, speed=1.0, pitch_scale=1.1)

    out_fast = os.path.join(os.path.dirname(output_audio), "hmm_tts_speed1.2x.wav")
    out_high = os.path.join(os.path.dirname(output_audio), "hmm_tts_pitch1.1x.wav")
    sf.write(out_fast, y_fast, sr, subtype="PCM_16")
    sf.write(out_high, y_high, sr, subtype="PCM_16")
    print(f"[*] Đã xuất biến thể can thiệp tham số HMM:")
    print(f"    -> Tốc độ 1.2x: {out_fast} ({len(y_fast)/sr:.2f}s)")
    print(f"    -> Dịch pitch 1.1x: {out_high}")

    # 5. Vẽ đồ thị quỹ đạo tham số HMM
    print(f"[*] Vẽ đồ thị quỹ đạo tham số âm học: {output_figure}...")
    plot_hmm_trajectories(tracks=tracks, y_audio=y_hmm, sr=sr, output_fig=output_figure)
    print(f"    -> Đã lưu biểu đồ thành công: {output_figure}")

    # Bảng tổng hợp số liệu
    summary_data = [
        ["Thời gian tổng hợp (Latency)", f"{t_synth:.2f} ms"],
        ["Thời lượng âm thanh (Duration)", f"{len(y_hmm)/sr:.2f} s"],
        ["Số lượng frame tham số (10ms)", f"{len(tracks['f0'])} frames"],
        ["F0 trung bình (Voiced)", f"{np.mean(tracks['f0'][tracks['f0'] > 0]):.1f} Hz"],
        ["Độ giảm bước nhảy đột ngột F2", f"{f2_jump_reduction:.1f} %"],
        ["Dung lượng lưu trữ mô hình", "~ 250 KB (Nhẹ hơn kho sóng mẫu)"],
    ]
    print("\n--- BẢNG THỐNG KÊ HIỆU NĂNG HMM SYNTHESIS ---")
    print(tabulate(summary_data, headers=["Chỉ số", "Giá trị"], tablefmt="fancy_grid"))
    print("\n✅ Hoàn thành Giai đoạn 5 thành công!")

    return {
        "status": "success",
        "output_audio": output_audio,
        "output_figure": output_figure,
        "latency_ms": t_synth,
        "duration_sec": len(y_hmm) / sr,
        "mlpg_smoothing_reduction": f2_jump_reduction,
        "tracks": tracks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Giai đoạn 5: HMM-based Parametric Synthesis")
    parser.add_argument("--text", default="Mời khách hàng số 23 đến quầy 5.", help="Văn bản cần đọc")
    parser.add_argument("--output-audio", default="audio_output/hmm_tts.wav", help="File âm thanh xuất ra")
    parser.add_argument("--output-fig", default="figures/04_hmm_trajectories.png", help="File ảnh quỹ đạo tham số")
    args = parser.parse_args()

    run_phase5(
        text=args.text,
        output_audio=args.output_audio,
        output_figure=args.output_fig,
    )
