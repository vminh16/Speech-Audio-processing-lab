"""
Script thực thi Giai đoạn 4: Prosody Modification, Source-Filter Model và Voice Conversion.
CSE457 - Lab 04: Text-to-Speech

Thực hiện:
- Kiểm chứng thực nghiệm công thức F0 = Fs / N_p trên khung nguyên âm hữu thanh.
- Thí nghiệm 3: Co giãn thời lượng phát âm (0.8x và 1.2x tốc độ) giữ nguyên cao độ.
- Thí nghiệm 4: Dịch chuyển cao độ (-2.0 và +2.0 semitones) giữ nguyên thời lượng.
- Biến đổi giọng nói (Voice Conversion):
  + Giọng Nữ giả lập (+5.5 semitones + Formant Warping).
  + Giọng Miền Nam mô phỏng (Kéo dài nguyên âm cuối + Làm mượt đường nét ngữ điệu).
- Xuất Hình 04 (figures/04_prosody_analysis.png) và các file âm thanh vào audio_output/.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf
from tabulate import tabulate

from src.prosody import (
    pitch_shift,
    time_stretch,
    verify_f0_formula,
    voice_conversion_female,
    voice_conversion_southern,
)
from src.utils import plot_prosody_analysis


def run_phase4(
    input_audio: str = "audio_output/best_concat.wav",
    sample_unit: str = "data/units/ba_01.wav",
    output_dir: str = "audio_output",
    output_figure: str = "figures/04_prosody_analysis.png",
    sr: int = 16000,
) -> Dict[str, Any]:
    """
    Chạy toàn bộ pipeline Giai đoạn 4 và trả về bảng kết quả chi tiết.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_figure)), exist_ok=True)

    print("=" * 70)
    print("🚀 GIAI ĐOẠN 4: PROSODY MODIFICATION & VOICE CONVERSION")
    print("=" * 70)

    # 1. Đọc file âm thanh đầu vào
    if not os.path.exists(input_audio):
        if os.path.exists("audio_output/raw_concat.wav"):
            input_audio = "audio_output/raw_concat.wav"
        else:
            raise FileNotFoundError(f"Không tìm thấy file âm thanh đầu vào: {input_audio}")

    print(f"[*] Đọc file âm thanh đầu vào: {input_audio}")
    y_orig, in_sr = sf.read(input_audio)
    if y_orig.ndim > 1:
        y_orig = np.mean(y_orig, axis=1)
    if in_sr != sr:
        y_orig = librosa.resample(y_orig.astype(np.float32), orig_sr=in_sr, target_sr=sr)
    y_orig = y_orig.astype(np.float32)

    # 2. Kiểm chứng công thức F0 = Fs / N_p
    print(f"[*] Kiểm chứng thực nghiệm công thức F0 = Fs / N_p trên: {sample_unit}")
    if not os.path.exists(sample_unit):
        # Fallback tìm 1 file wav bất kỳ trong data/units
        fallback_units = list(Path("data/units").glob("*.wav"))
        if fallback_units:
            sample_unit = str(fallback_units[0])
            print(f"    [!] Dùng file thay thế: {sample_unit}")
        else:
            sample_unit = input_audio

    f0_res = verify_f0_formula(sample_unit, sr=sr)
    voiced_frame = f0_res["sample_frame"]
    n_p = f0_res["n_p"]
    f0_formula = f0_res["f0_formula"]

    print("\n--- BẢNG KIỂM CHỨNG TẦN SỐ CƠ BẢN F0 ---")
    f0_table = [
        ["Tần số lấy mẫu (Fs)", f"{sr} Hz"],
        ["Chu kỳ pitch ước lượng (Np)", f"{n_p} samples"],
        ["F0 theo công thức (Fs / Np)", f"{f0_formula:.2f} Hz"],
        ["F0 thuật toán YIN đối chứng", f"{f0_res['f0_yin']:.2f} Hz"],
        ["Sai số tương đối", f"{f0_res['error_pct']:.2f} %"],
    ]
    print(tabulate(f0_table, headers=["Tham số", "Giá trị"], tablefmt="fancy_grid"))

    # 3. Thí nghiệm 3: Co giãn thời lượng (Time Stretch)
    print("\n[*] Thực hiện Thí nghiệm 3: Co giãn thời lượng (Phase Vocoder)...")
    y_slow = time_stretch(y_orig, rate=0.8, sr=sr)
    y_fast = time_stretch(y_orig, rate=1.2, sr=sr)

    path_slow = os.path.join(output_dir, "speed_0.8x.wav")
    path_fast = os.path.join(output_dir, "speed_1.2x.wav")
    sf.write(path_slow, y_slow, sr, subtype="PCM_16")
    sf.write(path_fast, y_fast, sr, subtype="PCM_16")
    print(f"    -> Đã xuất: {path_slow} (Thời lượng: {len(y_slow)/sr:.2f}s)")
    print(f"    -> Đã xuất: {path_fast} (Thời lượng: {len(y_fast)/sr:.2f}s)")

    # 4. Thí nghiệm 4: Dịch chuyển cao độ (Pitch Shift)
    print("\n[*] Thực hiện Thí nghiệm 4: Dịch chuyển cao độ (Source-Filter Pitch Shift)...")
    y_pitch_low = pitch_shift(y_orig, n_steps=-2.0, sr=sr)
    y_pitch_high = pitch_shift(y_orig, n_steps=+2.0, sr=sr)

    path_p_low = os.path.join(output_dir, "pitch_minus2.wav")
    path_p_high = os.path.join(output_dir, "pitch_plus2.wav")
    sf.write(path_p_low, y_pitch_low, sr, subtype="PCM_16")
    sf.write(path_p_high, y_pitch_high, sr, subtype="PCM_16")
    print(f"    -> Đã xuất: {path_p_low} (-2 semitones, giọng trầm)")
    print(f"    -> Đã xuất: {path_p_high} (+2 semitones, giọng bổng)")

    # 5. Voice Conversion: Giọng Nữ giả lập và Giọng Miền Nam
    print("\n[*] Thực hiện Voice Conversion (Cách 1: Xử lý Tín hiệu & Prosody)...")
    y_female = voice_conversion_female(y_orig, sr=sr, pitch_shift_steps=5.5)
    y_southern = voice_conversion_southern(y_orig, sr=sr)

    path_female = os.path.join(output_dir, "voice_female_sim.wav")
    path_southern = os.path.join(output_dir, "voice_southern_sim.wav")
    sf.write(path_female, y_female, sr, subtype="PCM_16")
    sf.write(path_southern, y_southern, sr, subtype="PCM_16")
    print(f"    -> Đã xuất: {path_female} (Giọng Nữ giả lập, F0 +5.5st & Formant Warping)")
    print(f"    -> Đã xuất: {path_southern} (Giọng Miền Nam mô phỏng, giãn nguyên âm đuôi)")

    # 6. Vẽ Hình 04: Phân tích Prosody & Biến đổi giọng nói
    print(f"\n[*] Vẽ Hình 04 phân tích ngữ điệu: {output_figure}...")
    plot_prosody_analysis(
        voiced_frame=voiced_frame,
        n_p=n_p,
        f0_formula=f0_formula,
        y_orig=y_orig,
        y_slow=y_slow,
        y_fast=y_fast,
        y_pitch_low=y_pitch_low,
        y_pitch_high=y_pitch_high,
        y_female=y_female,
        y_southern=y_southern,
        sr=sr,
        output_fig=output_figure,
    )
    print(f"    -> Đã lưu biểu đồ thành công: {output_figure}")

    # Bảng tổng hợp các biến thể âm thanh
    exp_summary = [
        ["Gốc (Original)", f"{len(y_orig)/sr:.2f}s", "Baseline", "audio_output/best_concat.wav"],
        ["Chậm 0.8x (TN3)", f"{len(y_slow)/sr:.2f}s", "+25% thời lượng, giữ pitch", path_slow],
        ["Nhanh 1.2x (TN3)", f"{len(y_fast)/sr:.2f}s", "-17% thời lượng, giữ pitch", path_fast],
        ["Hạ pitch -2st (TN4)", f"{len(y_pitch_low)/sr:.2f}s", "-2 semitones (~ -10.9%)", path_p_low],
        ["Nâng pitch +2st (TN4)", f"{len(y_pitch_high)/sr:.2f}s", "+2 semitones (~ +12.2%)", path_p_high],
        ["Giọng Nữ giả lập", f"{len(y_female)/sr:.2f}s", "+5.5 semitones + High-shelf", path_female],
        ["Giọng Miền Nam", f"{len(y_southern)/sr:.2f}s", "Giãn 1.25x âm cuối + Pitch lướt", path_southern],
    ]
    print("\n--- BẢNG TỔNG HỢP CÁC BIẾN THỂ ÂM THANH XUẤT RA ---")
    print(tabulate(exp_summary, headers=["Biến thể âm thanh", "Thời lượng", "Đặc trưng biến đổi", "File âm thanh"], tablefmt="fancy_grid"))
    print("\n✅ Hoàn thành Giai đoạn 4 thành công!")

    return {
        "status": "success",
        "f0_verification": f0_res,
        "outputs": {
            "speed_0.8x": path_slow,
            "speed_1.2x": path_fast,
            "pitch_minus2": path_p_low,
            "pitch_plus2": path_p_high,
            "voice_female": path_female,
            "voice_southern": path_southern,
        },
        "output_figure": output_figure,
        "summary": exp_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Giai đoạn 4: Prosody & Voice Conversion")
    parser.add_argument("--input", default="audio_output/best_concat.wav", help="File âm thanh đầu vào")
    parser.add_argument("--sample-unit", default="data/units/ba_01.wav", help="File nguyên âm để kiểm chứng F0")
    parser.add_argument("--output-dir", default="audio_output", help="Thư mục xuất âm thanh")
    parser.add_argument("--output-fig", default="figures/04_prosody_analysis.png", help="File ảnh đồ thị")
    args = parser.parse_args()

    run_phase4(
        input_audio=args.input,
        sample_unit=args.sample_unit,
        output_dir=args.output_dir,
        output_figure=args.output_fig,
    )
