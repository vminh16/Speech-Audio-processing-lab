"""
Script and module to process, validate, normalize, and extract metadata from the Word-Unit Inventory.
CSE457 - Lab 04: Text-to-Speech
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import librosa
import numpy as np
import soundfile as sf

from src.utils import (
    compute_rms,
    match_rms,
    trim_silence,
    extract_inventory_metadata,
    plot_inventory_stats,
)


def process_all_units(
    units_dir: str = "data/units",
    output_metadata_csv: str = "data/metadata.csv",
    output_figure: str = "figures/01_inventory_stats.png",
    target_sr: int = 16000,
    target_rms: float = 0.08,
) -> Dict[str, Any]:
    """
    Chuẩn hóa toàn bộ kho inventory WAV:
    1. Chuyển thành Mono nếu là Stereo.
    2. Resample về 16000 Hz nếu khác sample rate.
    3. Cắt tỉa silence dư thừa ở đầu và đuôi.
    4. Cân bằng năng lượng RMS về target_rms (0.08).
    5. Ghi đè file WAV chuẩn 16-bit PCM.
    6. Trích xuất metadata.csv và vẽ biểu đồ 01_inventory_stats.png.
    """
    units_path = Path(units_dir)
    wav_files = sorted(list(units_path.glob("*.wav")))

    if not wav_files:
        print(f"Cảnh báo: Không tìm thấy file .wav nào trong thư mục {units_dir}")
        return {"total_files": 0, "processed": 0}

    print(f"Bắt đầu xử lý {len(wav_files)} file unit trong {units_dir}...")

    processed_count = 0
    for wav_file in wav_files:
        try:
            # 1. Đọc file
            y, sr = sf.read(str(wav_file))
            if y.ndim > 1:
                y = np.mean(y, axis=1)  # Chuyển mono

            # 2. Resample nếu khác 16kHz
            if sr != target_sr:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # 3. Trim silence
            y = trim_silence(y, sr=target_sr, threshold_db=30.0, margin_ms=25.0)

            # 4. Chuẩn hóa RMS
            y = match_rms(y, target_rms=target_rms)

            # 5. Ghi lại chuẩn 16-bit PCM
            sf.write(str(wav_file), y, target_sr, subtype="PCM_16")
            processed_count += 1
        except Exception as e:
            print(f"Lỗi khi xử lý file {wav_file}: {e}")

    # 6. Trích xuất metadata CSV
    df = extract_inventory_metadata(units_dir=str(units_path), output_csv=output_metadata_csv)
    print(f"Đã xuất metadata thành công vào: {output_metadata_csv} ({len(df)} units)")

    # 7. Vẽ biểu đồ thống kê Hình 01
    plot_inventory_stats(metadata_csv=output_metadata_csv, output_fig=output_figure)
    print(f"Đã vẽ và lưu biểu đồ phân bố vào: {output_figure}")

    return {
        "total_files": len(wav_files),
        "processed": processed_count,
        "metadata_csv": output_metadata_csv,
        "figure_path": output_figure,
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hóa inventory và trích xuất metadata Lab 4 TTS")
    parser.add_argument("--units-dir", default="data/units", help="Đường dẫn thư mục chứa WAV units")
    parser.add_argument("--output-csv", default="data/metadata.csv", help="Đường dẫn file metadata.csv đầu ra")
    parser.add_argument("--output-fig", default="figures/01_inventory_stats.png", help="Đường dẫn biểu đồ đầu ra")
    parser.add_argument("--sr", type=int, default=16000, help="Tần số lấy mẫu mục tiêu (16000)")
    parser.add_argument("--rms", type=float, default=0.08, help="Mức năng lượng RMS mục tiêu (0.08)")

    args = parser.parse_args()
    process_all_units(
        units_dir=args.units_dir,
        output_metadata_csv=args.output_csv,
        output_figure=args.output_fig,
        target_sr=args.sr,
        target_rms=args.rms,
    )


if __name__ == "__main__":
    main()
