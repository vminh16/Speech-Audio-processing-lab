"""
Master Pipeline Orchestrator for Lab 04 Text-to-Speech (TTS).
CSE457 - Lab 04: Text-to-Speech
"""

import argparse
import sys
from pathlib import Path

from src.benchmark import run_tts_benchmark
from src.process_inventory import process_all_units
from src.run_phase2 import run_phase2
from src.run_phase3 import run_phase3
from src.run_phase4 import run_phase4
from src.run_phase5 import run_phase5
from src.run_phase6 import run_phase6
from src.evaluation import prepare_evaluation


def main():
    parser = argparse.ArgumentParser(description="CSE457 Lab 04 - Pipeline Điều phối Tổng hợp Tiếng nói")
    parser.add_argument("--benchmark", action="store_true", help="Chạy chế độ đo kiểm hiệu năng (Benchmark)")
    parser.add_argument("--process-data", action="store_true", help="Thực hiện chuẩn hóa kho inventory data/units/")
    parser.add_argument("--text", default="Mời khách hàng số 23 đến quầy 5.", help="Văn bản thông báo tổng hợp")
    parser.add_argument("--units-dir", default="data/units", help="Thư mục chứa units âm thanh")
    parser.add_argument("--audio-dir", default="audio_output", help="Thư mục xuất file âm thanh")
    parser.add_argument("--figures-dir", default="figures", help="Thư mục xuất file biểu đồ")

    args = parser.parse_args()

    # 1. Chạy Benchmark nếu có cờ --benchmark
    if args.benchmark:
        run_tts_benchmark(units_dir=args.units_dir, verbose=True)
        return

    # 2. Xử lý chuẩn hóa inventory nếu được yêu cầu hoặc nếu chưa có metadata.csv
    meta_path = Path("data/metadata.csv")
    fig1_path = Path(args.figures_dir) / "01_inventory_stats.png"
    if args.process_data or not meta_path.exists():
        print("=== [Pipeline] Chuẩn hóa kho Inventory âm thanh và xuất Metadata ===")
        process_all_units(
            units_dir=args.units_dir,
            output_metadata_csv=str(meta_path),
            output_figure=str(fig1_path),
        )

    # 3. Chạy ghép nối Baseline & Làm mượt biên nối (Giai đoạn 2)
    fig2_path = Path(args.figures_dir) / "02_boundary_spec.png"
    run_phase2(
        text=args.text,
        units_dir=args.units_dir,
        output_audio_dir=args.audio_dir,
        output_figure=str(fig2_path),
    )

    # 4. Chạy tối ưu hóa Unit Selection bằng Viterbi (Giai đoạn 3)
    fig3_path = Path(args.figures_dir) / "03_unit_selection.png"
    best_audio_path = Path(args.audio_dir) / "best_concat.wav"
    run_phase3(
        text=args.text,
        units_dir=args.units_dir,
        output_audio=str(best_audio_path),
        output_figure=str(fig3_path),
    )

    # Measured HMM and offline pretrained Piper backends.
    run_phase4(input_audio=str(best_audio_path), output_dir=args.audio_dir,
               output_figure=str(Path(args.figures_dir) / "04_prosody_analysis.png"))
    run_phase5(text=args.text, output_audio=str(Path(args.audio_dir) / "hmm_tts.wav"),
               output_figure=str(Path(args.figures_dir) / "04_hmm_trajectories.png"))
    run_phase6(output_audio=str(Path(args.audio_dir) / "neural_tts.wav"),
               output_figure=str(Path(args.figures_dir) / "05_neural_melspec.png"))
    prepare_evaluation(units_dir=args.units_dir)


if __name__ == "__main__":
    main()

