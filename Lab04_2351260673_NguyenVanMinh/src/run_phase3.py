"""
Execution Script for Phase 3: Unit Selection and Viterbi Optimization (Experiment 2).
CSE457 - Lab 04: Text-to-Speech
"""

import argparse
import os
from pathlib import Path

import soundfile as sf
from tabulate import tabulate

from src.smoothing import crossfade_sequence
from src.text_norm import normalize_queue_text
from src.unit_selection import compare_candidate_strategies
from src.utils import plot_unit_selection_comparison


def run_phase3(
    text: str = "Mời khách hàng số 23 đến quầy 5.",
    units_dir: str = "data/units",
    output_audio: str = "audio_output/best_concat.wav",
    output_figure: str = "figures/03_unit_selection.png",
    weight_join: float = 1.0,
    sr: int = 16000,
) -> dict:
    """
    Thực thi toàn bộ Giai đoạn 3:
    1. Chuẩn hóa chuỗi văn bản.
    2. Chạy thuật toán Viterbi tìm chuỗi candidate tối ưu.
    3. Đối sánh với phương án cố định _01 và _02 (Thí nghiệm 2).
    4. Ghép nối chuỗi tối ưu kết hợp Cross-fade 15ms -> audio_output/best_concat.wav.
    5. Vẽ và xuất Hình 03 -> figures/03_unit_selection.png.
    """
    print(f"\n=== [Giai đoạn 3] Tối ưu hóa Unit Selection bằng Quy hoạch động Viterbi ===")
    tokens = normalize_queue_text(text)
    print(f"Câu kiểm thử: '{text}'")
    print(f"Tokens: {tokens}")

    # 1. Chạy so sánh các chiến lược candidate
    comparison = compare_candidate_strategies(
        tokens=tokens,
        units_dir=units_dir,
        weight_join=weight_join,
        sr=sr,
    )

    f01 = comparison["fixed_01"]
    f02 = comparison["fixed_02"]
    opt = comparison["optimal_viterbi"]

    # 2. In bảng kết quả Thí nghiệm 2
    table_data = [
        [
            "Fixed _01 (Baseline)",
            " -> ".join(f01["path"]),
            f01["target_cost"],
            f01["join_cost"],
            f01["total_cost"],
            "0.0%",
        ],
        [
            "Fixed _02",
            " -> ".join(f02["path"]),
            f02["target_cost"],
            f02["join_cost"],
            f02["total_cost"],
            f"{((f01['total_cost'] - f02['total_cost'])/f01['total_cost']*100):.1f}%",
        ],
        [
            "Viterbi Optimal (Toàn cục)",
            " -> ".join(opt["path"]),
            opt["target_cost"],
            opt["join_cost"],
            opt["total_cost"],
            f"-{opt['reduction_pct']:.1f}%",
        ],
    ]

    print("\n--- KẾT QUẢ THÍ NGHIỆM 2: SO SÁNH CÁC PHƯƠNG ÁN CHỌN CANDIDATE ---")
    print(tabulate(
        table_data,
        headers=["Chiến lược", "Chuỗi Candidate đã chọn", "Target Cost", "Join Cost", "Total Cost", "Mức cải thiện"],
        tablefmt="github",
    ))

    # 3. Ghép nối chuỗi tối ưu kết hợp Cross-fade 15 ms
    optimal_audio_units = [u["audio"] for u in opt["units"]]
    y_best, boundaries_best = crossfade_sequence(optimal_audio_units, sr=sr, xfade_ms=15.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_audio)), exist_ok=True)
    sf.write(output_audio, y_best, sr)
    print(f"\n-> Đã cập nhật file âm thanh tối ưu: {output_audio} ({len(y_best)/sr:.2f}s)")

    # 4. Vẽ biểu đồ Hình 03
    plot_unit_selection_comparison(
        comparison_data=comparison,
        tokens=tokens,
        output_fig=output_figure,
    )
    print(f"-> Đã xuất biểu đồ phân tích Hình 03: {output_figure}")

    return {
        "status": "success",
        "comparison": comparison,
        "best_audio": output_audio,
        "figure": output_figure,
        "optimal_path": opt["path"],
        "cost_reduction_pct": opt["reduction_pct"],
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy Giai đoạn 3 - Unit Selection Viterbi")
    parser.add_argument("--text", default="Mời khách hàng số 23 đến quầy 5.", help="Văn bản kiểm thử")
    parser.add_argument("--units-dir", default="data/units", help="Thư mục chứa units")
    parser.add_argument("--out-audio", default="audio_output/best_concat.wav", help="File wav đầu ra tối ưu")
    parser.add_argument("--out-fig", default="figures/03_unit_selection.png", help="File ảnh đồ thị Hình 03")
    parser.add_argument("--weight-join", type=float, default=1.0, help="Trọng số lambda cho Join Cost")

    args = parser.parse_args()
    run_phase3(
        text=args.text,
        units_dir=args.units_dir,
        output_audio=args.out_audio,
        output_figure=args.out_fig,
        weight_join=args.weight_join,
    )


if __name__ == "__main__":
    main()
