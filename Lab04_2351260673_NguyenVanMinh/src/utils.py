"""
Audio Utilities and Metadata Extraction for Lab 04 TTS.
CSE457 - Lab 04: Text-to-Speech
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def compute_rms(y: np.ndarray) -> float:
    """
    Tính năng lượng hiệu dụng RMS (Root Mean Square) của tín hiệu âm thanh.
    Công thức: RMS(y) = sqrt( (1/N) * sum(y^2) + 1e-12 )
    """
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))


def match_rms(y: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """
    Chuẩn hóa năng lượng RMS của tín hiệu về mức target_rms (mặc định 0.08).
    Đồng thời giới hạn biên độ chống tràn clipping trong khoảng [-0.99, 0.99].
    """
    if len(y) == 0:
        return y
    current_rms = compute_rms(y)
    if current_rms < 1e-8:
        return y

    gain = target_rms / current_rms
    y_norm = y * gain
    return np.clip(y_norm, -0.99, 0.99).astype(np.float32)


def check_audio_format(file_path: str, expected_sr: int = 16000) -> Dict[str, Union[bool, int, float, str]]:
    """
    Kiểm tra định dạng file WAV theo tiêu chuẩn kỹ thuật Lab 4:
    - Định dạng: WAV
    - Tần số lấy mẫu: 16000 Hz
    - Số kênh: 1 (Mono)
    - Kiểu dữ liệu: 16-bit PCM
    """
    if not os.path.exists(file_path):
        return {
            "is_valid": False,
            "error_msg": f"File không tồn tại: {file_path}",
            "samplerate": 0,
            "channels": 0,
            "duration": 0.0,
            "subtype": "",
        }

    try:
        info = sf.info(file_path)
        is_valid = True
        errors = []

        if info.samplerate != expected_sr:
            is_valid = False
            errors.append(f"Tần số lấy mẫu {info.samplerate}Hz != {expected_sr}Hz")

        if info.channels != 1:
            is_valid = False
            errors.append(f"Số kênh {info.channels} != 1 (yêu cầu mono)")

        return {
            "is_valid": is_valid,
            "error_msg": "; ".join(errors) if errors else "Hợp lệ",
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "subtype": info.subtype,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_msg": str(e),
            "samplerate": 0,
            "channels": 0,
            "duration": 0.0,
            "subtype": "",
        }


def trim_silence(
    y: np.ndarray,
    sr: int = 16000,
    threshold_db: float = 20.0,
    margin_ms: float = 40.0,
) -> np.ndarray:
    """
    Cắt khoảng lặng (silence) ở hai đầu tín hiệu âm thanh sử dụng librosa.effects.trim,
    giữ lại một khoảng đệm an toàn (margin_ms) để tránh mất phụ âm đầu và âm đuôi.
    """
    if len(y) == 0:
        return y
    try:
        import librosa
        yt, idx = librosa.effects.trim(y, top_db=threshold_db)
        margin_samples = int(sr * margin_ms / 1000.0)
        start_idx = max(0, idx[0] - margin_samples)
        end_idx = min(len(y), idx[1] + margin_samples)
        return y[start_idx:end_idx]
    except Exception:
        return y



def extract_inventory_metadata(units_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Quét toàn bộ kho unit WAV, đo lường các đặc trưng âm học và xuất ra file metadata.csv.
    Cột dữ liệu: label, candidate_id, speaker, duration, rms, file_path.
    """
    units_path = Path(units_dir)
    records = []

    for wav_file in sorted(units_path.glob("*.wav")):
        stem = wav_file.stem
        # Phân tích tên file ví dụ: moi_01.wav -> label='moi', candidate_id='01'
        if "_" in stem:
            parts = stem.rsplit("_", 1)
            label = parts[0]
            candidate_id = parts[1]
        else:
            label = stem
            candidate_id = "01"

        try:
            data, sr = sf.read(str(wav_file))
            if data.ndim > 1:
                data = data[:, 0]  # Chuyển mono nếu 2 kênh

            dur = float(len(data) / sr)
            rms_val = compute_rms(data)

            records.append({
                "label": label,
                "candidate_id": candidate_id,
                "speaker": "default_speaker",
                "duration": round(dur, 4),
                "rms": round(rms_val, 6),
                "samplerate": sr,
                "file_path": str(wav_file),
            })
        except Exception as e:
            print(f"Lỗi khi đọc file {wav_file}: {e}")

    df = pd.DataFrame(records)
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def plot_inventory_stats(metadata_csv: str, output_fig: str) -> None:
    """
    Vẽ Hình 01: Histogram phân bố thời lượng (Duration) và Năng lượng RMS của inventory.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Không tìm thấy file: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError("File metadata.csv rỗng, không thể vẽ biểu đồ.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Histogram Duration
    axes[0].hist(df["duration"], bins=15, color="#2b5c8f", edgecolor="black", alpha=0.8)
    dur_mean = df["duration"].mean()
    dur_std = df["duration"].std()
    axes[0].axvline(dur_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {dur_mean:.2f}s")
    axes[0].set_title("Phân bố Thời lượng Unit (Duration Distribution)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời lượng (giây)")
    axes[0].set_ylabel("Số lượng Unit")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Histogram RMS Energy
    axes[1].hist(df["rms"], bins=15, color="#1e824c", edgecolor="black", alpha=0.8)
    rms_mean = df["rms"].mean()
    axes[1].axvline(rms_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean RMS = {rms_mean:.4f}")
    axes[1].axvline(0.08, color="orange", linestyle=":", linewidth=2, label="Target RMS = 0.0800")
    axes[1].set_title("Phân bố Năng lượng RMS (Energy Distribution)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mức năng lượng RMS")
    axes[1].set_ylabel("Số lượng Unit")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.suptitle("Hình 01: Thống kê Đặc tính Âm học Kho Word-Unit Inventory", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()


def plot_boundary_spec(
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    boundaries_raw: List[float],
    boundaries_smooth: List[float],
    sr: int = 16000,
    output_fig: str = "figures/02_boundary_spec.png",
    tokens: Optional[List[str]] = None,
) -> None:
    """
    Vẽ Hình 02: So sánh Waveform và Spectrogram giữa Baseline (xfade=0) và Cross-fade (xfade=15ms)
    có đánh dấu chính xác ranh giới ghép nối giữa các từ.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    time_raw = np.linspace(0, len(y_raw) / sr, len(y_raw), endpoint=False)
    time_smooth = np.linspace(0, len(y_smooth) / sr, len(y_smooth), endpoint=False)

    # 1. Waveform Baseline (xfade = 0 ms)
    axes[0].plot(time_raw, y_raw, color="#c0392b", linewidth=0.8, alpha=0.9, label="Raw Waveform (xfade=0)")
    for b in boundaries_raw:
        axes[0].axvline(b, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[0].set_title("1. Baseline Concatenation - Dạng sóng (Waveform, xfade = 0 ms)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Biên độ")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right")

    # 2. Spectrogram Baseline (xfade = 0 ms)
    Pxx_raw, freqs, bins_raw, im_raw = axes[1].specgram(y_raw, NFFT=512, Fs=sr, noverlap=256, cmap="magma")
    for b in boundaries_raw:
        axes[1].axvline(b, color="cyan", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1].set_title("2. Baseline Concatenation - Phổ biên độ (Spectrogram, xfade = 0 ms - vạch đứt gãy)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Tần số (Hz)")
    axes[1].set_ylim(0, 8000)

    # 3. Waveform Smoothed (xfade = 15 ms)
    axes[2].plot(time_smooth, y_smooth, color="#27ae60", linewidth=0.8, alpha=0.9, label="Smoothed Waveform (xfade=15ms)")
    for b in boundaries_smooth:
        axes[2].axvline(b, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[2].set_title("3. Làm mượt biên nối - Dạng sóng (Waveform, Linear Cross-fade = 15 ms)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Biên độ")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper right")

    # 4. Spectrogram Smoothed (xfade = 15 ms)
    Pxx_smooth, freqs, bins_smooth, im_smooth = axes[3].specgram(y_smooth, NFFT=512, Fs=sr, noverlap=256, cmap="magma")
    for b in boundaries_smooth:
        axes[3].axvline(b, color="lightgreen", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[3].set_title("4. Làm mượt biên nối - Phổ biên độ (Spectrogram, Cross-fade = 15 ms - liên tục và trơn)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Thời gian (giây)")
    axes[3].set_ylabel("Tần số (Hz)")
    axes[3].set_ylim(0, 8000)

    plt.suptitle("Hình 02: Phân tích Biên nối Ghép âm trước và sau Cross-Fade (Waveform & Spectrogram)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()


def plot_unit_selection_comparison(
    comparison_data: Dict[str, Any],
    tokens: List[str],
    output_fig: str = "figures/03_unit_selection.png",
) -> None:
    """
    Vẽ Hình 03: So sánh Chi phí và Lựa chọn Candidate giữa Fixed _01, Fixed _02 và Viterbi Search.
    Gồm 2 đồ thị:
    1. Grouped Bar Chart so sánh Target Cost, Join Cost và Total Cost.
    2. Trellis Path Diagram thể hiện đường đi candidate được thuật toán Viterbi lựa chọn.
    """
    fig, (ax_bar, ax_path) = plt.subplots(1, 2, figsize=(15, 6))

    strategies = ["Fixed _01 (Baseline)", "Fixed _02", "Viterbi Optimal"]
    target_costs = [
        comparison_data["fixed_01"]["target_cost"],
        comparison_data["fixed_02"]["target_cost"],
        comparison_data["optimal_viterbi"]["target_cost"],
    ]
    join_costs = [
        comparison_data["fixed_01"]["join_cost"],
        comparison_data["fixed_02"]["join_cost"],
        comparison_data["optimal_viterbi"]["join_cost"],
    ]
    total_costs = [
        comparison_data["fixed_01"]["total_cost"],
        comparison_data["fixed_02"]["total_cost"],
        comparison_data["optimal_viterbi"]["total_cost"],
    ]

    # 1. Bar Chart so sánh chi phí
    x = np.arange(len(strategies))
    width = 0.25

    rects1 = ax_bar.bar(x - width, target_costs, width, label="Target Cost", color="#3498db", alpha=0.9)
    rects2 = ax_bar.bar(x, join_costs, width, label="Join Cost", color="#e67e22", alpha=0.9)
    rects3 = ax_bar.bar(x + width, total_costs, width, label="Total Cost", color="#2ecc71", alpha=0.9)

    ax_bar.set_title("1. So sánh Chi phí giữa các Chiến lược chọn Candidate", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Chi phí (Cost)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(strategies, fontweight="bold")
    ax_bar.legend()
    ax_bar.grid(True, linestyle=":", alpha=0.6)

    # Hiển thị số liệu lên đầu cột Total Cost
    red_pct = comparison_data["optimal_viterbi"].get("reduction_pct", 0.0)
    for rect, val in zip(rects3, total_costs):
        ax_bar.annotate(
            f"{val:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_bar.text(
        0.5, 0.92,
        f"Mức giảm tổng chi phí (Viterbi vs Fixed _01): -{red_pct:.1f}%",
        transform=ax_bar.transAxes,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f5", edgecolor="#2ecc71"),
    )

    # 2. Trellis Path Diagram thể hiện đường đi của chuỗi candidate
    opt_path = comparison_data["optimal_viterbi"]["path"]
    fixed_path = comparison_data["fixed_01"]["path"]

    num_tokens = len(tokens)
    x_coords = np.arange(num_tokens)

    # Chuyển candidate_id thành số nguyên
    def parse_cid(cid):
        try:
            return int(cid)
        except:
            return 1

    opt_y = [parse_cid(c) for c in opt_path]
    fixed_y = [parse_cid(c) for c in fixed_path]

    # Vẽ nền các nút ứng viên
    for i in range(num_tokens):
        ax_path.scatter([i] * 5, [0, 1, 2, 3, 4], color="#bdc3c7", s=60, alpha=0.5)

    # Đường đi Fixed _01 (Đỏ nét đứt)
    ax_path.plot(x_coords, fixed_y, "r--o", linewidth=1.5, alpha=0.7, label="Đường đi Fixed _01 (Cố định)")

    # Đường đi Viterbi Optimal (Xanh lá đậm nét liền)
    ax_path.plot(x_coords, opt_y, "#27ae60", marker="s", markersize=8, linewidth=2.5, label="Đường đi Viterbi Optimal (Tối ưu)")

    for i, (tok, cid) in enumerate(zip(tokens, opt_path)):
        ax_path.text(i, parse_cid(cid) + 0.18, f"_{cid}", ha="center", fontsize=9, fontweight="bold", color="#1e824c")

    ax_path.set_title("2. Sơ đồ Lưới (Trellis) và Tuyến đường Chọn Candidate Tối ưu", fontsize=12, fontweight="bold")
    ax_path.set_xticks(x_coords)
    ax_path.set_xticklabels(tokens, rotation=25, ha="right", fontweight="bold")
    ax_path.set_ylabel("Candidate ID (_00 đến _04)")
    ax_path.set_yticks([0, 1, 2, 3, 4])
    ax_path.set_yticklabels(["_00", "_01", "_02", "_03", "_04"])
    ax_path.grid(True, linestyle=":", alpha=0.6)
    ax_path.legend(loc="upper left")

    plt.suptitle("Hình 03: Đánh giá Hiệu quả Thuật toán Unit Selection (Thí nghiệm 2)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()


def plot_prosody_analysis(
    voiced_frame: np.ndarray,
    n_p: int,
    f0_formula: float,
    y_orig: np.ndarray,
    y_slow: np.ndarray,
    y_fast: np.ndarray,
    y_pitch_low: np.ndarray,
    y_pitch_high: np.ndarray,
    y_female: np.ndarray,
    y_southern: np.ndarray,
    sr: int = 16000,
    output_fig: str = "figures/04_prosody_analysis.png",
) -> None:
    """
    Vẽ Hình 04: Phân tích Toàn diện Ngữ điệu & Mô hình Nguồn-Lọc (Giai đoạn 4).
    Gồm 4 khung đồ thị:
    1. Kiểm chứng F0: Waveform đoạn voiced & Đồ thị Autocorrelation R_xx[k] đánh dấu N_p.
    2. Thí nghiệm 3: So sánh Spectrogram 3 mức tốc độ (0.8x, 1.0x, 1.2x).
    3. Thí nghiệm 4: So sánh Spectrogram 3 mức cao độ (-2, 0, +2 semitones).
    4. Voice Conversion: Đối chiếu phổ Giọng Gốc (Nam) vs Giọng Nữ Giả lập vs Giọng Miền Nam.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Khung 1: Autocorrelation & Pitch F0
    ax_corr = fig.add_subplot(gs[0, 0])
    x_clean = voiced_frame - np.mean(voiced_frame)
    corr = np.correlate(x_clean, x_clean, mode="full")
    corr = corr[len(corr) // 2 :]
    corr = corr / (corr[0] + 1e-12)

    lags = np.arange(min(len(corr), int(sr * 0.025)))
    ax_corr.plot(lags, corr[:len(lags)], color="#2980b9", linewidth=1.5, label="Autocorrelation R_xx[k]")
    ax_corr.axvline(n_p, color="red", linestyle="--", linewidth=1.5, label=f"N_p = {n_p} mẫu (F0 = {f0_formula:.1f} Hz)")
    ax_corr.set_title("1. Kiểm chứng F0 qua Autocorrelation (F0 ≈ Fs / N_p)", fontsize=11, fontweight="bold")
    ax_corr.set_xlabel("Độ trễ lag k (mẫu)")
    ax_corr.set_ylabel("Hệ số tự tương quan")
    ax_corr.grid(True, linestyle=":", alpha=0.6)
    ax_corr.legend(loc="upper right")

    # Khung 2: Thí nghiệm 3 - Spectrogram Co giãn Thời lượng (0.8x, 1.0x, 1.2x)
    ax_speed = fig.add_subplot(gs[0, 1])
    # Vẽ dạng sóng chồng lớp thể hiện co giãn trục thời gian
    t_orig = np.linspace(0, len(y_orig) / sr, len(y_orig))
    t_slow = np.linspace(0, len(y_slow) / sr, len(y_slow))
    t_fast = np.linspace(0, len(y_fast) / sr, len(y_fast))
    ax_speed.plot(t_slow, y_slow * 0.7, color="#e67e22", alpha=0.5, label=f"0.8x (Chậm, {len(y_slow)/sr:.2f}s)")
    ax_speed.plot(t_orig, y_orig * 0.7, color="#2c3e50", alpha=0.8, label=f"1.0x (Gốc, {len(y_orig)/sr:.2f}s)")
    ax_speed.plot(t_fast, y_fast * 0.7, color="#27ae60", alpha=0.6, label=f"1.2x (Nhanh, {len(y_fast)/sr:.2f}s)")
    ax_speed.set_title("2. Thí nghiệm 3: Co giãn Thời lượng (Duration / Speed Scaling)", fontsize=11, fontweight="bold")
    ax_speed.set_xlabel("Thời gian (giây)")
    ax_speed.set_ylabel("Biên độ")
    ax_speed.grid(True, linestyle=":", alpha=0.6)
    ax_speed.legend(loc="upper right")

    # Khung 3: Thí nghiệm 4 - Phổ Spectrogram Dịch Pitch (-2 vs Gốc vs +2 semitone)
    ax_pitch1 = fig.add_subplot(gs[1, 0])
    ax_pitch1.specgram(y_pitch_low, NFFT=512, Fs=sr, noverlap=256, cmap="inferno")
    ax_pitch1.set_title("3a. Thí nghiệm 4: Pitch -2 semitone (Trầm hơn, Formant giữ nguyên)", fontsize=10, fontweight="bold")
    ax_pitch1.set_ylabel("Tần số (Hz)")
    ax_pitch1.set_ylim(0, 5000)

    ax_pitch2 = fig.add_subplot(gs[1, 1])
    ax_pitch2.specgram(y_pitch_high, NFFT=512, Fs=sr, noverlap=256, cmap="inferno")
    ax_pitch2.set_title("3b. Thí nghiệm 4: Pitch +2 semitone (Cao hơn, Formant giữ nguyên)", fontsize=10, fontweight="bold")
    ax_pitch2.set_ylabel("Tần số (Hz)")
    ax_pitch2.set_ylim(0, 5000)

    # Khung 4: Voice Conversion - So sánh Giọng Nam Gốc vs Giọng Nữ vs Giọng Miền Nam
    ax_vc1 = fig.add_subplot(gs[2, 0])
    ax_vc1.specgram(y_female, NFFT=512, Fs=sr, noverlap=256, cmap="viridis")
    ax_vc1.set_title("4a. Giả lập Giọng Nữ (Pitch +5.5st & Formant Warping nâng cao)", fontsize=10, fontweight="bold")
    ax_vc1.set_xlabel("Thời gian (giây)")
    ax_vc1.set_ylabel("Tần số (Hz)")
    ax_vc1.set_ylim(0, 6000)

    ax_vc2 = fig.add_subplot(gs[2, 1])
    ax_vc2.specgram(y_southern, NFFT=512, Fs=sr, noverlap=256, cmap="viridis")
    ax_vc2.set_title("4b. Mô phỏng Sắc thái Miền Nam (Kéo dài âm ngân & Pitch softening)", fontsize=10, fontweight="bold")
    ax_vc2.set_xlabel("Thời gian (giây)")
    ax_vc2.set_ylabel("Tần số (Hz)")
    ax_vc2.set_ylim(0, 6000)

    plt.suptitle("Hình 04: Nghiên cứu Ngữ điệu, Mô hình Source–Filter và Voice Conversion (Giai đoạn 4)", fontsize=13, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()



