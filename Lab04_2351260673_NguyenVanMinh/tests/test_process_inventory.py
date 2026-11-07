"""
Test inventory processing and Figure 01 generation.
"""

import numpy as np
import pytest
import soundfile as sf
from src.process_inventory import process_all_units


def test_process_all_units(tmp_path):
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    fig_path = tmp_path / "01_inventory_stats.png"
    meta_path = tmp_path / "metadata.csv"

    # Tạo 2 file wav mẫu với tần số khác để test tự động chuẩn hóa về 16kHz mono
    sr_test = 44100
    t = np.linspace(0, 0.3, int(sr_test * 0.3), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

    sf.write(str(units_dir / "moi_01.wav"), y, sr_test)
    sf.write(str(units_dir / "moi_02.wav"), y * 0.8, sr_test)

    summary = process_all_units(
        units_dir=str(units_dir),
        output_metadata_csv=str(meta_path),
        output_figure=str(fig_path),
        target_sr=16000,
        target_rms=0.08,
    )

    assert summary["total_files"] == 2
    assert meta_path.exists()
    assert fig_path.exists()

    # Kiểm tra file đã được chuẩn hóa về 16kHz mono
    info = sf.info(str(units_dir / "moi_01.wav"))
    assert info.samplerate == 16000
    assert info.channels == 1
