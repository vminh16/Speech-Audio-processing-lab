"""Bước 7 - Xuất hình (results/*.png) và bảng tổng hợp (results/summary.md) từ các file kết quả."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # chạy headless; phải đặt trước khi import pyplot qua src.plots
import matplotlib.pyplot as plt  # noqa: E402

from src.acoustic import load_meta, load_models  # noqa: E402
from src.config import LABELS, MODELS_DIR, RESULTS_DIR, FeatureConfig  # noqa: E402
from src.plots import plot_transmats, save  # noqa: E402
from src.reporting import build_figures, build_summary  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    fcfg = FeatureConfig(**load_meta(args.models_dir)["feature_config"])
    figs = build_figures(RESULTS_DIR, fcfg)
    figs["transmats"] = plot_transmats({l: m.transmat_ for l, m in load_models(args.models_dir, LABELS).items()})
    for name, fig in figs.items():
        save(fig, RESULTS_DIR / f"{name}.png")
        plt.close(fig)
    log.info("Đã lưu %d hình vào %s: %s", len(figs), RESULTS_DIR, ", ".join(sorted(figs)))

    summary = RESULTS_DIR / "summary.md"
    summary.write_text(build_summary(RESULTS_DIR), encoding="utf-8")
    log.info("Đã ghi %s", summary)
    return {"figures": sorted(figs), "summary": summary}


if __name__ == "__main__":
    main()
