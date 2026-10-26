"""Bước 2 - Trích đặc trưng tập train và huấn luyện 5 Whole-word Left-to-Right Gaussian HMM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.acoustic import check_topology, save_models, train_models
from src.config import LABELS, MODELS_DIR, TRAIN_DIR, FeatureConfig, HMMConfig
from src.data import list_split
from src.features import build_featset, by_label
from src.utils import Timer, get_logger, save_json

log = get_logger()


def main(argv: list[str] | None = None) -> dict:
    d = HMMConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=d.n_states)
    parser.add_argument("--cov-type", choices=["diag", "full"], default=d.cov_type)
    parser.add_argument("--n-iter", type=int, default=d.n_iter)
    parser.add_argument("--init", choices=["uniform", "kmeans"], default=d.init)
    parser.add_argument("--no-delta", action="store_true", help="chỉ dùng 13 MFCC tĩnh")
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    args = parser.parse_args(argv)

    fcfg = FeatureConfig(use_delta=not args.no_delta)
    hcfg = HMMConfig(n_states=args.n_states, cov_type=args.cov_type, n_iter=args.n_iter, init=args.init)

    train = list_split(TRAIN_DIR)
    if not train:
        raise SystemExit(f"Không có dữ liệu train trong {TRAIN_DIR}")
    with Timer() as t_feat:
        featset = build_featset(train, fcfg)
    log.info("Đặc trưng: %d utterance, mỗi frame %d chiều, %.1f s", len(featset), featset[0][1].shape[1], t_feat.elapsed)

    groups = by_label(featset)
    models, infos = train_models(groups, hcfg)

    topology = {}
    for label in LABELS:
        topo, info = check_topology(models[label]), infos[label]
        topology[label] = topo
        log.info(
            "%-6s | %d utt, %4d frame | EM %2d vòng, converged=%s | LL/frame=%.2f | %.3f s",
            label, info.n_seqs, info.n_frames, info.n_iter_run, info.converged, info.train_ll_per_frame, info.train_time_s,
        )
        if not (topo["no_backward"] and topo["no_skip"] and topo["rows_sum_to_1"]):
            raise RuntimeError(f"Topology của mô hình '{label}' bị phá vỡ sau khi huấn luyện: {topo}")
        if not topo["converged"]:
            log.warning("Mô hình '%s' chưa hội tụ sau %d vòng EM.", label, info.n_iter_run)
    log.info("transmat_ của '%s':\n%s", LABELS[0], np.array2string(models[LABELS[0]].transmat_, precision=3, suppress_small=True))

    paths = save_models(models, args.out_dir)
    meta = {
        "feature_config": asdict(fcfg),
        "hmm_config": asdict(hcfg),
        "train_per_class": {l: len(s) for l, s in groups.items()},
        "feature_time_s": t_feat.elapsed,
        "train_time_s": sum(i.train_time_s for i in infos.values()),
        "train_info": {l: asdict(i) for l, i in infos.items()},
        "topology": topology,
    }
    save_json(meta, Path(args.out_dir) / "meta.json")
    log.info("Đã lưu %d mô hình vào %s", len(paths), args.out_dir)
    return {"models": models, "infos": infos, "meta": meta}


if __name__ == "__main__":
    main()
