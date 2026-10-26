"""Bước 5 - Lexicon, bigram LM và thí nghiệm E6 (MAP re-ranking theo λ_LM)."""
from __future__ import annotations

import argparse

import pandas as pd

from src.acoustic import load_meta, load_models
from src.config import BOS, CORPUS_PATH, LABELS, LEXICON_PATH, MODELS_DIR, RESULTS_DIR, TEST_DIR, FeatureConfig, LMConfig
from src.data import list_split
from src.experiments import e6_lm
from src.lm import BigramLM, parse_lexicon, read_corpus
from src.utils import get_logger

log = get_logger()


def main(argv: list[str] | None = None) -> dict:
    d = LMConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--k", type=float, default=d.k, help="hệ số làm trơn Add-k")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(d.lambdas))
    parser.add_argument("--repeats", type=int, default=3, help="số lần lặp nhiễu cho mỗi mức SNR")
    args = parser.parse_args(argv)

    lexicon = parse_lexicon(LEXICON_PATH)
    missing = [w for w in LABELS if w not in lexicon]
    if missing:
        raise SystemExit(f"{LEXICON_PATH} thiếu các từ: {missing}")
    for word, phones in lexicon.items():
        log.info("lexicon | %-6s -> %s", word, " ".join(phones))

    lm = BigramLM(read_corpus(CORPUS_PATH), list(LABELS), k=args.k)
    table = pd.DataFrame(lm.table(), index=[BOS, *LABELS], columns=LABELS).rename_axis("prev").reset_index()
    log.info("P(cur | prev), k=%g:\n%s", args.k, table.set_index("prev").round(3).to_string())

    fcfg = FeatureConfig(**load_meta(args.models_dir)["feature_config"])
    models = load_models(args.models_dir, LABELS)
    df = e6_lm(models, lm, list_split(TEST_DIR), fcfg, lambdas=tuple(args.lambdas), repeats=args.repeats)
    log.info("E6 - accuracy theo λ_LM:\n%s", df.pivot(index="condition", columns="lambda_lm", values="accuracy").round(3).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "bigram_table.csv", index=False)
    df.to_csv(RESULTS_DIR / "e6_lm.csv", index=False)
    return {"lexicon": lexicon, "lm": lm, "bigram_table": table, "e6": df}


if __name__ == "__main__":
    main()
