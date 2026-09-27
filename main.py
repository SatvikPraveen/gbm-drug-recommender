#!/usr/bin/env python
"""
Run the GBM drug-response analysis pipeline.

Examples
--------
    python main.py                       # every stage, full settings (GNN included)
    python main.py --quick               # smoke run: 1 CV repeat, no GNN, small MCS set (~2 min)
    python main.py --stages benchmark    # only the benchmark (and the stages it depends on)
    python main.py --no-gnn --cv-repeats 3

Inputs are the committed tables under data/processed and data/smiles (see data/README.md).
Outputs go to results/ ; results/RESULTS.md and results/metadata.json describe the run.
"""

from __future__ import annotations

import argparse
import logging
import sys

from gbm_drug import config as cfg
from gbm_drug.pipeline import STAGES, Options, run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=STAGES,
        metavar="STAGE",
        help=f"stages to run (dependencies are added automatically). Choices: {', '.join(STAGES)}",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke run: no GNN, 1 CV repeat, 5 scramble rounds, <=40 drugs for MCS",
    )
    p.add_argument("--no-gnn", action="store_true", help="skip the graph neural network models")
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="use fixed hyper-parameters instead of nested grouped CV tuning",
    )
    p.add_argument(
        "--gnn-repeats",
        type=int,
        default=1,
        help="CV repeats for the GNN models (default 1; tabular models use --cv-repeats)",
    )
    p.add_argument(
        "--cv-repeats",
        type=int,
        default=cfg.CV_REPEATS,
        help=f"repeats of grouped {cfg.CV_FOLDS}-fold CV (default {cfg.CV_REPEATS})",
    )
    p.add_argument(
        "--scramble-rounds",
        type=int,
        default=cfg.Y_SCRAMBLE_ROUNDS,
        help="label-permutation rounds for the null model",
    )
    p.add_argument(
        "--mcs-max-drugs",
        type=int,
        default=cfg.MCS_MAX_DRUGS,
        help="cap on the candidate set for the O(n^2) MCS computation",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=cfg.TOP_N_DRUGS,
        help="top drugs by GBM selectivity added to the candidate set",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="torch device for the GNN"
    )
    p.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=cfg.LOG_FORMAT,
        handlers=[logging.FileHandler(cfg.LOG_FILE), logging.StreamHandler(sys.stdout)],
    )
    options = Options(
        stages=args.stages,
        quick=args.quick,
        include_gnn=not args.no_gnn,
        gnn_repeats=args.gnn_repeats,
        cv_repeats=args.cv_repeats,
        scramble_rounds=args.scramble_rounds,
        mcs_max_drugs=args.mcs_max_drugs,
        top_n=args.top_n,
        device=args.device,
        seed=args.seed,
        tune=not args.no_tune,
    )
    ctx = run(options)
    print(f"\nDone. Stages: {', '.join(ctx.timings)}  |  total {sum(ctx.timings.values()):.0f}s")
    print(f"Results: {cfg.RESULTS_DIR}  (see RESULTS.md and metadata.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
