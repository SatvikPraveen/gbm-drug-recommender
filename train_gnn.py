#!/usr/bin/env python
"""
Train and evaluate the graph neural network on its own.

Runs grouped and scaffold cross-validation for one GNN configuration against the
dummy baseline (same harness as the full benchmark), then fits a final model on all
labelled drugs and saves it with its training curve.

Examples
--------
    python train_gnn.py                              # GCN, primary task (mean GBM z-score)
    python train_gnn.py --gnn-type gat --task gbm_potency --repeats 3
    python train_gnn.py --hidden-channels 64 --num-gnn-layers 2 --epochs 100 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from gbm_drug import config as cfg
from gbm_drug import evaluation as ev
from gbm_drug.models.gnn_model import GNNDrugPredictor
from gbm_drug.pipeline import TASKS, Context, Options, stage_data, stage_features
from gbm_drug.utils import run_metadata, write_json
from gbm_drug.utils import visualization as viz


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=TASKS[0].name, choices=[t.name for t in TASKS])
    p.add_argument("--gnn-type", default=cfg.GNN_TYPE, choices=["gcn", "gat"])
    p.add_argument("--hidden-channels", type=int, default=cfg.GNN_HIDDEN_CHANNELS)
    p.add_argument("--num-gnn-layers", type=int, default=cfg.GNN_NUM_GNN_LAYERS)
    p.add_argument("--num-mlp-layers", type=int, default=cfg.GNN_NUM_MLP_LAYERS)
    p.add_argument("--dropout", type=float, default=cfg.GNN_DROPOUT)
    p.add_argument("--pooling", default=cfg.GNN_POOLING, choices=["mean", "max", "add"])
    p.add_argument("--learning-rate", type=float, default=cfg.GNN_LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=cfg.GNN_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=cfg.GNN_EPOCHS)
    p.add_argument("--patience", type=int, default=cfg.GNN_EARLY_STOPPING_PATIENCE)
    p.add_argument("--repeats", type=int, default=1, help="repeats of grouped CV")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)]
    )
    cfg.ensure_directories()

    ctx = Context(options=Options(include_gnn=True, device=args.device, seed=args.seed))
    stage_data(ctx)
    stage_features(ctx)
    task = next(t for t in TASKS if t.name == args.task)

    hp = dict(
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        pooling=args.pooling,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device,
        random_state=args.seed,
    )
    label = f"GNN-{args.gnn_type.upper()} h{args.hidden_channels} L{args.num_gnn_layers}"
    spec = ev.ModelSpec(
        label,
        lambda kind: GNNDrugPredictor(task=kind, **hp),
        "smiles",
        ("neural", "graph"),
        repeats=args.repeats,
    )

    fold_scores, oof = ev.run_benchmark(
        [task], [spec], ctx.features, ctx.targets, ctx.groups, n_repeats=args.repeats, random_state=args.seed
    )
    summary = ev.summarize_scores(fold_scores)
    out = cfg.BENCHMARK_RESULTS_DIR
    tag = f"gnn_{args.gnn_type}_{task.name}"
    fold_scores.to_csv(out / f"{tag}_fold_scores.csv", index=False)
    summary.to_csv(out / f"{tag}_summary.csv", index=False)
    oof.to_csv(out / f"{tag}_oof.csv", index=False)

    metric = ev.PRIMARY_METRIC[task.kind]
    print(f"\n{task.name} — {metric} (mean [95% CI]):")
    view = summary[summary["metric"] == metric][["model", "strategy", "mean", "ci_low", "ci_high", "n_folds"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    final = GNNDrugPredictor(task=task.kind, **hp).fit(
        ctx.smiles_list, ctx.targets[task.target].to_numpy(dtype=float)
    )
    model_path = cfg.MODEL_RESULTS_DIR / f"{tag}.pt"
    final.save(model_path)
    viz.gnn_training_curve(final.history_, name=f"{tag}_training_history")
    write_json(
        run_metadata(
            {
                "script": "train_gnn.py",
                "hyperparameters": hp,
                "task": task.name,
                "n_epochs": final.n_epochs_,
                "best_val_loss": final.best_val_loss_,
            }
        ),
        cfg.MODEL_RESULTS_DIR / f"{tag}_metadata.json",
    )
    print(
        f"\nFinal model saved to {model_path} ({final.n_epochs_} epochs, best val loss {final.best_val_loss_:.4f})"
    )
    pd.DataFrame(
        {"drug_name": ctx.targets["drug_name"], "prediction": final.predict(ctx.smiles_list)}
    ).to_csv(cfg.MODEL_RESULTS_DIR / f"{tag}_predictions_in_sample.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
