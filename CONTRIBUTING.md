# Contributing

Thanks for your interest. This project values correctness over feature count:
a smaller analysis whose claims survive scrutiny beats a larger one that does not.

## Setup

```bash
./setup.sh                      # creates .venv and installs the package with dev extras
source .venv/bin/activate
pytest                          # ~15 s; GNN tests run on CPU
python main.py --quick          # ~2 min smoke run of every stage
```

The committed data tables are enough to run everything; the raw GDSC files are
only needed to regenerate them (`python scripts/download_gdsc.py`).

## Ground rules

- **No leakage.** Any new model must be evaluated through
  `gbm_drug.evaluation.run_benchmark` so it shares folds and baselines with
  the others. If you need a new split strategy, add it to `iter_splits` and a
  test that `assert_no_group_leakage` holds.
- **Baselines and nulls.** A reported score without its dummy baseline and,
  for the headline model, a y-scrambling p-value is not a result.
- **No hand-edited results.** `results/RESULTS.md` is generated; change the
  code or the config and re-run. `docs/RESULTS.md` is where interpretation goes.
- **Configuration, not constants.** Thresholds live in `gbm_drug/config.py`
  and are justified in `docs/METHODS.md`. If you change one, update both.
- **Data provenance.** New external data needs an entry in `data/MANIFEST.json`
  (URL, checksum, licence) and a script that fetches it.
- **Tests and lint.** `ruff check . && ruff format --check . && pytest` must
  pass; CI runs them on Python 3.11 and 3.12 with CPU torch.

## Good first contributions

- Nested cross-validation for hyper-parameter tuning (currently fixed a priori).
- Batch-correcting ln IC50 between GDSC1 and GDSC2.
- Curated pharmacokinetic annotations for the interaction screen.
- More gene-set libraries or a better target-to-gene alias table
  (`gbm_drug/pathway_analysis.py`; see `results/pathways/unmapped_targets.csv`).
- Multi-omics features per cell line (expression, mutation) for a
  drug × cell-line model, with cell-line-grouped folds.

## Reporting problems

Open an issue with the command you ran, `results/metadata.json` if a run was
involved, and the relevant part of `pipeline.log`.
