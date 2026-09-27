---
title: GBM Drug Recommender
---

# GBM Drug Recommender

*Structure, selectivity and combination hypotheses for glioblastoma, from the
GDSC drug-sensitivity screens — reproducibly, with null models.*

[Source and README](https://github.com/SatvikPraveen/gbm-drug-recommender) ·
[Methods](METHODS.html) ·
[Results: interpretation](RESULTS.html) ·
[Generated results (RESULTS.md)](https://github.com/SatvikPraveen/gbm-drug-recommender/blob/main/results/RESULTS.md) ·
[Data provenance](https://github.com/SatvikPraveen/gbm-drug-recommender/blob/main/data/README.md)

---

## In brief

The Genomics of Drug Sensitivity in Cancer (GDSC) screens contain 20,329
dose–response curves for 542 compounds across 34 glioblastoma cell lines. This
project turns them into three answers, each backed by a leakage-free
evaluation and a permutation null:

1. **Which compounds are GBM-selective?** 24 of 542 at FDR < 0.05, led by the
   PI3Kβ inhibitors TGX-221 and AZD6482 — consistent with the PI3Kβ dependence
   of PTEN-deficient GBM.
2. **Does chemical structure predict GBM response?** Modestly for potency
   (Spearman ρ ≈ 0.4 on scaffold-disjoint folds), weakly for selectivity
   (ρ ≈ 0.2); GDSC's own target-pathway annotation predicts selectivity as well
   as any structural model. Graph neural networks do not beat fingerprints at
   ~430 molecules.
3. **Which pairs deserve a combination screen?** A transparent heuristic over
   target diversity, pathway complementarity, potency and structural novelty —
   hypotheses, not synergy predictions.

Every number is produced by code from the checksummed GDSC release; nothing is
typed in by hand.

---

## Figures

<p align="center">
<img src="https://raw.githubusercontent.com/SatvikPraveen/gbm-drug-recommender/main/results/figures/selectivity_volcano.png" width="640" alt="Selectivity volcano"><br>
<em>GBM-selective sensitivity: mean GDSC z-score over GBM lines against BH-adjusted p; the 24 selective compounds in red.</em>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/SatvikPraveen/gbm-drug-recommender/main/results/figures/benchmark_gbm_selectivity_spearman_rho.png" width="720" alt="Benchmark: GBM selectivity"><br>
<em>Predicting selectivity from structure and annotation: Spearman ρ, mean ± 95 % CI over folds, grouped and scaffold CV.</em>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/SatvikPraveen/gbm-drug-recommender/main/results/figures/benchmark_gbm_potency_spearman_rho.png" width="720" alt="Benchmark: GBM potency"><br>
<em>Predicting potency: structure carries more signal for cytotoxic potency than for GBM selectivity.</em>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/SatvikPraveen/gbm-drug-recommender/main/results/figures/y_scrambling.png" width="760" alt="Permutation null"><br>
<em>Permutation null: score distribution under label permutation with the real score marked, per task.</em>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/SatvikPraveen/gbm-drug-recommender/main/results/figures/clustering_umap.png" width="760" alt="Descriptor-space clustering"><br>
<em>Descriptor space: silhouette-chosen K-means clusters (left) and the same embedding coloured by GBM selectivity (right).</em>
</p>

---

## Reproduce

```bash
git clone https://github.com/SatvikPraveen/gbm-drug-recommender.git
cd gbm-drug-recommender
./setup.sh && source .venv/bin/activate
pytest                    # unit tests
python main.py --quick    # every stage without the GNN, ~2 min
python main.py            # full run with nested tuning and GNNs
```

A laptop is sufficient; no cluster or GPU is required.

---

## Citing

See [CITATION.cff](https://github.com/SatvikPraveen/gbm-drug-recommender/blob/main/CITATION.cff).
GDSC data are provided for academic, non-commercial use (Yang *et al.* 2013;
Iorio *et al.* 2016); structures come from PubChem (Kim *et al.* 2023). Code is
MIT-licensed.

<sub>This is a research tool, not a clinical decision aid.</sub>
