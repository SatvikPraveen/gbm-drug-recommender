# Methods

This document records every modelling decision in the pipeline and why it was
made. Numbers produced by a run are in `results/RESULTS.md` (auto-generated) and
interpreted in `docs/RESULTS.md`.

## 1. Data

**Source.** GDSC release 8.5 (27 Oct 2023): fitted dose–response tables for
GDSC1 and GDSC2, the screened-compound annotation and the cell-line annotation.
`data/MANIFEST.json` pins URLs and SHA-256 checksums; `scripts/download_gdsc.py`
refuses to build from files that do not match.

**Cohort.** Cell lines with GDSC's own `TCGA_DESC == "GBM"` (34 lines). An
earlier version matched cell-line names from a hand-written list and found 4.

**Unit of analysis.** One row per *drug name*, aggregated over GBM cell lines.
Duplicate screens of one drug on one cell line (GDSC1 and GDSC2, or two
`DRUG_ID`s) are averaged per cell line first, so each line contributes one
observation per drug to every statistic.

**Response variables.**

| Variable | Definition | Used for |
|---|---|---|
| `z_mean` | mean over GBM lines of GDSC's `Z_SCORE` (ln IC50 standardised per drug across *all* screened lines) | primary regression target; "GBM-selective sensitivity" |
| `ln_ic50_mean` | mean ln IC50 (µM) over GBM lines | secondary regression target; "GBM potency" |
| `gbm_selective` | `z_q < 0.05` **and** `z_mean ≤ −0.3` **and** `n_cell_lines ≥ 5`, where `z_q` is the BH-adjusted p of a one-sample t-test of the drug's GBM z-scores against 0 | classification target; candidate set |
| `potent` | `ln_ic50_mean < 0` (geometric-mean IC50 < 1 µM) | descriptive only |

*Why z-scores rather than an IC50 cut-off.* An absolute threshold such as
"IC50 < 10 µM" mixes general cytotoxicity with GBM-specific activity and depends
on each drug's tested concentration range. The z-score asks the question a
GBM-specific recommender should ask: are GBM lines more sensitive to this drug
than cancer lines in general? The one-sample t-test makes that a statement with
an error rate; the effect-size floor (−0.3 SD, "small" by Cohen's convention)
excludes drugs whose significance comes from tiny consistent shifts. With this
definition 24 of 542 drugs qualify; the count is insensitive to the effect-size
floor anywhere between 0 and −0.3.

*Extrapolated IC50s.* The median drug has only 17 % of its GBM curves with a
fitted IC50 inside the tested concentration range; `ic50_within_range` is
carried through so this can be audited. Potency results should be read with
that in mind. Z-scores and AUC are less affected.

## 2. Structures

SMILES come from PubChem by name, then by GDSC synonym, then by vendor-code
spelling variants, each hit validated with RDKit (`scripts/fetch_smiles.py`).
Because name lookup can silently return the wrong compound, an audit of the
riskier resolutions (non-primary-name matches, very small or very large
molecules, name/title mismatches) was performed and the corrections are in
`data/smiles/manual_overrides.csv` with a reason for each. Biologics and
Sanger-internal codes with no public structure are excluded (113 of 542 drugs;
`data/smiles/unresolved.txt`).

Salts are reduced to the largest organic fragment (`smiles_parent`); metal
complexes (cisplatin, oxaliplatin) are kept whole. All featurisation uses the
parent structure.

## 3. Features

| Block | Content |
|---|---|
| `descriptors` | 13 RDKit descriptors: MolWt, MolLogP, MolMR, HBD, HBA, TPSA, rotatable bonds, aromatic rings, Fsp3, heavy atoms, ring count, heteroatoms, QED. Median-imputed, standardised inside each model pipeline. |
| `morgan` | Morgan fingerprint, radius 2, 2048 bits (ECFP4-like). |
| `smiles` | Parent SMILES, converted to graphs by the GNN (atom: element, degree, charge, hybridisation, H count, chirality, aromaticity, ring, mass; bond: order, conjugation, ring). |
| `pathway` | One-hot of GDSC's `PATHWAY_NAME` annotation (~25 columns). Contains no chemistry; included so "what the drug targets" can be compared with "what the drug looks like" on identical folds. |
| `descriptors_pathway` | Concatenation of `descriptors` and `pathway`. |

## 4. Evaluation protocol

**No leakage by construction.** Folds are built over molecule *groups*:

* `grouped`: InChIKey connectivity block, so salts, stereoisomers and the same
  compound screened under two names (e.g. `Bleomycin (10 uM)` / `(50 uM)`)
  never straddle a fold boundary. Stratified for classification. Repeated with
  different seeds (5 × 5-fold by default).
* `scaffold`: Bemis–Murcko scaffolds assigned to folds largest-first into the
  currently smallest fold (a k-fold generalisation of the DeepChem scaffold
  split). This is the out-of-distribution setting: test molecules share no
  scaffold with training molecules. Deterministic, run once.

`assert_no_group_leakage` raises if any group appears on both sides of a split.

**Metrics.** Regression: Spearman ρ (primary; the use case is ranking), Pearson
r, R², RMSE, MAE. Classification: ROC-AUC (primary), PR-AUC, balanced accuracy,
MCC, Brier. Each is reported as mean ± SD and a 95 % percentile-bootstrap CI
over fold scores (1000 resamples). Fold scores are not independent, so the CI
is a description of run-to-run spread, not a confidence interval in the
strict sense; the y-scrambling p-value below is the inferential statement.

**Baselines and null models.** Every task includes a dummy predictor (mean or
class prior). The best tabular model per task is re-evaluated with labels
permuted across molecules (20 rounds); the empirical p is the fraction of
permuted rounds that match or beat the real score. A model whose real score is
not clearly outside the permuted distribution has learned nothing generalisable.

**Hyper-parameters** of the tabular models are selected by *nested*
cross-validation: inside each outer training fold, a 3-fold inner CV grouped by
the same molecule ids (stratified for classification) scores every setting of a
small search space (1–2 axes, 3–8 settings; `gbm_drug/models/zoo.py`) by the
task's primary metric, and the best is refit on the whole training fold. Test
molecules never influence the choice. The selected setting and inner score are
recorded per outer fold (`results/benchmark/tuning.csv`). The permutation null
and the final refit use the same procedure. GNN settings are fixed
(`config.GNN_*`); tuning them at ~430 molecules would cost far more than it
would reveal. `--no-tune` restores fixed settings for every model.

**Class imbalance.** ~4 % positives. Linear/SVM/RF use `class_weight="balanced"`;
XGBoost sets `scale_pos_weight` from the training fold; the GNN re-weights the
positive class in the BCE loss.

## 5. Models

Linear (ridge / logistic), KNN with Jaccard distance on fingerprints, random
forest and XGBoost on both feature blocks, RBF-SVM, MLP, and two GNNs (GCN and
GAT encoders, mean pooling, MLP head; standardised targets; internal 15 %
validation split for early stopping with best-weight restore; Adam,
weight decay 1e-5). GNN folds use one repeat by default for run time; the
`--gnn-repeats` flag raises it.

## 6. Downstream analyses

**One-class novelty (exploratory).** A One-Class SVM fitted on the
GBM-selective drugs scores every drug out-of-fold; separation of held-out
selective drugs from the rest is reported as ROC-AUC / PR-AUC. Retained for
continuity with earlier versions; the supervised benchmark supersedes it.

**Clustering (descriptive).** Standardised descriptors; K-means with k chosen
by silhouette over 2–10; Ward linkage and DBSCAN as alternative views; PCA and
UMAP for display. No claim is made beyond "these drugs are neighbours in
descriptor space".

**Similarity.** Tanimoto on Morgan fingerprints for all drugs; MCS-Tanimoto
(|MCS| / (|A| + |B| − |MCS|) in heavy atoms, ring-only, complete rings, 1 s
timeout per pair) and cosine similarity of GNN embeddings for the candidate
set. The three views are compared with a Mantel test (Spearman, 999
permutations) because pairwise entries of a similarity matrix are not
independent observations.

**Combination hypotheses.** Pairs among candidates are scored by a weighted sum
of target-set diversity (1 − overlap coefficient of GDSC target tokens),
pathway complementarity (GDSC pathway differs), single-agent potency (mean
percentile of −z), and structural novelty (1 − Tanimoto), with weights in
`config.COMBINATION_WEIGHTS`. Pairs above Tanimoto 0.7 are excluded as
near-duplicates. This is hypothesis generation from single-agent data and has
not been validated against any combination screen; `bliss_independence` is
provided for when such data exist.

**Structural screen of pairs.** A handful of functional-group rules (acid +
aliphatic amine, Michael acceptor + thiol, nitro + thiol, aldehyde + primary
amine, platinum + thiol) and a lipophilicity-mismatch rule. Output is an alert
level; `no_flag` means no rule fired. Pharmacokinetic (CYP450) interactions are
not modelled unless a curated annotation table is supplied.

**Pathway enrichment.** GDSC free-text targets are mapped to HGNC symbols
through an explicit alias table (unmapped tokens are reported). Targets of
GBM-selective drugs are tested for over-representation in KEGG 2021, Reactome
2022 and WikiPathways 2023 gene sets by the hypergeometric test with **the
targets of all screened drugs as background**, BH-corrected. Testing against
the genome would only rediscover that oncology screening libraries are built
from kinase inhibitors and DNA-damaging agents.

## 7. Reproducibility

* `python scripts/download_gdsc.py && python scripts/build_gbm_dataset.py &&
  python scripts/fetch_smiles.py` regenerates every committed data table.
* `python main.py` regenerates every result; `results/metadata.json` records the
  git commit, whether the tree was dirty, Python and dependency versions,
  platform, all configuration values and per-stage timings.
* Seeds: `RANDOM_STATE` for splits, models and permutations; the GNN seeds
  torch and its data loaders. GPU (MPS/CUDA) kernels are not bit-reproducible
  across devices; CPU runs are.
* `requirements.lock` is the exact environment the committed results were
  produced with; `pyproject.toml` gives the compatible ranges.

## 8. Known limitations

* 34 cell lines and ~430 structures is small data; scaffold-split performance
  is the honest estimate for new chemotypes and it is expected to be modest.
* Most fitted IC50s are extrapolated beyond the tested concentration range.
* GDSC1 and GDSC2 used different viability assays; the z-score is computed
  within each screen, ln IC50 is not batch-corrected.
* Target annotations are GDSC's free text, mapped heuristically; ~20 % of
  tokens are not gene symbols.
* Nothing here is validated experimentally. Cell-line sensitivity is not
  clinical efficacy, and the combination scores are not synergy predictions.
