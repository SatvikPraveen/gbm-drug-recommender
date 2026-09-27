# Results: interpretation

`results/RESULTS.md` is generated from the artifacts and holds every table.
This page says what those numbers mean and what they do not. Numbers quoted
here are from the committed full run (`results/metadata.json`); if you re-run
with different settings, trust the generated file over this one.

## 1. Which drugs are GBM-selective?

Of 542 drugs screened on 34 GBM lines, **24** are significantly more effective
against GBM lines than against the pan-cancer panel (one-sample t-test of GDSC
z-scores, BH FDR < 0.05, mean z ≤ −0.3). **139** drugs are significantly *less*
effective in GBM than average, a reminder that GBM lines are, on the whole, a
resistant group.

The ranking is biologically coherent, which is the main sanity check on the
endpoint:

- The two most selective compounds are **PI3Kβ inhibitors** (TGX-221, mean z
  = −1.35; AZD6482, −0.48, FDR 6 × 10⁻⁴). PTEN loss is present in roughly
  a third of GBMs and PTEN-null cells depend on PI3Kβ (p110β) rather than
  p110α — a known and clinically pursued vulnerability.
- **mTOR/PI3K axis** more broadly: torin2, FS112 (p70S6K), the mTOR pathway
  is the most represented pathway among selective drugs.
- **ROCK1/2** (GSK269962A), **HSP90** (tanespimycin), **TBK1/PDK1** (BX795),
  **SRC/LCK** (WH-4-023), **PDGFR/KIT/CSF1R** (pazopanib), broad-spectrum
  kinase inhibitors (staurosporine, midostaurin) and DNA-damaging bleomycin
  complete the top of the list.
- Temozolomide, the standard-of-care alkylator, is **not** GBM-selective in
  this cell-line panel (its 72-h in-vitro effect is weak and not GBM-specific),
  which is consistent with the literature and a useful reminder that
  cell-line selectivity and clinical utility are different things.

Note the ~30× spread of geometric-mean IC50 among the selective drugs (27 nM
for staurosporine to 31 µM for CAP-232): selectivity and potency are separate
axes, and the pipeline treats them as separate tasks.

## 2. Does structure predict GBM response?

Three tasks, 14 models, identical molecule-grouped folds (5 × 5-fold) and
scaffold-disjoint folds, dummy baselines and a permutation null.

**Potency (mean ln IC50): yes, modestly.** The best model reaches Spearman
ρ ≈ 0.40 [0.35, 0.43] under grouped CV and ≈ 0.41 on scaffold-disjoint folds
(Random Forest on descriptors + pathway one-hot). Descriptors alone give
ρ ≈ 0.35, fingerprints ≈ 0.35. This is the expected signal: lipophilicity,
size and chemotype correlate with cytotoxic potency in any cell line.

**Selectivity (mean GBM z-score): weakly.** The best structural model manages
ρ ≈ 0.22 [0.18, 0.25] (KNN with Jaccard distance on Morgan fingerprints), and
**a 25-column one-hot of GDSC's pathway annotation does exactly as well
(ρ ≈ 0.21–0.22) with no chemistry in it at all.** Combining descriptors with
the annotation does not add anything (ρ ≈ 0.20). The permutation null
confirms the signal is real (p = 0.048, the floor for 20 rounds), but a ρ of
0.2 explains ~4 % of rank variance: structure is not how GBM selectivity
should be predicted. What a drug *targets* matters; what it *looks like*
barely does, beyond the extent to which look encodes target class.

**Classification of the 19 selective structures**: Random Forest on Morgan
fingerprints reaches ROC-AUC ≈ 0.75 [0.71, 0.80] and PR-AUC ≈ 0.33 against a
positive rate of 0.044 (7× enrichment at the top of the ranking). On
scaffold-disjoint folds the best is ≈ 0.72. With 19 positives the CIs are
wide and the scaffold estimate spans 0.59–0.85; treat this as "there is
signal", not as a deployable classifier.

**Graph neural networks**: GCN and GAT encoders on molecular graphs sit in
the middle of the pack or below on every task (GCN: selectivity ρ = 0.18 [0.14, 0.22],
potency 0.28 [0.24, 0.32], classification AUC 0.70 [0.64, 0.74]; GAT is
weaker on all three) and never beat gradient boosting or random forests on
fingerprints. At ~430 molecules this is the expected outcome and
the reason they are reported as a controlled comparison rather than a
headline. The committed results use five grouped-CV repeats for the GNNs
(`--gnn-repeats 5`), the same as the tabular models.

**Model ranking is not stable enough to name a winner.** Within each task the
top four or five models have overlapping CIs. The robust conclusions are
about *feature families* (annotation ≈ fingerprints > descriptors for
selectivity; descriptors ≈ fingerprints for potency), not about algorithms.

## 3. Exploratory and descriptive analyses

- **One-class novelty** (OC-SVM on the selective drugs, scored out-of-fold):
  ROC-AUC ≈ 0.47, PR-AUC ≈ 0.05 — no better than chance. The earlier
  version of this project reported this model's *training* drugs as its
  discoveries; evaluated properly it has nothing to say. Kept only so the
  comparison is on record.
- **Clustering** in descriptor space finds two well-separated groups
  (silhouette 0.63): essentially small/medium drug-like molecules versus
  large natural-product-derived ones (taxanes, macrolides, bleomycin). GBM
  selectivity does not concentrate in either.
- **Similarity views** on the 30-drug candidate set agree: Tanimoto and MCS
  similarity are strongly correlated (Mantel r ≈ 0.68, p = 0.001), and the
  embeddings of the GNN trained on GBM selectivity correlate with Tanimoto
  (r ≈ 0.73) and MCS (r ≈ 0.58), i.e. the network's notion of similarity is
  still mostly structural. Candidates are structurally diverse: only
  a handful of pairs exceed Tanimoto 0.7.

## 4. Combination hypotheses

Pairs among the candidate set are ranked by target diversity, pathway
complementarity, single-agent potency and structural novelty. The top of the
list pairs the PI3Kβ inhibitor TGX-221 with drugs acting through different
pathways (SRC/LCK, ROCK, HSP90, PDGFR/KIT). These are reasonable things to
put on a combination screen; they are **not** synergy predictions. Nothing in
the score uses combination data, and pairs of drugs that are individually
GBM-selective will always rank high regardless of any interaction between
them. A structural screen of the top 100 pairs raised low-level flags
(acid/amine, lipophilicity mismatch) on 28 and no moderate or high flags;
"no flag" is not "safe".

## 5. Pathway enrichment

Targets of the 24 selective drugs (36 genes) were tested against the targets
of all screened drugs (285 genes). The terms that reach FDR < 0.05 are almost
all **Reactome immune-signalling sets** ("TCR signaling", "Adaptive immune
system", "FCERI-mediated NF-κB activation"). This is an artefact of gene-set
membership, not evidence of immune biology in a cell-line screen: the genes
driving it are LCK, SRC, CHUK/IKBKB and TBK1, kinases that happen to be
annotated to T-cell signalling. Read those rows as "GBM-selective drugs are
enriched for SRC-family / IKK-family kinase inhibitors". The more
mechanistically interesting KEGG terms — **NF-κB signalling, mTOR signalling,
base-excision repair (PARP1/2, FEN1), cytosolic DNA sensing** — are
suggestive but do not survive correction (q ≈ 0.10–0.23) with a 36-gene query.

Because the background is the screening library itself, even a significant
term only says "among the kinds of targets GDSC screened, this class is
over-represented in GBM-selective drugs", which is narrower and more
defensible than genome-background enrichment. 81 drugs have target tokens the
alias table could not map to genes; they are listed in
`results/pathways/unmapped_targets.csv`.

## 6. What would change these conclusions

- **More GBM lines or patient-derived models.** 34 lines and 19 selective
  structures bound everything above.
- **Per-cell-line modelling** with expression/mutation features (drug × line
  rows, line-grouped folds) would test whether PTEN status explains the PI3Kβ
  signal directly.
- **Batch correction** between GDSC1 and GDSC2 ln IC50 might sharpen the
  potency task.
- **Hyper-parameter tuning under nested CV** could move individual models by
  a few hundredths of ρ; it would not turn a 0.2 into a 0.5.
