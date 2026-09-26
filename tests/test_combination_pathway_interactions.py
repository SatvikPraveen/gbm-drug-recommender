import numpy as np
import pandas as pd
import pytest

from gbm_drug import combination_therapy as ct
from gbm_drug import drug_interactions as di
from gbm_drug import pathway_analysis as pa

# ---------------------------------------------------------------------------
# Combination scoring
# ---------------------------------------------------------------------------


def _candidates():
    return pd.DataFrame(
        {
            "drug_name": ["Afatinib", "Gefitinib", "Etoposide", "Mystery"],
            "putative_target": ["EGFR, ERBB2, ERBB4", "EGFR", "TOP2", np.nan],
            "pathway_name": ["EGFR signaling", "EGFR signaling", "DNA replication", "Other"],
            "z_mean": [-0.6, -0.5, -0.3, 0.2],
            "gbm_selective": [True, True, False, False],
        }
    )


def _tanimoto():
    names = ["Afatinib", "Gefitinib", "Etoposide", "Mystery"]
    a = np.full((4, 4), 0.1)
    a[0, 1] = a[1, 0] = 0.45
    a[0, 2] = a[2, 0] = 0.05
    a[1, 2] = a[2, 1] = 0.04
    np.fill_diagonal(a, 1.0)
    return pd.DataFrame(a, index=names, columns=names)


def test_parse_target_tokens():
    assert ct.parse_target_tokens("EGFR, ERBB2, ERBB4") == {"EGFR", "ERBB2", "ERBB4"}
    assert ct.parse_target_tokens("PI3K/MTOR") == {"PI3K", "MTOR"}
    assert ct.parse_target_tokens("PKC, others") == {"PKC"}
    assert ct.parse_target_tokens(np.nan) == set()


def test_two_egfr_inhibitors_score_below_egfr_plus_topoisomerase():
    pairs = ct.score_pairs(_candidates(), _tanimoto()).set_index(["drug_a", "drug_b"])
    same = pairs.loc[("Afatinib", "Gefitinib")]
    diff = pairs.loc[("Afatinib", "Etoposide")]
    assert same["target_diversity"] == 0.0 and same["pathway_complementarity"] == 0.0
    assert diff["target_diversity"] == 1.0 and diff["pathway_complementarity"] == 1.0
    assert diff["total_score"] > same["total_score"]
    assert "identical target profile" in same["rationale"] and "both GBM-selective" in same["rationale"]
    assert "distinct targets" in diff["rationale"]


def test_unannotated_targets_get_neutral_score_and_note():
    pairs = ct.score_pairs(_candidates(), _tanimoto()).set_index(["drug_a", "drug_b"])
    row = pairs.loc[("Etoposide", "Mystery")]
    assert row["target_diversity"] == 0.5 and row["pathway_complementarity"] == 0.5
    assert "unannotated" in row["rationale"]


def test_near_duplicates_are_excluded_but_kept_auditable():
    t = _tanimoto()
    t.loc["Afatinib", "Gefitinib"] = t.loc["Gefitinib", "Afatinib"] = 0.9
    pairs = ct.score_pairs(_candidates(), t).set_index(["drug_a", "drug_b"])
    row = pairs.loc[("Afatinib", "Gefitinib")]
    assert np.isnan(row["total_score"]) and "near-duplicate" in row["excluded_reason"]
    assert pd.isna(row["rank"])


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ct.score_pairs(
            _candidates(),
            weights={
                "target_diversity": 1,
                "pathway_complementarity": 1,
                "potency": 0,
                "structural_novelty": 0,
            },
        )


def test_pair_matrix_symmetric():
    pairs = ct.score_pairs(_candidates(), _tanimoto())
    m = ct.pair_matrix(pairs)
    assert m.shape == (4, 4) and np.allclose(m.fillna(-1).to_numpy(), m.fillna(-1).to_numpy().T)


def test_bliss_independence_calls():
    a, b = np.array([0.3, 0.4]), np.array([0.3, 0.4])
    expected = a + b - a * b
    assert ct.bliss_independence(a, b, expected + 0.2)["call"] == "synergistic"
    assert ct.bliss_independence(a, b, expected - 0.2)["call"] == "antagonistic"
    assert ct.bliss_independence(a, b, expected)["call"] == "additive"


# ---------------------------------------------------------------------------
# Pathway enrichment
# ---------------------------------------------------------------------------


def test_targets_to_genes_aliases_and_slash_expansion():
    genes, unmapped = pa.targets_to_genes("PI3Kbeta, CDK4/6, mTOR, Broad spectrum kinase inhibitor")
    assert genes == {"PIK3CB", "CDK4", "CDK6", "MTOR"}
    assert unmapped == {"Broad spectrum kinase inhibitor"}
    assert pa.targets_to_genes("BCR-ABL, SRC")[0] == {"ABL1", "SRC"}
    assert pa.targets_to_genes(np.nan) == (set(), set())


def test_enrich_recovers_planted_signal_with_background():
    rng = np.random.default_rng(0)
    background = [f"G{i}" for i in range(200)]
    pathway_a = set(background[:20])
    gene_sets = {
        "A": pathway_a,
        "B": set(background[100:130]),
        "C": set(rng.choice(background, 25, replace=False)),
    }
    query = list(pathway_a)[:12] + background[150:153]  # 12/15 query genes from pathway A
    res = pa.enrich(query, background, gene_sets, min_overlap=1).set_index("term")
    assert res.loc["A", "overlap"] == 12 and res.loc["A", "q"] < 1e-6
    assert res.loc["A", "fold_enrichment"] > 5
    assert "B" not in res.index  # zero overlap => dropped
    # Genes outside the background are ignored in query and sets.
    res2 = pa.enrich(query + ["NOT_IN_BG"], background, gene_sets, min_overlap=1)
    assert (res2["query_size"] == 15).all()


def test_enrich_handles_empty_query():
    assert pa.enrich([], ["A", "B"], {"t": {"A"}}).empty


def test_load_gmt_and_gbm_filter(tmp_path):
    gmt = tmp_path / "lib.gmt"
    gmt.write_text(
        "EGFR tyrosine kinase inhibitor resistance\tdesc\tEGFR\tERBB2,1.0\tKDR\nOther\tdesc\tTP53\n\n"
    )
    sets = pa.load_gmt(gmt)
    assert sets["EGFR tyrosine kinase inhibitor resistance"] == {"EGFR", "ERBB2", "KDR"}
    df = pd.DataFrame({"term": list(sets), "q": [0.001, 0.001]})
    assert pa.gbm_relevant(df)["term"].tolist() == ["EGFR tyrosine kinase inhibitor resistance"]


# ---------------------------------------------------------------------------
# Interaction screen
# ---------------------------------------------------------------------------


def test_structural_rules_fire_symmetrically():
    acid, amine = "CC(=O)O", "CCN"
    assert di.structural_flags(acid, amine)[0]["rule"] == "acid_base_pair"
    assert di.structural_flags(amine, acid)[0]["rule"] == "acid_base_pair"
    assert di.structural_flags("c1ccccc1", "CCO") == []
    assert di.structural_flags("C1CC", "CCO") == []  # invalid SMILES => nothing, not an error


def test_screen_pairs_levels_and_vocabulary():
    smiles = {"acid": "CC(=O)O", "amine": "CCN", "benzene": "c1ccccc1", "greasy": "CCCCCCCCCCCCCCCCCCCCCCCC"}
    out = di.screen_pairs(
        [("acid", "amine"), ("benzene", "acid"), ("acid", "greasy"), ("acid", "missing")], smiles
    )
    by = out.set_index(["drug_a", "drug_b"])
    assert by.loc[("acid", "amine"), "alert_level"] == "low"
    assert by.loc[("benzene", "acid"), "alert_level"] == "no_flag"
    assert "logp_mismatch" in by.loc[("acid", "greasy"), "rules"]
    assert not by.loc[("acid", "missing"), "structures_available"]
    assert "safe" not in " ".join(out["alert_level"].unique()).lower()
    s = di.summarize(out)
    assert s["n_pairs"] == 4 and s["n_without_structures"] == 1


def test_pk_flags_only_with_annotations():
    ann = pd.DataFrame(
        {"drug_name": ["A", "B"], "enzyme": ["CYP3A4", "CYP3A4"], "role": ["substrate", "inhibitor"]}
    )
    assert di.pk_flags("A", "B", None) == []
    assert di.pk_flags("A", "B", ann)[0]["rule"] == "CYP3A4_substrate_inhibitor"
    assert di.pk_flags("A", "A", ann) == []
