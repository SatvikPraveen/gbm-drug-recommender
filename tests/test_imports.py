"""Smoke tests: the package imports cleanly and config has no import-time side effects."""

import importlib
import subprocess
import sys

import pytest

MODULES = [
    "gbm_drug",
    "gbm_drug.config",
    "gbm_drug.data_processing",
    "gbm_drug.feature_extraction",
    "gbm_drug.similarity",
    "gbm_drug.models",
    "gbm_drug.models.gnn_model",
    "gbm_drug.pathway_analysis",
    "gbm_drug.combination_therapy",
    "gbm_drug.drug_interactions",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_config_import_is_silent_and_does_not_import_torch():
    code = (
        "import sys, gbm_drug.config as c; "
        "assert 'torch' not in sys.modules, 'config must not import torch'; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
    assert out.stderr.strip() == ""
