#!/usr/bin/env bash
# Create a virtual environment and install the package in editable mode.
# Usage: ./setup.sh            (uses `uv` if present, otherwise python -m venv + pip)
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -e ".[dashboard,notebook,dev]"
else
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e ".[dashboard,notebook,dev]"
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:             pytest            # unit tests"
echo "                      python main.py    # full pipeline"
