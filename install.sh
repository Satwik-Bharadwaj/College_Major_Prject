#!/usr/bin/env bash
# TrainOS one-shot installer. Creates a virtualenv, installs deps, trains the
# model if missing. Works on Ubuntu and macOS (for testing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"

echo "==> TrainOS installer"
echo "    dir: $HERE"
echo "    python: $($PYTHON --version)"

if [ ! -d venv ]; then
  echo "==> creating virtualenv"
  "$PYTHON" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> installing dependencies"
python -m pip install --quiet -r requirements.txt

echo "==> installing trainos package"
python -m pip install --quiet -e .

if [ ! -f trainos/artifacts/trainos_rf.joblib ]; then
  echo "==> training model (first run)"
  python -m trainos.cli train
fi

echo ""
echo "==> Done. Try:"
echo "    ./venv/bin/python -m trainos.cli selftest"
echo "    ./venv/bin/python -m trainos.cli benchmark"
echo "    sudo ./venv/bin/python -m trainos.cli run --dry-run   # (Linux)"
