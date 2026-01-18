#!/usr/bin/env bash

# Setup script for Binance ML Training Platform Python virtual environment.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#
# This script will:
# - Create a Python virtual environment in .venv (if it does not already exist)
# - Upgrade pip inside the venv
# - Install the dependencies from requirements.txt

set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: No suitable Python interpreter found in PATH." >&2
  echo "Install Python 3.11 (recommended) and retry." >&2
  echo "macOS (Homebrew): brew install python@3.11" >&2
  echo "Ubuntu: sudo apt-get install python3.11 python3.11-venv" >&2
  exit 1
fi

"${PYTHON_BIN}" - << 'EOF'
import sys

min_required = (3, 9)
max_allowed = (3, 12)
if sys.version_info < min_required:
    raise SystemExit(
        f"Error: Python {min_required[0]}.{min_required[1]} or higher is required, "
        f"but found {sys.version_info.major}.{sys.version_info.minor}.",
    )
if sys.version_info >= max_allowed:
    raise SystemExit(
        f"Error: Python < {max_allowed[0]}.{max_allowed[1]} is required due to MLflow/TensorFlow compatibility, "
        f"but found {sys.version_info.major}.{sys.version_info.minor}.",
    )
EOF

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment ${VENV_DIR} already exists. Reusing it."
fi

# If an existing venv uses an unsupported Python version, recreate it.
if [ -x "${VENV_DIR}/bin/python" ]; then
  "${VENV_DIR}/bin/python" - << 'EOF'
import sys

min_required = (3, 9)
max_allowed = (3, 12)
if sys.version_info < min_required or sys.version_info >= max_allowed:
    raise SystemExit(1)
EOF
  if [ "$?" -ne 0 ]; then
    echo "Existing virtual environment uses an unsupported Python version; recreating..."
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

# Activate the virtual environment (for the rest of this script)
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found in current directory." >&2
  exit 1
fi

# Prefer binary wheels for pyarrow to avoid source builds on macOS.
pip install --only-binary=pyarrow -r requirements.txt

# Verify installed dependencies are consistent
python -m pip check

echo "Virtual environment setup complete. To use it, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run tests (including hypothesis property tests):"
echo "  python -m unittest discover -s tests -v"
echo ""
echo "To reduce hypothesis examples in CI:"
echo "  CI=true python -m unittest discover -s tests -v"
