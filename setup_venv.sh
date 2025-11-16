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
PYTHON_BIN="python3"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found in PATH. Please install Python 3." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment ${VENV_DIR} already exists. Reusing it."
fi

# Activate the virtual environment (for the rest of this script)
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found in current directory." >&2
  exit 1
fi

pip install -r requirements.txt

echo "Virtual environment setup complete. To use it, run:"
echo "  source ${VENV_DIR}/bin/activate"
