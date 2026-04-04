#!/usr/bin/env bash

# Render Mermaid figures for the LaTeX spec.
#
# This script renders Notes/tex/figures/*.mmd to Notes/tex/figures/*.png using
# the public Kroki API.
#
# Requirements:
# - curl
# - network access to https://kroki.io

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG_DIR="${SCRIPT_DIR}/figures"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found in PATH." >&2
  exit 1
fi

if [ ! -d "${FIG_DIR}" ]; then
  echo "Error: figures directory not found: ${FIG_DIR}" >&2
  exit 1
fi

shopt -s nullglob

sources=("${FIG_DIR}"/*.mmd)
if [ ${#sources[@]} -eq 0 ]; then
  echo "No Mermaid sources found under ${FIG_DIR} (*.mmd)." >&2
  exit 1
fi

for src in "${sources[@]}"; do
  out="${src%.mmd}.png"
  echo "Rendering $(basename "${src}") -> $(basename "${out}")"
  curl -fsSL \
    -H "Content-Type: text/plain" \
    --data-binary @"${src}" \
    "https://kroki.io/mermaid/png" \
    -o "${out}"
done

echo "Done."
