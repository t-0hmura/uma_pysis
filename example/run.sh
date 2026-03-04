#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== small (gas phase) ==="
cd "${SCRIPT_DIR}/small/"
uma_pysis input.yaml | tee pysis.log
python3 example.py

echo ""
echo "=== large (gas phase) ==="
cd "${SCRIPT_DIR}/large/"
uma_pysis input.yaml | tee pysis.log

echo ""
echo "=== solvent_alpb (water, ALPB) ==="
cd "${SCRIPT_DIR}/solvent_alpb/"
uma_pysis input.yaml | tee pysis.log

# NOTE: solvent_cpcmx requires xTB built with -DWITH_CPCMX=ON
# Uncomment below if CPCM-X is available:
# echo ""
# echo "=== solvent_cpcmx (water, CPCM-X) ==="
# cd "${SCRIPT_DIR}/solvent_cpcmx/"
# uma_pysis input.yaml | tee pysis.log
