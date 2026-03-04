#!/bin/bash
set -e

echo "=== small (gas phase) ==="
cd ./small/
uma_pysis input.yaml | tee pysis.log
python3 example.py

echo ""
echo "=== large (gas phase) ==="
cd ../large/
uma_pysis input.yaml | tee pysis.log

echo ""
echo "=== solvent_alpb (water, ALPB) ==="
cd ../solvent_alpb/
uma_pysis input.yaml | tee pysis.log

# NOTE: solvent_cpcmx requires xTB built with -DWITH_CPCMX=ON
# Uncomment below if CPCM-X is available:
# echo ""
# echo "=== solvent_cpcmx (water, CPCM-X) ==="
# cd ../solvent_cpcmx/
# uma_pysis input.yaml | tee pysis.log
