# Solvent Effects (xTB Implicit-Solvent Delta Correction)

`uma_pysis` supports an implicit-solvent correction that adds only solvent
contributions to MLIP vacuum predictions.

## Solvent Delta Terms Added to MLIP Outputs

For each geometry `R`, the wrapper evaluates xTB twice:

- vacuum
- implicit solvent (`solvent: <name>`, model selected by `solvent_model`)

The solvent delta terms are defined as:

- `dE(R) = E_xTB(solv) - E_xTB(vac)`
- `dF(R) = F_xTB(solv) - F_xTB(vac)`
- `dH(R) = H_xTB(solv) - H_xTB(vac)`

The calculator then returns:

- `E_total = E_MLIP(vac) + dE`
- `F_total = F_MLIP(vac) + dF`
- `H_total = H_MLIP(vac) + dH`

This keeps the MLIP model in vacuum mode and adds only solvent-induced correction terms.

## Installation

Install xTB in your conda environment (or build from source):

```bash
conda install xtb
```

## YAML Usage

```yaml
calc:
 type: uma_pysis
 charge: 0
 spin: 1
 solvent: water           # e.g. water, thf, toluene, ...
 solvent_model: alpb      # alpb (default) or cpcmx
 xtb_cmd: xtb             # path to xTB executable (default: xtb)
 xtb_acc: 0.2             # xTB accuracy (default: 0.2)
```

## Python API Usage

```python
from uma_pysis import uma_pysis
from uma_pysis.solvent import SolventCorrectedCalculator

base = uma_pysis(charge=0, spin=1, model="uma-s-1p1", device="auto")
calc = SolventCorrectedCalculator(base, solvent="water", solvent_model="alpb")

# Use calc as a normal Pysisyphus calculator
from pysisyphus.io.xyz import geom_from_xyz
geom = geom_from_xyz('reac.xyz')
geom.set_calculator(calc)

E = geom.energy     # Hartree (MLIP + solvent correction)
F = geom.forces     # Hartree·Bohr⁻¹ (MLIP + solvent correction)
H = geom.hessian    # (3N × 3N) Hessian (MLIP + solvent correction)
```

Alternatively, when using the YAML interface (`uma_pysis input.yaml`), the solvent correction is applied automatically when `solvent` is specified in `calc:`.

## CPCM-X Setup

CPCM-X requires xTB to be built from source with CPCM-X linked in.
The conda-forge `xtb` package does not include CPCM-X support.

**Step 1: Build xTB with `-DWITH_CPCMX=ON`**

CPCM-X is bundled in the xTB source tree (`subprojects/cpx.wrap`) and is fetched automatically during the CMake configure step. Requires GCC >= 10 (gfortran 8 causes internal compiler errors).

```bash
git clone --depth 1 https://github.com/grimme-lab/xtb.git
cd xtb
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPCMX=ON \
  -DBLAS_LIBRARIES=/path/to/libblas.so \
  -DLAPACK_LIBRARIES=/path/to/liblapack.so
make -C build tblite-lib -j8   # build tblite first to avoid a parallel build race
make -C build xtb-exe -j8
```

**Step 2: Use the custom xTB via `xtb_cmd`**

```yaml
calc:
 type: uma_pysis
 charge: 0
 spin: 1
 solvent: water
 solvent_model: cpcmx
 xtb_cmd: /path/to/xtb
```

`CPXHOME` must be set at runtime to point to the CPCM-X source directory (containing `DB/`). When xTB fetches CPCM-X during build, the source is placed under `build/_deps/cpcmx-src/`.

For full details, see:
- https://github.com/grimme-lab/xtb
- https://github.com/grimme-lab/CPCM-X

## Performance Notes

- Solvent correction runs two xTB calculations per geometry point (vacuum + solvated state).
- Hessian correction is expensive: each state needs an xTB Hessian.
- The vacuum and solvated xTB calculations are parallelized using `ThreadPoolExecutor(max_workers=2)`.

## Citation

This implementation follows the solvent-correction approach described in:
Zhang, C., Leforestier, B., Besnard, C., & Mazet, C. (2025). Pd-catalyzed regiodivergent arylation of cyclic allylboronates. Chemical Science, 16, 22656-22665. https://doi.org/10.1039/d5sc07577g

If citing this correction in a paper, you can use the following:
`Implicit solvent effects were accounted for by integrating the ALPB [or CPCM-X] solvation model from the xtb package as an additional correction to UMA-generated energies, gradients, and Hessians.`

## Troubleshooting

- **xTB command not found**:
  - Install xTB in the active environment.
  - Or set `xtb_cmd: /full/path/to/xtb` in YAML.
- **xTB solvent correction failed**:
  - Verify the solvent spelling (`water`, `thf`, `toluene`, ...).
  - For `solvent_model: cpcmx`, use an xTB build with CPCM-X support
    (see https://github.com/grimme-lab/CPCM-X).
