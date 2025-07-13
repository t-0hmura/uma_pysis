# UMA – PySisyphus Interface

A thin wrapper that lets you call [Meta’s **UMA**](https://github.com/facebookresearch/fairchem) *machine‑learned interatomic potentials* to use with the [**PySisyphus**](https://pysisyphus.readthedocs.io/), a software-suite for the exploration of potential energy surfaces.  
In this implementation, UMA can provides energies, forces and **Analytic Hessians** by back‑propagating *twice* through its neural network. This interface connect them to PySisyphus so you can carry out transition‑state searches, growing‑string calculations, vibrational analysis, etc., with one of the latest MLIPs.

---

## 1 · Installation

### CUDA 12.6

```bash
pip install git+https://github.com/t-0hmura/uma_pysis.git
huggingface-cli login    # required to download the pretrained UMA checkpoints
```

### CUDA 12.8 (needed for RTX 50 series)

```bash
pip install git+https://github.com/t-0hmura/uma_pysis.git
pip install --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
huggingface-cli login
```

**Dependencies**

| package | version |
|---------|---------|
| Python  | ≥ 3.11  |
| PyTorch | 2.6 + (*or* 2.7.0 for CUDA 12.8) |
| fairchem‑core | latest |
| PySisyphus | latest |
| ASE, NumPy | — |

---

## 2 · Quick Start

### Command‑line

```bash
uma_pysis input.yaml
```

The script registers a calculator called `uma_pysis`, so any YAML pysisyphus input that lists this name will automatically pull in UMA energies, forces and Hessians.

### Python

```python
from uma_pysis import uma_pysis

calc = uma_pysis(model="uma-s-1p1")  # choose checkpoint as desired

E = calc.get_energy(elem, coords)["energy"]     # Hartree
F = calc.get_forces(elem, coords)["forces"]     # Hartree·Bohr⁻¹
H = calc.get_hessian(elem, coords)["hessian"]   # (3N × 3N) analytic Hessian
```

---

## 3 · Examples

The **examples** directory shows a single‑step *growing‑string* TS search:

```
examples/
├─ run.sh          # one‑liner that launches the job
├─ input.yaml      # PySisyphus workflow
├─ reac.xyz        # reactant geometry
└─ prod.xyz        # product geometry
```

Running

```bash
cd examples
bash run.sh
```

optimises the Aromatic Claisen rearrangement from allyl phenyl ether to 6-(prop-2-en-1-yl) cyclohexa-2,4-dien-1-one.