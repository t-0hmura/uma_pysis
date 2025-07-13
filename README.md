# UMA – PySisyphus Interface

A thin wrapper that lets you call [Meta’s **UMA**](https://github.com/facebookresearch/fairchem) *machine‑learned interatomic potentials* to use with the [**PySisyphus**](https://pysisyphus.readthedocs.io/), a software-suite for the exploration of potential energy surfaces.  
In this implementation, UMA can provides energies, forces and **Analytic Hessians** by back‑propagating *twice* through its neural network. This interface connect them to PySisyphus so you can carry out transition‑state searches, growing‑string calculations, vibrational analysis, etc., with one of the latest MLIPs.

---

## 1 · Installation (including fairchem and pysisyphus installation)

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
from pysisyphus.io.xyz import geom_from_xyz

geom = geom_from_xyz('reac.xyz')
calc = uma_pysis(model="uma-s-1p1")  # choose checkpoint as desired

geom.set_calculator(calc)

E = geom.energy     # Hartree
F = geom.forces     # Hartree·Bohr⁻¹
H = geom.hessian    # (3N × 3N) analytic Hessian

print(f'Energy: {E:.6f} Hartree')
print(f'Max Force : {F.max():.6f} Hartree·Bohr⁻¹')
print(f"Hessian shape: {H.shape}")
```

---

## 3 · Examples

The **examples** directory shows a single‑step *growing‑string* TS search:

```
examples/
├─ example.py      # Exmaple for Python API
├─ run.sh          # One‑liner that launches the job
├─ input.yaml      # PySisyphus workflow
├─ reac.xyz        # Reactant geometry
└─ prod.xyz        # Product geometry
```

Running

```bash
cd examples
bash run.sh
```

optimises the Aromatic Claisen rearrangement from allyl phenyl ether to 6-(prop-2-en-1-yl) cyclohexa-2,4-dien-1-one.

## References
[1] Wood, B. M., Dzamba, M., Fu, X., Gao, M., Shuaibi, M., Barroso-Luque, L., Abdelmaqsoud, K., Gharakhanyan, V., Kitchin, J. R., Levine, D. S., Michel, K., Sriram, A., Cohen, T., Das, A., Rizvi, A., Sahoo, S. J., Ulissi, Z. W., & Zitnick, C. L. (2025). UMA: A Family of Universal Models for Atoms. http://arxiv.org/abs/2506.23971   
[2] Steinmetzer, J., Kupfer, S., & Gräfe, S. (2021). pysisyphus: Exploring potential energy surfaces in ground and excited states. International Journal of Quantum Chemistry, 121(3). https://doi.org/10.1002/qua.26390