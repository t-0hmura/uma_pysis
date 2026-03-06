# uma_pysis Options

For most users, the defaults in the [Quick Start](README.md) are sufficient.

## Calculator Parameters (YAML `calc:` block)

All parameters listed below can be set in the Pysisyphus YAML input file under `calc:`.
They can also be passed as keyword arguments to `uma_pysis(...)` in the Python API.

### Basic

| Parameter | Description | Default |
|-----------|-------------|---------|
| `type` | Calculator type for Pysisyphus. | `uma_pysis` |
| `charge` | Total system charge. | `0` |
| `spin` | Spin multiplicity (2S+1). | `1` |
| `model` | UMA pretrained model name. | `"uma-s-1p2"` |
| `task_name` | Task tag recorded in UMA batches. | `"omol"` |
| `device` | `"auto"`, `"cpu"`, or `"cuda"`. | `"auto"` |

Available models:

| Model | Description |
|-------|-------------|
| `uma-s-1p1` | Small model, fastest while still SOTA on most benchmarks |
| `uma-s-1p2` | Small model v1.2, ~50% faster & ~40% more accurate on OMol (6.6M/290M active/total params) |
| `uma-m-1p1` | Best across all metrics, slower and more memory-intensive |

Available tasks: `oc20`, `omat`, `omol`, `odac`, `omc`.

### Hessian

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hessian_calc_mode` | `"Analytical"` or `"FiniteDifference"`. | `"Analytical"` |
| `hessian_double` | Assemble and return the Hessian in float64 precision. | `true` |
| `out_hess_torch` | Return Hessians as `torch.Tensor` objects instead of numpy. | `false` |

- **Analytical**: Uses second-order autograd (double back-propagation) through the neural network. Only available when `workers` is 1.
- **FiniteDifference**: Central-difference on GPU. Works with any `workers` setting. Slower but more memory-efficient for large systems.

When `workers > 1`, analytical Hessians are automatically disabled and FiniteDifference is used regardless of `hessian_calc_mode`.

### Parallelism

| Parameter | Description | Default |
|-----------|-------------|---------|
| `workers` | Number of predictor workers. | `1` |
| `workers_per_node` | Workers per compute node (for distributed setup). | `1` |

When `workers > 1`, the `ParallelMLIPPredictUnit` from fairchem is used. This requires `fairchem-core[extras]` to be installed.

> **Note**: When `workers > 1`, analytical Hessians are not available.

For HPC multi-node setup (PBS + Ray), see [`HPC_MULTI_WORKER.md`](HPC_MULTI_WORKER.md).

### Graph Construction

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_neigh` | Override graph neighbor cap. | Model default |
| `radius` | Override graph cutoff radius (Angstrom). | Model default |
| `r_edges` | Enable distance edge attributes. | `false` |

### Implicit Solvent Correction

| Parameter | Description | Default |
|-----------|-------------|---------|
| `solvent` | Solvent name (e.g. `"water"`, `"thf"`) or `"none"`. | `"none"` |
| `solvent_model` | `"alpb"` or `"cpcmx"`. | `"alpb"` |
| `xtb_cmd` | xTB executable path or command. | `"xtb"` |
| `xtb_acc` | xTB `--acc` value. | `0.2` |

When `solvent` is not `"none"`, xTB must be installed and available. See [`SOLVENT_EFFECTS.md`](SOLVENT_EFFECTS.md) for details.

### Logging

| Parameter | Description | Default |
|-----------|-------------|---------|
| `print_timing` | Print Hessian computation timing. | `false` |

## YAML Example (full)

```yaml
calc:
 type: uma_pysis
 charge: 0
 spin: 1
 model: uma-s-1p2
 task_name: omol
 device: auto
 hessian_calc_mode: Analytical
 hessian_double: true
 workers: 1
 solvent: none
 print_timing: false
```

## Python API Example

```python
from uma_pysis import uma_pysis

calc = uma_pysis(
    charge=0,
    spin=1,
    model="uma-s-1p2",
    task_name="omol",
    device="auto",
    hessian_calc_mode="Analytical",
    workers=1,
)
```
