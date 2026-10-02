"""
- uma_pysis.py

UMA Calculator Wrapper for PySisyphus.
Return Energy, Force, Analytic/FiniteDifference Hessian.
Supports multi-worker inference and xTB implicit-solvent correction.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater

# Optional: only needed when workers > 1
try:
    from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
except (ImportError, ModuleNotFoundError):
    ParallelMLIPPredictUnit = None
    guess_inference_settings = None

from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG, ANG2BOHR, AU2EV
from pysisyphus import run

# ------------ unit conversion constants ----------------------------
EV2AU          = 1.0 / AU2EV                     # eV → Hartree
F_EVAA_2_AU    = EV2AU / ANG2BOHR                # eV Å⁻¹ → Hartree Bohr⁻¹
H_EVAA_2_AU    = EV2AU / ANG2BOHR / ANG2BOHR     # eV Å⁻² → Hartree Bohr⁻²


# ===================================================================
#                         UMA core wrapper
# ===================================================================
class UMAcore:
    """Thin wrapper around fairchem-UMA predict_unit.

    If ``workers > 1``, uses ``ParallelMLIPPredictUnit`` which does not
    expose ``predictor.model``.  Analytical Hessians are unavailable in
    that mode.
    """

    def __init__(
        self,
        elem: Sequence[str],
        *,
        charge: int = 0,
        spin: int = 1,
        model: str = "uma-s-1p1",
        task_name: str = "omol",
        device: str = "auto",
        workers: int = 1,
        workers_per_node: int = 1,
        max_neigh: Optional[int] = None,
        radius:    Optional[float] = None,
        r_edges:   bool = False,
        weights_file: Optional[str] = None,
    ):
        # Select device ------------------------------------------------
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.device = torch.device(device)

        self.workers = max(int(workers or 1), 1)
        self.workers_per_node = max(int(workers_per_node or 1), 1)
        self.parallel_predict = self.workers > 1

        self._AtomicData = AtomicData
        self._collater   = data_list_collater
        self.model_name = model
        self.weights_file = None
        if weights_file is not None:
            self.weights_file = os.path.abspath(os.path.expanduser(os.fspath(weights_file)))
            if not os.path.isfile(self.weights_file):
                raise FileNotFoundError(f"Weights file does not exist: {self.weights_file}")
        self._hessian_ready = False

        # Predictor ----------------------------------------------------
        self.predict = self._load_predictor()

        self.has_torch_model = hasattr(self.predict, "model") and isinstance(
            getattr(self.predict, "model", None), nn.Module
        )
        if self.has_torch_model:
            self.predict.model.eval()
            for m in self.predict.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0

        self.elem      = [e.capitalize() for e in elem]
        self.charge    = charge
        self.spin      = spin
        self.task_name = task_name

        self._max_neigh = max_neigh
        self._radius    = radius
        self._r_edges   = r_edges

    def _load_predictor(self, inference_settings="default"):
        if self.parallel_predict:
            if (ParallelMLIPPredictUnit is None) or (guess_inference_settings is None):
                raise ImportError(
                    "workers>1 requested, but ParallelMLIPPredictUnit/guess_inference_settings "
                    "could not be imported from fairchem. Please ensure your FAIR-Chem installation "
                    "includes `fairchem-core[ray]`."
                )
            ckpt_path = self.weights_file
            atom_refs = None
            form_elem_refs = None
            if ckpt_path is None:
                ckpt_path = pretrained_mlip.pretrained_checkpoint_path_from_name(self.model_name)
                atom_refs = pretrained_mlip.get_reference_energies(self.model_name, reference_type="atom_refs")
                try:
                    form_elem_refs = pretrained_mlip.get_reference_energies(
                        self.model_name, reference_type="form_elem_refs",
                    )["refs"]
                except (TypeError, KeyError):
                    form_elem_refs = None
            return ParallelMLIPPredictUnit(
                inference_model_path=str(ckpt_path),
                device=self.device_str,
                inference_settings=guess_inference_settings(inference_settings),
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
                num_workers=self.workers,
                num_workers_per_node=self.workers_per_node,
            )
        if self.weights_file is not None:
            from fairchem.core.units.mlip_unit import load_predict_unit

            return load_predict_unit(
                self.weights_file, device=self.device_str, workers=self.workers,
                inference_settings=inference_settings,
            )
        return pretrained_mlip.get_predict_unit(
            self.model_name, device=self.device_str, workers=self.workers,
            inference_settings=inference_settings,
        )

    # ----------------------------------------------------------------
    def _model_backbone(self):
        if not self.has_torch_model:
            return None
        mdl = self.predict.model
        mod = getattr(mdl, "module", mdl)
        return getattr(mod, "backbone", None)

    # ----------------------------------------------------------------
    def _ase_to_batch(self, atoms: Atoms):
        """Convert ASE Atoms → UMA AtomicData(Batch)."""

        backbone = self._model_backbone()
        default_max_neigh = getattr(backbone, "max_neighbors", None) if backbone is not None else None
        default_radius    = getattr(backbone, "cutoff", None)        if backbone is not None else None
        if default_radius is None:
            default_radius = 6.0

        max_neigh = self._max_neigh if self._max_neigh is not None else default_max_neigh
        radius    = self._radius    if self._radius    is not None else default_radius
        r_edges   = self._r_edges

        atoms.info.update({"charge": self.charge, "spin": self.spin})
        data = self._AtomicData.from_ase(
            atoms,
            r_data_keys=["spin", "charge"],
            max_neigh=max_neigh,
            radius   =radius,
            r_edges  =r_edges,
        )
        data.dataset = self.task_name
        batch = self._collater([data], otf_graph=True)
        # FAIR-Chem keeps its model on CPU until the first prediction initializes it.
        if not self.parallel_predict and getattr(self.predict, "lazy_model_intialized", True):
            batch = batch.to(self.device)
        return batch

    # ----------------------------------------------------------------
    def compute(
        self,
        coord_ang: np.ndarray,
        *,
        forces: bool = False,
        hessian: bool = False,
    ) -> Dict[str, Any]:
        """
        coord_ang : (N,3) Å
        forces / hessian : return toggles
        Returns dict with keys energy (eV), forces (eV/Å), hessian (torch)
        """
        if hessian and not self.parallel_predict and not self._hessian_ready:
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

            settings = InferenceSettings(
                activation_checkpointing=False, merge_mole=False, compile=False,
            )
            if hasattr(settings, "execution_mode"):
                settings.execution_mode = "general"
            self.predict = self._load_predictor(settings)
            self.has_torch_model = isinstance(getattr(self.predict, "model", None), nn.Module)
            if hasattr(self.predict, "move_to_device"):
                self.predict.move_to_device()
            if self.has_torch_model:
                for module in self.predict.model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
            self._hessian_ready = True

        atoms = Atoms(self.elem, positions=coord_ang)
        batch = self._ase_to_batch(atoms)

        if self.parallel_predict or (not self.has_torch_model):
            if hessian:
                raise RuntimeError(
                    "Analytical Hessian is not available when predictor workers > 1 "
                    "or when predictor.model is not exposed. Use FiniteDifference Hessian."
                )
            res = self.predict.predict(batch)
            energy = float(res["energy"].squeeze().detach().item())
            forces_np = res["forces"].detach().cpu().numpy() if forces else None
            return {"energy": energy, "forces": forces_np, "hessian": None}

        batch.pos.requires_grad_(True)
        res      = self.predict.predict(batch)
        energy   = float(res["energy"].squeeze().detach().item())
        forces_np = res["forces"].detach().cpu().numpy() if (forces or hessian) else None

        if hessian:
            p_flags = [p.requires_grad for p in self.predict.model.parameters()]
            for p in self.predict.model.parameters():
                p.requires_grad_(False)
            self.predict.model.train()
            try:
                def e_fn(flat):
                    batch.pos = flat.view(-1, 3)
                    return self.predict.predict(batch)["energy"].squeeze()
                H = torch.autograd.functional.hessian(e_fn, batch.pos.view(-1), vectorize=False)
                H = H.view(len(atoms), 3, len(atoms), 3).detach()
            finally:
                self.predict.model.eval()
                for p, flag in zip(self.predict.model.parameters(), p_flags):
                    p.requires_grad_(flag)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            H = None

        return {"energy": energy, "forces": forces_np, "hessian": H}


# ===================================================================
#                    PySisyphus calculator class
# ===================================================================
class uma_pysis(Calculator):
    """PySisyphus-compatible UMA calculator.

    Supports Analytical and FiniteDifference Hessian modes,
    multi-worker inference, and xTB implicit-solvent correction
    (when wrapped via ``SolventCorrectedCalculator``).
    """

    implemented_properties = ["energy", "forces", "hessian"]

    def __init__(
        self,
        *,
        charge: int = 0,
        spin: int = 1,
        model: str = "uma-s-1p1",
        task_name: str = "omol",
        device: str = "auto",
        workers: int = 1,
        workers_per_node: int = 1,
        hessian_calc_mode: str = "Analytical",
        hessian_double: bool = True,
        out_hess_torch: bool = False,
        print_timing: bool = False,
        max_neigh: Optional[int] = None,
        radius:    Optional[float] = None,
        r_edges:   bool = False,
        weights_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(charge=charge, mult=spin, **kwargs)
        self._core: Optional[UMAcore] = None
        self._core_kw = dict(
            charge=charge,
            spin=spin,
            model=model,
            weights_file=weights_file,
            task_name=task_name,
            device=device,
            workers=workers,
            workers_per_node=workers_per_node,
            max_neigh=max_neigh,
            radius=radius,
            r_edges=r_edges,
        )
        self.hessian_calc_mode = hessian_calc_mode
        self.hessian_double = bool(hessian_double)
        self.out_hess_torch = out_hess_torch
        self.print_timing = bool(print_timing)

    # ---------- helpers ---------------------------------------------
    def _ensure_core(self, elem: Sequence[str]):
        if self._core is None:
            self._core = UMAcore(elem, **self._core_kw)

    @staticmethod
    def _au_energy(e_eV: float) -> float:
        return e_eV * EV2AU

    @staticmethod
    def _au_forces(f_eV_A: np.ndarray) -> np.ndarray:
        return (f_eV_A * F_EVAA_2_AU).reshape(-1)

    def _au_hessian(self, H_eV_AA: torch.Tensor):
        """Convert Hessian from eV/Å² to Hartree/Bohr² (torch version)."""
        n = H_eV_AA.size(0)
        H = H_eV_AA.view(n * 3, n * 3)
        _t = H.T.clone()
        H = 0.5 * (H + _t)
        del _t
        H = H * H_EVAA_2_AU
        if self.hessian_double:
            H = H.to(dtype=torch.float64)
        if self.out_hess_torch:
            return H.detach()
        else:
            return H.detach().cpu().numpy()

    # ---------- GPU Finite-Difference Hessian -----------------------
    def _build_fd_hessian_gpu(
        self,
        elem: Sequence[str],
        coord_ang: np.ndarray,
        *,
        eps_ang: float = 1.0e-3,
    ) -> Dict[str, Any]:
        """Assemble central-difference Hessian on GPU."""
        self._ensure_core(elem)
        core = self._core
        assert core is not None
        dev = core.device

        n_atoms = len(elem)
        dof = n_atoms * 3

        res0 = core.compute(coord_ang, forces=True, hessian=False)
        energy0_eV = res0["energy"]
        F0 = res0["forces"]

        force_dtype = torch.from_numpy(F0).dtype
        hessian_dtype = torch.float64 if self.hessian_double else force_dtype
        H = torch.zeros((dof, dof), device=dev, dtype=hessian_dtype)

        coord_plus = coord_ang.copy()
        coord_minus = coord_ang.copy()

        for k in range(dof):
            a = k // 3
            c = k % 3
            coord_plus[a, c] = coord_ang[a, c] + eps_ang
            res_p = core.compute(coord_plus, forces=True, hessian=False)
            Fp = res_p["forces"].reshape(-1)

            coord_minus[a, c] = coord_ang[a, c] - eps_ang
            res_m = core.compute(coord_minus, forces=True, hessian=False)
            Fm = res_m["forces"].reshape(-1)

            Fp_t = torch.from_numpy(Fp).to(dev, dtype=hessian_dtype)
            Fm_t = torch.from_numpy(Fm).to(dev, dtype=hessian_dtype)
            col = -(Fp_t - Fm_t) / (2.0 * eps_ang)
            H[:, k] = col

            coord_plus[a, c] = coord_ang[a, c]
            coord_minus[a, c] = coord_ang[a, c]

        H = H.view(n_atoms, 3, n_atoms, 3)
        return {"energy": energy0_eV, "forces": F0, "hessian": H}

    # ---------- PySisyphus API --------------------------------------
    def get_energy(self, elem, coords):
        self._ensure_core(elem)
        coord_ang = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * BOHR2ANG
        res = self._core.compute(coord_ang)
        return {"energy": self._au_energy(res["energy"])}

    def get_forces(self, elem, coords):
        self._ensure_core(elem)
        coord_ang = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * BOHR2ANG
        res = self._core.compute(coord_ang, forces=True)
        return {
            "energy": self._au_energy(res["energy"]),
            "forces": self._au_forces(res["forces"]),
        }

    def get_hessian(self, elem, coords):
        self._ensure_core(elem)
        coord_ang = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * BOHR2ANG

        core = self._core
        assert core is not None
        force_fd = (core.parallel_predict or (not core.has_torch_model))

        hess_total_start = time.perf_counter()
        mode_elapsed_s = 0.0
        mode_label = "FiniteDifference"

        mode = (self.hessian_calc_mode or "Analytical").strip().lower()
        if (not force_fd) and (mode in ("analytical", "analytic")):
            mode_label = "Analytical"
            t0 = time.perf_counter()
            try:
                res = self._core.compute(coord_ang, forces=True, hessian=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e).lower()
                if "out of memory" in msg and "cuda" in msg:
                    raise RuntimeError(
                        "Analytical Hessian computation failed due to CUDA out-of-memory. "
                        "Please switch to `hessian_calc_mode: FiniteDifference`."
                    ) from e
                raise
            mode_elapsed_s = time.perf_counter() - t0
            out = {
                "energy": self._au_energy(res["energy"]),
                "forces": self._au_forces(res["forces"]),
                "hessian": self._au_hessian(res["hessian"]),
            }
        else:
            t0 = time.perf_counter()
            res = self._build_fd_hessian_gpu(elem, coord_ang)
            mode_elapsed_s = time.perf_counter() - t0
            out = {
                "energy": self._au_energy(res["energy"]),
                "forces": self._au_forces(res["forces"]),
                "hessian": self._au_hessian(res["hessian"]),
            }

        if self.print_timing:
            total_elapsed_s = time.perf_counter() - hess_total_start
            print(
                f"[HessianTiming] mode: {mode_label} | "
                f"elapsed: {mode_elapsed_s:.2f} s | total: {total_elapsed_s:.2f} s"
            )
        return out


# ---------- CLI / YAML factory ------------------------------------
def _uma_pysis_factory(**kwargs):
    """Factory for YAML usage. Extracts solvent keys and wraps if needed."""
    solvent = kwargs.pop("solvent", "none")
    solvent_model = kwargs.pop("solvent_model", "alpb")
    xtb_cmd = kwargs.pop("xtb_cmd", "xtb")
    xtb_acc = kwargs.pop("xtb_acc", 0.2)

    calc = uma_pysis(**kwargs)

    from .solvent import SolventCorrectedCalculator, solvent_correction_enabled
    if solvent_correction_enabled(solvent):
        calc = SolventCorrectedCalculator(
            calc,
            solvent=solvent,
            solvent_model=solvent_model,
            xtb_cmd=xtb_cmd,
            xtb_acc=xtb_acc,
        )
    return calc


def run_pysis():
    """Enable `uma_pysis input.yaml`"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-w", "--weights-file", default=None, help="Path to downloaded UMA weights.")
    args, remaining = parser.parse_known_args()
    weights_file = args.weights_file
    if weights_file is not None:
        weights_file = os.path.abspath(os.path.expanduser(weights_file))
        if not os.path.isfile(weights_file):
            parser.error(f"Weights file does not exist: {weights_file}")

    def factory(*factory_args, **kwargs):
        if weights_file is not None:
            configured = kwargs.get("weights_file")
            if configured is not None and os.path.abspath(os.path.expanduser(configured)) != weights_file:
                raise ValueError("--weights-file conflicts with calc.weights_file.")
            kwargs["weights_file"] = weights_file
        return _uma_pysis_factory(*factory_args, **kwargs)

    run.CALC_DICT["uma_pysis"] = factory
    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        run.run()
    except SystemExit as exc:
        if exc.code == 0 and any(flag in remaining for flag in ("-h", "--help")):
            parser.print_help()
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    run_pysis()
