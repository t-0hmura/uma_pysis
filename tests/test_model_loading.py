from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    for name in list(sys.modules):
        if name == "uma_pysis" or name.startswith("uma_pysis."):
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("uma_pysis.uma_pysis")


@pytest.fixture
def weights(tmp_path):
    filename = tmp_path / "local weights.pt"
    filename.write_bytes(b"offline fixture")
    return filename


@pytest.fixture
def loader(monkeypatch, module):
    import fairchem.core.units.mlip_unit as unit
    calls = []
    def load(path, **kwargs):
        predict = SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            move_to_device=lambda: None,
            lazy_model_intialized=False,
        )
        def evaluate(batch):
            if not predict.lazy_model_intialized:
                assert batch.pos.device.type == "cpu"
                predict.lazy_model_intialized = True
            return {"energy": batch.pos.square().sum().reshape(1),
                    "forces": -2 * batch.pos}
        predict.predict = evaluate
        calls.append((str(path), kwargs, predict))
        return predict
    def download(*a, **k):
        raise AssertionError("Offline weights must not download model files or references")
    monkeypatch.setattr(unit, "load_predict_unit", load)
    monkeypatch.setattr(module.pretrained_mlip, "get_predict_unit", download)
    monkeypatch.setattr(module.pretrained_mlip, "get_reference_energies", download)
    monkeypatch.setattr(module.pretrained_mlip, "pretrained_checkpoint_path_from_name", download)
    return calls


def test_core_offline_energy_forces_hessian(module, weights, loader):
    core = module.UMAcore(["H", "H"], device="cpu", weights_file=str(weights))
    coords = np.array([[0., 0., 0.], [1., 0., 0.]])
    result = core.compute(coords, forces=True)
    assert result["energy"] == 1
    np.testing.assert_array_equal(result["forces"], -2 * coords)
    result = core.compute(coords, forces=True, hessian=True)
    np.testing.assert_array_equal(result["hessian"].reshape(6, 6), 2 * np.eye(6))
    assert all(path == str(weights) for path, _, _ in loader)
    settings = loader[-1][1]["inference_settings"]
    assert settings.compile is False
    assert settings.activation_checkpointing is False


def test_python_api_offline_weights(module, weights, loader):
    calc = module.uma_pysis(device="cpu", weights_file=str(weights))
    coords = np.array([[0., 0., 0.], [1., 0., 0.]]).reshape(-1)
    result = calc.get_energy(["H", "H"], coords)
    assert np.isfinite(result["energy"])
    assert loader[0][0] == str(weights)


@pytest.mark.parametrize("flag", ["-w", "--weights-file"])
def test_cli_forwarding(module, weights, monkeypatch, flag):
    captured = []
    monkeypatch.setattr(module, "_uma_pysis_factory", lambda **kwargs: captured.append(kwargs))
    def run():
        assert sys.argv == ["uma_pysis", "input.yaml", "--restart"]
        module.run.CALC_DICT["uma_pysis"](charge=0)
    monkeypatch.setattr(module.run, "run", run)
    monkeypatch.setattr(sys, "argv", ["uma_pysis", flag, str(weights), "input.yaml", "--restart"])
    original = sys.argv
    module.run_pysis()
    assert sys.argv is original
    assert captured == [{"charge": 0, "weights_file": str(weights)}]


def test_cli_yaml_conflict(module, weights, monkeypatch, tmp_path):
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    monkeypatch.setattr(sys, "argv", ["uma_pysis", "-w", str(weights), "input.yaml"])
    monkeypatch.setattr(module.run, "run",
                        lambda: module.run.CALC_DICT["uma_pysis"](weights_file=str(other)))
    with pytest.raises(ValueError, match="conflicts"):
        module.run_pysis()


def test_missing_weights(module, loader, tmp_path):
    with pytest.raises(FileNotFoundError, match="Weights file does not exist"):
        module.UMAcore(["H", "H"], device="cpu", weights_file=str(tmp_path / "missing.pt"))
    assert loader == []


def test_initial_batch_stays_on_cpu(module, loader, weights):
    core = module.UMAcore(["H", "H"], device="cpu", weights_file=str(weights))
    core.device = torch.device("cuda")
    atoms = module.Atoms(["H", "H"], positions=[[0., 0., 0.], [1., 0., 0.]])
    batch = core._ase_to_batch(atoms)
    assert batch.pos.device.type == "cpu"


@pytest.mark.parametrize("charge,spin", [(0, 1), (-1, 2)])
def test_batch_preserves_charge_and_spin(module, loader, weights, charge, spin):
    core = module.UMAcore(["H", "H"], device="cpu", charge=charge, spin=spin,
                          weights_file=str(weights))
    atoms = module.Atoms(["H", "H"], positions=[[0., 0., 0.], [1., 0., 0.]])
    batch = core._ase_to_batch(atoms)
    assert batch.charge.item() == charge
    assert batch.spin.item() == spin
