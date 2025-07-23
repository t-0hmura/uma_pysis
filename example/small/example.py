from uma_pysis import uma_pysis
from pysisyphus.io.xyz import geom_from_xyz

geom = geom_from_xyz('reac.xyz')
calc = uma_pysis(charge=0, spin=1, model="uma-s-1p1", task_name="omol", device="auto")

geom.set_calculator(calc)

E = geom.energy     # Hartree
F = geom.forces     # Hartree·Bohr⁻¹
H = geom.hessian    # (3N × 3N) analytic Hessian

print(f'Energy: {E:.6f} Hartree')
print(f'Max Force : {F.max():.6f} Hartree·Bohr⁻¹')
print(f"Hessian shape: {H.shape}")