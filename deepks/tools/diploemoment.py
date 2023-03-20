"""
DSCF will give the dipole and force
"""
import argparse
import os
import time
import numpy as np
from numpy.linalg import norm
from deepks.utils import load_yaml
from deepks.scf.scf import DSCF
from pyscf import gto, lib
from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.hessian import thermo
from pyscf.geomopt import berny_solver,geometric_solver, as_pyscf_method

def readxyz(xyz):
    T=open(xyz)
    G=T.readlines()[2:]
    return "".join(G)

#print(readxyz("NH3.xyz"))
#T=readxyz("NH3.xyz")

parser = argparse.ArgumentParser(description="Calculate dipole moment for given xyz files.")
parser.add_argument("files", help="input xyz files")
parser.add_argument("-m","--model-file", help="file of trained model")
parser.add_argument("-v","--verbose", default=1, type=int, help="output calculation information")
parser.add_argument("-B", "--basis", default="def2-tzvp", type=str, help="basis used to do the calculation")

args = parser.parse_args()
files = args.files

xyz=readxyz(files)

mol = gto.M(atom=xyz,basis="def2-tzvp",verbose=args.verbose,parse_arg=False)

"""
mol1 = Mole()
mol1.atom=xyz
mol1.basis = 'def2-tzvp'
mol1.build()
HF=RHF(mol1).run()
print('')
print('HF dipole moment:')
hfdip = HF.dip_moment()
print('Absolute value: {0:.3f} Debye'.format(norm(hfdip)))
"""

model = args.model_file
deepks = DSCF(mol,model).run()

print(dir(deepks))
print('')
print('DeePKS dipole moment:')
dpksdip = deepks.dip_moment(verbose=5)   #
print('Absolute value: {0:.3f} Debye'.format(norm(dpksdip)))
print(mol.elements)
print("Charge",deepks.mulliken_pop(verbose=0)[1].tolist())

mo_energy=deepks.mo_energy
nocc=deepks.mol.nelectron//2
HOMO=mo_energy[nocc-1]
LUMO=mo_energy[nocc]
print("HOMO,LUMO,LUMO-HOMO GAP",HOMO,LUMO,LUMO-HOMO)

hessian=deepks.Hessian().kernel()
freq_info = thermo.harmonic_analysis(deepks.mol, hessian)
thermo_info = thermo.thermo(deepks, freq_info['freq_au'], 298.15, 101325)

print('Rotation constant')
print(thermo_info['rot_const'])

print('Zero-point energy')
print(thermo_info['ZPE'   ])

print('Internal energy at 0 K')
print(thermo_info['E_0K'  ])

print('Internal energy at 298.15 K')
print(thermo_info['E_tot' ])

print('Enthalpy energy at 298.15 K')
print(thermo_info['H_tot' ])

print('Gibbs free energy at 298.15 K')
print(thermo_info['G_tot' ])

print('Heat capacity at 298.15 K')
print(thermo_info['Cv_tot'])
