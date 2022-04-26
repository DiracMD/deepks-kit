"""
DSCF will give the energy and gradient 
"""

import time
import numpy as np
from deepks.utils import load_yaml
from deepks.scf.scf import DSCF
from pyscf import gto, lib
from pyscf.geomopt import berny_solver,geometric_solver, as_pyscf_method

try:
    from pyscf.geomopt.berny_solver import optimize
except ImportError:
    from pyscf.geomopt.geometric_solver import optimize

def readxyz(xyz):
    T=open(xyz)
    G=T.readlines()[2:]
    coor=[]
    for j in G:
        j1=j.split()[0]
        j2=j.split()[1:]
        j3=tuple([float(i) for i in j2])
        t=[j1,j3]
        coor.append(t)
    return coor

mol1 = gto.M(
    verbose = 5,
    atom = readxyz("atom.xyz"),
    basis = 'def2-tzvp'
)

def f(mol):
    cf = DSCF(mol,"model.pth")
    T=cf.nuc_grad_method().as_scanner()
    e,g =T(mol)
    return e,g

#fake_method = berny_solver.as_pyscf_method(mol1, f)

fake_method = geometric_solver.as_pyscf_method(mol1, f)

#new_mol = berny_solver.optimize(fake_method)
new_mol = geometric_solver.optimize(fake_method)

Bohr=0.52917721092
print('Old geometry (Ang)')
print(mol1.atom_coords()*Bohr)

print('New geometry (Ang)')
print(new_mol.atom_coords()*Bohr)

#cf = DSCF(mol,"model.pth")
#print(f(mol1))
#print(cf.kernel())
#print(cf.grads)
#print(optimize(cf))

