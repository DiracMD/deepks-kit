import numpy as np
import os
from glob import glob
#from multiprocessing import Pool
import os
import re

ELEMENTS = ['X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
CHARGES = dict(((str(i),x) for i,x in enumerate(ELEMENTS)))


def ChooseTest(atom,i):
    ChoseTest=np.load(atom)
    for j in range(len(ChoseTest)):
        mol1=np.array(["%f" %w for w in ChoseTest[j].reshape(ChoseTest[j].size)])
        mol2=mol1.reshape(ChoseTest[j].shape)
        for h in range(len(mol2[:,0])):
            mol2[:,0][h]=CHARGES[str(int(float(mol2[:,0][h])))]
        L=mol2.tolist()
        A=[" ".join(g) for g in L]
        xyz1=["**** ccpvdz mole"+str(j),"memory,100,m","geometry={"]
        xyz5=["}","basis=cc-pVDZ","hf","ccsd(t)"]
        input=xyz1+A+xyz5
        f=open("ChoseT"+str(i)+"/"+str(j)+"atom.xyz","w")
        f.write("\n".join(input))
        f.close()

pattern=re.compile(r"\-\d+\.?\d*")
def extract_E(i):
    with open(i) as f:
        for h in f:
            if "CCSD(T)/cc-pVDZ energy" in h:
                return float(pattern.findall(h)[0])

def energy(i):
    energy_list=os.listdir("./ChoseT"+str(i)+"/")
    ene=[]
    ene_arr=[]
    for e in energy_list:
        if os.path.splitext(e)[1]==".out":
            ene.append("./ChoseT"+str(i)+"/"+e)
    for j in ene:
        ene_arr.append([extract_E(j)])
    return np.array(ene_arr)

def testout(i):
    A=glob(r"iter."+str(i)+"/model.*/iter.00/01.train/")
    E=[]
    for path in A:
        os.system("deepks scf scf_input.yaml -m "+path+"/model.pth -s Tsys"+str(i)+" -d "+path+"test0")
        A=len(np.load(path+"test0/Tsys"+str(i)+"/dm_eig.npy"))
        np.save(path+"test0/Tsys"+str(i)+"/l_e_delta.npy",np.zeros((A,1)))
        os.system("deepks test -m "+path+"/model.pth -o test1/test -d "+path+"test0/Tsys"+str(i)+" -D dm_eig ")
        G=[np.loadtxt(T)[:,1] for T in glob(path+"test1/test.00.out")]
        E.append(G)
    return np.stack(E,-1)

def work(xyz):
    os.system("molpro "+xyz+" -n 16")
def pol(i):
    inp=[]
    f_list=os.listdir("./ChoseT"+str(i)+"/")
    for f in f_list:
        if os.path.splitext(f)[1]==".xyz":
            inp.append("./ChoseT"+str(i)+"/"+f)
    for j in inp:
        work(j)

def xyz(nsel,i):
    TRatom=np.load("Rsys"+str(i)+"/atom.npy")
    TRenergy=np.load("Rsys"+str(i)+"/energy.npy")
    TEatom=np.load("Tsys"+str(i)+"/atom.npy")
    TEenergy=np.load("Tsys"+str(i)+"/energy.npy")
    tst_res=testout(i)
    tst_std=np.std(tst_res,axis=-1)
    order=np.argsort(tst_std)[::-1]
    sel=order[0][:nsel]
    rst=np.sort(order[0][nsel:])
    if not os.path.exists("ChoseT"+str(i)):
        os.mkdir("ChoseT"+str(i))
    np.save("ChoseT"+str(i)+"/atom.npy",TEatom[sel])
    ChooseTest("ChoseT"+str(i)+"/atom.npy",i)
    pol(i)
    sel_arr=energy(i)
    New_trn_atom=np.concatenate([TRatom,TEatom[sel]])
    New_trn_energy=np.concatenate([TRenergy,sel_arr])
    New_tst_atom=TEatom[rst]
    #New_tst_energy=TEenergy[rst]
    if not os.path.exists("Rsys"+str(i+1)):
        os.mkdir("Rsys"+str(i+1))
    if not os.path.exists("Tsys"+str(i+1)):
        os.mkdir("Tsys"+str(i+1))
    np.save("Rsys"+str(i+1)+"/atom.npy",New_trn_atom)
    np.save("Rsys"+str(i+1)+"/energy.npy",New_trn_energy)
    np.save("Tsys"+str(i+1)+"/atom.npy",New_tst_atom)
    #np.save("Tsy"+str(i+1)+"/energy.npy",New_tst_energy)

def Iter(nsel,n):
    for i in range(n):
        for j in range(2):
            os.system("mkdir -p iter."+str(i)+"/model.0"+str(j)+
            " && cd iter."+str(i)+"/model.0"+str(j)+
            " && cp -r ../../args.yaml ./ && cp -r ../../Rsys"+str(i)+" ./" +
            " && cp -r ../../Tsys"+str(i)+" ./ && deepks iterate args.yaml -s Rsys"+str(i)+" -t Rsys"+str(i))
        xyz(nsel,i)
Iter(5,2)
