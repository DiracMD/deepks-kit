import numpy as np
import os
from glob import glob

def select(nsel,i):
    A=glob(r"iter."+str(i)+"/model.*/iter.00/01.train/")
    TRatom=np.load("Rsys"+str(i)+"/atom.npy")
    TRenergy=np.load("Rsys"+str(i)+"/energy.npy")

    TEatom=np.load("Tsys"+str(i)+"/atom.npy")
    TEenergy=np.load("Tsys"+str(i)+"/energy.npy")
    E=[]
    for path in A:
        os.system("deepqc scf scf_input.yaml -m "+path+"/model.pth -s Tsys"+str(i)+" -d "+path+"test0")
        A=len(np.load(path+"test0/Tsys"+str(i)+"/dm_eig.npy"))
        np.save(path+"test0/Tsys"+str(i)+"/l_e_delta.npy",np.zeros((A,1)))
        np.save(path+"test0/Tsys"+str(i)+"/conv.npy",np.array([[True]]*A))
        os.system("deepqc test -m "+path+"/model.pth -o test1/test -d "+path+"test0/Tsys"+str(i)+" -D dm_eig ")
        A=[np.loadtxt(T)[:,1] for T in glob(path+"test1/test.00.out")]
        E.append(A)
    tst_res=np.stack(E,-1)
    tst_std=np.std(tst_res,axis=-1)
    print(tst_std)
    order =np.argsort(tst_std)[::-1]
    print(order)
    sel=order[0][:nsel]
    print(sel)
    rst=np.sort(order[0][nsel:])
    print(rst)
    New_trn_atom=np.concatenate([TRatom,TEatom[sel]])
    New_trn_energy=np.concatenate([TRenergy,TEenergy[sel]])
    New_tst_atom=TEatom[rst]
    New_tst_energy=TEenergy[rst]
    os.mkdir("Rsys"+str(i+1))
    os.mkdir("Tsys"+str(i+1))
    np.save("Rsys"+str(i+1)+"/atom.npy",New_trn_atom)
    np.save("Rsys"+str(i+1)+"/energy.npy",New_trn_energy)
    np.save("Tsys"+str(i+1)+"/atom.npy",New_tst_atom)
    np.save("Tsys"+str(i+1)+"/energy.npy",New_tst_energy)

def Iter(n,nsel):
    for i in range(n):
        for j in range(2):
            os.system("mkdir -p iter."+str(i)+"/model.0"+str(j)+
            " && cd iter."+str(i)+"/model.0"+str(j)+
            " && cp -r ../../args.yaml ./ && cp -r ../../Rsys"+str(i)+" ./" +
            " && cp -r ../../Tsys"+str(i)+" ./ && deepqc iterate args.yaml -s Rsys"+str(i))
        select(nsel,i)
Iter(2,5)
