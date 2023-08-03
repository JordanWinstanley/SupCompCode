from mpi4py import MPI
import h5py
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
test = False
backstr = ""
x = os.listdir(os.getcwd())
while test == False:
    if "SupCompCode" in x:
        test = True
    else:
        backstr = backstr + "../"
        x = os.listdir(backstr)
x = backstr + "SupCompCode"
sys.path.insert(0,x)
import parameters as par
import AllFunc as af


def main():
    global COMM
    COMM = MPI.COMM_WORLD
    if COMM.rank == 0:
        snapshotlst = af.getsnapshots()
        plotfilepath,pltfiles = af.findplotdirec()
        plotfp = af.prepareplotfile(plotfilepath, pltfiles)
        indexdf = af.findmiddleparts(snapshotlst)
        k = len(snapshotlst)
        snapshotlst = np.array(list(enumerate(snapshotlst)))
        snapsplt = np.array_split(snapshotlst,COMM.size)
    else:
        snapshotlst = None
        snapsplt = None 
        indexdf = None
        k = None
        plotfp = None

    k = COMM.bcast(k, root=0)
    indexdf = COMM.bcast(indexdf, root=0)
    splt = COMM.scatter(snapsplt)
    fp = COMM.bcast(plotfp,root=0)
    COMM.Barrier()

    Comlist = np.array([]).reshape(0,3)
    inr200 = []; timinglist = np.empty(shape=(0,0))
    for i, fname in splt:
        df, time = af.datainitializing(fname)
        timinglist = np.append(timinglist, time)
        avgposdf, avgveldf = af.COMfind(df, indexdf)
        CircComdf, CircVeldf = af.shrinkingcircmethod(df, avgposdf)
        r = af.radiusprep()
        df2 = af.calcs(df, r, CircComdf, CircVeldf)

        Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        #Prep work
        boundlst = []; mdnlist = []; avglst = []
        x = [0,32,64,96,128,160,192,224,256]
        y = [0.05,0.10,0.25,0.5,0.75]
        condict2 = dict(); mdndict = dict()
        condict = {
        "50": [],
        "100": [],
        "200": [],
        "400": [],
        "800": [],
        "1600": [],
        "3200": [],
        "6400": []}
        i = int(i)
        """for num in y:
            key = int(round(num*N))
            condict2[str(key)] = []
            mdndict[str(key)] = [[],[]]"""

        fn = str(i)
        if len(fn) == 1:
            fn = "00" + fn
        if len(fn) == 2:
            fn = "0" + fn

        if i % 32 == 0:
            af.poscircleplot(df,i,fp,CircComdf,condict,fn,k)
            af.position(df,fn,fp,i,k)
            af.phase(df,fn,fp,i,k)
            af.phaseCOM(df,fn,fp,i,k)
            af.density(df2,fn,fp,i,k)
            af.mass(df2,fn,fp,i,k)
            af.avgvelplot(df2,fn,fp,i,k)
            af.avgvelplot2(df2,fn,fp,i,k)
            af.densitywithhalo(df2,fn,fp,i,k)
            af.masswithhalo(df2,fn,fp,i,k)
            af.sigmavrplot(df2,fn,fp,i,k)
            af.sigmatotplot(df2,fn,fp,i,k)
            af.NpartCOM(df,fn,fp,i,k)

        inr200 = af.inr200func(df,CircComdf,inr200,fp)
        condict = af.maxraddict(df,condict,CircComdf)
        condict2 = af.maxraddict(df,condict2,CircComdf)
        mdndict = af.medavgcalc(df,mdndict,fp)

        del df
        del df2

    inr200 = COMM.gather(np.array(inr200), root=0)
    #condict = COMM.gather(condict, root=0)
    #condict2 = COMM.gather(condict2, root=0)
    #mdndict = COMM.gather(mdndict, root=0)
    Comlist = COMM.gather(Comlist, root=0)
    timinglist = COMM.gather(timinglist, root=0)
    if COMM.rank == 0:
        inr200 = np.concatenate(inr200)
        Comlist = np.concatenate(Comlist)
        timinglist = np.concatenate(timinglist)
        af.inr200plot(inr200, fp, timinglist)
        af.COMplot(Comlist, fp, timinglist)
        af.COMradplot(Comlist, fp, timinglist)
        af.COMtestplot(Comlist, fp, timinglist)

    print(f"Rank {COMM.rank} Finished")
    COMM.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()