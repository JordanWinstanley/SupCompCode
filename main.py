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
        if len(df['posx']) > 10_000_000:
            df = df[df.index > 10_000_000]
        
        timinglist = np.append(timinglist, time)
        avgposdf, avgveldf = af.COMfind(df, indexdf)
        CircComdf, CircVeldf = af.shrinkingcircmethod(df, avgposdf)
        r = af.radiusprep()
        df2 = af.calcs(df, r, CircComdf, CircVeldf)
        Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        i = int(i)
        fn = str(i)
        if len(fn) == 1:
            fn = "00" + fn
        if len(fn) == 2:
            fn = "0" + fn
        
        if i % 32 == 0:
            #af.poscircleplot(df,i,fp,CircComdf,condict,fn,time)
            af.position(df,fn,fp,i,time)
            af.phase(df,fn,fp,i,time)
            af.phaseCOM(df,fn,fp,i,time)
            af.density(df2,fn,fp,i,time)
            af.mass(df2,fn,fp,i,time)
            af.avgvelplot(df2,fn,fp,i,time)
            af.avgvelplot2(df2,fn,fp,i,time)
            af.densitywithhalo(df2,fn,fp,i,time)
            af.masswithhalo(df2,fn,fp,i,time)
            af.sigmavrplot(df2,fn,fp,i,time)
            af.sigmatotplot(df2,fn,fp,i,time)
            af.NpartCOM(df,fn,fp,i,time)

        inr200 = af.inr200func(df,CircComdf,inr200,fp)
        #condict = af.maxraddict(df,condict,CircComdf)
        #condict2 = af.maxraddict(df,condict2,CircComdf)
        #mdndict = af.medavgcalc(df,mdndict,fp)

        del df
        del df2

    inr200 = COMM.gather(np.array(inr200), root=0)
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