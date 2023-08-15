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
COMM = MPI.COMM_WORLD

global M200; global beta
fpexists = True
simloc = "/1r200/ExtraSims/nocut/1e12/Plunging/b0/Analytical"
M200 = 1e12
beta = 0

import AllFunc as af
def main():
    fpm = "../.." + simloc
    if COMM.rank == 0:
        snapshotlst = af.getsnapshots(fpm,fpexists)
        plotfilepath,pltfiles = af.findplotdirec(fpm,fpexists)
        if fpexists == True:
            pltfiles = simloc
        print(plotfilepath)
        print(pltfiles)
        plotfp = af.prepareplotfile(plotfilepath, pltfiles,fpexists)
        indexdf,Ntot = af.findmiddleparts(snapshotlst,fpm,fpexists)
        k = len(snapshotlst)
        snapshotlst = np.array(list(enumerate(snapshotlst)))
        snapsplt = np.array_split(snapshotlst,COMM.size)
    else:
        snapshotlst = None
        snapsplt = None 
        indexdf = None
        k = None
        plotfp = None
        Ntot = None

    k = COMM.bcast(k, root=0)
    indexdf = COMM.bcast(indexdf, root=0)
    splt = COMM.scatter(snapsplt)
    fp = COMM.bcast(plotfp,root=0)
    Ntot = COMM.bcast(Ntot, root=0)
    COMM.Barrier()

    Comlist = np.array([]).reshape(0,3)
    inr200 = []; timinglist = np.empty(shape=(0,0))
    in2r200 = [];
    for i, fname in splt:
        df, time = af.datainitializing(fname,fpm,fpexists)
        if len(df['posx']) > 10_000_000:
            df = af.recenterdf(df)
            fulldf = df[df.index <= 10_000_000]
            df = df[df.index > 10_000_000]
            Live = True
        else:
            fulldf = None
            Live = False
        timinglist = np.append(timinglist, time)
        #CircComdf, CircVeldf = af.COMfind(df, indexdf)
        #CircComdf, CircVeldf = af.shrinkingcircmethod(df, CircComdf)
        CircComdf, CircVeldf = af.findcenterhist(df)
        r = af.radiusprep()
        df, df2 = af.calcs(df, r, CircComdf, CircVeldf)
        Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        i = int(i)
        fn = str(i)
        if len(fn) == 1:
            fn = "00" + fn
        if len(fn) == 2:
            fn = "0" + fn
        
        if i % 32 == 0:
            #af.poscircleplot(df,i,fp,CircComdf,condict,fn,time)
            af.position(df,fn,fp,i,time,CircComdf)
            if Live == True:
                af.liveposition(df,fulldf,fn,fp,i,time,CircComdf)
            af.velhist(df,fn,fp,i,time)
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

        inr200 = af.inr200func(df,inr200)
        in2r200 = af.in2r200func(df,in2r200)
        #condict = af.maxraddict(df,condict,CircComdf)
        #condict2 = af.maxraddict(df,condict2,CircComdf)
        #mdndict = af.medavgcalc(df,mdndict,fp)

        del df
        del df2

    inr200 = COMM.gather(np.array(inr200), root=0)
    in2r200 = COMM.gather(np.array(in2r200), root=0)
    Comlist = COMM.gather(Comlist, root=0)
    timinglist = COMM.gather(timinglist, root=0)
    COMM.Barrier()
    
    if COMM.rank == 0:
        inr200 = np.concatenate(inr200)
        in2r200 = np.concatenate(in2r200)
        Comlist = np.concatenate(Comlist)
        timinglist = np.concatenate(timinglist)
        af.inr200plot(inr200, fp, timinglist,Ntot)
        af.in2r200plot(inr200,in2r200, fp, timinglist,Ntot)
        af.COMplot(Comlist, fp, timinglist)
        af.COMradplot(Comlist, fp, timinglist)
        af.COMtestplot(Comlist, fp, timinglist)

    print(f"Rank {COMM.rank} Finished")
    COMM.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()