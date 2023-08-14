from mpi4py import MPI
import h5py
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


#fp1 = "/Users/jordan.winstanley/Library/CloudStorage/OneDrive-Personal/AAA - Uni/Project - Masters/Simulations/Original functions/1r200/0deg/nodisk/m1e12/Circular/b0/Analytical"
fp1 = "../../1r200/ExtraSims/nocut/1e12/Circular/b0/Analytical"
#fp2 = "/Users/jordan.winstanley/Library/CloudStorage/OneDrive-Personal/AAA - Uni/Project - Masters/Simulations/Original functions/1r200/0deg/nodisk/m1e12/Circular/b3/Analytical"
fp2 = "../../1r200/ExtraSims/nocut/1e12/Intermediate/b0/Analytical"
#fp3 = "/Users/jordan.winstanley/Library/CloudStorage/OneDrive-Personal/AAA - Uni/Project - Masters/Simulations/Original functions/1r200/0deg/nodisk/m1e12/Circular/b5/Analytical"
fp3 = "../../1r200/ExtraSims/nocut/1e12/Plunging/b0/Analytical"

plotfp = "../../1r200/Plots/ExtraPlots/"
label1 = "Circular"
label2 = "Intermediate"
label3 = "Plunging"

a2f = "1e12difforbit"

M1 = 1e12
M2 = 1e12
M3 = 1e12

title = "Mass: 1e12, Different Orbits"


COMM = MPI.COMM_WORLD

def main():
    if COMM.rank == 0:
        snapshotlst1 = getsnapshotsnew(fp1)
        snapshotlst2 = getsnapshotsnew(fp2)
        snapshotlst3 = getsnapshotsnew(fp3)
        indexdf1,Ntot1 = findmiddleparts(snapshotlst1,fp1)
        indexdf2,Ntot2 = findmiddleparts(snapshotlst2,fp2)
        indexdf3,Ntot3 = findmiddleparts(snapshotlst3,fp3)
        snapshotlst1 = np.array(list(enumerate(snapshotlst1)))
        snapshotlst2 = np.array(list(enumerate(snapshotlst2)))
        snapshotlst3 = np.array(list(enumerate(snapshotlst3)))
        snapsplt1 = np.array_split(snapshotlst1,COMM.size)
        snapsplt2 = np.array_split(snapshotlst2,COMM.size)
        snapsplt3 = np.array_split(snapshotlst3,COMM.size)
    else:
        snapshotlst1 = None
        snapsplt1 = None 
        indexdf1 = None
        Ntot1 = None
        snapshotlst2 = None
        snapsplt2 = None 
        indexdf2 = None
        Ntot2 = None
        snapshotlst3 = None
        snapsplt3 = None 
        indexdf3 = None
        Ntot3 = None
    

    
    indexdf1 = COMM.bcast(indexdf1, root=0)
    splt1 = COMM.scatter(snapsplt1)
    Ntot1 = COMM.bcast(Ntot1, root=0)

    indexdf2 = COMM.bcast(indexdf2, root=0)
    splt2 = COMM.scatter(snapsplt2)
    Ntot2 = COMM.bcast(Ntot2, root=0)

    indexdf3 = COMM.bcast(indexdf3, root=0)
    splt3 = COMM.scatter(snapsplt3)
    Ntot3 = COMM.bcast(Ntot3, root=0)


    Comlist1 = np.array([]).reshape(0,3)
    Comlist2 = np.array([]).reshape(0,3)
    Comlist3 = np.array([]).reshape(0,3)
    inr2001 = []; inr2002 = []; inr2003 = [];
    in2r2001 = []; in2r2002 = []; in2r2003 = [];
    timinglist1 = np.empty(shape=(0,0))
    timinglist2 = np.empty(shape=(0,0))
    timinglist3 = np.empty(shape=(0,0))

    
    for i, fname in splt1:
        M200 = 1e12
        df, time = datainitializing(fname,fp1)
        if len(df['posx']) > 10_000_000:
            df = df[df.index > 10_000_000]
        timinglist1 = np.append(timinglist1, time)
        #avgposdf, avgveldf = COMfind(df, indexdf1)
        #CircComdf, CircVeldf = shrinkingcircmethod(df, avgposdf)
        CircComdf, CircVeldf = findcenterhist(df)
        df = bonuscalc(df,CircComdf, CircVeldf)
        #Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
    
        Comlist1 = np.concatenate((Comlist1, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        inr2001 = inr200func(df,inr2001,M1)
        in2r2001 = in2r200func(df,in2r2001,M1)

        del df

    for i, fname in splt2:
        M200 = 1e12
        df, time = datainitializing(fname,fp2)
        if len(df['posx']) > 10_000_000:
            df = df[df.index > 10_000_000]
        timinglist2 = np.append(timinglist2, time)
        avgposdf, avgveldf = COMfind(df, indexdf2)
        CircComdf, CircVeldf = shrinkingcircmethod(df, avgposdf)
        CircComdf, CircVeldf = findcenterhist(df)
        df = bonuscalc(df,CircComdf, CircVeldf)
        #Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
    
        Comlist2 = np.concatenate((Comlist2, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        inr2002 = inr200func(df,inr2002,M2)
        in2r2002 = in2r200func(df,in2r2002,M2)

        del df

    for i, fname in splt3:
        M200 = 1e12
        df, time = datainitializing(fname,fp3)
        if len(df['posx']) > 10_000_000:
            df = df[df.index > 10_000_000]
        timinglist3 = np.append(timinglist3, time)
        avgposdf, avgveldf = COMfind(df, indexdf3)
        CircComdf, CircVeldf = shrinkingcircmethod(df, avgposdf)
        CircComdf, CircVeldf = findcenterhist(df)
        df = bonuscalc(df,CircComdf, CircVeldf)
        #Comlist = np.concatenate((Comlist, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)

        Comlist3 = np.concatenate((Comlist3, np.array(CircComdf.iloc[0]).reshape(1,3)),axis=0)
        inr2003 = inr200func(df,inr2003,M3)
        in2r2003 = in2r200func(df,in2r2003,M3)

        del df

    COMM.Barrier()

    inr2001 = COMM.gather(np.array(inr2001), root=0)
    in2r2001 = COMM.gather(np.array(in2r2001), root=0)
    timinglist1 = COMM.gather(timinglist1, root=0)
    Comlist1 = COMM.gather(Comlist1, root=0)

    inr2002 = COMM.gather(np.array(inr2002), root=0)
    in2r2002 = COMM.gather(np.array(in2r2002), root=0)
    timinglist2 = COMM.gather(timinglist2, root=0)
    Comlist2 = COMM.gather(Comlist2, root=0)

    inr2003 = COMM.gather(np.array(inr2003), root=0)
    in2r2003 = COMM.gather(np.array(in2r2003), root=0)
    timinglist3 = COMM.gather(timinglist3, root=0)
    Comlist3 = COMM.gather(Comlist3, root=0)

    COMM.Barrier()

    if COMM.rank == 0:
        inr2001 = np.concatenate(inr2001)
        in2r2001 = np.concatenate(in2r2001)
        timinglist1 = np.concatenate(timinglist1)
        Comlist1 = np.concatenate(Comlist1)

        inr2002 = np.concatenate(inr2002)
        in2r2002 = np.concatenate(in2r2002)
        timinglist2 = np.concatenate(timinglist2)
        Comlist2 = np.concatenate(Comlist2)

        inr2003 = np.concatenate(inr2003)
        in2r2003 = np.concatenate(in2r2003)
        timinglist3 = np.concatenate(timinglist3)
        Comlist3 = np.concatenate(Comlist3)

        Rlist1 = np.sqrt(Comlist1[:,0]**2 + Comlist1[:,1]**2 + Comlist1[:,2]**2)
        Rlist2 = np.sqrt(Comlist2[:,0]**2 + Comlist2[:,1]**2 + Comlist2[:,2]**2)
        Rlist3 = np.sqrt(Comlist3[:,0]**2 + Comlist3[:,1]**2 + Comlist3[:,2]**2)

        M = 1e12
        fig = plt.figure()
        gs = fig.add_gridspec(2,2)
        (ax1, ax2), (ax3, ax4) = gs.subplots()
        ax1.scatter(Comlist1[:,0],Comlist1[:,1],s=0.2,c="black")
        ax2.scatter(Comlist1[:,0],Comlist1[:,2],s=0.2,c="black", label=label1)
        ax3.scatter(Comlist1[:,1],Comlist1[:,2],s=0.2,c="black")
        ax4.plot(timinglist1, Rlist1, c="black")

        ax1.scatter(Comlist2[:,0],Comlist2[:,1],s=0.2,c="red")
        ax2.scatter(Comlist2[:,0],Comlist2[:,2],s=0.2,c="red",label=label2)
        ax3.scatter(Comlist2[:,1],Comlist2[:,2],s=0.2,c="red")
        ax4.plot(timinglist2, Rlist2, c="red")

        ax1.scatter(Comlist3[:,0],Comlist3[:,1],s=0.2,c="blue")
        ax2.scatter(Comlist3[:,0],Comlist3[:,2],s=0.2,c="blue",label=label3)
        ax3.scatter(Comlist3[:,1],Comlist3[:,2],s=0.2,c="blue")
        ax4.plot(timinglist3, Rlist3, c="blue")

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax3.set_xlabel("Y")
        ax3.set_ylabel("Z")
        ax4.set_xlabel("T (Gyr)")
        ax4.set_ylabel("Radius")
        #plt.legend(loc='best')
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(plotfp + "3FILECOM_"+a2f+".png",dpi=600)
        plt.close()


        
        plt.plot(timinglist1,Rlist1,color='black',label=label1)
        plt.plot(timinglist2,Rlist2,color='red',label=label2)
        plt.plot(timinglist3,Rlist3,color='blue',label=label3)
        plt.ylabel("rad")
        plt.xlabel("Time in Gyr")
        plt.legend(loc='best')
        plt.title(title)
        #plt.title(f"Mass {M:.1e}")
        plt.tight_layout()
        plt.savefig(plotfp + "3FILERAD_"+a2f+".png",dpi=600)
        plt.close()



        plt.plot(timinglist1, inr2001/Ntot1,c='black', label=label1,zorder=1)
        plt.plot(timinglist1, in2r2001/Ntot1,c='black',linestyle='dashed',zorder=1)

        plt.plot(timinglist2, inr2002/Ntot2,c='red',label=label2,zorder=0)
        plt.plot(timinglist2, in2r2002/Ntot2,c='red',linestyle='dashed',zorder=0)

        plt.plot(timinglist3, inr2003/Ntot3,c='blue',label=label3,zorder=0)
        plt.plot(timinglist3, in2r2003/Ntot3,c='blue',linestyle='dashed',zorder=0)

        plt.xlabel("Time (Gyr)")
        plt.ylabel(r"frac contained within $R_{x}$")
        plt.legend(loc='best')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(plotfp + "3FILEinr200_"+a2f+".png",dpi=600)
        plt.close()


    print(f"Rank {COMM.rank} Finished")
    COMM.Barrier()
    MPI.Finalize()

def bonuscalc(df, COM, VEL):
    df['posxCOM'] = df['posx']-COM['posx'].iloc[0] 
    df['posyCOM'] = df['posy']-COM['posy'].iloc[0]  
    df['poszCOM'] = df['posz']-COM['posz'].iloc[0]  
    df['velxCOM'] = df['velx']-VEL['velx'].iloc[0] 
    df['velyCOM'] = df['vely']-VEL['vely'].iloc[0] 
    df['velzCOM'] = df['velz']-VEL['velz'].iloc[0] 
    df['radius'] = np.sqrt(df['posx'] ** 2 + df['posy'] ** 2 + df['posz'] ** 2)
    df['radiusCOM'] = np.sqrt(df['posxCOM']**2 + df['posyCOM']**2 + df['poszCOM']**2)
    df['modvel'] = np.sqrt(df['velx'] ** 2 + df['vely'] ** 2 + df['velz'] ** 2)
    df['modvelCOM'] = np.sqrt(df['velxCOM'] ** 2 + df['velyCOM'] ** 2 + df['velzCOM'] ** 2)
    df['vr'] = df['posx'] * df['velx'] + df['posy'] * df['vely'] + df['posz'] * df['velz']; df['vr'] /= df['radius']
    df['vrCOM'] = df['posxCOM'] * df['velxCOM'] + df['posyCOM'] * df['velyCOM'] + df['poszCOM'] * df['velzCOM']; df['vrCOM'] /= df['radiusCOM']
    return df

def getsnapshotsnew(fp):
    files = sorted(os.listdir(fp + "/output/"))
    snapshotlst = []
    for item in files:
        if "snapshot" in item:
            snapshotlst.append(item)
    return sorted(snapshotlst)


def datainitializing(filename,fp): 
    with h5py.File(fp + "/output/"+ filename, 'r') as f:
        NumPart = f['Header'].attrs['NumPart_Total'][()]
        pos = f['PartType1/Coordinates'][()]
        vel = f['PartType1/Velocities'][()]
        pids = f['PartType1/ParticleIDs'][()]
        masses = f['PartType1/Masses'][()]
        data = {
            "posx": pos[:, 0],
            "posy": pos[:, 1],
            "posz": pos[:, 2],
            "velx": vel[:, 0],
            "vely": vel[:, 1],
            "velz": vel[:, 2], 
            "mpp": masses
        }
        df = pd.DataFrame(data, index = pids)
        time = f['Header'].attrs['Time']
    return df, time


def findmiddleparts(snapshotlst,fp,i=10):
    with h5py.File(fp + "/output/"+ snapshotlst[0], 'r') as f:
        NumPart = f['Header'].attrs['NumPart_Total'][()]
        pos = f['PartType1/Coordinates'][()]
        vel = f['PartType1/Velocities'][()]
        pids = f['PartType1/ParticleIDs'][()]
        data = {
            "posx": pos[:, 0],
            "posy": pos[:, 1],
            "posz": pos[:, 2],
            "velx": vel[:, 0],
            "vely": vel[:, 1],
            "velz": vel[:, 2]
        }
        df = pd.DataFrame(data, index=pids)
    if len(df['posx']) > 10_000_000:
         df = df[df.index > 10_000_000]
    Ntot = len(df['posx'])
    df['posx'] = df['posx'] - df['posx'].mean()
    df['posy'] = df['posy'] - df['posx'].mean()
    df['posz'] = df['posz'] - df['posz'].mean()
    df['r'] = np.sqrt(df['posx']**2 + df['posy']**2 + df['posz']**2)
    temp = df.copy()    
    temp = temp.sort_values(by=['r']).head(i)
    z = temp.index
    dict = {"i":list(z)}
    temp = pd.DataFrame(dict)
    return temp, Ntot


def COMfind(df, indexdf):
    avgposdf = pd.DataFrame(columns = ['posx','posy','posz'])
    avgveldf = pd.DataFrame(columns = ['velx','vely','velz'])
    temp = df[df.index.isin(list(indexdf['i']))]
    avgposdf.loc[len(avgposdf)] = [temp['posx'].mean(),temp['posy'].mean(),temp['posz'].mean()]
    avgveldf.loc[len(avgveldf)] = [temp['velx'].mean(),temp['vely'].mean(),temp['velz'].mean()]
    return avgposdf, avgveldf


def shrinkingcircmethod(df, COM):
    radlist = 10**np.arange(3,0,-0.1)
    CircComdf = pd.DataFrame(columns = ['posx','posy','posz'])
    CircVeldf = pd.DataFrame(columns = ['velx','vely','velz'])
    posx = COM['posx'].iloc[0]; posy = COM['posy'].iloc[0]; posz = COM['posz'].iloc[0];


    for radii in radlist:
        temp = df.copy()
        temp2 = df.copy()
        temp['posx'] -= posx; temp['posy'] -= posy; temp['posz'] -= posz;
        temp['rad'] = np.sqrt(temp['posx']**2 + temp['posy']**2 + temp['posz']**2)
        temp2 = temp2[temp['rad']<= radii]
        if len(temp2['posx'])==0:
            break
        posx = temp2['posx'].mean(); posy = temp2['posy'].mean(); posz = temp2['posz'].mean()
        velx = temp2['velx'].mean(); vely = temp2['vely'].mean(); velz = temp2['velz'].mean()
    CircComdf.loc[len(CircComdf)] = [posx,posy,posz]
    CircVeldf.loc[len(CircVeldf)] = [velx,vely,velz] 
    return CircComdf, CircVeldf


rhocrit = 2.7755e11


def inr200func(df,inrhalf,M200=1e12):
    temp = df.copy(); tot = len(df['posx'])
    r200 = (3/4/np.pi/200/rhocrit*M200)**(1/3)*1000
    inrhalf.append(len(temp[temp['radiusCOM']<=r200]))
    return inrhalf

def in2r200func(df,inrhalf,M200=1e12):
    temp = df.copy(); tot = len(df['posx'])
    r200 = (3/4/np.pi/200/rhocrit*M200)**(1/3)*1000
    inrhalf.append(len(temp[temp['radiusCOM']<=2*r200]))
    return inrhalf

def findcenterhist(df):
    CircComdf = pd.DataFrame(columns = ['posx','posy','posz'])
    CircVeldf = pd.DataFrame(columns = ['velx','vely','velz'])
    temp = df.copy()
    temp['rad'] = np.sqrt(temp['posx']**2 + temp['posy']**2 + temp['posz']**2)
    cut = 500 
    temp = temp[temp['rad'] <= cut]
    nb = 1000
    H1, xedges1, yedges1 = np.histogram2d(temp['posx'],temp['posy'],bins=(nb,nb))
    H2, xedges2, yedges2 = np.histogram2d(temp['posx'],temp['posz'],bins=(nb,nb))
    H3, xedges3, yedges3 = np.histogram2d(temp['posy'],temp['posz'],bins=(nb,nb))

    for x, y in np.argwhere(H1 == H1.max()):
        # center is between x and x+1
        xpos1 = np.average(xedges1[x:x + 2])
        ypos1 = np.average(yedges1[y:y + 2])

    for x, z in np.argwhere(H2 == H2.max()):
        # center is between x and x+1
        xpos2 = np.average(xedges2[x:x + 2])
        zpos1 = np.average(yedges2[z:z + 2])

    for y, z in np.argwhere(H3 == H3.max()):
        # center is between x and x+1
        ypos2 = np.average(xedges3[y:y + 2])
        zpos2 = np.average(yedges3[z:z + 2])

    posx = (xpos1 + xpos2)/2
    posy = (ypos1 + ypos2)/2
    posz = (zpos1 + zpos2)/2


    CircComdf.loc[len(CircComdf)] = [posx,posy,posz]
    del temp    
    temp = df.copy()
    temp['posx'] -= posx
    temp['posy'] -= posy
    temp['posz'] -= posz
    temp['rad'] = np.sqrt(temp['posx']**2 + temp['posy']**2 + temp['posz']**2)
    temp = temp[temp['rad']<1]
    velx = temp['velx'].mean(); vely = temp['vely'].mean(); velz = temp['velz'].mean()
    CircVeldf.loc[len(CircVeldf)] = [velx,vely,velz]
    del temp
    return CircComdf, CircVeldf



if __name__ == "__main__":
    main()



