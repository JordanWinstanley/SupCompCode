from mpi4py import MPI
import h5py
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0,os.getcwd())
import parameters as par
massunit = 1e10
kpctom = 3.086e+19 
smtokg = 1.9889e+30
G = 6.67430 * (10 ** (-11)); 
rhocrit = 2.7755e11
    

def prepareplotfile(plotfilepath, pltfiles):
    x = pltfiles.split("/")
    x.pop()
    sparestr = ""
    for item in x:
        if os.path.exists(plotfilepath + sparestr + item) == False:
            os.mkdir(plotfilepath + sparestr + item)
        sparestr = sparestr + item + "/"
    plotfilepath2 = plotfilepath + sparestr
    directorycheck(plotfilepath2)
    return plotfilepath2


def getsnapshots():
    files = sorted(os.listdir(os.getcwd() + "/output/"))
    snapshotlst = []
    for item in files:
        if "snapshot" in item:
            snapshotlst.append(item)
    return sorted(snapshotlst)


def findplotdirec():
    test = False
    backstr = ""
    x = os.listdir(os.getcwd())
    pltfiles = ""
    while test == False:
        if "Plots" in x:
            test = True
        else:
            backstr = backstr + "../"
            x = os.listdir(backstr)
            for item in x:
                if item in os.getcwd():
                    pltfiles = item + "/" + pltfiles
    return backstr + "Plots/", pltfiles


def directorycheck(fp): 
	dict = {
	"plots": ["bound","COM","Density","mass","phase","pos","maxr","sigmar",'sigmatot',"phaseCOM","circleplot","inrhalf","densitywithhalo",\
	"masswithhalo","avgvel","avgvelsquared", "mdnmean", "RmassEnc","inr200","npartfromcent","velhist"],

	"quickpos":["eps","pos","half","with dot","extra", "half/png","half/eps","with dot/png", \
	"with dot/eps", "extra/png","extra/eps","partpath", "partpathwhole", "hist","half with dot","half with dot/eps","half with dot/png",\
	"galwithDM", "galwithnoDM"],

	}

	list1 = dict.keys()
	for item in list1:
		if not os.path.exists(fp + item):
			try: 
				os.mkdir(fp + "/" + item)
			except:
				print("Exists")
			else:
				list2 = dict.get(item)
				if len(list2) != 0:
					for item2 in list2:
						try: 
							os.mkdir(fp + item + "/" + item2)
						except:
							pass
		else: 
			list2 = dict.get(item)
			if len(list2) != 0:
				for item2 in list2:
					if not os.path.exists(fp + item + "/" + item2):
						try:
							os.mkdir(fp + item + "/" + item2)
						except:
							pass
			else:
				pass
			

#Main Bulk of Functions for central location Calculation


def datainitializing(filename): 
    with h5py.File("output/"+ filename, 'r') as f:
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


def findmiddleparts(snapshotlst):
    with h5py.File("output/"+ snapshotlst[0], 'r') as f:
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
    if len(df['posx'] > 10_000_000):
         df = df[df.index > 10_000_000]
    i = 10
    df['posx'] = df['posx'] - df['posx'].mean()
    df['posy'] = df['posy'] - df['posx'].mean()
    df['posz'] = df['posz'] - df['posz'].mean()
    df['r'] = np.sqrt(df['posx']**2 + df['posy']**2 + df['posz']**2)
    temp = df.copy()    
    temp = temp.sort_values(by=['r']).head(i)
    z = temp.index
    dict = {"i":list(z)}
    temp = pd.DataFrame(dict)
    return temp


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


#Functions for Bulk Calcs
def radiusprep(x=-1,y=3,z=0.1):
    r = 10 ** np.arange(x,y+z,z)
    r = np.insert(r, 0, 0)
    return r


def calcs(df, r, COM, VEL):
    npart = np.empty(shape=(0,0)); mass = np.empty(shape=(0,0)); dens = np.empty(shape=(0,0)); 
    rmax = np.empty(shape=(0,0)); meanrad = np.empty(shape=(0,0)); rmin = np.empty(shape=(0,0));
    avgvelx = np.empty(shape=(0,0)); avgvely = np.empty(shape=(0,0)); avgvelz = np.empty(shape=(0,0))
    avgvelx2 = np.empty(shape=(0,0)); avgvely2 = np.empty(shape=(0,0)); avgvelz2 = np.empty(shape=(0,0))
    sigmax = np.empty(shape=(0,0)); sigmay = np.empty(shape=(0,0)); sigmaz = np.empty(shape=(0,0))
    sigmavr = np.empty(shape=(0,0))
    r200 = (3/4/np.pi/200/rhocrit*par.M200)**(1/3)*1000
    rs = r200 / par.c * np.sqrt(2*(np.log(1+par.c)-par.c/(1+par.c)))
    halodens = par.M200 / 2 / np.pi / rs**3

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

    tempdf = df.copy()
    for i in range(0, len(r)-1):
        radii = tempdf['radiusCOM']
        mpp = df['mpp'].iloc[0]
        try:
            x = radii.between(r[i], r[i + 1]).value_counts().values[1] 
        except IndexError:
            try:
                x = radii.between(r[i], r[i+1]).value_counts().values[1] 
            except IndexError:
                x = 0
            finally:
                npart = np.append(npart,x)
                mass = np.append(mass, x*massunit*mpp)
                dens = np.append(dens,(x * massunit * mpp) / ((4 / 3) * np.pi * ((r[i+1] ** 3 - r[i] ** 3))))
                rmax = np.append(rmax, r[i+1])
                rmin = np.append(rmin, r[i])
                meanrad = np.append(meanrad,(r[i]+r[i+1])/2)
                avgvelx = np.append(avgvelx, np.nan) #np.nan
                avgvely = np.append(avgvely, np.nan)
                avgvelz = np.append(avgvelz, np.nan)
                avgvelx2 = np.append(avgvelz2, np.nan)
                avgvely2 = np.append(avgvely2, np.nan)
                avgvelz2 = np.append(avgvelz2, np.nan)
                sigmax = np.append(sigmax,np.nan)
                sigmay = np.append(sigmay,np.nan)
                sigmaz = np.append(sigmaz,np.nan)
                sigmavr = np.append(sigmavr,np.nan)
        else:
            npart = np.append(npart,x)
            mass = np.append(mass, x * massunit * mpp)
            dens = np.append(dens,(x * massunit* mpp) / ((4 / 3) * np.pi * (r[i + 1] ** 3 - r[i] ** 3)))
            rmax = np.append(rmax, r[i+1])
            rmin = np.append(rmin, r[i])
            z = df[radii.between(r[i], r[i+1])]['radiusCOM'].mean()
            meanrad = np.append(meanrad,z)
            avgvelx = np.append(avgvelx, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velxCOM'].mean())
            avgvely = np.append(avgvely, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velyCOM'].mean())
            avgvelz = np.append(avgvelz, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velzCOM'].mean())
            avgvelx2 = np.append(avgvelx2, (tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velxCOM']**2).mean())
            avgvely2 = np.append(avgvely2, (tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velyCOM']**2).mean())
            avgvelz2 = np.append(avgvelz2, (tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velzCOM']**2).mean())
            sigmax = np.append(sigmax, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velxCOM'].var())
            sigmay = np.append(sigmay, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velyCOM'].var())
            sigmaz = np.append(sigmaz, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['velzCOM'].var())
            sigmavr = np.append(sigmavr, tempdf[tempdf['radiusCOM'].between(r[i],r[i+1])]['vrCOM'].var())
    data2 = {
        'rmax': rmax,
        'rmin': rmin,
        'num': npart,
        'mass': mass,
        'dens': dens,
        'meanradius': meanrad,
        'avgvelx': avgvelx,
        'avgvely': avgvely,
        'avgvelz': avgvelz,
        'avgvelx2': avgvelx2,
        'avgvely2': avgvely2,
        'avgvelz2': avgvelz2,
        'sigmax': sigmax,
        'sigmay': sigmay,
        'sigmaz': sigmaz,
        'sigmavr': sigmavr,
        }
    df2 = pd.DataFrame(data2)
    #df2['r'] = r
    #df2 = sigmacalc(df,df2)
    df2['summass'] = df2['mass'].cumsum()
    df2['sumnum'] = df2['num'].cumsum()
    #df2['exactdens'] = (G * M * rs)/(2 * np.pi * df2['meanradius'] * (rs + df2['meanradius'])**3)
    #df2['resid'] = df2['dens'] - df2['exactdens']
    df2['halodens'] = halodens / (df2['meanradius']/rs) / ((1 + df2['meanradius']/rs)**3) 
    df2['haloMass'] = 4 * np.pi * halodens * rs**3 * ((df2['meanradius']/rs)**2/(2 *(1 + df2['meanradius']/rs)**2))
    df2['sigmatot'] = df2['sigmax'] + df2['sigmay'] + df2['sigmaz']
    df2['avgvel'] = np.sqrt(df2['avgvelx']**2 + df2['avgvely']**2 + df2['avgvelz']**2)
    df2['avgvel2'] = np.sqrt(df2['avgvelx2']**2 + df2['avgvely2']**2 + df2['avgvelz2']**2)
    return df2


def inr200func(df,COM,inrhalf,fp):
    temp = df.copy(); tot = len(df['posx'])
    r200 = (3/4/np.pi/200/rhocrit*par.M200)**(1/3)*1000
    inrhalf.append(len(temp[temp['radiusCOM']<=r200]))
    return inrhalf


def maxraddict(df,condict,COM):
    temp=df.copy()
    temp = temp.sort_values(by='radiusCOM')
    for i in list(condict.keys()):
        x = condict.get(i)
        x.append(max(temp['radiusCOM'].head(int(i))))
        condict.update({i:x})
    return condict


def medavgcalc(df, mdndict, fp):
    for key in list(mdndict.keys()):
        temp = df.copy()
        x = mdndict.get(key)
        temp = temp.sort_values(by=['radiusCOM'])
        temp = temp.head(int(key))
        mdn = temp['radiusCOM'].median()
        mean = temp['radiusCOM'].mean()
        x[0].append(mdn)
        x[1].append(mean)
        mdndict.update({key:x})
    return mdndict


#Plotting
def position(df,filename,fp,i,k):
    lims = 2000
    plt.scatter(df['posx'], df['posy'], s=0.1, color='black')
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/pos/"+"position_"+filename,dpi=600)
    plt.close()


def phase(df,filename,fp,i,k):
    plt.scatter(df['radius'],df['vr'],s=0.1, color='black')
    plt.xlabel("r")
    plt.ylabel("vr")
    #plt.xlim(0, 500)
    #plt.ylim(-600,600)
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/phase/"+"phase_"+filename,dpi=600)
    plt.close()


def phaseCOM(df,filename,fp,i,k):
    plt.scatter(df[df['radiusCOM']<1000]['radiusCOM'],df[df['radiusCOM']<1000]['vrCOM'],s=0.1, color='black')
    plt.xlabel("r")
    plt.ylabel("vr")
    #plt.xlim(0, 500)
    #plt.ylim(-600,600)
    plt.title(f"t = {round(k,1)} Gyr, Snap: {i}")
    plt.savefig(fp+"plots/phaseCOM/"+"phase_"+filename,dpi=600)
    plt.close()

def velhist(df,filename,fp,i,k):
    xcounts, xbins = np.histogram(df['velx'],bins=100)
    ycounts, ybins = np.histogram(df['vely'],bins=100)
    zcounts, zbins = np.histogram(df['velz'],bins=100)
    fig, axs = plt.subplots(3)
    axs[0].hist(xbins[:-1],xbins, weights=xcounts,color='black')
    axs[1].hist(ybins[:-1],ybins, weights=ycounts,color='black')
    axs[2].hist(zbins[:-1],zbins, weights=zcounts,color='black')
    axs[0].set_xlabel(r"$V_{x}$")
    axs[1].set_xlabel(r"$V_{y}$")
    axs[2].set_xlabel(r"$V_{z}$")
    fig.supylabel("Counts")
    fig.suptitle(f"t = {round(k,1)} Gyr, Snap: {i}")
    fig.tight_layout()
    plt.savefig(fp+"plots/velhist/"+"velhist_"+filename,dpi=600)

def density(df2,filename,fp,i,k):
    plt.loglog()
    #print(df2)
    plt.scatter(df2["meanradius"], df2['dens'], s=5, color='black')
    plt.plot(df2["meanradius"], df2['dens'], color='red')
    #plt.ylim(1E0, 1E12)
    plt.xlim(1E-1, 2000)
    plt.ylabel("Density")
    plt.xlabel("Radius")
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/density/"+ "density_"+filename,dpi=600)
    plt.close()


def mass(df2,filename,fp,i,k):
    plt.loglog()
    plt.scatter(df2['meanradius'], df2['summass'], s=5, color='black')
    plt.plot(df2['meanradius'], df2['summass'], color='red')
    plt.ylabel("Enclosed Mass")
    plt.xlabel("Radius")
    plt.ylim(1E0, 1E12)
    plt.xlim(1E-1, 2000)
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/mass/"+ "mass_"+filename,dpi=600)
    plt.close()


def densitywithhalo(df2,filename,fp,i,k):
    plt.loglog()
    plt.plot(df2["meanradius"], df2['dens'], color='black', label = 'N-Body')
    plt.plot(df2["meanradius"], df2['halodens'], color='red',label='rs')
    #plt.plot(df2["meanradius"], df2['halodens2.0'], color='green',label='updated rs')
    #plt.plot(df2["meanradius"], df2['halodens2'], color='green')
    #plt.plot(df2["meanradius"], df2['halodens3'], color='purple')
    #plt.plot(df2["meanradius"], df2['halodensNFW'], color='orange')
    plt.ylim(1E0, 1E10)
    plt.xlim(1E0, 2000)
    plt.ylabel("Density")
    plt.xlabel("Radius")
    plt.legend(loc='best')
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/densitywithhalo/"+ "densitywithhalo_"+filename,dpi=600)
    plt.close()


def masswithhalo(df2,filename,fp,i,k):
    plt.loglog()
    plt.plot(df2['meanradius'], df2['summass'], color='black', label='N-body')
    plt.plot(df2['meanradius'], df2['haloMass'], color='red', label='rs')
    #plt.plot(df2['meanradius'], df2['haloMass2'], color='green',label='Updated rs')
    #rhocrit = 2.7755e11
    #r200 = (3/4/np.pi/200/rhocrit*1e12)**(1/3)*1000
    #print((df2['summass']/df2['haloMass']).mean())
    #plt.plot(df2['meanradius'],df2['summass']/df2['haloMass'])
    #plt.hlines(1e12, xmin = df2['meanradius'].min(),xmax=df2['meanradius'].max(),color='purple',linestyles='dashed')
    #plt.vlines(r200,ymin=0,ymax=1e12,color='purple',linestyles='dashed')
    plt.ylabel("Enclosed Mass")
    plt.xlabel("Radius")
    plt.legend(loc='best')
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/masswithhalo/"+ "masswithhalo_"+filename,dpi=600)
    plt.close()


def COMplot(COMlist,fp,timinglist):

    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    (ax1, ax2), (ax3, ax4) = gs.subplots()
    ax1.scatter(COMlist[:,0],COMlist[:,1],s=0.2,c=timinglist)
    ax2.scatter(COMlist[:,0],COMlist[:,2],s=0.2,c=timinglist)
    ax3.scatter(COMlist[:,1],COMlist[:,2],s=0.2,c=timinglist)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    fig.tight_layout()
    #fig.colorbar()
    plt.xlabel("x")
    plt.savefig(fp+"plots/COM/COM.png",dpi=600)
    plt.close()
    #plt.style.use("default")

def COMtestplot(COMlist,fp,timinglist):
    Rlist = np.sqrt(COMlist[:,0]**2 + COMlist[:,1]**2 + COMlist[:,2]**2)
    fig = plt.figure()
    gs = fig.add_gridspec(4, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(timinglist,COMlist[:,0])
    axs[1].plot(timinglist,COMlist[:,1])
    axs[2].plot(timinglist,COMlist[:,2])
    axs[3].plot(timinglist,Rlist)
    axs[0].set_ylabel("x")
    axs[1].set_ylabel("y")
    axs[2].set_ylabel("z")
    axs[3].set_ylabel("R")
    axs[3].set_xlabel("Time (Gyr)")
    plt.savefig(fp+"plots/COM/COMtest.png",dpi=600)
    plt.close()



def COMradplot(COMlist,fp,timinglist):
    Rlist = np.sqrt(COMlist[:,0]**2 + COMlist[:,1]**2 + COMlist[:,2]**2)
    plt.plot(timinglist,Rlist,color='red',zorder=0)
    plt.scatter(timinglist,Rlist,s=0.2,color='black',zorder=1)
    plt.ylabel("rad")
    plt.xlabel("Time in Gyr")
    plt.savefig(fp+"plots/COM/COMrad2.png",dpi=600)
    plt.close()


def inr200plot(r200,fp,timinglist):
    plt.plot(timinglist,r200,c='red',zorder=0)
    plt.scatter(timinglist, r200, s=1, color='black',zorder=1)
    plt.ylabel("Num part in r200")
    plt.xlabel("Gyr")
    plt.savefig(fp+"plots/inr200/inr200.png",dpi=600)
    plt.close()


def conwithin(condict,fp):
    x = np.arange(0,15,15/len(condict.get("50")))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, condict.get("50"), s=1, color='black',label="50")
    ax1.scatter(x, condict.get("1600"), s=1, color='red',label="1600")
    ax1.scatter(x, condict.get("6400"), s=1, color='blue',label="6400")
    plt.ylabel("Max r")
    plt.xlabel("Gyr")
    plt.legend(loc='upper left')
    plt.savefig(fp+"plots/maxr/maxr3.png",dpi=600)

    plt.close()


def conwithin3(condict, fp):
    x = np.arange(0,15,15/len(condict.get("50")))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(x, condict.get("50"), color='black',label="50")
    ax1.plot(x, condict.get("100"),  color='purple',label="100")
    ax1.plot(x, condict.get("200"), color='orange',label="200")
    ax1.plot(x, condict.get("400"),  color='green',label="400")
    ax1.plot(x, condict.get("800"),  color='red',label="800")
    ax1.plot(x, condict.get("1600"), color='blue',label="1600")

    plt.ylabel("Max r")
    plt.xlabel("Gyr")
    plt.legend(loc='upper left')
    plt.savefig(fp+"plots/maxr/maxrrefresh.png",dpi=600)

    plt.close()


def conwithin2(condict, fp):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    c = ['black','purple','orange','green','red','blue']
    keys = list(condict.keys())
    x = np.arange(0,15,15/len(condict.get(keys[0])))
    for i, y  in enumerate(keys):
        ax1.plot(x, condict.get(y), color=c[i],label=y)
    plt.ylabel("Max r")
    plt.xlabel("Gyr")
    plt.legend(loc='upper left')
    plt.savefig(fp+"plots/maxr/maxrsomeperc.png",dpi=600)

    plt.close()


def poscircleplot(df,i,fp,COM,condict,fn,k):
    temp=df.copy()
    temp['posx'] = temp['posx'] - COM['posx']; 
    temp['posy'] = temp['posy'] - COM['posy']; 
    temp['posz'] = temp['posz'] - COM['posz'];
    temp.drop('radius', axis = 1, inplace = True)
    temp['radius'] = np.sqrt(temp['posx']**2 + temp['posy']**2 + temp['posz']**2)

    ax1 = plt.gca()
    ax1.cla()
    colors = ['brown','purple','orange','green','red','blue','yellow','pink']
    lims = 100
    ax1.set_xlim(COM['posx'].iloc[0]-lims,COM['posx'].iloc[0]+lims)
    ax1.set_ylim(COM['posy'].iloc[0]-lims,COM['posy'].iloc[0]+lims)

    ax1.scatter(df['posx'], df['posy'], s=0.1, color='black')
    for a,b in zip(list(condict.keys()),colors):
        r = max((temp.sort_values(by='radius')['radius']).head(int(a)))
        circ = plt.Circle((COM['posx'].iloc[0],COM['posy'].iloc[0]),r,fill=False, color = b,label=a,clip_on=False,lw=0.2)
        ax1.add_patch(circ)
    plt.legend(loc='upper left')
    plt.title("t = " + str(round(k,1)) + " Gyr")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fp+"plots/circleplot/circleplotcut_" + fn + ".png",dpi=600)
    lims = 2000
    ax1.set_xlim(-lims,lims)
    ax1.set_ylim(-lims,lims)
    plt.savefig(fp+"plots/circleplot/circleplotfull_" + fn + ".png",dpi=600)
    plt.close()


def sigmaplot(df2,filename,fp,i,k):
    plt.plot(df2['meanradius'], df2['sigma2'], color='red')
    plt.scatter(df2['meanradius'], df2['sigma2'], s=5, color='black')
    plt.ylabel("Sigma")
    plt.xlabel("Mean Radius")
    plt.title(f"t = {round(i*15/k,2)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/sigma/"+ "sigma_"+filename,dpi=600)
    plt.close()


def avgvelplot(df2, filename,fp,i,k):
    plt.plot(df2['meanradius'],df2['avgvel'],c='red')
    plt.scatter(df2['meanradius'],df2['avgvel'],s=5,c='black')
    plt.ylabel("Avg vel")
    plt.xlabel("Radius")
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/avgvel/"+ "avgvel_"+filename,dpi=600)
    plt.close()


def avgvelplot2(df2, filename,fp,i,k):
    plt.plot(df2['meanradius'],df2['avgvel2'],c='red')
    plt.scatter(df2['meanradius'],df2['avgvel2'],s=5,c='black')
    plt.ylabel("Avg vel ** 2")
    plt.xlabel("Radius")
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/avgvelsquared/"+ "avgvelsquared_"+filename,dpi=600)
    plt.close()


def mdnmeanplot(mdndict, fp):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    c = ['black','purple','orange','green','red','blue']
    keys = list(mdndict.keys())
    x = np.arange(0,15,15/len(mdndict.get(keys[0])[0]))
    for i, y  in enumerate(keys):
        z = mdndict.get(y)
        ax1.plot(x, z[0], color=c[i],label=y)
    plt.ylabel("Max r")
    plt.xlabel("Gyr")
    plt.legend(loc='upper left')
    plt.savefig(fp+"plots/mdnmean/mdnplot.png",dpi=600)
    plt.close()


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, y  in enumerate(keys):
        z = mdndict.get(y)
        ax1.plot(x, z[1], color=c[i],label=y)
    plt.ylabel("Max r")
    plt.xlabel("Gyr")
    plt.legend(loc='upper left')
    plt.savefig(fp+"plots/mdnmean/meanplot.png",dpi=600)
    plt.close()


def sigmavrplot(df2, filename,fp,i,k):
    plt.plot(df2['meanradius'],np.sqrt(df2['sigmavr']))
    plt.xlabel('R')
    plt.ylabel(r"$\sigma_{r}$")
    plt.xlim(0,500)
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/sigmar/"+ "sigmar_"+filename,dpi=600)
    plt.close()


def sigmatotplot(df2, filename,fp,i,k):
    plt.plot(df2['meanradius'],1-(df2['sigmatot'])/2/df2['sigmavr'] + 0.5)
    plt.xlabel('R')
    plt.ylabel(r"$\sigma_{tot}/\sigma_{r}^{2}$")
    plt.xlim(0,500)
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/sigmatot/sigmatotr_"+filename,dpi=600)
    plt.close()


def NpartCOM(df,filename,fp,i,k):
    #r = 10**np.arange(-1,3.6,0.1)
    r = np.arange(0,1510,10)
    npart = np.empty([0,0])
    meanrad = np.empty([0,0])
    radii = df['radius']
    for spare in range(len(r)-1):
        y = radii.between(r[spare], r[spare + 1])
        try:
            x = y.value_counts().values[1]
        except IndexError:
            x = 0
            npart = np.append(npart,x)
            meanrad = np.append(meanrad, ((r[spare]+r[spare+1])/2))
        else:
            npart = np.append(npart,x)
            meanrad = np.append(meanrad, radii[y].mean())
    plt.plot(meanrad,npart)
    plt.xlabel('radius from (0,0,0)')
    plt.ylabel("Number of Particles")
    plt.title(f"t = {round(k,1)} Gyr, snap: {i}")
    plt.savefig(fp+"plots/npartfromcent/npartsfc_"+filename,dpi=600)
    plt.close()