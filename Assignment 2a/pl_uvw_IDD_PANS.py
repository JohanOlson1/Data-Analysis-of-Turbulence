import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

plt.interactive(True)


dx=0.05
dz=0.025

ni=34
nj=49
nk=34


viscos=1./5200.

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_1 & transform v_1 to a 3D array (file 1)
uvw = sio.loadmat('u1_IDD_PANS.mat')
ttu=uvw['u1_IDD_PANS']
u3d1= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v1_IDD_PANS.mat')
tt=uvw['v1_IDD_PANS']
v3d1= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w1_IDD_PANS.mat')
tt=uvw['w1_IDD_PANS']
w3d1= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te1_IDD_PANS.mat')
tt=uvw['te1_IDD_PANS']
te3d1= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps1_IDD_PANS.mat')
tt=uvw['eps1_IDD_PANS']
eps3d1= np.reshape(tt,(nk,nj,ni))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_2 & transform v_2 to a 3D array (file 2)
uvw = sio.loadmat('u2_IDD_PANS.mat')
ttu=uvw['u2_IDD_PANS']
u3d2= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v2_IDD_PANS.mat')
tt=uvw['v2_IDD_PANS']
v3d2= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w2_IDD_PANS.mat')
tt=uvw['w2_IDD_PANS']
w3d2= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te2_IDD_PANS.mat')
tt=uvw['te2_IDD_PANS']
te3d2= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps2_IDD_PANS.mat')
tt=uvw['eps2_IDD_PANS']
eps3d2= np.reshape(tt,(nk,nj,ni))


# merge 2 files. This means that new ni = 2*ni
u3d=np.concatenate((u3d1, u3d2), axis=0)
v3d=np.concatenate((v3d1, v3d2), axis=0)
w3d=np.concatenate((w3d1, w3d2), axis=0)
te3d=np.concatenate((te3d1, te3d2), axis=0)
eps3d=np.concatenate((eps3d1, eps3d2), axis=0)


# x coordinate direction = index 0, first index
# y coordinate direction = index 1, second index
# z coordinate direction = index 2, third index



ni=len(u3d)

x=dx*ni
z=dz*nk


umean=np.mean(u3d, axis=(0,2))
vmean=np.mean(v3d, axis=(0,2))
wmean=np.mean(w3d, axis=(0,2))
temean=np.mean(te3d, axis=(0,2))
epsmean=np.mean(eps3d, axis=(0,2))

# face coordinates
yc = np.loadtxt("yc.dat")

# cell cener coordinates
y= np.zeros(nj)
dy=np.zeros(nj)
for j in range (1,nj-1):
# dy = cell width
   dy[j]=yc[j]-yc[j-1]
   y[j]=0.5*(yc[j]+yc[j-1])

y[nj-1]=yc[nj-1]
tauw=viscos*umean[1]/y[1]
ustar=tauw**0.5
yplus=y*ustar/viscos

DNS=np.genfromtxt("LM_Channel_5200_mean_prof.dat", dtype=None,comments="%")
y_DNS=DNS[:,0]
yplus_DNS=DNS[:,1]
u_DNS=DNS[:,2]

DNS=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat", dtype=None,comments="%")

u2_DNS=DNS[:,2]
v2_DNS=DNS[:,3]
w2_DNS=DNS[:,4]
uv_DNS=DNS[:,5]

k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2

############################### U
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.semilogx(yplus,umean/ustar,'b-')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.axis([1, 8000, 0, 31])
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")

plt.savefig('u_log_zonal_python.eps',bbox_inches='tight')
