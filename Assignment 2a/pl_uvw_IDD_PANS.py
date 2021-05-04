import scipy.io as sio
import numpy as np
import tkinter as tk
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

#----  read v_1 & transform v_1 to a 3D array (file 1)
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


#----  read v_2 & transform v_2 to a 3D array (file 2)
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
   
# ---- Plot

## U2
def plot_mean_velocity_profile():
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    
    plt.semilogx(yplus,umean/ustar,'b-')
    plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
    plt.axis([1, 8000, 0, 31])
    plt.ylabel("$U^+$")
    plt.xlabel("$y^+$")
    
uvmean1= np.mean((u3d-umean[None,:,None])*(v3d-vmean[None,:,None]), axis=(0,2))
uumean = np.mean((u3d-umean[None,:,None])*(u3d-umean[None,:,None]), axis=(0,2))
vvmean = np.mean((v3d-vmean[None,:,None])*(v3d-vmean[None,:,None]), axis=(0,2))
wwmean = np.mean((w3d-wmean[None,:,None])*(w3d-wmean[None,:,None]), axis=(0,2))

te_resolved = 0.5*(uumean + vvmean + wwmean)

## U3

def uv_stress_resolved():
    plt.figure("uv_Stress_resolved")
    plt.plot(yplus, uvmean1)
    plt.title('Resolved uv Stress')
    plt.ylabel("$U^+$")
    plt.xlabel("$y^+$")

## U4
def te_plot():
    plt.figure("Turbulent Kinetic Energy")
    plt.plot(yplus, temean, label='$k_{modelled}$')
    plt.plot(yplus, te_resolved, label='$k_{resolved}$')
    plt.title('Turbulent Kinetic Energy')
    plt.ylabel("$U^+$")
    plt.xlabel("$y^+$")
    plt.legend()
    
## U5
Cmu = 0.09

nu_t = Cmu*np.divide(np.multiply(temean, temean), epsmean)
    
dudx, dudy, dudz=np.gradient(u3d,dx,y,dz)

dvdx, dvdy, dvdz=np.gradient(v3d,dx,y,dz)

dwdx, dwdy, dwdz=np.gradient(w3d,dx,y,dz)

# Time average

dudymean = np.mean(dudy, axis=(0,2))
dvdxmean = np.mean(dvdx, axis=(0,2))

tau12 = - np.multiply(nu_t, dudymean + dvdxmean)

def turbulent_shear():
    plt.figure("Turbulent Shear")
    plt.plot(yplus, tau12, label='$\\tau_{12}$')
    plt.plot(yplus, uvmean1, label='$\\tau_{resolved}$')
    plt.title('Turbulent Kinetic Energy')
    plt.ylabel("Shear")
    plt.xlabel("$y^+$")
    plt.legend()
    
## U6

# some calc.

def boundary_interface():
    plt.figure("boundary interface")

## U7

kappa = 0.41

zeta = 1

# 1D

dumeandy = np.gradient(umean, y)
dumeandy2 = np.gradient(dumeandy, y)

L_v_K_1D = kappa * np.abs(np.divide(dumeandy, dumeandy2))

S_1D = 2*np.multiply(dumeandy2, dumeandy2)

L_1D = (Cmu**(3/4))*np.multiply(np.power(temean, 3/2), epsmean)

T_1_1D = zeta*kappa*np.divide(np.multiply(np.multiply(S_1D,S_1D), L_1D), L_v_K_1D)

# 3D

# u
dudx2, empty1, empty2 = np.gradient(dudx,dx,y,dz)

empty1, dudy2, empty2 = np.gradient(dudy,dx,y,dz)

empty1, empty2, dudz2 = np.gradient(dudz,dx,y,dz)
# v
dvdx2, empty1, empty2 = np.gradient(dvdx,dx,y,dz)

empty1, dvdy2, empty2 = np.gradient(dvdy,dx,y,dz)

empty1, empty2, dvdz2 = np.gradient(dvdz,dx,y,dz)
# w
dwdx2, empty1, empty2 = np.gradient(dwdx,dx,y,dz)

empty1, dwdy2, empty2 = np.gradient(dwdy,dx,y,dz)

empty1, empty2, dwdz2 = np.gradient(dwdz,dx,y,dz)

# Partials
Lambda1 = dudx2 + dudy2 + dudz2
Lambda2 = dvdx2 + dvdy2 + dvdz2
Lambda3 = dwdx2 + dwdy2 + dwdz2

Gamma1 = dudx2 + dudy2 + dudz2
Gamma2 = dvdx2 + dvdy2 + dvdz2
Gamma3 = dwdx2 + dwdy2 + dwdz2


S = np.zeros((ni,nj,nk))
Ubiss_1 = np.zeros((ni,nj,nk))

for i in range(ni):
    for j in range(nj):
        for k in range(nk):
            S[i,j,k] += dudx[i,j,k]*dudx[i,j,k]
            S[i,j,k] += dudy[i,j,k]*dudy[i,j,k]
            S[i,j,k] += dudz[i,j,k]*dudz[i,j,k]
            S[i,j,k] += dvdx[i,j,k]*dvdx[i,j,k]
            S[i,j,k] += dvdy[i,j,k]*dvdy[i,j,k]
            S[i,j,k] += dvdz[i,j,k]*dvdz[i,j,k]
            S[i,j,k] += dwdx[i,j,k]*dwdx[i,j,k]
            S[i,j,k] += dwdy[i,j,k]*dwdy[i,j,k]
            S[i,j,k] += dwdz[i,j,k]*dwdz[i,j,k]
            
            Ubiss_1[i,j,k] += Gamma1[i,j,k]*Lambda1[i,j,k]
            Ubiss_1[i,j,k] += Gamma2[i,j,k]*Lambda2[i,j,k]
            Ubiss_1[i,j,k] += Gamma3[i,j,k]*Lambda3[i,j,k]
            
for i in range(ni):
    for j in range(nj):
        for k in range(nk):
            S[i,j,k] = np.sqrt(2*S[i,j,k])            
            Ubiss_1[i,j,k] = np.sqrt(Ubiss_1[i,j,k]) 

L_v_K_3D = kappa*np.abs(np.divide(S, Ubiss_1))

L = (Cmu**(3/4))*np.multiply(np.power(te3d, 3/2), eps3d)

T_1_3D_1 = zeta*kappa*np.divide(np.multiply(np.multiply(S,S), L), L_v_K_3D) 

# Plot
L_v_K_1D_mean_after = np.mean(L_v_K_3D , axis=(0,2))

def length_scale_compare():
    plt.figure("Length_scale_compare")
    plt.plot(yplus, L_v_K_1D , label='1D before')
    plt.plot(yplus, L_v_K_1D_mean_after , label='1D After')
    plt.title('Length_scale_compare')
    plt.ylabel("Length Scale")
    plt.xlabel("$y^+$")
    plt.legend()
    
# ---- GUI Append

def close_fig():
    plt.close()
    
root = tk.Tk()
close_button = tk.Button(root, text='Close plot', command = close_fig)
close_button.grid(row=0, column=0)


## Overview Plots
# U2
label_overview = tk.Label(text="U2", background="grey")
label_overview.grid(row=0, column=1, sticky='nesw')

button_mean_velocity_profile = tk.Button(root, text= 'Mean velocity Profile v_1', command = plot_mean_velocity_profile)
button_mean_velocity_profile.grid(row=1, column=1, sticky='nesw')

# U3
label_overview = tk.Label(text="U3", background="grey")
label_overview.grid(row=0, column=2, sticky='nesw')

button_uv = tk.Button(root, text= 'uv Resolved', command = uv_stress_resolved)
button_uv.grid(row=1, column=2, sticky='nesw')

# U4
label_overview = tk.Label(text="U4", background="grey")
label_overview.grid(row=0, column=3, sticky='nesw')

button_te = tk.Button(root, text= 'Resolved tubulent Kinetic Energy', command = te_plot)
button_te.grid(row=1, column=3, sticky='nesw')

# U5
label_overview = tk.Label(text="U5", background="grey")
label_overview.grid(row=0, column=4, sticky='nesw')

button_turbulent_shear = tk.Button(root, text= 'Turbulent Shear', command = turbulent_shear)
button_turbulent_shear.grid(row=1, column=4, sticky='nesw')

# U6
label_overview = tk.Label(text="U6", background="grey")
label_overview.grid(row=0, column=5, sticky='nesw')

button_boundary_interface = tk.Button(root, text= 'Boundary Interface DDES', command = boundary_interface)
button_boundary_interface.grid(row=1, column=5, sticky='nesw')

# U7
label_overview = tk.Label(text="U7", background="grey")
label_overview.grid(row=0, column=6, sticky='nesw')

button_length_scale_compare = tk.Button(root, text= 'Length_scale_compare', command = length_scale_compare)
button_length_scale_compare.grid(row=1, column=6, sticky='nesw')

root.mainloop()

