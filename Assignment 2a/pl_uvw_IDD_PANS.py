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

#----  read v_3 & transform v_3 to a 3D array (file 3)
uvw = sio.loadmat('u3_IDD_PANS.mat')
ttu=uvw['u3_IDD_PANS']
u3d3= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v3_IDD_PANS.mat')
tt=uvw['v3_IDD_PANS']
v3d3= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w3_IDD_PANS.mat')
tt=uvw['w3_IDD_PANS']
w3d3= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te3_IDD_PANS.mat')
tt=uvw['te3_IDD_PANS']
te3d3= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps3_IDD_PANS.mat')
tt=uvw['eps3_IDD_PANS']
eps3d3= np.reshape(tt,(nk,nj,ni))


#----  read v_4 & transform v_4 to a 3D array (file 4)
uvw = sio.loadmat('u4_IDD_PANS.mat')
ttu=uvw['u4_IDD_PANS']
u3d4= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v4_IDD_PANS.mat')
tt=uvw['v4_IDD_PANS']
v3d4= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w4_IDD_PANS.mat')
tt=uvw['w4_IDD_PANS']
w3d4= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te4_IDD_PANS.mat')
tt=uvw['te4_IDD_PANS']
te3d4= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps4_IDD_PANS.mat')
tt=uvw['eps4_IDD_PANS']
eps3d4= np.reshape(tt,(nk,nj,ni))


#----  read v_5 & transform v_5 to a 3D array (file 5)
uvw = sio.loadmat('u5_IDD_PANS.mat')
ttu=uvw['u5_IDD_PANS']
u3d5= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v5_IDD_PANS.mat')
tt=uvw['v5_IDD_PANS']
v3d5= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w5_IDD_PANS.mat')
tt=uvw['w5_IDD_PANS']
w3d5= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te5_IDD_PANS.mat')
tt=uvw['te5_IDD_PANS']
te3d5= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps5_IDD_PANS.mat')
tt=uvw['eps5_IDD_PANS']
eps3d5= np.reshape(tt,(nk,nj,ni))


#----  read v_6 & transform v_6 to a 3D array (file 6)
uvw = sio.loadmat('u6_IDD_PANS.mat')
ttu=uvw['u6_IDD_PANS']
u3d6= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v6_IDD_PANS.mat')
tt=uvw['v6_IDD_PANS']
v3d6= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w6_IDD_PANS.mat')
tt=uvw['w6_IDD_PANS']
w3d6= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te6_IDD_PANS.mat')
tt=uvw['te6_IDD_PANS']
te3d6= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps6_IDD_PANS.mat')
tt=uvw['eps6_IDD_PANS']
eps3d6= np.reshape(tt,(nk,nj,ni))


#----  read v_7 & transform v_7 to a 3D array (file 7)
uvw = sio.loadmat('u7_IDD_PANS.mat')
ttu=uvw['u7_IDD_PANS']
u3d7= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v7_IDD_PANS.mat')
tt=uvw['v7_IDD_PANS']
v3d7= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w7_IDD_PANS.mat')
tt=uvw['w7_IDD_PANS']
w3d7= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te7_IDD_PANS.mat')
tt=uvw['te7_IDD_PANS']
te3d7= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps7_IDD_PANS.mat')
tt=uvw['eps7_IDD_PANS']
eps3d7= np.reshape(tt,(nk,nj,ni))


#----  read v_8 & transform v_8 to a 3D array (file 8)
uvw = sio.loadmat('u8_IDD_PANS.mat')
ttu=uvw['u8_IDD_PANS']
u3d8= np.reshape(ttu,(nk,nj,ni))
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

uvw = sio.loadmat('v8_IDD_PANS.mat')
tt=uvw['v8_IDD_PANS']
v3d8= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('w8_IDD_PANS.mat')
tt=uvw['w8_IDD_PANS']
w3d8= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('te8_IDD_PANS.mat')
tt=uvw['te8_IDD_PANS']
te3d8= np.reshape(tt,(nk,nj,ni))

uvw = sio.loadmat('eps8_IDD_PANS.mat')
tt=uvw['eps8_IDD_PANS']
eps3d8= np.reshape(tt,(nk,nj,ni))


# merge 2 files. This means that new ni = 2*ni
u3d=np.concatenate((u3d1, u3d2, u3d3, u3d4, u3d5, u3d6, u3d7, u3d8), axis=0)
v3d=np.concatenate((v3d1, v3d2, v3d3, v3d4, v3d5, v3d6, v3d7, v3d8), axis=0)
w3d=np.concatenate((w3d1, w3d2, w3d3, w3d4, w3d5, w3d6, w3d7, w3d8), axis=0)
te3d=np.concatenate((te3d1, te3d2, te3d3, te3d4, te3d5, te3d6, te3d7, te3d8), axis=0)
eps3d=np.concatenate((eps3d1, eps3d2, eps3d3, eps3d4, eps3d5, eps3d6, eps3d7, eps3d8), axis=0)


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

# ---- U2
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

# ---- U3

def uv_stress_resolved():
    plt.figure("uv_Stress_resolved")
    plt.plot(yplus, uvmean1)
    plt.title('Resolved uv Stress')
    plt.ylabel("$u^\prime v^\prime$")
    plt.xlabel("$y^+$")

# ---- U4
def te_plot():
    plt.figure("Turbulent Kinetic Energy")
    plt.plot(yplus, temean, label='$k_{modelled}$')
    plt.plot(yplus, te_resolved, label='$k_{resolved}$')
    plt.plot(yplus, te_resolved + temean, label='$k_{tot}$')
    plt.title('Turbulent Kinetic Energy')
    plt.ylabel("$k$")
    plt.xlabel("$y^+$")
    plt.legend()

line08 = np.ones(np.size(yplus))
def te_plot_ratio():
    plt.figure("Turbulent Kinetic Energy ratio")
    plt.plot(yplus, te_resolved/(te_resolved + temean))
    plt.plot(yplus, 0.8*line08)
    plt.title('k ratio')
    plt.ylabel("$Ratio$")
    plt.xlabel("$y^+$")
    
# ---- U5
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
    plt.title('Turbulent Shear Ratio')
    plt.ylabel("$\\tau$")
    plt.xlabel("$y^+$")
    plt.legend()

def turbulent_shear_ratio():
    plt.figure("Turbulent Shear Ratio")
    plt.plot(yplus, uvmean1/(tau12 + uvmean1), label='$\\tau_{ratio}$')
    plt.plot(yplus, 0.8*line08)
    plt.title('Turbulent Shear Ratio')
    plt.ylabel("$\\tau$ Ratio")
    plt.xlabel("$y^+$")
    plt.legend()
    
# ---- U6

L_t = np.divide(np.power(temean, 3/2), epsmean)

F_DES = (1/(0.61*dx))*L_t

arg1 = 2*np.divide(L_t, y)
arg2 = 500*viscos*Cmu*np.divide(temean, np.multiply(epsmean,np.power(y, 2)))
eta = np.maximum(arg1, arg2)
F_S = np.tanh(np.power(eta,2))
F_DDES = (1/(0.61*dx))*np.multiply(L_t, 1 - F_S)

def boundary_interface_DES():
    plt.figure("DES")
    plt.plot(yplus, F_DES)
    plt.plot(yplus, np.ones(np.size(yplus)), 'k-')
    plt.title('DES $F_{DES}$')
    plt.ylabel("f")
    plt.xlabel("$y^+$")


def boundary_interface_DDES():
    plt.figure("DDES")
    plt.plot(yplus, F_DDES)
    plt.plot(yplus, np.ones(np.size(yplus)), 'k-')
    plt.title('DDES $F_{DDES}$')
    plt.ylabel("f")
    plt.xlabel("$y^+$")

# ---- U7

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
dudx2, dudxdy, dudxdz = np.gradient(dudx,dx,y,dz)

dudydx, dudy2, dudydz = np.gradient(dudy,dx,y,dz)

dudzdx, dudzdy, dudz2 = np.gradient(dudz,dx,y,dz)
# v
dvdx2, dvdxdy, dvdxdz = np.gradient(dvdx,dx,y,dz)

dvdydx, dvdy2, dvdydz = np.gradient(dvdy,dx,y,dz)

dvdzdx, dvdzdy, dvdz2 = np.gradient(dvdz,dx,y,dz)
# w
dwdx2, dwdxdy, dwdxdz = np.gradient(dwdx,dx,y,dz)

dwdydx, dwdy2, dwdydz = np.gradient(dwdy,dx,y,dz)

dwdzdx, dwdzdy, dwdz2 = np.gradient(dwdz,dx,y,dz)

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

#T_1_3D_1 = zeta*kappa*np.divide(np.multiply(np.multiply(S,S), L), L_v_K_3D) 

# Alternative

Ubiss_2 = np.zeros((ni,nj,nk))

for i in range(ni):
    for j in range(nj):
        for k in range(nk):
            Ubiss_2[i,j,k] += dudx2[i,j,k] + dudxdy[i,j,k] + dudxdz[i,j,k]
            Ubiss_2[i,j,k] += dudydx[i,j,k] + dudy2[i,j,k] + dudydz[i,j,k]
            Ubiss_2[i,j,k] += dudzdx[i,j,k] + dudzdy[i,j,k] + dudz2[i,j,k]
            Ubiss_2[i,j,k] += dvdx2[i,j,k] + dvdxdy[i,j,k] + dvdxdz[i,j,k]
            Ubiss_2[i,j,k] += dvdydx[i,j,k] + dvdy2[i,j,k] + dvdydz[i,j,k]
            Ubiss_2[i,j,k] += dvdzdx[i,j,k] + dvdzdy[i,j,k] + dvdz2[i,j,k]
            Ubiss_2[i,j,k] += dwdx2[i,j,k] + dwdxdy[i,j,k] + dwdxdz[i,j,k]
            Ubiss_2[i,j,k] += dwdydx[i,j,k] + dwdy2[i,j,k] + dwdydz[i,j,k]
            Ubiss_2[i,j,k] += dwdzdx[i,j,k] + dwdzdy[i,j,k] + dwdz2[i,j,k]
            
for i in range(ni):
    for j in range(nj):
        for k in range(nk):
            S[i,j,k] = np.sqrt(2*S[i,j,k])            

L_v_K_3D_alt = kappa*np.abs(np.divide(S, Ubiss_2))

# Plot
L_v_K_3D_mean = np.mean(L_v_K_3D , axis=(0,2))
L_v_K_3D_mean_alt = np.mean(L_v_K_3D_alt , axis=(0,2))

def length_scale_compare():
    plt.figure("Length_scale_comparison")
    plt.plot(yplus, L_v_K_1D , label='1D')
    plt.plot(yplus, L_v_K_3D_mean , label='3D')
    plt.plot(yplus, L_v_K_3D_mean_alt , label='3D Alternative')
    plt.title('Length_scale_compare')
    plt.axis([0, 5200, 0, 0.5])
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

button_te_ratio = tk.Button(root, text= 'Tubulent Kinetic Energy resolved ratio', command = te_plot_ratio)
button_te_ratio.grid(row=2, column=3, sticky='nesw')

# U5
label_overview = tk.Label(text="U5", background="grey")
label_overview.grid(row=0, column=4, sticky='nesw')

button_turbulent_shear = tk.Button(root, text= 'Turbulent Shear', command = turbulent_shear)
button_turbulent_shear.grid(row=1, column=4, sticky='nesw')

button_turbulent_shear_ratio = tk.Button(root, text= 'Turbulent Shear Ratio', command = turbulent_shear_ratio)
button_turbulent_shear_ratio.grid(row=2, column=4, sticky='nesw')

# U6
label_overview = tk.Label(text="U6", background="grey")
label_overview.grid(row=0, column=5, sticky='nesw')

button_boundary_interface_DES = tk.Button(root, text= 'Boundary Interface DES', command = boundary_interface_DES)
button_boundary_interface_DES.grid(row=1, column=5, sticky='nesw')

button_boundary_interface_DDES = tk.Button(root, text= 'Boundary Interface DDES', command = boundary_interface_DDES)
button_boundary_interface_DDES.grid(row=2, column=5, sticky='nesw')

# U7
label_overview = tk.Label(text="U7", background="grey")
label_overview.grid(row=0, column=6, sticky='nesw')

button_length_scale_compare = tk.Button(root, text= 'Length_scale_compare', command = length_scale_compare)
button_length_scale_compare.grid(row=1, column=6, sticky='nesw')

root.mainloop()

