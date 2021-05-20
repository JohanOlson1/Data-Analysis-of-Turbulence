import scipy.io as sio
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
plt.rcParams.update({'font.size': 22})
plt.interactive(True)


re =9.36e+5
viscos =1/re

nk=66
nk=34
dz=0.2/(nk-2)

xy_hump_fine = np.loadtxt("xy_hump_fine.dat")
x1=xy_hump_fine[:,0]
y1=xy_hump_fine[:,1]


nim1=int(x1[0])
njm1=int(y1[0])

ni=nim1+1
nj=njm1+1


x=x1[1:]
y=y1[1:]

x_2d=np.reshape(x,(njm1,nim1))
y_2d=np.reshape(y,(njm1,nim1))

x_2d=np.transpose(x_2d)
y_2d=np.transpose(y_2d)

# compute cell centers
xp2d= np.zeros((ni,nj))
yp2d= np.zeros((ni,nj))

for jj in range (0,nj):
   for ii in range (0,ni):

      im1=max(ii-1,0)
      jm1=max(jj-1,0)

      i=min(ii,nim1-1)
      j=min(jj,njm1-1)


      xp2d[ii,jj]=0.25*(x_2d[i,j]+x_2d[im1,j]+x_2d[i,jm1]+x_2d[im1,jm1])
      yp2d[ii,jj]=0.25*(y_2d[i,j]+y_2d[im1,j]+y_2d[i,jm1]+y_2d[im1,jm1])


#  read v_1 & transform v_1 to a 3D array (file 1)
uvw = sio.loadmat('u1_pans_iddes.mat')
tt=uvw['u1_pans_iddes']
u3d1= np.reshape(tt,(ni,nj,nk))  #v_1 velocity

uvw = sio.loadmat('v1_pans_iddes.mat')
tt=uvw['v1_pans_iddes']
v3d1= np.reshape(tt,(ni,nj,nk))  #v_2 velocity

uvw = sio.loadmat('w1_pans_iddes.mat')
tt=uvw['w1_pans_iddes']
w3d1= np.reshape(tt,(ni,nj,nk))  #v_3 velocity

#  read v_2 & transform v_2 to a 3D array (file 2)
uvw = sio.loadmat('u2_pans_iddes.mat')
tt=uvw['u2_pans_iddes']
u3d2= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v2_pans_iddes.mat')
tt=uvw['v2_pans_iddes']
v3d2= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w2_pans_iddes.mat')
tt=uvw['w2_pans_iddes']
w3d2= np.reshape(tt,(ni,nj,nk))

#  read v_3 & transform v_3 to a 3D array (file 3)
uvw = sio.loadmat('u3_pans_iddes.mat')
tt=uvw['u3_pans_iddes']
u3d3= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v3_pans_iddes.mat')
tt=uvw['v3_pans_iddes']
v3d3= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w3_pans_iddes.mat')
tt=uvw['w3_pans_iddes']
w3d3= np.reshape(tt,(ni,nj,nk))

#  read v_4 & transform v_4 to a 3D array (file 4)
uvw = sio.loadmat('u4_pans_iddes.mat')
tt=uvw['u4_pans_iddes']
u3d4= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v4_pans_iddes.mat')
tt=uvw['v4_pans_iddes']
v3d4= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w4_pans_iddes.mat')
tt=uvw['w4_pans_iddes']
w3d4= np.reshape(tt,(ni,nj,nk))

#  read v_5 & transform v_5 to a 3D array (file 5)
uvw = sio.loadmat('u5_pans_iddes.mat')
tt=uvw['u5_pans_iddes']
u3d5= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v5_pans_iddes.mat')
tt=uvw['v5_pans_iddes']
v3d5= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w5_pans_iddes.mat')
tt=uvw['w5_pans_iddes']
w3d5= np.reshape(tt,(ni,nj,nk))

#  read v_6 & transform v_6 to a 3D array (file 6)
uvw = sio.loadmat('u6_pans_iddes.mat')
tt=uvw['u6_pans_iddes']
u3d6= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v6_pans_iddes.mat')
tt=uvw['v6_pans_iddes']
v3d6= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w6_pans_iddes.mat')
tt=uvw['w6_pans_iddes']
w3d6= np.reshape(tt,(ni,nj,nk))

#  read v_7 & transform v_7 to a 3D array (file 7)
uvw = sio.loadmat('u7_pans_iddes.mat')
tt=uvw['u7_pans_iddes']
u3d7= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v7_pans_iddes.mat')
tt=uvw['v7_pans_iddes']
v3d7= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w7_pans_iddes.mat')
tt=uvw['w7_pans_iddes']
w3d7= np.reshape(tt,(ni,nj,nk))

#  read v_8 & transform v_8 to a 3D array (file 8)
uvw = sio.loadmat('u8_pans_iddes.mat')
tt=uvw['u8_pans_iddes']
u3d8= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('v8_pans_iddes.mat')
tt=uvw['v8_pans_iddes']
v3d8= np.reshape(tt,(ni,nj,nk))

uvw = sio.loadmat('w8_pans_iddes.mat')
tt=uvw['w8_pans_iddes']
w3d8= np.reshape(tt,(ni,nj,nk))

# merge 2 files. This means than new nk = 2*nk
u3d=np.concatenate((u3d1, u3d2, u3d3, u3d4, u3d5, u3d6, u3d7, u3d8), axis=2)
v3d=np.concatenate((v3d1, v3d2, v3d3, v3d4, v3d5, v3d6, v3d7, v3d8), axis=2)
w3d=np.concatenate((w3d1, w3d2, w3d3, w3d4, w3d5, w3d6, w3d7, w3d8), axis=2)

idum,idum,nk=u3d.shape

#u3d = u3d1
#v3d = v3d1
#w3d = w3d1


# x coordinate direction = index 0, first index
# y coordinate direction = index 1, second index
# z coordinate direction = index 2, third index

ni=len(u3d)

#vmean=np.mean(v3d, axis=(2))
#wmean=np.mean(w3d, axis=(2))

dudx= np.zeros((ni,nj,nk))
dudy= np.zeros((ni,nj,nk))

d2udx2= np.zeros((ni,nj,nk))
d2udy2= np.zeros((ni,nj,nk))
d2udxy= np.zeros((ni,nj,nk))

dvdx= np.zeros((ni,nj,nk))
dvdy= np.zeros((ni,nj,nk))

d2vdx2= np.zeros((ni,nj,nk))
d2vdy2= np.zeros((ni,nj,nk))
d2vdxy= np.zeros((ni,nj,nk))

dwdx= np.zeros((ni,nj,nk))
dwdy= np.zeros((ni,nj,nk))

d2wdx2= np.zeros((ni,nj,nk))
d2wdy2= np.zeros((ni,nj,nk))
d2wdxy= np.zeros((ni,nj,nk))

dphidx= np.zeros((ni,nj))
dphidy= np.zeros((ni,nj))
#dummyx= np.zeros((ni,nj))
#dummyy= np.zeros((ni,nj))

# compute x, y gradients ar grid plane k
for k in range (1,nk-1):
   u2d=u3d[:,:,k]
   v2d=v3d[:,:,k]
   w2d=w3d[:,:,k]

# u
   dphidx,dphidy=dphidx_dy(x_2d,y_2d,u2d)
   dudx[:,:,k]=dphidx
   dudy[:,:,k]=dphidy

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidx)
   d2udx2[:,:,k]=dummyx

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidy)
   d2udy2[:,:,k]=dummyy

# v
   dphidx,dphidy=dphidx_dy(x_2d,y_2d,v2d)
   dvdx[:,:,k]=dphidx
   dvdy[:,:,k]=dphidy

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidx)
   d2vdx2[:,:,k]=dummyx

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidy)
   d2vdy2[:,:,k]=dummyy

# w
   dphidx,dphidy=dphidx_dy(x_2d,y_2d,w2d)
   dwdx[:,:,k]=dphidx
   dwdy[:,:,k]=dphidy

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidx)
   d2wdx2[:,:,k]=dummyx

   dummyx,dummyy=dphidx_dy(x_2d,y_2d,dphidy)
   d2wdy2[:,:,k]=dummyy

   print('derivatives computed for plane k',k,'out of',nk-2,'planes')
 

# compute z gradient (note the the command gradient cannot be used
# in x and y direction since the mesh is not Cartesian
# compute 3D instantaneous gradients
dudz=np.gradient(u3d,dz,axis=2)
dvdz=np.gradient(v3d,dz,axis=2)
dwdz=np.gradient(w3d,dz,axis=2)

d2udxz=np.gradient(dudx,dz,axis=2)
d2udyz=np.gradient(dudy,dz,axis=2)
d2udz2=np.gradient(dudz,dz,axis=2)

d2vdxz=np.gradient(dvdx,dz,axis=2)
d2vdyz=np.gradient(dvdy,dz,axis=2)
d2vdz2=np.gradient(dvdz,dz,axis=2)

d2wdxz=np.gradient(dwdx,dz,axis=2)
d2wdyz=np.gradient(dwdy,dz,axis=2)
d2wdz2=np.gradient(dwdz,dz,axis=2)

# U7
s11=dudx
s12=0.5*(dudy+dvdx)
s13=0.5*(dudz+dwdx)
s21=s12
s22=dvdy
s23=0.5*(dvdz+dwdy)
s31=s13
s32=s23
s33=dwdz

ss=(2*(s11**2+s12**2+s13**2+s21**2+s22**2+s23**2+s31**2+s32**2+s33**2)**0.5)

termu=(d2udx2+d2udy2+d2udz2)**2
termv=(d2vdx2+d2vdy2+d2vdz2)**2
termw=(d2wdx2+d2wdy2+d2wdz2)**2

termu_b=d2udx2**2+d2udy2**2+2*d2udxy**2
termv_b=d2vdx2**2+d2vdy2**2+2*d2vdxy**2
termw_b=d2wdx2**2+d2wdy2**2+2*d2wdxy**2

ubis=(termu+termv+termw)**0.5

ubis_b=(termu_b+termv_b+termw_b)**0.5

kappa = 0.41
L_vk3d=kappa*ss/ubis
L_vk3d_b=kappa*ss/ubis_b

L_vk3d_spanz=np.mean(L_vk3d[1:-2,1:-2,1:-2], axis=(2))
L_vk3d_b_spanz=np.mean(L_vk3d_b[1:-2,1:-2,1:-2], axis=(2))

# read mean velocities
# read data file
vectz = np.loadtxt("vectz_fine.dat")
ntstep=vectz[0]
ni=int(vectz[1])
nj=int(vectz[2])
nk=int(vectz[3])
n=len(vectz)

nn=14
nst=3
iu=range(nst+1,n,nn)
iv=range(nst+2,n,nn)
iuu=range(nst+4,n,nn)
ivv=range(nst+5,n,nn)
iww=range(nst+6,n,nn)
iuv=range(nst+7,n,nn)
ik=range(nst+9,n,nn)
ivis=range(nst+10,n,nn)
idiss=range(nst+11,n,nn)

u=vectz[iu]/ntstep
v=vectz[iv]/ntstep
uu=vectz[iuu]/ntstep
vv=vectz[ivv]/ntstep
ww=vectz[iww]/ntstep
uv=vectz[iuv]/ntstep
k_model=vectz[ik]/ntstep
vis=vectz[ivis]/ntstep
diss=vectz[idiss]/ntstep

# uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u**2
vv=vv-v**2
#ww=ww-w**2 no w exists...
uv=uv-u*v

umean_2d=np.reshape(u,(ni,nj))
vmean_2d=np.reshape(v,(ni,nj))
uu_2d=np.reshape(uu,(ni,nj))
uv_2d=np.reshape(uv,(ni,nj))
vv_2d=np.reshape(vv,(ni,nj))
ww_2d=np.reshape(ww,(ni,nj))
k_model_2d=np.reshape(k_model,(ni,nj))
vis_2d=np.reshape(vis,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb
diss_2d=np.reshape(diss,(ni,nj)) 

dudx_mean,dudy_mean=dphidx_dy(x_2d,y_2d,umean_2d)

d2udx2_mean,dummy=dphidx_dy(x_2d,y_2d,dudx_mean)

dummy,d2udy2_mean=dphidx_dy(x_2d,y_2d,dudy_mean)

dvdx_mean,dvdy_mean=dphidx_dy(x_2d,y_2d,vmean_2d)

d2vdx2_mean,dummy=dphidx_dy(x_2d,y_2d,dvdx_mean)

dummy,d2vdy2_mean=dphidx_dy(x_2d,y_2d,dvdy_mean)

s11_mean=dudx_mean
s12_mean=0.5*(dudy_mean+dvdx_mean)
s21_mean=s12_mean
s22_mean=dvdy_mean

ss_mean=(2*(s11_mean**2+s12_mean**2+s21_mean**2+s22_mean**2)**0.5)

termu_mean=(d2udx2_mean+d2udy2_mean)**2
termv_mean=(d2vdx2_mean+d2vdy2_mean)**2

ubis_mean=(termu_mean+termv_mean)**0.5

L_vk1d=kappa*ss_mean/ubis_mean

C_DES = 0.65

L_DES = C_DES*np.maximum(x_2d, y_2d)

k_res = 0.5*(uu_2d + vv_2d + ww_2d)

L_RANS = (k_res**1.5)/diss_2d

L_PANS = (k_model_2d**1.5)/diss_2d

def Lvk_065():
    plt.figure("Figure 065")
    plt.clf() #clear the figure
    xx=0.65;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(y_2d[i1,:], L_vk1d[i1,0:-1],'r-', label = '1D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_spanz[i1,:],'b-', label = '3D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_b_spanz[i1,:],'g-', label = '3D Alt.') 
    plt.plot(y_2d[i1,0:-1], L_DES[i1,0:-1],'k-', label = '$C_{DES}\\Delta$')
    plt.plot(y_2d[i1,:], L_PANS[i1,0:-1], 'y-', label = '$L_{PANS}$')
    plt.plot(y_2d[i1,:], L_RANS[i1,0:-1], 'c-', label = '$L_{RANS}$')
    plt.xlabel("y")
    plt.ylabel("$Length Scales$")
    plt.title("$x=0.65$")
    plt.axis([np.min(y_2d[i1,0]), 0.5, 0, 1])
    plt.legend()
    
def Lvk_080():
    plt.figure("Figure 080")
    plt.clf() #clear the figure
    xx=0.80;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(y_2d[i1,:], L_vk1d[i1,0:-1],'r-', label = '1D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_spanz[i1,:],'b-', label = '3D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_b_spanz[i1,:],'g-', label = '3D Alt.') 
    plt.plot(y_2d[i1,0:-1], L_DES[i1,0:-1],'k-', label = '$C_{DES}\\Delta$')
    plt.plot(y_2d[i1,:], L_PANS[i1,0:-1], 'y-', label = '$L_{PANS}$')
    plt.plot(y_2d[i1,:], L_RANS[i1,0:-1], 'c-', label = '$L_{RANS}$')
    plt.xlabel("y")
    plt.ylabel("$Length Scales$")
    plt.title("$x=0.80$")
    plt.axis([np.min(y_2d[i1,0]), 0.5, 0, 1])
    plt.legend()
    
def Lvk_090():
    plt.figure("Figure 090")
    plt.clf() #clear the figure
    xx=0.90;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(y_2d[i1,:], L_vk1d[i1,0:-1],'r-', label = '1D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_spanz[i1,:],'b-', label = '3D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_b_spanz[i1,:],'g-', label = '3D Alt.') 
    plt.plot(y_2d[i1,0:-1], L_DES[i1,0:-1],'k-', label = '$C_{DES}\\Delta$')
    plt.plot(y_2d[i1,:], L_PANS[i1,0:-1], 'y-', label = '$L_{PANS}$')    
    plt.plot(y_2d[i1,:], L_RANS[i1,0:-1], 'c-', label = '$L_{RANS}$')
    plt.xlabel("y")
    plt.ylabel("$Length Scales$")
    plt.title("$x=0.90$")
    plt.axis([np.min(y_2d[i1,0]), 0.5, 0, 1])
    plt.legend()

def Lvk_130():
    plt.figure("Figure 130")
    plt.clf() #clear the figure
    xx=1.30;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(y_2d[i1,:], L_vk1d[i1,0:-1],'r-', label = '1D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_spanz[i1,:],'b-', label = '3D')
    plt.plot(y_2d[i1,1:-1], L_vk3d_b_spanz[i1,:],'g-', label = '3D Alt.') 
    plt.plot(y_2d[i1,0:-1], L_DES[i1,0:-1],'k-', label = '$C_{DES}\\Delta$')
    plt.plot(y_2d[i1,:], L_PANS[i1,0:-1], 'y-', label = '$L_{PANS}$')    
    plt.plot(y_2d[i1,:], L_RANS[i1,0:-1], 'c-', label = '$L_{RANS}$')
    plt.xlabel("y")
    plt.ylabel("$Length Scales$")
    plt.title("$x=1.30$")
    plt.axis([np.min(y_2d[i1,0]), 0.5, 0, 1])
    plt.legend()

def close_fig():
    plt.close()

root = tk.Tk()
close_button = tk.Button(root, text='Close plot', command = close_fig)
close_button.grid(row=0, column=0)

# V.6

label_overview = tk.Label(text="V.6", background="grey")
label_overview.grid(row=0, column=1, sticky='nesw')

button_Lvk_065 = tk.Button(root, text= 'Lvk_065', command = Lvk_065)
button_Lvk_065.grid(row=1, column=1, sticky='nesw')

button_Lvk_080 = tk.Button(root, text= 'Lvk_080', command = Lvk_080)
button_Lvk_080.grid(row=2, column=1, sticky='nesw')

button_Lvk_090 = tk.Button(root, text= 'Lvk_090', command = Lvk_090)
button_Lvk_090.grid(row=3, column=1, sticky='nesw')

button_Lvk_130 = tk.Button(root, text= 'Lvk_130', command = Lvk_130)
button_Lvk_130.grid(row=4, column=1, sticky='nesw')

root.mainloop()

