# MTF271 - Assignment 1
# Johan Olson, Alexander Rodin
import tkinter as tk
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

# import a lot of daataa and shuffle
# ---- Import data
x=tec[:,0]
y=tec[:,1]
p=tec[:,2]
u=tec[:,3]
v=tec[:,4]
uu=tec[:,5]
vv=tec[:,6]
ww=tec[:,7]
uv=tec[:,8]
k=0.5*(uu+vv+ww)

if max(y) == 1.:
   ni=170
   nj=194
   nu=1./10000.
else:
   nu=1./10595.
   if max(x) > 8.:
     nj=162
     ni=162
   else:
     ni=402
     nj=162

viscos=nu

u2d=np.reshape(u,(nj,ni))
v2d=np.reshape(v,(nj,ni))
p2d=np.reshape(p,(nj,ni))
x2d=np.reshape(x,(nj,ni))
y2d=np.reshape(y,(nj,ni))
uu2d=np.reshape(uu,(nj,ni)) #=mean{v'_1v'_1}
uv2d=np.reshape(uv,(nj,ni)) #=mean{v'_1v'_2}
vv2d=np.reshape(vv,(nj,ni)) #=mean{v'_2v'_2}
k2d=np.reshape(k,(nj,ni))

u2d=np.transpose(u2d)
v2d=np.transpose(v2d)
p2d=np.transpose(p2d)
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)
uu2d=np.transpose(uu2d)
vv2d=np.transpose(vv2d)
uv2d=np.transpose(uv2d)
k2d=np.transpose(k2d)

# set periodic b.c on west boundary
#u2d[0,:]=u2d[-1,:]
#v2d[0,:]=v2d[-1,:]
#p2d[0,:]=p2d[-1,:]
#uu2d[0,:]=uu2d[-1,:]

# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("k_eps_RANS.dat")
k_RANS=k_eps_RANS[:,0]
diss_RANS=k_eps_RANS[:,1]
vist_RANS=k_eps_RANS[:,2]

ntstep=k_RANS[0]

k_RANS_2d=np.reshape(k_RANS,(ni,nj))/ntstep
diss_RANS_2d=np.reshape(diss_RANS,(ni,nj))/ntstep
vist_RANS_2d=np.reshape(vist_RANS,(ni,nj))/ntstep

# set small values on k & eps at upper and lower boundaries to prevent NaN on division
diss_RANS_2d[:,0]= 1e-10
k_RANS_2d[:,0]= 1e-10
vist_RANS_2d[:,0]= nu
diss_RANS_2d[:,-1]= 1e-10
k_RANS_2d[:,-1]= 1e-10
vist_RANS_2d[:,-1]= nu

# set Neumann of p at upper and lower boundaries
p2d[:,0]=p2d[:,1]
p2d[:,-1]=p2d[:,-1-1]

# Set Neumann u2d at upper
u2d[:,-1]=u2d[:,-2]

uu2d[:,-1] = uu2d[:,-2]

# x and y are of the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("xc_yc.dat")
xf=xc_yc[:,0]
yf=xc_yc[:,1]
xf2d=np.reshape(xf,(nj,ni))
yf2d=np.reshape(yf,(nj,ni))
xf2d=np.transpose(xf2d)
yf2d=np.transpose(yf2d)

# delete last row
xf2d = np.delete(xf2d, -1, 0)
yf2d = np.delete(yf2d, -1, 0)
# delete last columns
xf2d = np.delete(xf2d, -1, 1)
yf2d = np.delete(yf2d, -1, 1)

# compute the gradient dudx, dudy at point P
dudx = np.zeros((ni,nj))
dudy = np.zeros((ni,nj))
dvdx = np.zeros((ni,nj))
dvdy = np.zeros((ni,nj))

dudx, dudy = dphidx_dy(xf2d,yf2d,u2d)
dvdx, dvdy = dphidx_dy(xf2d,yf2d,v2d)


# ---- Plots 
################################ Velocity Field
def plot_velocity_field():
    plt.figure("Figure velocity")
    k=10    # plot every tenth
    plt.quiver(x2d[::k,::k],y2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.005)
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("vector plot")
    plt.colorbar()
    plt.tight_layout()        
    plt.show() 

################################ Pressure Plot
def plot_pressure():
    plt.figure("Figure p")
    plt.contourf(x2d,y2d,p2d, 50)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("contour pressure plot")
    plt.colorbar()
    plt.tight_layout()
    plt.show() 

################################# Example Turbulent energy 
def plot_k():
    plt.figure("Figure k")
    plt.contourf(x2d,y2d,k_RANS_2d, 50)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("Contour k RANS plot")
    plt.colorbar()
    plt.tight_layout()
    plt.show() 

################################ Example uu plot
def plot_uu_stress():
    plt.figure("Figure uu Stress")
    plt.contourf(x2d,y2d,uu2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel('$\overline{u^\prime u^\prime}$')
    plt.ylabel("$y$")
    plt.title("uu counter plot")
    plt.colorbar()
    plt.tight_layout()
    plt.show() 

################################ Example vv plot
def plot_vv_stress():
    plt.figure("Figure vv Stress")
    plt.contourf(x2d,y2d,vv2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel('$\overline{v^\prime v^\prime}$')
    plt.ylabel("$y$")
    plt.title("vv counter plot")
    plt.colorbar()
    plt.tight_layout()
    plt.show() 

################################ Example uv plot
def plot_uv_Stress():
    plt.figure("Figure uv Stress")
    plt.contourf(x2d,y2d,uv2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel('$\overline{u^\prime v^\prime}$')
    plt.ylabel("$y$")
    plt.title("uv counter plot")
    plt.colorbar()
    plt.tight_layout()
    plt.show() 

# ---- A_1.1)

# Line close to inlet
def plot_1_close():    
    plt.figure("Figure 1.1a")
    i=5 # 0.09862734
    plt.plot(uu2d[i,:],y2d[i,:],'b-', label='$\overline{u^{\prime 2}}$')
    plt.plot(uv2d[i,:],y2d[i,:],'r-', label='$\overline{u^\prime v^\prime}$')
    plt.plot(vv2d[i,:],y2d[i,:],'k-', label='$\overline{v^{\prime 2}}$')
    plt.xlabel('$\overline{u_i^\prime u_j^\prime}$')
    plt.ylabel('y/H')
    plt.title('1.1 (x = 0.10) vertical line', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show() 

# Line at turbulence
def plot_1_turb():
    plt.figure("Figure 1.1b")
    i=50 # 1.069507
    plt.plot(uu2d[i,:],y2d[i,:],'b-', label='$\overline{u^{\prime 2}}$')
    plt.plot(uv2d[i,:],y2d[i,:],'r-', label='$\overline{u^\prime v^\prime}$')
    plt.plot(vv2d[i,:],y2d[i,:],'k-', label='$\overline{v^{\prime 2}}$')
    plt.xlabel('$\overline{u_i^\prime u_j^\prime}$')
    plt.ylabel('y/H')
    plt.title('1.1 (x = 1.07) vertical line', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show() 

# ---- A_1.2) 
# Eq. (R.1)
# Convection Terms 
du_udx, du_udy = dphidx_dy(xf2d,yf2d, np.multiply(u2d,u2d))
du_vdx, du_vdy = dphidx_dy(xf2d,yf2d, np.multiply(u2d,v2d))
dv_vdx, dv_vdy = dphidx_dy(xf2d,yf2d, np.multiply(v2d,v2d))

# Pressure Terms
dpdx,   dpdy   = dphidx_dy(xf2d,yf2d, p2d)

# Diffusion Terms
dudx,   dudy   = dphidx_dy(xf2d,yf2d, u2d)
du2dx2, NaN1   = dphidx_dy(xf2d,yf2d, dudx)
NaN2,   du2dy2 = dphidx_dy(xf2d,yf2d, dudy)

dvdx,   dvdy   = dphidx_dy(xf2d,yf2d, v2d)
dv2dx2, NaN3   = dphidx_dy(xf2d,yf2d, dvdx)
NaN4,   dv2dy2 = dphidx_dy(xf2d,yf2d, dvdy)

# Turbulence Terms
duudx,  duudy  = dphidx_dy(xf2d,yf2d, uu2d)
duvdx,  duvdy  = dphidx_dy(xf2d,yf2d, uv2d)
dvvdx,  dvvdy  = dphidx_dy(xf2d,yf2d, vv2d)

    ## Close to inlet
    # x-dir
def plot_2_close_x():
    plt.figure("Figure 1.2a")
    i=5
    plt.plot(du_udx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(du_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdx[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duudx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*du2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(duvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size $v_x$-direction (Eq. R.1)')
    plt.ylabel('y')
    plt.title('1.2 (x = 0.10) vertical line', fontsize=20)
    plt.axis([-0.2, 0.2, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    # y-dir
def plot_2_close_y():
    plt.figure("Figure 1.2b")
    i=5
    plt.plot(du_vdx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(dv_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdy[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duvdx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*dv2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(dvvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size $v_y$-direction (Eq. R.1)')
    plt.ylabel('y')
    plt.title('1.2 (x = 0.10) vertical line', fontsize=20)
    plt.axis([-0.1, 0.1, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    ## Turbulent Region
    # x-dir
def plot_2_turb_x():    
    plt.figure("Figure 1.2c")
    i=50
    plt.plot(du_udx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(du_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdx[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duudx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*du2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(duvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size $v_x$-direction (Eq. R.1)')
    plt.ylabel('y')
    plt.title('1.2 (x = 1.07) vertical line', fontsize=20)
    plt.axis([-0.25, 0.25, np.min(y2d), 1])
    plt.legend()
    plt.tight_layout()
    
    # y-dir
def plot_2_turb_y():    
    plt.figure("Figure 1.2d")
    i=50
    plt.plot(du_vdx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(dv_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdy[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duvdx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*dv2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(dvvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size (Eq. R.1)')
    plt.ylabel('y')
    plt.title('1.2 $v_y$-direction (x = 1.07) vertical line', fontsize=20)
    plt.axis([-0.45, 0.28, np.min(y2d), 1])
    plt.legend()
    plt.tight_layout()
    
    ## Zoom in; Turbulent Region
    # x-dir
def plot_2_turb_x_zoom():
    plt.figure("Figure 1.2e")
    i=50
    plt.plot(du_udx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(du_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdx[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duudx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*du2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(duvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size (Eq. R.1)')
    plt.ylabel('y/H')
    plt.title('1.2 $v_x$-direction (x = 1.07) vertical line', fontsize=20)
    plt.axis([-0.3, 0.25, np.min(y2d), 0.1])
    plt.legend()
    plt.tight_layout()

    # y-dir
def plot_2_turb_y_zoom(): 
    plt.figure("Figure 1.2f")
    i=50
    plt.plot(du_vdx[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(dv_vdy[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(dpdy[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(-nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
    plt.plot(duvdx[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-nu*dv2dy2[i,:],y2d[i,:],'k-',label='6')
    plt.plot(dvvdy[i,:],y2d[i,:],'c-',   label='7')
    plt.xlabel('Term size $v_y$-direction (Eq. R.1)')
    plt.ylabel('y/H')
    plt.title('1.2 (x = 1.07) vertical line', fontsize=20)
    plt.axis([-0.45, 0.28, np.min(y2d), 0.1])
    plt.legend()
    plt.tight_layout()

# ---- A_1.3)

# Last part, part of next Question
F_N = np.zeros((ni,nj,2))
F_S = np.zeros((ni,nj,2))

F_N[:,:,0] = -duudx
F_N[:,:,1] = -dvvdy

F_S[:,:,0] = -duvdy
F_S[:,:,1] = -duvdx

# Normalize N
F_N_x_norm = np.divide(F_N[:,:,0], np.sqrt(F_N[:,:,0]**2 + F_N[:,:,1]**2))
F_N_y_norm = np.divide(F_N[:,:,1], np.sqrt(F_N[:,:,0]**2 + F_N[:,:,1]**2))
# Normalize S
F_S_x_norm = np.divide(F_S[:,:,0], np.sqrt(F_S[:,:,0]**2 + F_S[:,:,1]**2))
F_S_y_norm = np.divide(F_S[:,:,1], np.sqrt(F_S[:,:,0]**2 + F_S[:,:,1]**2))

def plot_3_normal_stress():    
    plt.figure("Figure 1.3a")
    k=10# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], F_N_x_norm[::k,::k], F_N_y_norm[::k,::k], scale=10)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.3) Normal Stresses Vector Plot", fontsize=20)
    plt.tight_layout()
    
    # Zooom
def plot_3_normal_stress_zoom():
    plt.figure("Figure 1.3b")
    k=10# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], F_N_x_norm[::k,::k], F_N_y_norm[::k,::k], width = 0.007, scale=13)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.3) Zoom in, Normal Stresses Vector Plot", fontsize=20)
    plt.axis([0, 2, -0.3, 0.2])
    plt.tight_layout()

# ---- A_1.4) 
    # F_S
def plot_4_shear_stress():    
    plt.figure("Figure 1.4a")
    k=10# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], F_S_x_norm[::k,::k], F_S_y_norm[::k,::k],width=0.005)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.4) Shear Stress Vector Plot", fontsize=20)
    plt.tight_layout()    
    
    # Zooom
def plot_4_shear_stress_zoom():    
    plt.figure("Figure 1.4b")
    k=10# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], F_S_x_norm[::k,::k], F_S_y_norm[::k,::k], width = 0.007, scale=13)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.4) Zoom in, Shear Stress Vector Plot", fontsize=20)
    plt.axis([0, 2, -0.3, 0.2])
    plt.tight_layout()


dp_norm = np.sqrt(np.multiply(dpdx,dpdx) + np.multiply(dpdy,dpdy))
#dpdx[:,0] = np.divide(dpdx[:,0], dp_norm[:,0])
#dpdy[:,0] = np.divide(dpdy[:,0], dp_norm[:,0])
    
    # Pressure
def plot_4_pressure_grad():    
    plt.figure("Figure 1.4c")
    k=10# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], -dpdx[::k,::k], -dpdy[::k,::k], width = 0.007)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.4) Pressure Gradient Vector Plot", fontsize=20)
    plt.axis([0, 2, -0.3, 0.2])
    plt.tight_layout()
    
    # Viscous x
def plot_4_Visc_D_x():    
    plt.figure("Figure 1.4d")
    k=20# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], nu*du2dx2[::k,::k], nu*dv2dx2[::k,::k], width = 0.005, scale = 0.05)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.4) Viscous Diffusion x", fontsize=20)
    plt.tight_layout()
    
    # Viscous y
def plot_4_Visc_D_y():
    plt.figure("Figure 1.4e")
    k=20# plot every
    plt.quiver(x2d[::k,::k], y2d[::k,::k], nu*du2dy2[::k,::k], nu*dv2dy2[::k,::k], width = 0.005, scale = 1)
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("1.4) Viscous Diffusion y", fontsize=20)
    plt.tight_layout()

# ---- A_1.5)

# Production term
P_k = np.zeros((ni,nj,6))

P_k[:,:,0] = - np.multiply(uu2d,dudx)
P_k[:,:,1] = - np.multiply(uv2d,dudy)
P_k[:,:,2] = - np.multiply(uv2d,dvdx)
P_k[:,:,3] = - np.multiply(vv2d,dvdy)
P_k[:,:,4] = P_k[:,:,0] + P_k[:,:,1] + P_k[:,:,2] + P_k[:,:,3]
P_k[:,:,5] = P_k[:,:,0] + P_k[:,:,3]
 
# Close to inlet
def plot_5_k_close():
    plt.figure("Figure 1.5a")
    i=5
    plt.plot(P_k[i,:,0],y2d[i,:],'g-',label='$P_{11}$')
    plt.plot(P_k[i,:,1],y2d[i,:],'b-',label='$P_{12}$')
    plt.plot(P_k[i,:,2],y2d[i,:],'r-',label='$P_{21}$')
    plt.plot(P_k[i,:,3],y2d[i,:],'y-',label='$P_{22}$')
    plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Total')
    plt.xlabel('Turbulent Energy Production')
    plt.ylabel('y/H')
    plt.title('1.5) x = 0.10 vertical line', fontsize=20)
    plt.axis([-0.02, 0.02, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    # At recirculation
def plot_5_k_turb():
    plt.figure("Figure 1.5b")
    i=50
    plt.plot(P_k[i,:,0],y2d[i,:],'g-',label='$P_{11}$')
    plt.plot(P_k[i,:,1],y2d[i,:],'b-',label='$P_{12}$')
    plt.plot(P_k[i,:,2],y2d[i,:],'r-',label='$P_{21}$')
    plt.plot(P_k[i,:,3],y2d[i,:],'y-',label='$P_{22}$')
    plt.plot(P_k[i,:,4],y2d[i,:],'k-',label='Total')
    plt.xlabel('Turbulent Energy Production')
    plt.ylabel('y/H')
    plt.title('1.5) x = 1.07 vertical line', fontsize=20)
    plt.axis([-0.03, 0.045, np.min(y2d), 1])
    plt.legend()
    plt.tight_layout()

# ---- A_1.6)
    
    # Close to inlet, Compare with data for Eps
def plot_6_prod_close():    
    plt.figure("Figure 1.6a")
    i=5
    plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Production')
    plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
    plt.xlabel('Turbulent Energy')
    plt.ylabel('y')
    plt.title('1.6) x = 0.10 vertical line', fontsize=20)
    plt.axis([-0.05, 0.05, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    # At recircualtion, Compare with data foe Eps
def plot_6_prod_turb():    
    plt.figure("Figure 1.6b")
    i=50
    plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Production')
    plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
    plt.xlabel('Turbulent Energy')
    plt.ylabel('y')
    plt.title('1.6) x = 1.07 vertical line', fontsize=20)
    plt.axis([-0.05, 0.05, np.min(y2d), 1])
    plt.legend()
    plt.tight_layout()

# ---- A_1.7)
i=50
Cmu = 0.09; C1 = 1.5; C2 = 0.6; C1w = 0.5; C2w = 0.3; sigma_k = 1; rho = 1;

# Convection Term
# 11
duuudx, empty  = dphidx_dy(xf2d,yf2d, np.multiply(u2d, uu2d))
empty,  dvuudy = dphidx_dy(xf2d,yf2d, np.multiply(v2d, uu2d))
# 12
duuvdx, empty  = dphidx_dy(xf2d,yf2d, np.multiply(u2d, uv2d))
empty,  dvuvdy = dphidx_dy(xf2d,yf2d, np.multiply(v2d, uv2d))
# 22
duvvdx, empty  = dphidx_dy(xf2d,yf2d, np.multiply(u2d, vv2d))
empty,  dvvvdy = dphidx_dy(xf2d,yf2d, np.multiply(v2d, vv2d))

C11 = duuudx + dvuudy
C12 = duuvdx + dvuvdy
C22 = duvvdx + dvvvdy

# Viscous Diffusion
# 11
duudx2, empty = dphidx_dy(xf2d,yf2d, duudx)
empty,  duudy2 = dphidx_dy(xf2d,yf2d, duudy)
# 12
duvdx2, empty = dphidx_dy(xf2d,yf2d, duvdx)
empty,  duvdy2 = dphidx_dy(xf2d,yf2d, duvdy)
# 22
dvvdx2, empty = dphidx_dy(xf2d,yf2d, dvvdx)
empty, dvvdy2 = dphidx_dy(xf2d,yf2d, dvvdy)

D11v = nu*duudx2 + nu*duudy2
D12v = nu*duvdx2 + nu*duvdy2
D22v = nu*dvvdx2 + nu*dvvdy2

# Production
# 11
P11 = -2*(np.multiply(uu2d, dudx) + np.multiply(uv2d, dudy))
# 12
P12 = -(np.multiply(uu2d, dvdx) + np.multiply(vv2d, dudy) + np.multiply(uv2d, dudx + dvdy))
# 22
P22 = -2*(np.multiply(uv2d, dvdx) + np.multiply(vv2d, dvdy)) 

# Pressure-Strain
## 11
Phi121 = -C1*rho*np.multiply(np.divide(diss_RANS_2d,k_RANS_2d), uu2d-(2/3)*k_RANS_2d)
#
Phi122 = -C2*rho*(P11-(2/3)*P_k[i,:,5])

## 12
Phi111 = -C1*rho*np.multiply(np.divide(diss_RANS_2d,k_RANS_2d), uv2d)
#
Phi112 = -C2*rho*P12

## 22
Phi221 = -C1*rho*np.multiply(np.divide(diss_RANS_2d,k_RANS_2d), vv2d-(2/3)*k_RANS_2d)
#
Phi222 = -C2*rho*(P22-(2/3)*P_k[i,:,5])

Phi11  = Phi111 + Phi112
Phi12  = Phi121 + Phi122
Phi22  = Phi221 + Phi222

## Turbulent Diffusion
nu_t = Cmu*np.divide(np.multiply(k_RANS_2d, k_RANS_2d), diss_RANS_2d)
# 11 
D111, empty = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, duudx))
empty, D112 = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, duudy))
# 12
D121, empty = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, duvdx))
empty, D122 = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, duvdy))
# 22 
D221, empty = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, dvvdx))
empty, D222 = dphidx_dy(xf2d,yf2d, np.multiply(vist_RANS_2d/sigma_k, dvvdy))

D11t = D111 + D112
D12t = D121 + D122
D22t = D221 + D222

## Dissipation
# 11
Eps11 = (2/3)*diss_RANS_2d
# 12
Eps12 = np.zeros((ni,nj))
# 22
Eps22 = (2/3)*diss_RANS_2d

## Pressure Strain Wall

## west face
# parallell vector
sx_w=np.diff(x2d,axis=1) 
sy_w=np.diff(y2d,axis=1) 
   
# duplicate last column and put it at the end
sx_w=np.insert(sx_w,-1,sx_w[:,-1],axis=1)
sy_w=np.insert(sy_w,-1,sy_w[:,-1],axis=1)

# normalize
d=np.sqrt(sx_w**2+sy_w**2)
nx_w=sy_w/d # approx. west face as east (small CV) TODO: fix afterwards
ny_w=-sx_w/d

# normalvector products
n1n1 = np.multiply(nx_w, nx_w)
n1n2 = np.multiply(nx_w, ny_w)
n2n2 = np.multiply(ny_w, ny_w)

f = np.minimum((1/2.55)*np.divide(np.divide(np.power(k_RANS_2d,3/2),diss_RANS_2d), d), np.ones((ni,nj)))

K1 = np.multiply(f, C1w*np.divide(diss_RANS_2d, k_RANS_2d))
K2 = C2w*f

Phi111W = K1*( - 2*np.multiply(uu2d,n1n1) - np.multiply(uv2d,n1n2) + np.multiply(vv2d,n2n2))
Phi112W = K2*( - 2*np.multiply(Phi112, n1n1) - np.multiply(Phi122, n1n2) + np.multiply(Phi222,n2n2))
Phi121W = -1.5*K1*(2*np.multiply(k_RANS_2d, n1n1) + np.multiply(uv2d, n1n1 + n2n2))
Phi122W = -1.5*K2*(np.multiply(n1n2, Phi112 + Phi222) + np.multiply(Phi122, n1n1+n2n2))

#Phi11  += Phi111W + Phi112W
#Phi12  += Phi121W + Phi122W

# Total

Total_Stress11 = -C11 + D11v + P11 + Phi11 + D11t - Eps11
Total_Stress12 = -C12 + D12v + P12 + Phi12 + D12t - Eps12
Total_Stress22 = -C22 + D22v + P22 + Phi22 + D22t - Eps22
    
    # Plot Close to inlet 11
def plot_7_close_11():    
    plt.figure("Figure 1.7a")
    i=5
    plt.plot(-C11[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(D11v[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(P11[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(Phi11[i,:],y2d[i,:],'y-',label='4')
    plt.plot(D11t[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-Eps11[i,:],y2d[i,:],'k-',label='6')
    plt.plot(Total_Stress11[i,:],y2d[i,:],'c-',label='7')
    plt.xlabel('Reynold Stress terms 11')
    plt.ylabel('y')
    plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
    plt.axis([-0.03, 0.03, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    # Plot Close to inlet 12
def plot_7_close_12():    
    plt.figure("Figure 1.7b")
    i=5
    plt.plot(-C12[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(D12v[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(P12[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(Phi12[i,:],y2d[i,:],'y-',label='4')
    plt.plot(D12t[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-Eps12[i,:],y2d[i,:],'k-',label='6')
    plt.plot(Total_Stress12[i,:],y2d[i,:],'c-',label='7')
    plt.xlabel('Reynold Stress Terms 12')
    plt.ylabel('y')
    plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
    plt.axis([-0.06, 0.06, 0, 1])
    plt.legend()
    plt.tight_layout()
    
    # Plot Close to inlet 22
def plot_7_close_22():
    plt.figure("Figure 1.7c")
    i=5
    plt.plot(-C22[i,:],y2d[i,:],'b-',   label='1')
    plt.plot(D22v[i,:],y2d[i,:],'g-',   label='2')
    plt.plot(P22[i,:],y2d[i,:],'r-',    label='3')
    plt.plot(Phi22[i,:],y2d[i,:],'y-',label='4')
    plt.plot(D22t[i,:],y2d[i,:],'m-',   label='5')
    plt.plot(-Eps22[i,:],y2d[i,:],'k-',label='6')
    plt.plot(Total_Stress22[i,:],y2d[i,:],'c-',label='7')
    plt.xlabel('Reynold Stress terms 22')
    plt.ylabel('y')
    plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
    plt.axis([-0.01, 0.02, 0, 1])
    plt.legend()
    plt.tight_layout()

# ---- A_1.8)
Bouss11 = np.multiply(-nu_t, 2*dudx) + (2/3)*k_RANS_2d
Bouss12 = np.multiply(-nu_t, (dudy + dvdx))
Bouss22 = np.multiply(-nu_t, 2*dvdy) + (2/3)*k_RANS_2d

def plot_8a():
    plt.figure("1.8a")
    plt.contourf(x2d,y2d, Bouss11, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. uu")
    plt.colorbar()
    plt.tight_layout()

def plot_8b():    
    plt.figure("1.8b")
    plt.contourf(x2d,y2d, Bouss12, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. uv")
    plt.colorbar()
    plt.tight_layout()

def plot_8c():
    plt.figure("1.8c")
    plt.contourf(x2d,y2d, Bouss22, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. vv")
    plt.colorbar()
    plt.tight_layout()
    
def plot_8a_zoom():
    plt.figure("1.8a")
    plt.contourf(x2d,y2d, Bouss11, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. uu")
    plt.axis([0, 3, -0.15, 0.2])
    plt.colorbar()
    plt.tight_layout()

def plot_8b_zoom():    
    plt.figure("1.8b")
    plt.contourf(x2d,y2d, Bouss12, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. uv")
    plt.axis([0, 3, -0.15, 0.2])
    plt.colorbar()
    plt.tight_layout()

def plot_8c_zoom():
    plt.figure("1.8c")
    plt.contourf(x2d,y2d, Bouss22, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
    plt.xlabel('$x$')
    plt.ylabel("$y$")
    plt.title("Bouss Approx. vv")
    plt.axis([0, 3, -0.15, 0.2])
    plt.colorbar()
    plt.tight_layout()

# ---- A_1.9)
P_tot = P11 + P22
    
def plot_9a():
    plt.figure("Figure 1.9a")
    plt.contourf(x2d,y2d, P_tot, 50)
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("Production")
    plt.colorbar()

def plot_9b():    
    plt.figure("Figure 1.9b")
    plt.contourf(x2d,y2d, P12, 50)
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("Robin Hood")
    plt.colorbar()
    
def plot_9c():    
    plt.figure("Figure 1.9c")
    plt.contourf(x2d,y2d, P_tot, 50, levels = np.linspace(np.min(P_tot),0,50))
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.xlabel("$x$"); plt.ylabel("$y$")
    plt.title("Negative Production")
    plt.colorbar()

# ---- A_1.10)
Eigenvalues = np.zeros((ni,nj,2))

BigSMatrix = np.zeros((ni,nj,2,2))

s11 = 0.5*(dudx + dudx)
s12 = 0.5*(dudy + dvdx)
s21 = 0.5*(dudy + dvdx)
s22 = 0.5*(dvdy + dvdy)

s_tot = np.multiply(s11,s11) + np.multiply(s12,s12) + np.multiply(s21,s21) + np.multiply(s22,s22)

for i in range(ni):
    for j in range(nj):
        S = np.array([[s11[i,j], s12[i,j]], [s21[i,j], s22[i,j]]])
        eigs, empty = np.linalg.eig(S)
        Eigenvalues[i,j,0] = eigs[0]
        Eigenvalues[i,j,1] = eigs[1] 
        
limiter = np.divide(k_RANS_2d, 3*np.abs(Eigenvalues[:,:,0])) # numerically
limiter = np.multiply(np.sqrt(2*np.divide(1,s_tot)),k_RANS_2d/3) # algebraic

#diff_nu_t = vist_RANS_2d - limiter
diff_nu_t = nu_t - limiter

def plot_10a():    
    plt.figure("Figure 1.10a")
    i=5
    plt.plot(diff_nu_t[i,:], y2d[i,:],'r-')
    plt.plot(np.zeros((nj)), y2d[i,:],'k-')
    plt.xlabel('Difference nu_t vs limiter')
    plt.ylabel('y')
    plt.title('1.10) (x = 0.10) vertical line', fontsize=20)
    #plt.axis([np.min(diff_nu_t), np.max(diff_nu_t), 0, 1])
    plt.tight_layout()    

def plot_10b():
    plt.figure("Figure 1.10b")
    i=50
    plt.plot(diff_nu_t[i,:], y2d[i,:],'r-')
    plt.plot(np.zeros((nj)), y2d[i,:],'k-')
    plt.xlabel('Difference nu_t vs limiter')
    plt.ylabel('y')
    plt.title('1.10) (x = 1.07) vertical line', fontsize=20)
    #plt.axis([np.min(diff_nu_t), np.max(diff_nu_t), 0, 1])
    plt.tight_layout() 
    
def plot_10c():
    plt.figure("Figure 10c")
    plt.contourf(x2d,y2d, diff_nu_t, 50, levels = np.linspace(0, np.max(diff_nu_t), 50))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.title("Diff")
    plt.colorbar()   
    
def plot_10d():
    plt.figure("Figure 10d")
    plt.contourf(x2d,y2d, nu_t, 50)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.plot(x2d[:,0],y2d[:,0], 'k-')
    plt.title("Nu_t")
    plt.axis([0, np.max(x2d), -0.15, 0.1])
    plt.colorbar()   


# ---- GUI Commands
def close_fig():
    plt.close()

root = tk.Tk()
close_button = tk.Button(root, text='close plot', command = close_fig)
close_button.grid(row=0, column=0)

# Overwiew Plots
label1 = tk.Label(text="Overview Plots", background="yellow")
label1.grid(row=0, column=1, sticky='nesw')

button1 = tk.Button(root, text='Velocity Plot', command = plot_velocity_field)
button1.grid(row=1, column=1, sticky='nesw')

button2 = tk.Button(root, text='Pressure Plot', command = plot_pressure)
button2.grid(row=2, column=1, sticky='nesw')

button3 = tk.Button(root, text='k Plot', command = plot_k)
button3.grid(row=3, column=1, sticky='nesw')

button4 = tk.Button(root, text='uu Plot', command = plot_uu_stress)
button4.grid(row=4, column=1, sticky='nesw')

button5 = tk.Button(root, text='vv Plot', command = plot_vv_stress)
button5.grid(row=5, column=1, sticky='nesw')

button6 = tk.Button(root, text='uv Plot', command = plot_uv_Stress)
button6.grid(row=6, column=1, sticky='nesw')

# 1.1
label2 = tk.Label(text="1.1", background="yellow")
label2.grid(row=0, column=2, sticky='nesw')

button7 = tk.Button(root, text='Close', command = plot_1_close)
button7.grid(row=1, column=2, sticky='nesw')

button8 = tk.Button(root, text='Turbulent', command = plot_1_turb)
button8.grid(row=2, column=2, sticky='nesw')

# 1.2
label3 = tk.Label(text="1.2", background="yellow")
label3.grid(row=0, column=3, sticky='nesw')

button9 = tk.Button(root, text='Close v_x', command = plot_2_close_x)
button9.grid(row=1, column=3, sticky='nesw')

button10 = tk.Button(root, text='Close v_y', command = plot_2_close_y)
button10.grid(row=2, column=3, sticky='nesw')

button11 = tk.Button(root, text='Turb v_x', command = plot_2_turb_x)
button11.grid(row=3, column=3, sticky='nesw')

button12 = tk.Button(root, text='Turb v_y', command = plot_2_turb_y)
button12.grid(row=4, column=3, sticky='nesw')

button13 = tk.Button(root, text='Turb v_x zoom', command = plot_2_turb_x_zoom)
button13.grid(row=5, column=3, sticky='nesw')

button14 = tk.Button(root, text='Turb v_y zoom', command = plot_2_turb_y_zoom)
button14.grid(row=6, column=3, sticky='nesw')

# 1.3
label4 = tk.Label(text="1.3", background="yellow")
label4.grid(row=0, column=4, sticky='nesw')

button15 = tk.Button(root, text='Normal Stress', command = plot_3_normal_stress)
button15.grid(row=1, column=4, sticky='nesw')

button16 = tk.Button(root, text='Normal Stress zoom', command = plot_3_normal_stress_zoom)
button16.grid(row=2, column=4, sticky='nesw')

# 1.4
label5 = tk.Label(text="1.4", background="yellow")
label5.grid(row=0, column=5, sticky='nesw')

button17 = tk.Button(root, text='Shear Stresses', command = plot_4_shear_stress)
button17.grid(row=1, column=5, sticky='nesw')

button18 = tk.Button(root, text='Shear Stresses Zoom', command = plot_4_shear_stress_zoom)
button18.grid(row=2, column=5, sticky='nesw')

button19 = tk.Button(root, text='Pressure Grad', command = plot_4_pressure_grad)
button19.grid(row=3, column=5, sticky='nesw')

button20 = tk.Button(root, text='Visous Terms x-dir', command = plot_4_Visc_D_x)
button20.grid(row=4, column=5, sticky='nesw')

button21 = tk.Button(root, text='Visous Terms y-dir', command = plot_4_Visc_D_y)
button21.grid(row=5, column=5, sticky='nesw')

# 1.5

label6 = tk.Label(text="1.5", background="yellow")
label6.grid(row=0, column=6, sticky='nesw')

button22 = tk.Button(root, text='k close', command = plot_5_k_close)
button22.grid(row=1, column=6, sticky='nesw')

button23 = tk.Button(root, text='k turbulent', command = plot_5_k_turb)
button23.grid(row=2, column=6, sticky='nesw')

# 1.6

label7 = tk.Label(text="1.6", background="yellow")
label7.grid(row=0, column=7, sticky='nesw')

button24 = tk.Button(root, text='prod. close', command = plot_6_prod_close)
button24.grid(row=1, column=7, sticky='nesw')

button25 = tk.Button(root, text='prod. turbulent', command = plot_6_prod_turb)
button25.grid(row=2, column=7, sticky='nesw')

# 1.7

label8 = tk.Label(text="1.7", background="yellow")
label8.grid(row=0, column=8, sticky='nesw')

button26 = tk.Button(root, text='Stress 11', command = plot_7_close_11)
button26.grid(row=1, column=8, sticky='nesw')

button27 = tk.Button(root, text='Stress 12', command = plot_7_close_12)
button27.grid(row=2, column=8, sticky='nesw')

button28 = tk.Button(root, text='Stress 22', command = plot_7_close_22)
button28.grid(row=3, column=8, sticky='nesw')

# 1.8

label9 = tk.Label(text="1.8", background="yellow")
label9.grid(row=0, column=9, sticky='nesw')

button29 = tk.Button(root, text='Bouss 11', command = plot_8a)
button29.grid(row=1, column=9, sticky='nesw')

button30 = tk.Button(root, text='Bouss 12', command = plot_8b)
button30.grid(row=2, column=9, sticky='nesw')

button31 = tk.Button(root, text='Bouss 22', command = plot_8c)
button31.grid(row=3, column=9, sticky='nesw')

button29a = tk.Button(root, text='Bouss 11 zoom', command = plot_8a_zoom)
button29a.grid(row=4, column=9, sticky='nesw')

button30b = tk.Button(root, text='Bouss 12 zoom', command = plot_8b_zoom)
button30b.grid(row=5, column=9, sticky='nesw')

button31c = tk.Button(root, text='Bouss 22 zoom', command = plot_8c_zoom)
button31c.grid(row=6, column=9, sticky='nesw')

# 1.9

label10 = tk.Label(text="1.9", background="yellow")
label10.grid(row=0, column=10, sticky='nesw')

button32 = tk.Button(root, text='Production', command = plot_9a)
button32.grid(row=1, column=10, sticky='nesw')

button33 = tk.Button(root, text='Robin Hood Term', command = plot_9b)
button33.grid(row=2, column=10, sticky='nesw')

button34 = tk.Button(root, text='Negative Production', command = plot_9c)
button34.grid(row=3, column=10, sticky='nesw')


# 1.10

label11 = tk.Label(text="1.10", background="yellow")
label11.grid(row=0, column=11, sticky='nesw')

button35 = tk.Button(root, text='nu_t vs limiter close', command = plot_10a)
button35.grid(row=1, column=11, sticky='nesw')

button36 = tk.Button(root, text='nu_t vs limiter turbulent', command = plot_10b)
button36.grid(row=2, column=11, sticky='nesw')

button37 = tk.Button(root, text='Positive diff', command = plot_10c)
button37.grid(row=3, column=11, sticky='nesw')

button38 = tk.Button(root, text='nu_t', command = plot_10d)
button38.grid(row=4, column=11, sticky='nesw')

root.mainloop()


