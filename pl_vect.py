# MTF271 - Assignment 1
# Johan Olson, Alexander Rodin
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
p2d[:,1]=p2d[:,2]
p2d[:,-1]=p2d[:,-1-1]

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
fig1 = plt.figure("Figure velocity")
k=10    # plot every tenth
ss=2.2 #vector length
plt.quiver(x2d[::k,::k],y2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.005)
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("vector plot")
plt.colorbar()
plt.tight_layout()

################################ Pressure Plot
fig2 = plt.figure("Figure p")
plt.contourf(x2d,y2d,p2d, 50)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("contour pressure plot")
plt.colorbar()
plt.tight_layout()

################################# Example Turbulent energy 
fig3 = plt.figure("Figure k")
plt.contourf(x2d,y2d,k_RANS_2d, 50)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("Contour k RANS plot")
plt.colorbar()
plt.tight_layout()

################################ Example uu plot
fig4 = plt.figure("Figure uu Stress")
plt.contourf(x2d,y2d,uu2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel('$\overline{u^\prime u^\prime}$')
plt.ylabel("$y$")
plt.title("uu counter plot")
plt.colorbar()
plt.tight_layout()

################################ Example vv plot
fig5 = plt.figure("Figure vv Stress")
plt.contourf(x2d,y2d,vv2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel('$\overline{v^\prime v^\prime}$')
plt.ylabel("$y$")
plt.title("vv counter plot")
plt.colorbar()
plt.tight_layout()

################################ Example uv plot
fig6 = plt.figure("Figure uv Stress")
plt.contourf(x2d,y2d,uv2d, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel("$y$")
plt.title("uv counter plot")
plt.colorbar()
plt.tight_layout()

# ---- A_1.1)

# Line close to inlet
fig11 = plt.figure("Figure 1.1a")
i=5 # 0.09862734
plt.plot(uu2d[i,:],y2d[i,:],'b-', label='$\overline{u^{\prime 2}}$')
plt.plot(uv2d[i,:],y2d[i,:],'r-', label='$\overline{u^\prime v^\prime}$')
plt.plot(vv2d[i,:],y2d[i,:],'k-', label='$\overline{v^{\prime 2}}$')
plt.xlabel('$\overline{u_i^\prime u_j^\prime}$')
plt.ylabel('y/H')
plt.title('1.1 (x = 0.10) vertical line', fontsize=20)
plt.legend()
plt.tight_layout()

# Line at turbulence
fig12 = plt.figure("Figure 1.1b")
i=50 # 1.069507
plt.plot(uu2d[i,:],y2d[i,:],'b-', label='$\overline{u^{\prime 2}}$')
plt.plot(uv2d[i,:],y2d[i,:],'r-', label='$\overline{u^\prime v^\prime}$')
plt.plot(vv2d[i,:],y2d[i,:],'k-', label='$\overline{v^{\prime 2}}$')
plt.xlabel('$\overline{u_i^\prime u_j^\prime}$')
plt.ylabel('y/H')
plt.title('1.1 (x = 1.07) vertical line', fontsize=20)
plt.legend()
plt.tight_layout()

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
fig21 = plt.figure("Figure 1.2a")
i=5
plt.plot(du_udx[i,:],y2d[i,:],'b-',   label='1')
plt.plot(du_vdy[i,:],y2d[i,:],'g-',   label='2')
plt.plot(dpdx[i,:],y2d[i,:],'r-',    label='3')
plt.plot(-nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(duudx[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-nu*du2dy2[i,:],y2d[i,:],'k-',label='6')
plt.plot(duvdy[i,:],y2d[i,:],'c-',   label='7')
plt.xlabel('Term size $v_x$-direction (Eq. R.1)')
plt.ylabel('y/H')
plt.title('1.2 (x = 0.10) vertical line', fontsize=20)
plt.axis([-0.2, 0.2, 0, 1])
plt.legend()
plt.tight_layout()
# y-dir
fig22 = plt.figure("Figure 1.2b")
plt.plot(du_vdx[i,:],y2d[i,:],'b-',   label='1')
plt.plot(dv_vdy[i,:],y2d[i,:],'g-',   label='2')
plt.plot(dpdy[i,:],y2d[i,:],'r-',    label='3')
plt.plot(-nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(duvdx[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-nu*dv2dy2[i,:],y2d[i,:],'k-',label='6')
plt.plot(dvvdy[i,:],y2d[i,:],'c-',   label='7')
plt.xlabel('Term size $v_y$-direction (Eq. R.1)')
plt.ylabel('y/H')
plt.title('1.2 (x = 0.10) vertical line', fontsize=20)
plt.axis([-0.1, 0.1, 0, 1])
plt.legend()
plt.tight_layout()

## Turbulent Region
# x-dir
fig23 = plt.figure("Figure 1.2c")
i=50
plt.plot(du_udx[i,:],y2d[i,:],'b-',   label='1')
plt.plot(du_vdy[i,:],y2d[i,:],'g-',   label='2')
plt.plot(dpdx[i,:],y2d[i,:],'r-',    label='3')
plt.plot(-nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(duudx[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-nu*du2dy2[i,:],y2d[i,:],'k-',label='6')
plt.plot(duvdy[i,:],y2d[i,:],'c-',   label='7')
plt.xlabel('Term size $v_x$-direction (Eq. R.1)')
plt.ylabel('y/H')
plt.title('1.2 (x = 1.07) vertical line', fontsize=20)
plt.axis([-0.25, 0.1, 0, 1])
plt.legend()
plt.tight_layout()
# y-dir
fig24 = plt.figure("Figure 1.2d")
plt.plot(du_vdx[i,:],y2d[i,:],'b-',   label='1')
plt.plot(dv_vdy[i,:],y2d[i,:],'g-',   label='2')
plt.plot(dpdy[i,:],y2d[i,:],'r-',    label='3')
plt.plot(-nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(duvdx[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-nu*dv2dy2[i,:],y2d[i,:],'k-',label='6')
plt.plot(dvvdy[i,:],y2d[i,:],'c-',   label='7')
plt.xlabel('Term size (Eq. R.1)')
plt.ylabel('y/H')
plt.title('1.2 $v_y$-direction (x = 1.07) vertical line', fontsize=20)
plt.axis([-0.11, 0.15, 0, 1])
plt.legend()
plt.tight_layout()

## Zoom in; Turbulent Region
# x-dir
fig25 = plt.figure("Figure 1.2e")
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
plt.axis([-0.1, 0.1, 0, 0.01])
plt.legend()
plt.tight_layout()
# y-dir
fig26 = plt.figure("Figure 1.2f")
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
plt.axis([-0.11, 0.15, 0, 0.01])
plt.legend()
plt.tight_layout()

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

# ---- A_1.3)
fig31 = plt.figure("Figure 1.3a")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], F_N_x_norm[::k,::k], F_N_y_norm[::k,::k], scale=10)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.3) Normal Stresses Vector Plot", fontsize=20)
plt.tight_layout()

# Zooom
fig32 = plt.figure("Figure 1.3b")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], F_N_x_norm[::k,::k], F_N_y_norm[::k,::k], width = 0.007, scale=13)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.3) Zoom in, Normal Stresses Vector Plot", fontsize=20)
plt.axis([0, 2, -0.3, 0.2])
plt.tight_layout()

# ---- A_1.4) 
# F_S
fig41 = plt.figure("Figure 1.4a")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], F_S_x_norm[::k,::k], F_S_y_norm[::k,::k],width=0.005)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.4) Shear Stress Vector Plot", fontsize=20)
plt.tight_layout()    

# Zooom
fig42 = plt.figure("Figure 1.4b")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], F_S_x_norm[::k,::k], F_S_y_norm[::k,::k], width = 0.007, scale=13)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.4) Zoom in, Normal Stresses Vector Plot", fontsize=20)
plt.axis([0, 2, -0.3, 0.2])
plt.tight_layout()

# Pressure
fig43 = plt.figure("Figure 1.4c")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], dpdx[::k,::k], dpdy[::k,::k], width = 0.007, scale=95)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.4) Pressure Gradient Vector Plot", fontsize=20)
plt.tight_layout()

# Viscous x
fig44 = plt.figure("Figure 1.4d")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], nu*du2dx2[::k,::k], nu*du2dy2[::k,::k], width = 0.007, scale=1)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.4) Viscous Diffusion x", fontsize=20)
plt.tight_layout()

# Viscous y
fig45 = plt.figure("Figure 1.4e")
k=10# plot every
plt.quiver(x2d[::k,::k], y2d[::k,::k], nu*dv2dx2[::k,::k], nu*dv2dy2[::k,::k], width = 0.007, scale=0.1)
plt.plot(x2d[:,0],y2d[:,0], 'k-')
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("1.4) Viscous Diffusion y", fontsize=20)
plt.tight_layout()

plt.show()
plt.close('all')

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
fig51 = plt.figure("Figure 1.5a")
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
fig52 = plt.figure("Figure 1.5b")
i=50
plt.plot(P_k[i,:,0],y2d[i,:],'g-',label='$P_{11}$')
plt.plot(P_k[i,:,1],y2d[i,:],'b-',label='$P_{12}$')
plt.plot(P_k[i,:,2],y2d[i,:],'r-',label='$P_{21}$')
plt.plot(P_k[i,:,3],y2d[i,:],'y-',label='$P_{22}$')
plt.plot(P_k[i,:,4],y2d[i,:],'k-',label='Total')
plt.xlabel('Turbulent Energy Production')
plt.ylabel('y/H')
plt.title('1.5) x = 1.07 vertical line', fontsize=20)
plt.axis([-0.03, 0.045, 0, 1])
plt.legend()
plt.tight_layout()

# ---- A_1.6)

# Close to inlet, Compare with data for Eps
fig61 = plt.figure("Figure 1.6a")
i=5
plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Production')
plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
plt.xlabel('Turbulent Energy')
plt.ylabel('y/H')
plt.title('1.6) x = 0.10 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

# At recircualtion, Compare with data foe Eps
fig62 = plt.figure("Figure 1.6b")
i=50
plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Production')
plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
plt.xlabel('Turbulent Energy')
plt.ylabel('y/H')
plt.title('1.6) x = 1.07 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

# ---- A_1.7)

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
#nu_t = Cmu*np.divide(np.multiply(k_RANS_2d, vist_RANS_2d), diss_RANS_2d)
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
Phi112W = K2*( -2*np.multiply(Phi112, n1n1) - np.multiply(Phi122, n1n2) + np.multiply(Phi222,n2n2))
Phi121W = -1.5*K1*(2*np.multiply(k_RANS_2d, n1n1) + np.multiply(uv2d, n1n1 + n2n2))
Phi122W = -1.5*K2*(np.multiply(n1n2, Phi112 + Phi222) + np.multiply(Phi122, n1n1+n2n2))

Phi11  += Phi111W + Phi112W
Phi12  += Phi121W + Phi122W

# Total

Total_Stress11 = -C11 + D11v + P11 + Phi11 + D11t - Eps11
Total_Stress12 = -C12 + D12v + P12 + Phi12 + D12t - Eps12
Total_Stress22 = -C22 + D22v + P22 + Phi22 + D22t - Eps22

# Plot Close to inlet 11
fig71 = plt.figure("Figure 1.7a")
i=5
plt.plot(-C11[i,:],y2d[i,:],'b-',   label='1')
plt.plot(D11v[i,:],y2d[i,:],'g-',   label='2')
plt.plot(P11[i,:],y2d[i,:],'r-',    label='3')
plt.plot(Phi11[i,:],y2d[i,:],'y-',label='4')
plt.plot(D11t[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-Eps11[i,:],y2d[i,:],'k-',label='6')
plt.plot(Total_Stress11[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('Reynold Stress terms 11')
plt.ylabel('y/H')
plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
plt.axis([-0.03, 0.03, 0, 1])
plt.legend()
plt.tight_layout()

# Plot Close to inlet 12
fig72 = plt.figure("Figure 1.7b")
i=5
plt.plot(-C12[i,:],y2d[i,:],'b-',   label='1')
plt.plot(D12v[i,:],y2d[i,:],'g-',   label='2')
plt.plot(P12[i,:],y2d[i,:],'r-',    label='3')
plt.plot(Phi12[i,:],y2d[i,:],'y-',label='4')
plt.plot(D12t[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-Eps12[i,:],y2d[i,:],'k-',label='6')
plt.plot(Total_Stress12[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('Reynold Stress Terms 12')
plt.ylabel('y/H')
plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
plt.axis([-0.06, 0.06, 0, 1])
plt.legend()
plt.tight_layout()

# Plot Close to inlet 22
fig73 = plt.figure("Figure 1.7c")
i=5
plt.plot(-C22[i,:],y2d[i,:],'b-',   label='1')
plt.plot(D22v[i,:],y2d[i,:],'g-',   label='2')
plt.plot(P22[i,:],y2d[i,:],'r-',    label='3')
plt.plot(Phi22[i,:],y2d[i,:],'y-',label='4')
plt.plot(D22t[i,:],y2d[i,:],'m-',   label='5')
plt.plot(-Eps22[i,:],y2d[i,:],'k-',label='6')
plt.plot(Total_Stress22[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('Reynold Stress terms 22')
plt.ylabel('y/H')
plt.title('1.7 (x = 0.10) vertical line', fontsize=20)
plt.axis([-0.01, 0.02, 0, 1])
plt.legend()
plt.tight_layout()

# ---- A_1.8)
Bouss11 = np.multiply(-vist_RANS_2d, 2*dudx) + (2/3)*k_RANS_2d
Bouss12 = np.multiply(-vist_RANS_2d, (dudy + dvdx))
Bouss22 = np.multiply(-vist_RANS_2d, 2*dvdy) + (2/3)*k_RANS_2d

fig81 = plt.figure("1.8a")
plt.contourf(x2d,y2d, Bouss11, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.xlabel('$x$')
plt.ylabel("$y$")
plt.title("Bouss11 Approx.")
plt.colorbar()
plt.tight_layout()

fig82 = plt.figure("1.8b")
plt.contourf(x2d,y2d, Bouss12, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.xlabel('$x$')
plt.ylabel("$y$")
plt.title("Bouss12 Approx.")
plt.colorbar()
plt.tight_layout()

fig83 = plt.figure("1.8c")
plt.contourf(x2d,y2d, Bouss22, 50, levels = np.linspace(np.min(uv2d), np.max(uu2d), 50))
plt.xlabel('$x$')
plt.ylabel("$y$")
plt.title("Bouss22 Approx.")
plt.colorbar()
plt.tight_layout()

# ---- A_1.9)
P_tot = P11 + P22

fig91 = plt.figure("Figure 1.9a")
plt.contourf(x2d,y2d, P_tot, 50)
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("Production")
plt.colorbar()

fig92 = plt.figure("Figure 1.9b")
plt.contourf(x2d,y2d, P12, 50)
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.title("Robin Hood")
plt.colorbar()

fig93 = plt.figure("Figure 1.9c")
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

for i in range(ni):
    for j in range(nj):
        S = np.array([[s11[i,j], s12[i,j]], [s21[i,j], s22[i,j]]])
        eigs, empty = np.linalg.eig(S)
        Eigenvalues[i,j,0] = eigs[0]
        Eigenvalues[i,j,1] = eigs[1] 
        
limiter = np.divide(k_RANS_2d, 3*np.abs(Eigenvalues[:,:,0])) # numerically
limiter = np.multiply(np.sqrt(2*np.divide(1,np.multiply(s11,s11))),k_RANS_2d/3) # algebraic

diff_nu_t = vist_RANS_2d - limiter

fig101 = plt.figure("Figure 1.10a")
i=5
plt.plot(diff_nu_t[i,:], y2d[i,:],'r-')
plt.plot(np.zeros((nj)), y2d[i,:],'k-')
plt.xlabel('Difference nu_t vs limiter')
plt.ylabel('y/H')
plt.title('1.10) (x = 0.10) vertical line', fontsize=20)
#plt.axis([np.min(diff_nu_t), np.max(diff_nu_t), 0, 1])
plt.tight_layout()    

fig102 = plt.figure("Figure 1.10b")
i=50
plt.plot(diff_nu_t[i,:], y2d[i,:],'r-')
plt.plot(np.zeros((nj)), y2d[i,:],'k-')
plt.xlabel('Difference nu_t vs limiter')
plt.ylabel('y/H')
plt.title('1.10) (x = 1.07) vertical line', fontsize=20)
#plt.axis([np.min(diff_nu_t), np.max(diff_nu_t), 0, 1])
plt.tight_layout() 

fig103 = plt.figure("Figure 10c")
plt.contourf(x2d,y2d, diff_nu_t, 50, levels = np.linspace(0, 0.90*np.max(vist_RANS_2d - limiter), 20))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Diff")
plt.colorbar()   
      
plt.show()


