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
### Import data
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
dudx= np.zeros((ni,nj))
dudy= np.zeros((ni,nj))
dvdx= np.zeros((ni,nj))
dvdy= np.zeros((ni,nj))

dudx,dudy=dphidx_dy(xf2d,yf2d,u2d)
dvdx,dvdy=dphidx_dy(xf2d,yf2d,v2d)


### Plots 
################################ Plot 1
# =============================================================================
# fig1 = plt.figure("Figure 1")
# k=6# plot every forth vector
# ss=3.2 #vector length
# plt.quiver(x2d[::k,::k],y2d[::k,::k],u2d[::k,::k],v2d[::k,::k],width=0.01)
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("vector plot")
# plt.savefig('vect_python.eps',bbox_inches='tight')
# =============================================================================

# =============================================================================
# ################################ Pressure
# fig2 = plt.figure("Figure 2")
# plt.contourf(x2d,y2d,p2d, 50)
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("contour pressure plot")
# plt.savefig('piso_python.eps',bbox_inches='tight')
# 
# =============================================================================
################################# Turbulent energy 
fig3 = plt.figure("Figure 3")
plt.contourf(x2d,y2d,k_RANS_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("contour k RANS plot")
plt.savefig('k_rans.eps',bbox_inches='tight')

# =============================================================================
# ################################ uv plot
# fig4 = plt.figure("Figure 4")
# i=10
# plt.plot(uv2d[i,:],y2d[i,:],'b-')
# plt.xlabel('$\overline{u^\prime v^\prime}$')
# plt.ylabel('y/H')
# plt.savefig('uv_python.eps',bbox_inches='tight')
# =============================================================================

################################ A_1.1)

# Line close to inlet
fig5 = plt.figure("Figure 5")
i=5 # 0.09862734
plt.plot(uu2d[i,:],y2d[i,:],'b-')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.1 x = 0.10 vertical line', fontsize=20)
plt.tight_layout()

# Line at recirculation
fig6 = plt.figure("Figure 6")
i=50 # 1.069507
plt.plot(uu2d[i,:],y2d[i,:],'b-')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.1 x = 1.07 vertical line', fontsize=20)
plt.tight_layout()

################################ A_1.2) 
# Eq. (R.1)
# Convection Terms 
du_udx,du_udy=dphidx_dy(xf2d,yf2d, np.multiply(u2d,u2d))
du_vdx,du_vdy=dphidx_dy(xf2d,yf2d, np.multiply(u2d,v2d))
dv_vdx,dv_vdy=dphidx_dy(xf2d,yf2d, np.multiply(v2d,v2d))

# Pressure Terms
dpdx,dpdy=dphidx_dy(xf2d,yf2d, p2d)

# Diffusion Terms
dudx,dudy=dphidx_dy(xf2d,yf2d, u2d)
du2dx2,NaN1=dphidx_dy(xf2d,yf2d, dudx)
NaN2,du2dy2=dphidx_dy(xf2d,yf2d, dudy)

dvdx,dvdy=dphidx_dy(xf2d,yf2d, v2d)
dv2dx2,NaN3=dphidx_dy(xf2d,yf2d, dvdx)
NaN4,dv2dy2=dphidx_dy(xf2d,yf2d, dvdy)

# Turbulence Terms
duudx,NaN5=dphidx_dy(xf2d,yf2d, uu2d)
duvdx,duvdy=dphidx_dy(xf2d,yf2d, uv2d)
NaN6,dvvdy=dphidx_dy(xf2d,yf2d, vv2d)

# Close to inlet
fig7 = plt.figure("Figure 7")
i=5
plt.plot(du_udx[i,:],y2d[i,:],'b-',label='1')
plt.plot(du_vdy[i,:],y2d[i,:],'g-',label='2')
plt.plot(-dpdx[i,:],y2d[i,:],'r-',label='3')
plt.plot(nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(nu*du2dy2[i,:],y2d[i,:],'k-',label='5')
plt.plot(-duudx[i,:],y2d[i,:],'m-',label='6')
plt.plot(-duvdy[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.2 x = 0.10 vertical line', fontsize=20)
plt.axis([-0.1, 0.1, 0, 1])
plt.legend()
plt.tight_layout()

# At recirculation
fig8 = plt.figure("Figure 8")
i=50
plt.plot(du_udx[i,:],y2d[i,:],'b-',label='1')
plt.plot(du_vdy[i,:],y2d[i,:],'g-',label='2')
plt.plot(-dpdx[i,:],y2d[i,:],'r-',label='3')
plt.plot(nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(nu*du2dy2[i,:],y2d[i,:],'k-',label='5')
plt.plot(-duudx[i,:],y2d[i,:],'m-',label='6')
plt.plot(-duvdy[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.2 x = 1.07 vertical line', fontsize=20)
plt.axis([-0.1, 0.1, 0, 1])
plt.legend()
plt.tight_layout()

# Close to wall
fig9 = plt.figure("Figure 9")
i=5
plt.plot(du_udx[i,:],y2d[i,:],'b-',label='1')
plt.plot(du_vdy[i,:],y2d[i,:],'g-',label='2')
plt.plot(-dpdx[i,:],y2d[i,:],'r-',label='3')
plt.plot(nu*du2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(nu*du2dy2[i,:],y2d[i,:],'k-',label='5')
plt.plot(-duudx[i,:],y2d[i,:],'m-',label='6')
plt.plot(-duvdy[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.2 x = 0.10 vertical line', fontsize=20)
plt.axis([-0.1, 0.1, 0, 0.01])
plt.legend()
plt.tight_layout()


# y Mom-Eq at recirculation
fig10 = plt.figure("Figure 10")
i=50
plt.plot(du_vdx[i,:],y2d[i,:],'g-',label='1')
plt.plot(dv_vdy[i,:],y2d[i,:],'b-',label='2')
plt.plot(-dpdy[i,:],y2d[i,:],'r-',label='3')
plt.plot(nu*dv2dx2[i,:],y2d[i,:],'y-',label='4')
plt.plot(nu*dv2dy2[i,:],y2d[i,:],'k-',label='5')
plt.plot(-duvdx[i,:],y2d[i,:],'m-',label='6')
plt.plot(-dvvdy[i,:],y2d[i,:],'c-',label='7')
plt.xlabel('$\overline{u^{\prime 2}}$')
plt.ylabel('y/H')
plt.title('1.2 x = 1.07 vertical line', fontsize=20)
plt.axis([-0.1, 0.1, 0, 1])
plt.legend()
plt.tight_layout()

# Last part, part of next Question
F_N = np.zeros((ni,nj,2))
F_S = np.zeros((ni,nj,2))

F_N[:,:,0] = -duudx
F_N[:,:,1] = -dvvdy

F_S[:,:,0] = -duvdx
F_S[:,:,1] = -duvdy

################################# A_1.3) vector plot
fig11 = plt.figure("Figure 11")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],F_N[::k,::k,0],F_N[::k,::k,1],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.3) Normal Stresses Vector Plot")

# Zooom
fig12 = plt.figure("Figure 12")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],F_N[::k,::k,0],F_N[::k,::k,1],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.3) Zoom in, Normal Stresses Vector Plot")
plt.axis([0, 3, -0.3, 0.2])

################################# A_1.4) 
# F_S
fig13 = plt.figure("Figure 13")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],F_S[::k,::k,0],F_S[::k,::k,1],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.4) Shear Stress Vector Plot")

# Pressure
fig14 = plt.figure("Figure 14")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],dpdx[::k,::k],dpdy[::k,::k],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.4) Pressure Gradient Vector Plot")

# Viscous x
fig14 = plt.figure("Figure 14")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],du2dx2[::k,::k],du2dy2[::k,::k],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.4) Viscous x")

# Viscous y
fig15 = plt.figure("Figure 15")
k=10# plot every
plt.quiver(x2d[::k,::k],y2d[::k,::k],dv2dx2[::k,::k],dv2dy2[::k,::k],width=0.005)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("1.4) Viscous Diffusion y")

################################# A_1.5)

# Production term
P_k = np.zeros((ni,nj,5))

P_k[:,:,0] = - np.multiply(uu2d,dudx)
P_k[:,:,1] = - np.multiply(uv2d,dudy)
P_k[:,:,2] = - np.multiply(uv2d,dvdx)
P_k[:,:,3] = - np.multiply(vv2d,dvdy)
P_k[:,:,4] = P_k[:,:,0] + P_k[:,:,1] + P_k[:,:,2] + P_k[:,:,3]
 
# Close to inlet
fig16 = plt.figure("Figure 16")
i=5
plt.plot(P_k[i,:,0],y2d[i,:],'g-',label='1')
plt.plot(P_k[i,:,1],y2d[i,:],'b-',label='2')
plt.plot(P_k[i,:,2],y2d[i,:],'r-',label='3')
plt.plot(P_k[i,:,3],y2d[i,:],'y-',label='4')
plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='Total')
plt.xlabel('change me')
plt.ylabel('y/H')
plt.title('1.5) x = 0.10 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

# At recirculation 
fig17 = plt.figure("Figure 17")
i=50
plt.plot(P_k[i,:,0],y2d[i,:],'g-',label='1')
plt.plot(P_k[i,:,1],y2d[i,:],'b-',label='2')
plt.plot(P_k[i,:,2],y2d[i,:],'r-',label='3')
plt.plot(P_k[i,:,3],y2d[i,:],'y-',label='4')
plt.plot(P_k[i,:,4],y2d[i,:],'k-',label='Total')
plt.xlabel('change me')
plt.ylabel('y/H')
plt.title('1.5) x = 1.07 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

################################# A_1.6)

# Close to inlet, Compare with data for Eps
fig18 = plt.figure("Figure 18")
i=5
plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='P^k')
plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
plt.xlabel('change me')
plt.ylabel('y/H')
plt.title('1.6) x = 0.10 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

# At recircualtion, Compare with data foe Eps
fig19 = plt.figure("Figure 19")
i=50
plt.plot(P_k[i,:,4],y2d[i,:],'y-',label='P^k')
plt.plot(diss_RANS_2d[i,:],y2d[i,:],'k-',label='Dissipation')
plt.xlabel('change me')
plt.ylabel('y/H')
plt.title('1.6) x = 1.07 vertical line', fontsize=20)
plt.axis([-0.05, 0.05, 0, 1])
plt.legend()
plt.tight_layout()

plt.show()



