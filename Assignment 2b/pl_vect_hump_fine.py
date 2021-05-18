import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dphidx_dy import dphidx_dy
from IPython import display
from matplotlib import cm 
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

re =9.36e+5
viscos =1/re

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

# read data file
vectz = np.loadtxt("vectz_fine.dat")
ntstep=vectz[0]
ni=int(vectz[1])
nj=int(vectz[2])
nk=int(vectz[3])
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)fk2d(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2d(i,j)
#            write(48,*)rk2d(i,j)
#            write(48,*)vis2d(i,j)
#            write(48,*)dissp2d(i,j)
#            write(48,*)wvec(i,j)
#            write(48,*)vtvec(i,j)
#            write(48,*)tvec(i,j)


nn=14
nst=3
iu=range(nst+1,n,nn)
iv=range(nst+2,n,nn)
ifk=range(nst+3,n,nn)
iuu=range(nst+4,n,nn)
ivv=range(nst+5,n,nn)
iww=range(nst+6,n,nn)
iuv=range(nst+7,n,nn)
ip=range(nst+8,n,nn)
ik=range(nst+9,n,nn)
ivis=range(nst+10,n,nn)
idiss=range(nst+11,n,nn)

u=vectz[iu]/ntstep
v=vectz[iv]/ntstep
fk=vectz[ifk]/ntstep
uu=vectz[iuu]/ntstep
vv=vectz[ivv]/ntstep
ww=vectz[iww]/ntstep
uv=vectz[iuv]/ntstep
p=vectz[ip]/ntstep
k_model=vectz[ik]/ntstep
vis=vectz[ivis]/ntstep
diss=vectz[idiss]/ntstep

# uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u**2
vv=vv-v**2
#ww=ww-w**2 no w exists...
uv=uv-u*v

p_2d=np.reshape(p,(ni,nj))
u_2d=np.reshape(u,(ni,nj))
v_2d=np.reshape(v,(ni,nj))
fk_2d=np.reshape(fk,(ni,nj))
uu_2d=np.reshape(uu,(ni,nj))
uv_2d=np.reshape(uv,(ni,nj))
vv_2d=np.reshape(vv,(ni,nj))
ww_2d=np.reshape(ww,(ni,nj))
k_model_2d=np.reshape(k_model,(ni,nj))
vis_2d=np.reshape(vis,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb
diss_2d=np.reshape(diss,(ni,nj)) 

# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

dz=0.2/nk

x065_off=np.genfromtxt("x065_off.dat", dtype=None,comments="%")

# compute the gradient
dudx,dudy=dphidx_dy(x_2d,y_2d,u_2d)
dvdx,dvdy=dphidx_dy(x_2d,y_2d,v_2d)


#*************************
# plot u
def u_plot():
    fig1,ax1 = plt.subplots()
    xx=0.65;
    i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(u_2d[i1,:],yp2d[i1,:],'b-')
    plt.plot(x065_off[:,2],x065_off[:,1],'bo')
    plt.xlabel("$U$")
    plt.ylabel("$y$")
    plt.title("$x=0.65$")
    plt.axis([0, 1.3,0.115,0.2])
    
    # Create inset of width 30% and height 40% of the parent axes' bounding box
    # at the lower left corner (loc=3)
    # upper left corner (loc=2)
    # use borderpad=1, i.e.
    # 22 points padding (as 22pt is the default fontsize) to the parent axes
    axins1 = inset_axes(ax1, width="40%", height="30%", loc=2, borderpad=1)
    plt.plot(u_2d[i1,:],yp2d[i1,:],'b-')
    plt.axis([0, 1.3,0.115,0.13])
    # reduce fotnsize 
    axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)
    
    # Turn ticklabels of insets off
    axins1.tick_params(labelleft=False, labelbottom=False)
    
    plt.plot(x065_off[:,2],x065_off[:,1],'bo')

def fk_contour():
    plt.figure("Figure fk contour")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, fk_2d[1:,1:], 50)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5,0,1])
    plt.title("contour $f_k$")
    plt.colorbar()

switch = np.zeros(ni-1)
for i in range(ni-1):
    for j in reversed(range(nj-1)):
        if fk_2d[i,j] < 0.4:
            switch[i] = y_2d[i,j] - y_2d[i,0]
            break
            
def fk_boundary():
    plt.figure("Figure fk contour boundary")
    plt.clf() #clear the figure
    plt.plot(x_2d[:,0], switch, 'k-')
    plt.xlabel("$x$")
    plt.ylabel("$y-y_{wall}$")
    plt.axis([0.6,1.5,0,0.2])
    plt.title("Wall distance $f_k=0.4$")
                
Cmu = 0.09            
        
dx = np.diff(x_2d, axis = 0)

dx = np.insert(dx,0, dx[-1,:], axis=0)

dy = np.diff(y_2d, axis = 1)

dy = np.insert(dy,0, dy[:,-1], axis = 1)

C_des = 0.65

dZ = np.ones((ni-1,nj-1))

# ---- Here
k_res = 0.5*(uu_2d + vv_2d + ww_2d) # ww_2d aswell? not defined the same

k_tot = k_model_2d + k_res

fk_def = k_model_2d/k_tot

L_t_alt = np.power(k_tot, 3/2)/diss_2d

Delta = np.minimum(dx, dy, dz*dZ)

Delta2 = np.power(Delta*dx*dy*dz, 1/3)

fk_alt = np.power(Delta2/L_t_alt[1:,1:], 2/3)/(np.sqrt(Cmu))


def fk_lines_given():
    plt.figure("Figure fk lines")
    plt.clf() #clear the figure
    
    xx = 0.65
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_2d[i1,1:],'r-', label = 'x = 0.65')
    
    xx = 0.80
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_2d[i1,1:],'b-', label = 'x = 0.80')
    
    xx = 0.90
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_2d[i1,1:],'g-', label = 'x = 0.90')
    
    xx = 1.30
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_2d[i1,1:],'y-', label = 'x = 1.30')
        
    plt.xlabel("$y-y_{wall}$")
    plt.ylabel("$f_k$")
    plt.axis([0, 0.2, 0, 1])
    plt.title("Wall distance")
    plt.legend() 
    
def fk_lines_computed():
    plt.figure("Figure fk lines")
    plt.clf() #clear the figure
    
    xx = 0.65
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_def[i1,1:],'r-', label = 'x = 0.65 by def')
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_alt[i1,:],'r--', label = 'x = 0.65 by alt def')
    
    xx = 0.80
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_def[i1,1:],'b-', label = 'x = 0.80 by def')
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_alt[i1,:],'b--', label = 'x = 0.80 by alt def')
    
    xx = 0.90
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_def[i1,1:],'g-', label = 'x = 0.90 by def')
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_alt[i1,:],'g--', label = 'x = 0.90 by alt def')
    
    xx = 1.30
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_def[i1,1:],'y-', label = 'x = 1.30 by def')
    plt.plot(y_2d[i1,:]-y_2d[i1,0], fk_alt[i1,:],'y--', label = 'x = 1.30 by alt def')
        
    plt.xlabel("$y-y_{wall}$")
    plt.ylabel("$f_k$")
    plt.axis([0, 0.2, 0, 1])
    plt.title("Wall distance")
    plt.legend()
    
# V.5 

length_scale_SA_DES = C_des * np.maximum(dx, dy, dz*dZ)

d_tilde = np.zeros((ni-1,nj-1))
for i in range(ni-1):
    for j in range(nj-1):
        d_tilde[i,j] = np.min((x_2d[i,j] - x_2d[:,0])**2 + (y_2d[i,j] - y_2d[:,0])**2)
        
d_tilde = np.sqrt(d_tilde)

y_boundary_SA_DES = np.zeros(ni-1)
for i in range(ni-1):
    for j in reversed(range(nj-1)):
        if d_tilde[i,j] < length_scale_SA_DES[i,j]:
            y_boundary_SA_DES[i] = y_2d[i,j]-y_2d[i,0]
            break
        
def d_boundary_plot():
    plt.figure("Figure d boundary")
    plt.clf() #clear the figure
    plt.plot(x_2d[:,0], y_boundary_SA_DES, 'k-')
    plt.xlabel("$x$")
    plt.ylabel("$y-y_{wall}$")
    plt.axis([0.6,1.5,0, 0.2])
    plt.title("Wall distance SA-DES")
    
    
l_PANS = np.power(k_model_2d, 3/2)/diss_2d
l_RANS = np.power(k_res, 3/2)/diss_2d

def length_scale_lines_PANS():
    plt.figure("Figure fk lines")
    plt.clf() #clear the figure
    
    xx = 0.65
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_PANS[i1,1:],'r-', label = 'x = 0.65 PANS')
    
    xx = 0.80
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_PANS[i1,1:],'b-', label = 'x = 0.80 PANS')
    
    xx = 1.00
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_PANS[i1,1:],'g-', label = 'x = 1.00 PANS')
    
    xx = 1.20
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_PANS[i1,1:],'y-', label = 'x = 1.20 PANS')
      
    xx = 2.00
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_PANS[i1,1:],'k-', label = 'x = 2.00 PANS')
        
    plt.xlabel("$y-y_{wall}$")
    plt.ylabel("$Length Scale$")
    plt.axis([0, 0.17, 0, 0.02])
    plt.title("Wall distance")
    plt.legend()
    
def length_scale_lines_RANS():
    plt.figure("Figure fk lines")
    plt.clf() #clear the figure
    
    xx = 0.65
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_RANS[i1,1:],'r--', label = 'x = 0.65 RANS')
    
    xx = 0.80
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_RANS[i1,1:],'b--', label = 'x = 0.80 RANS')
    
    xx = 1.00
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_RANS[i1,1:],'g--', label = 'x = 1.00 RANS')
    
    xx = 1.20
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_RANS[i1,1:],'y--', label = 'x = 1.20 RANS')
      
    xx = 2.00
    i1 = (np.abs(xx-x_2d[:,1])).argmin()
    plt.plot(y_2d[i1,:]-y_2d[i1,0], l_RANS[i1,1:],'k--', label = 'x = 2.00 RANS')
        
    plt.xlabel("$y-y_{wall}$")
    plt.ylabel("$Length Scale$")
    plt.axis([0, 0.17, 0, 1])
    plt.title("Wall distance")
    plt.legend()
    
omega = diss_2d/(Cmu * k_model_2d)

L_t = np.sqrt(k_model_2d[1:,1:])/(Cmu*omega[1:,1:])

length_scale_SST_DES = (0.61*np.maximum(dx, dy, dz*dZ))

F_DES = L_t/length_scale_SST_DES

y_boundary_alt = np.zeros(ni-1)
for i in range(ni-1):
    for j in reversed(range(nj-1)):
        if 1 < F_DES[i,j]:
            y_boundary_alt[i] = y_2d[i,j]-y_2d[i,0]
            break
        elif j == 0:
            y_boundary_alt[i] = y_2d[i,nj-2]
            
def d_boundary_alt_plot():
    plt.figure("Figure d boundary alt")
    plt.clf() #clear the figure
    plt.plot(x_2d[:,0], y_boundary_alt, 'k-')
    plt.xlabel("$x$")
    plt.ylabel("$y-y_{wall}$")
    plt.axis([0.6,1.5,0, 0.2])
    plt.title("Wall distance SST-DES")

arg1 = 2*L_t/d_tilde
arg2 = 500*viscos/(np.power(d_tilde, 2)*omega[1:,1:])

eta = np.maximum(arg1, arg2)
F_S = np.tanh(np.power(eta, 2))

F_DDES = (1.0-F_S)*L_t/length_scale_SST_DES
y_boundary_alt_2 = np.zeros(ni-1)
for i in range(ni-1):
    for j in reversed(range(nj-1)):
        if 1.0 < F_DDES[i,j]:
            y_boundary_alt_2[i] = y_2d[i,j]-y_2d[i,0]
            break
        elif j == 0:
            y_boundary_alt_2[i] = y_2d[i,nj-2]
        
def d_boundary_alt_2_plot():
    plt.figure("Figure d boundary alt 2")
    plt.clf() #clear the figure
    plt.plot(x_2d[:,0], y_boundary_alt_2, 'k-')
    plt.xlabel("$x$")
    plt.ylabel("$y-y_{wall}$")
    plt.axis([0.6,1.5,0, 0.2])
    plt.title("Wall distance SST-DDES")

def check_contour_plot():
    plt.figure("Check")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, np.maximum(F_DDES,1), 
                 levels = np.linspace(1,1.1,30),
                 cmap = cm.viridis, extend ='max')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5, 0, 1])
    plt.title("Check")
    plt.colorbar()
    
# V.7

nu_t = vis_2d-viscos

def nu_t_ratio():
    plt.figure("V.7.1")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, np.abs(nu_t[:-1,:-1])/viscos, 
                 levels = np.linspace(0, 10,30),
                 cmap = cm.viridis, extend ='max')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5, 0, 1])
    plt.title("$\\frac{\\nu_t}{\\nu}$")
    plt.colorbar()
    
# Modelled Shear Stress

uv_model_2d = np.abs(- nu_t*(dudy + dvdx))
uv_2d = np.abs(uv_2d)

shear_ratio = uv_2d/(uv_2d+uv_model_2d)

def shear_ratio_plot():
    plt.figure("V.7.2")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, shear_ratio[1:,1:], 
                 levels = np.linspace(0, 1,30))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5, 0, 1])
    plt.title("$\\frac{\\tau_{resolved}}{\\tau_{tot}}$")
    plt.colorbar()
    
k_ratio = k_res/(k_res+k_model_2d)

def k_plot():
    plt.figure("V.7.3")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, k_ratio[1:,1:], 
                 levels = np.linspace(0, 1,30))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5, 0, 1])
    plt.title("$\\frac{k_{resolved}}{k_{tot}}$")
    plt.colorbar()

def close_fig():
    plt.close()
    
root = tk.Tk()
close_button = tk.Button(root, text='Close plot', command = close_fig)
close_button.grid(row=0, column=0)

# V.4

label_overview = tk.Label(text="V.4", background="grey")
label_overview.grid(row=0, column=1, sticky='nesw')

button_u_plot = tk.Button(root, text= 'u plot', command = u_plot)
button_u_plot.grid(row=1, column=1, sticky='nesw')

button_switch_fk = tk.Button(root, text= 'fk contour', command = fk_contour)
button_switch_fk.grid(row=2, column=1, sticky='nesw')

button_switch_fk_boundary = tk.Button(root, text= 'fk contour boundary', command = fk_boundary)
button_switch_fk_boundary.grid(row=3, column=1, sticky='nesw')

button_fk_line_given = tk.Button(root, text= 'fk offsets given', command = fk_lines_given)
button_fk_line_given.grid(row=4, column=1, sticky='nesw')

button_fk_line_computed = tk.Button(root, text= 'fk offsets computed', command = fk_lines_computed)
button_fk_line_computed.grid(row=5, column=1, sticky='nesw')


# V.5

label_overview = tk.Label(text="V.5", background="grey")
label_overview.grid(row=0, column=2, sticky='nesw')

button_d_boundary = tk.Button(root, text= 'd boundary des', command = d_boundary_plot)
button_d_boundary.grid(row=1, column=2, sticky='nesw')

button_length_scale_line_PANS = tk.Button(root, text= 'length scale lines PANS', command = length_scale_lines_PANS)
button_length_scale_line_PANS.grid(row=2, column=2, sticky='nesw')

button_length_scale_line_RANS = tk.Button(root, text= 'length scale lines RANS', command = length_scale_lines_RANS)
button_length_scale_line_RANS.grid(row=3, column=2, sticky='nesw')

button_length_scale_line2 = tk.Button(root, text= 'd boundary alt', command = d_boundary_alt_plot)
button_length_scale_line2.grid(row=4, column=2, sticky='nesw')

button_length_scale_line3 = tk.Button(root, text= 'd boundary alt 2', command = d_boundary_alt_2_plot)
button_length_scale_line3.grid(row=5, column=2, sticky='nesw')

button_check_contour_plot = tk.Button(root, text= 'check contour plot', command = check_contour_plot)
button_check_contour_plot.grid(row=6, column=2, sticky='nesw')

# V.7

label_overview = tk.Label(text="V.7", background="grey")
label_overview.grid(row=0, column=3, sticky='nesw')

button_V_7_1 = tk.Button(root, text= 'V.7.1 ', command = nu_t_ratio)
button_V_7_1.grid(row=1, column=3, sticky='nesw')

button_V_7_2 = tk.Button(root, text= 'V.7.2 ', command = shear_ratio_plot)
button_V_7_2.grid(row=2, column=3, sticky='nesw')

button_V_7_3 = tk.Button(root, text= 'V.7.3 ', command = k_plot)
button_V_7_3.grid(row=3, column=3, sticky='nesw')

root.mainloop()
