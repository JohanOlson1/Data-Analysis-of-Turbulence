import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

re =9.36e+5
viscos =1/re

xy_hump_fine = np.loadtxt("xy_hump.dat")
x=xy_hump_fine[:,0]
y=xy_hump_fine[:,1]

ni=314
nj=122

nim1=ni-1
njm1=nj-1

# read data file
vectz=np.genfromtxt("vectz_aiaa_paper.dat",comments="%")
ntstep=vectz[0]
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)dummy(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2D(i,j)
#            write(48,*)rk2D(i,j)
#            write(48,*)vis2D(i,j)  
#            write(48,*)dissp2D(i,j)
#            write(48,*)uvturb(i,j)



nn=12
nst=0
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
iuv_model=range(nst+12,n,nn)

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
uv_model=vectz[iuv_model]/ntstep

# uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u**2
vv=vv-v**2
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
x_2d=np.transpose(np.reshape(x,(nj,ni)))
y_2d=np.transpose(np.reshape(y,(nj,ni)))

# ---- Test
#uv_model = 2*u*v - u*v_inst -u_inst*v 
uv_model_2d=np.transpose(np.reshape(uv_model,(nj,ni)))


# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

x065_off=np.genfromtxt("x065_off.dat",comments="%")
x080_off=np.genfromtxt("x080_off.dat",comments="%")
x090_off=np.genfromtxt("x090_off.dat",comments="%")
x100_off=np.genfromtxt("x100_off.dat",comments="%")
x110_off=np.genfromtxt("x110_off.dat",comments="%")
x120_off=np.genfromtxt("x120_off.dat",comments="%")
x130_off=np.genfromtxt("x130_off.dat",comments="%")

# the funtion dphidx_dy wants x and y arrays to be one cell smaller than u2d. Hence I take away the last row and column below
x_2d_new=np.delete(x_2d,-1,0)
x_2d_new=np.delete(x_2d_new,-1,1)
y_2d_new=np.delete(y_2d,-1,0)
y_2d_new=np.delete(y_2d_new,-1,1)
# compute the gradient
dudx,dudy=dphidx_dy(x_2d_new,y_2d_new,u_2d)
dvdx,dvdy=dphidx_dy(x_2d_new,y_2d_new,v_2d)

# ---- Plot 
# ---- V.1
def u065():
    plt.figure("Figure 1")
    plt.clf() #clear the figure
    xx=0.65;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(u_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x065_off[:,2],x065_off[:,1],'bo')
    plt.xlabel("$U$")
    plt.ylabel("$y$")
    plt.title("$x=0.65$")
    plt.axis([0, 1.3, np.min(y_2d[i1,0]), 0.4])
    
#*************************
# plot vv
def vv065():
    plt.figure("Figure vv065")
    plt.clf() #clear the figure
    xx=0.65;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x065_off[:,5],x065_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=0.65$")
    plt.axis([0, 0.01, np.min(y_2d[i1,0]), 0.4])
    
#*************************
# plot vv
def vv080():
    plt.figure("Figure vv080")
    plt.clf() #clear the figure
    xx=0.80;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x080_off[:,5],x080_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=0.80$")
    plt.axis([0, 0.05, np.min(y_2d[i1,0]), 0.3])

#*************************
# plot vv
def vv090():
    plt.figure("Figure vv090")
    plt.clf() #clear the figure
    xx=0.90;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x090_off[:,5],x090_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=0.80$")
    plt.axis([0, 0.05,np.min(y_2d[i1,0]),0.3])
    
#*************************
# plot vv
def vv100():
    plt.figure("Figure vv100")
    plt.clf() #clear the figure
    xx=1.00;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x100_off[:,5],x100_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.00$")
    plt.axis([0, 0.05,np.min(y_2d[i1,0]),0.3])    

#*************************
# plot vv
def vv110():
    plt.figure("Figure vv110")
    plt.clf() #clear the figure
    xx=1.10;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x110_off[:,5],x110_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.10$")
    plt.axis([0, 0.05,np.min(y_2d[i1,0]),0.3])    

#*************************
# plot vv
def vv120():
    plt.figure("Figure vv120")
    plt.clf() #clear the figure
    xx=1.20;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x120_off[:,5],x120_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.20$")
    plt.axis([0, 0.05,np.min(y_2d[i1,0]),0.3])

#*************************
# plot vv
def vv130():
    plt.figure("Figure vv130")
    plt.clf() #clear the figure
    xx=1.30;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x130_off[:,5],x130_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.30$")
    plt.axis([0, 0.05,np.min(y_2d[i1,0]),0.3])    
    
#*************************
# plot uv
def uv065():
    plt.figure("Figure uv065")
    plt.clf() #clear the figure
    xx=0.65;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(uv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x065_off[:,6],x065_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=0.65$")
    plt.axis([-0.02, 0.02,np.min(y_2d[i1,0]),0.3])
    
#*************************
# plot uv
def uv130():
    plt.figure("Figure uv130")
    plt.clf() #clear the figure
    xx=1.30;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    plt.plot(uv_2d[i1,:],y_2d[i1,:],'b-')
    plt.plot(x130_off[:,6],x130_off[:,1],'bo')
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.30$")
    plt.axis([-0.04, 0.02,np.min(y_2d[i1,0]),0.3])

# ---- V.2

visc_turb = vis_2d - viscos
    

#*************************
# plot uv
def compare_uv065():
    plt.figure("Figure uv065")
    plt.clf() #clear the figure
    xx=0.65;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    
    counter_065 = 0
    for i in range(nj):
        counter_065 += 1
        if visc_turb[i1,i]/viscos < 1.0 and i > 12:
            break
        
    plt.plot(uv_2d[i1,:],y_2d[i1,:],'b-', label= "Resolved")
    plt.plot(uv_model_2d[i1,:],y_2d[i1,:],'r-', label= "Modelled")
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=0.65$")
    plt.axis([-0.02, 0.02,np.min(y_2d[i1,0]), y_2d[i1,counter_065]])
    plt.legend()
    
#*************************
# plot uv
def compare_uv100():
    plt.figure("Figure uv100")
    plt.clf() #clear the figure
    xx=1.00;
    i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
    
    counter_100 = 0
    for i in range(nj):
        counter_100 += 1
        print(counter_100)
        if visc_turb[i1,i]/viscos < 1.0 and i > 12:
            break
        
    plt.plot(uv_2d[i1,:],y_2d[i1,:],'b-', label= "Resolved")
    plt.plot(uv_model_2d[i1,:],y_2d[i1,:],'r-', label = "Modelled")    
    plt.xlabel("$\overline{v'v'}$")
    plt.ylabel("$y$")
    plt.title("$x=1.00$")
    plt.axis([-0.04, 0.02,np.min(y_2d[i1,0]), y_2d[i1,counter_100]]) 
    plt.legend()
    
# ---- V.3

def nu_t_ratio_contour():
    plt.figure("Figure shear ratio")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, visc_turb/viscos, 50)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5,0,1])
    plt.title("contour $\\frac{\\nu_t}{\\nu}$")
    plt.colorbar()
    
nu_t_mod_xx, empty=dphidx_dy(x_2d_new,y_2d_new, visc_turb*dudx)
empty, nu_t_mod_xy =dphidx_dy(x_2d_new,y_2d_new, visc_turb*dudy)

nu_t_mod_yx, empty=dphidx_dy(x_2d_new,y_2d_new, visc_turb*dvdx)
empty, nu_t_mod_yy =dphidx_dy(x_2d_new,y_2d_new, visc_turb*dvdy)

shear_stress_model_x = np.abs(nu_t_mod_xx) + np.abs(nu_t_mod_xy)
shear_stress_model_y = np.abs(nu_t_mod_yx) + np.abs(nu_t_mod_yy)

duudx, empty  =dphidx_dy(x_2d_new,y_2d_new, uu_2d)
duvdx, duvdy  =dphidx_dy(x_2d_new,y_2d_new, uv_2d)
empty, dvvdy  =dphidx_dy(x_2d_new,y_2d_new, vv_2d)

shear_stress_resolved_x = np.abs(- duudx - duvdy)
shear_stress_resolved_y = np.abs(- duvdx - dvvdy)
    
shear_stress_x_tot = shear_stress_model_x + shear_stress_resolved_x
shear_stress_y_tot = shear_stress_model_y + shear_stress_resolved_y

def shear_stress_ratio_x_contour_plot():
    plt.figure("Figure nu_t ratio")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, shear_stress_resolved_x/shear_stress_x_tot, 
                 levels = np.linspace(0,1,30))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5,0,1])
    plt.title("contour $shear ratio$")
    plt.colorbar()

def shear_stress_ratio_y_contour_plot():
    plt.figure("Figure nu_t ratio")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d, shear_stress_resolved_y/shear_stress_y_tot, 
                 levels = np.linspace(0,1,30))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5,0,1])
    plt.title("contour $shear ratio$")
    plt.colorbar()

# ---- Extras Plots
################################ contour plot
def contour():
    plt.figure("Figure 4")
    plt.clf() #clear the figure
    plt.contourf(x_2d,y_2d,uu_2d, 50)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.clim(0,0.05)
    plt.axis([0.6,1.5,0,1])
    plt.title("contour $\overline{u'u'}$")

################################ vector plot
def velocity_vector():
    plt.figure("Figure 5")
    plt.clf() #clear the figure
    k=6# plot every forth vector
    ss=3.2 #vector length
    plt.quiver(x_2d[::k,::k],y_2d[::k,::k],u_2d[::k,::k],v_2d[::k,::k], width=0.01)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis([0.6,1.5,0,1])
    plt.title("vector plot")



def close_fig():
    plt.close()
    
root = tk.Tk()
close_button = tk.Button(root, text='Close plot', command = close_fig)
close_button.grid(row=0, column=0)

# V.1

label_overview = tk.Label(text="V.1", background="grey")
label_overview.grid(row=0, column=1, sticky='nesw')

button_u065 = tk.Button(root, text= 'u065', command = u065)
button_u065.grid(row=1, column=1, sticky='nesw')

button_vv065 = tk.Button(root, text= 'vv065', command = vv065)
button_vv065.grid(row=2, column=1, sticky='nesw')

button_vv080 = tk.Button(root, text= 'vv080', command = vv080)
button_vv080.grid(row=3, column=1, sticky='nesw')

button_vv090 = tk.Button(root, text= 'vv090', command = vv090)
button_vv090.grid(row=4, column=1, sticky='nesw')

button_vv100 = tk.Button(root, text= 'vv100', command = vv100)
button_vv100.grid(row=5, column=1, sticky='nesw')

button_vv110 = tk.Button(root, text= 'vv110', command = vv110)
button_vv110.grid(row=6, column=1, sticky='nesw')

button_vv120 = tk.Button(root, text= 'vv120', command = vv120)
button_vv120.grid(row=7, column=1, sticky='nesw')

button_vv130 = tk.Button(root, text= 'vv130', command = vv130)
button_vv130.grid(row=8, column=1, sticky='nesw')

button_uv065 = tk.Button(root, text= 'uv065', command = uv065)
button_uv065.grid(row=9, column=1, sticky='nesw')

button_uv130 = tk.Button(root, text= 'uv130', command = uv130)
button_uv130.grid(row=10, column=1, sticky='nesw')

# V.2

label_overview = tk.Label(text="V.2", background="grey")
label_overview.grid(row=0, column=2, sticky='nesw')

button_compare_uv065 = tk.Button(root, text= 'Comparison uv065', command = compare_uv065)
button_compare_uv065.grid(row=1, column=2, sticky='nesw')

button_compare_uv100 = tk.Button(root, text= 'Comparison uv100', command = compare_uv100)
button_compare_uv100.grid(row=2, column=2, sticky='nesw')

# V.3

label_overview = tk.Label(text="V.3", background="grey")
label_overview.grid(row=0, column=3, sticky='nesw')

button_nu_t_ratio = tk.Button(root, text= 'Nu_t ratio Contour Plot', command = nu_t_ratio_contour)
button_nu_t_ratio.grid(row=1, column=3, sticky='nesw')

button_shear_stress_ratio_x_contour_plot = tk.Button(root, text= 'Shear Stress Ratio x Contour', command = shear_stress_ratio_x_contour_plot)
button_shear_stress_ratio_x_contour_plot.grid(row=2, column=3, sticky='nesw')

button_shear_stress_ratio_y_contour_plot = tk.Button(root, text= 'Shear Stress Ratio y Contour', command = shear_stress_ratio_y_contour_plot)
button_shear_stress_ratio_y_contour_plot.grid(row=3, column=3, sticky='nesw')

# Extra Plots
button_contour = tk.Button(root, text= 'Contour Plot', command = contour)
button_contour.grid(row=1, column=0, sticky='nesw')

button_velocity = tk.Button(root, text= 'Velocity Vector Field', command = velocity_vector)
button_velocity.grid(row=2, column=0, sticky='nesw')



root.mainloop()