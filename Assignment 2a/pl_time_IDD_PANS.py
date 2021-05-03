import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hann
from IPython import display
plt.rcParams.update({'font.size': 22})


# needed when using ipython
plt.interactive(True)

# data from ~/noback/les_aalborg/channel_5200_pans_iddes_xmax3.2_ni-nk-32-friess_MTF271

# ***** read u
data = np.genfromtxt("u_w_time_5nodes_IDD_PANS.dat", dtype=None)


u1=data[:,0]   #v_1 at point 1
u2=data[:,1]   #v_1 at point 2
u3=data[:,2]   #v_1 at point 3
u4=data[:,3]   #v_1 at point 4
u5=data[:,4]   #v_1 at point 5

w1=data[:,5]   #w_1 at point 1
w2=data[:,6]   #w_1 at point 2
w3=data[:,7]   #w_1 at point 3
w4=data[:,8]   #w_1 at point 4
w5=data[:,9]   #w_1 at point 5

print("u1=",u1)


dx=3.2/32
dt= 0.25*dx/20
t_tot=dt*len(u1)

t = np.linspace(0,t_tot,len(u1))

# %%%%%%%%%%%%%%%% plotting section %%%%%%%%%%%%%%%%%%%%%%%%%%
# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(t,u1,'b--')
plt.plot(t,u4,'r-')
plt.xlabel("$t$")
plt.ylabel("$u$")

plt.savefig('utime_python.eps',bbox_inches='tight')

####################### # zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(t,u1,'b--')
plt.plot(t,u4,'r-')
plt.xlabel("$t$")
plt.ylabel("$u$")

plt.axis([6, 7, 7, 22])

plt.savefig('utime_zoom_python.eps',bbox_inches='tight')