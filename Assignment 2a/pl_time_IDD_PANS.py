import numpy as np
import pandas
import tkinter as tk
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

# ---- Plots 
def node_1_2_3_velocity_v1():
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,u1,'b-', label='$Node\:\: 1$')
    plt.plot(t,u2,'r-',  label='$Node\:\: 2$')
    plt.plot(t,u3,'k-',  label='$Node\:\: 3$')
    plt.title('$v_1$ each node')
    plt.xlabel("$t$")
    plt.ylabel("$v_1$")
    plt.legend()
    
    # zoom
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,u1,'b-', label='$Node\:\: 1$')
    plt.plot(t,u2,'r-',  label='$Node\:\: 2$')
    plt.plot(t,u3,'k-',  label='$Node\:\: 3$')
    plt.title('$v_1$ each node, zoom')
    plt.xlabel("$t$")
    plt.ylabel("$v_1$")
    plt.legend()
    
    plt.axis([6, 7, 7, 30])
    
def node_1_2_3_velocity_v3():
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,w1,'b-', label='$Node\:\: 1$')
    plt.plot(t,w2,'r-',  label='$Node\:\: 2$')
    plt.plot(t,w3,'k-',  label='$Node\:\: 3$')
    plt.title('$v_3$ each node')
    plt.xlabel("$t$")
    plt.ylabel("$v_3$")
    plt.legend()
    
    # zoom
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,w1,'b-', label='$Node\:\: 1$')
    plt.plot(t,w2,'r-',  label='$Node\:\: 2$')
    plt.plot(t,w3,'k-',  label='$Node\:\: 3$')
    plt.title('$v_3$ each node, zoom')
    plt.xlabel("$t$")
    plt.ylabel("$v_3$")
    plt.legend()
    
    plt.axis([6, 7, -3, 3])
    
def node_4_5_velocity_v1():
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,u4,'m-', label='$Node\:\: 4$')
    plt.plot(t,u5,'g-',  label='$Node\:\: 5$')
    plt.title('$v_1$ each node')
    plt.xlabel("$t$")
    plt.ylabel("$v_1$")
    plt.legend()
    
    # zoom
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,u4,'m-', label='$Node\:\: 4$')
    plt.plot(t,u5,'g-',  label='$Node\:\: 5$')
    plt.title('$v_1$ each node, zoom')
    plt.xlabel("$t$")
    plt.ylabel("$v_1$")
    plt.legend()
    
    plt.axis([6, 7, 7, 30])
    
def node_4_5_velocity_v3():
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,w4,'m-', label='$Node\:\: 4$')
    plt.plot(t,w5,'g-',  label='$Node\:\: 5$')
    plt.title('$v_3$ each node')
    plt.xlabel("$t$")
    plt.ylabel("$v_3$")
    plt.legend()
    
    # zoom
    fig1,ax1 = plt.subplots()
    plt.subplots_adjust(left=0.20,bottom=0.20)
    plt.plot(t,w4,'m-', label='$Node\:\: 4$')
    plt.plot(t,w5,'g-',  label='$Node\:\: 5$')
    plt.title('$v_3$ each node, zoom')
    plt.xlabel("$t$")
    plt.ylabel("$v_3$")
    plt.legend()
    
    plt.axis([6, 7, -3, 3])
        
# Autocorrelation

## node 1 v_1
u1_fluct=u1-np.mean(u1)
node1v1=np.correlate(u1_fluct,u1_fluct,'full')
# find max
nmax_node1v1=np.argmax(node1v1)
# and its value
node1v1_max=np.max(node1v1)   
# two_max is symmwetric. Pick the right half and normalize
node1v1_sym_norm=node1v1[nmax_node1v1:]/node1v1_max

int_T_1 = np.trapz(node1v1_sym_norm)*dt

## node 2 v_1
u2_fluct=u2-np.mean(u2)
node2v1=np.correlate(u2_fluct, u2_fluct,'full')
# find max
nmax_node2v1=np.argmax(node2v1)
# and its value
node2v1_max=np.max(node2v1)   
# two_max is symmwetric. Pick the right half and normalize
node2v1_sym_norm=node2v1[nmax_node2v1:]/node2v1_max

int_T_2 = np.trapz(node2v1_sym_norm)*dt

#3 node 3 v_1
u3_fluct=u3-np.mean(u3)
node3v1=np.correlate(u3_fluct, u3_fluct,'full')
# find max
nmax_node3v1=np.argmax(node3v1)
# and its value
node3v1_max=np.max(node3v1)   
# two_max is symmwetric. Pick the right half and normalize
node3v1_sym_norm=node3v1[nmax_node3v1:]/node3v1_max

int_T_3 = np.trapz(node3v1_sym_norm)*dt

## node 4 v_1
u4_fluct=u4-np.mean(u4)
node4v1=np.correlate(u4_fluct, u4_fluct,'full')
# find max
nmax_node4v1=np.argmax(node4v1)
# and its value
node4v1_max=np.max(node4v1)   
# two_max is symmwetric. Pick the right half and normalize
node4v1_sym_norm=node4v1[nmax_node4v1:]/node4v1_max

int_T_4 = np.trapz(node4v1_sym_norm)*dt

# node 5 v_1
u5_fluct=u5-np.mean(u5)
node5v1=np.correlate(u5_fluct, u5_fluct,'full')
# find max
nmax_node5v1=np.argmax(node5v1)
# and its value
node5v1_max=np.max(node5v1)   
# two_max is symmwetric. Pick the right half and normalize
node5v1_sym_norm=node5v1[nmax_node5v1:]/node5v1_max

int_T_5 = np.trapz(node5v1_sym_norm)*dt



imax=500;
def autocorr_node_1():
    plt.figure("autocorr_node_1")
    plt.title('Autocorrelation Node 1: $v_1$')
    plt.plot(t[0:imax],node1v1_sym_norm[0:imax],'b-')
    plt.xlabel('t')
    plt.ylabel('$B_{v_1 v_1}$')
    
def autocorr_node_2():
    plt.figure("autocorr_node_2")
    plt.title('Autocorrelation Node 2: $v_1$')
    plt.plot(t[0:imax],node2v1_sym_norm[0:imax],'b-')
    plt.xlabel('t')
    plt.ylabel('$B_{v_1 v_1}$')

def autocorr_node_3():
    plt.figure("autocorr_node_3")
    plt.title('Autocorrelation Node 2: $v_1$')
    plt.plot(t[0:imax],node3v1_sym_norm[0:imax],'b-')
    plt.xlabel('t')
    plt.ylabel('$B_{v_1 v_1}$')

def autocorr_node_4():
    plt.figure("autocorr_node_4")
    plt.title('Autocorrelation Node 2: $v_1$')
    plt.plot(t[0:imax],node4v1_sym_norm[0:imax],'b-')
    plt.xlabel('t')
    plt.ylabel('$B_{v_1 v_1}$')

def autocorr_node_5():
    plt.figure("autocorr_node_5")
    plt.title('Autocorrelation Node 2: $v_1$')
    plt.plot(t[0:imax],node5v1_sym_norm[0:imax],'b-')
    plt.xlabel('t')
    plt.ylabel('$B_{v_1 v_1}$')
    
def integral_timescales_v_1():
    node = ["Node 1:", "Node 2:", "Node 3:", "Node 4:", "Node 5:"]
    timescales = [int_T_1, int_T_2, int_T_3, int_T_4, int_T_5]
    
    table = pandas.DataFrame(timescales, node)
    
    print(table)
    
#---- GUI Append

def close_fig():
    plt.close()
    
root = tk.Tk()
close_button = tk.Button(root, text='Close plot', command = close_fig)
close_button.grid(row=0, column=0)

# Overview Plots
label_overview = tk.Label(text="U1", background="grey")
label_overview.grid(row=0, column=1, sticky='nesw')

button_U1_123_v1 = tk.Button(root, text= 'node 1, 2, 3, v_1', command = node_1_2_3_velocity_v1)
button_U1_123_v1.grid(row=1, column=1, sticky='nesw')

button_U1_123_v3 = tk.Button(root, text= 'node 1, 2, 3, v_3', command = node_1_2_3_velocity_v3)
button_U1_123_v3.grid(row=2, column=1, sticky='nesw')

button_U1_45_v1 = tk.Button(root, text= 'node 4, 5, v_1', command = node_4_5_velocity_v1)
button_U1_45_v1.grid(row=3, column=1, sticky='nesw')

button_U1_45_v3 = tk.Button(root, text= 'node 4, 5 v_3', command = node_4_5_velocity_v3)
button_U1_45_v3.grid(row=4, column=1, sticky='nesw')

button_U1_autocorr_N1_v_1 = tk.Button(root, text= 'Autocorr Node 1, v_1', command = autocorr_node_1)
button_U1_autocorr_N1_v_1.grid(row=5, column=1, sticky='nesw')

button_U1_autocorr_N2_v_1 = tk.Button(root, text= 'Autocorr Node 2, v_1', command = autocorr_node_2)
button_U1_autocorr_N2_v_1.grid(row=6, column=1, sticky='nesw')

button_U1_autocorr_N3_v_1 = tk.Button(root, text= 'Autocorr Node 3, v_1', command = autocorr_node_3)
button_U1_autocorr_N3_v_1.grid(row=7, column=1, sticky='nesw')

button_U1_autocorr_N4_v_1 = tk.Button(root, text= 'Autocorr Node 4, v_1', command = autocorr_node_4)
button_U1_autocorr_N4_v_1.grid(row=8, column=1, sticky='nesw')

button_U1_autocorr_N5_v_1 = tk.Button(root, text= 'Autocorr Node 5, v_1', command = autocorr_node_5)
button_U1_autocorr_N5_v_1.grid(row=9, column=1, sticky='nesw')

button_U1_timescale_table = tk.Button(root, text= 'Integral Timescales', command = integral_timescales_v_1)
button_U1_timescale_table.grid(row=10, column=1, sticky='nesw')

root.mainloop()