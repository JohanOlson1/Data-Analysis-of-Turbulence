import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


# x=0.65
#   do itstep= ....
#         i=36
#         do j=10,100,20
#         do k=2,nk/2
#            write(67,*) phi(i,j,k,w)
#         end do
#         end do
#         write(67,*)ivisz
#   end do


w_time = np.loadtxt("w_time_z65.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

w_time_org=w_time

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_65= np.swapaxes(w_y_z_t,0,2)  # the order of the indices are (k,j,t)
w_y_z_t_65= np.swapaxes(w_y_z_t_65,0,1)



# x=0.8
w_time = np.loadtxt("w_time_z80.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_80= np.swapaxes(w_y_z_t,0,2)
w_y_z_t_80= np.swapaxes(w_y_z_t_80,0,1)



# x=1.1 
w_time = np.loadtxt("w_time_z110.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_110= np.swapaxes(w_y_z_t,0,2)
w_y_z_t_110= np.swapaxes(w_y_z_t_110,0,1)


# x=1.3 
w_time = np.loadtxt("w_time_z130.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_130= np.swapaxes(w_y_z_t,0,2)
w_y_z_t_130= np.swapaxes(w_y_z_t_130,0,1)

z = np.linspace(0,0.1,nk)

dz = 0.2/32

# x = 0.65
B33_0 = np.zeros((nj, nk))
L_int_0 = np.zeros(nj)

# x = 0.80
B33_1 = np.zeros((nj, nk))
L_int_1 = np.zeros(nj)

# x = 1.10
B33_2 = np.zeros((nj, nk))
L_int_2 = np.zeros(nj)

# x = 1.30
B33_3 = np.zeros((nj, nk))
L_int_3 = np.zeros(nj)


for j in range(nj):
    for k in range(nk):
        for m in range(nk):
            for n in range(nt):
                B33_0[j,k] += w_y_z_t_65[j, m, n]*w_y_z_t_65[j, k, n]/nt  
                B33_1[j,k] += w_y_z_t_80[j, m, n]*w_y_z_t_80[j, k, n]/nt  
                B33_2[j,k] += w_y_z_t_110[j, m, n]*w_y_z_t_110[j, k, n]/nt  
                B33_3[j,k] += w_y_z_t_130[j, m, n]*w_y_z_t_130[j, k, n]/nt  
                
    B33_0[j,:] /= B33_0[j,0]
    B33_1[j,:] /= B33_1[j,0]
    B33_2[j,:] /= B33_2[j,0]
    B33_3[j,:] /= B33_3[j,0]

    L_int_0[j] = np.trapz(B33_0[j])*dz
    L_int_1[j] = np.trapz(B33_1[j])*dz
    L_int_2[j] = np.trapz(B33_2[j])*dz
    L_int_3[j] = np.trapz(B33_3[j])*dz

resolution_number = np.array([L_int_0, L_int_1, L_int_2, L_int_3])/dz


xy= np.loadtxt("hump_grid_nasa_les_coarse_noflow.dat")
x1=xy[:,0]
y1=xy[:,1]

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


dx = np.diff(x_2d, axis = 0)

dx = np.insert(dx,0, dx[-1,:], axis=0)

dy = np.diff(y_2d, axis = 1)

dy = np.insert(dy,0, dy[:,-1], axis = 1)



