import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.abspath('src'))
import fastpotential as ts
import wavefunc as w
from utils import find_nearest,x_deriv, y_deriv, create_wall, Y
#If you have ffmpeg, you can use this and the two lines at the bottom to save an animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
 
# Update psi using a single RK4 step with timestep dt.

def update_psi_rk4(psi, dt):
    k1 = -(1. / (2.j)) * ts.laplace(psi,Npoints,dx) + (1. / 1.j)*V*(psi)
    k2 = -(1. / (2.j)) * ts.laplace(psi + 0.5 * dt * k1,Npoints,dx)+ (1. / 1.j)*V*(psi + 0.5 * dt * k1)
    k3 = -(1. / (2.j)) * ts.laplace(psi + 0.5 * dt * k2,Npoints,dx)+ (1. / 1.j)*V*(psi + 0.5 * dt * k2)
    k4 = -(1. / (2.j)) * ts.laplace(psi + dt * k3,Npoints,dx)+ (1. / 1.j)*V*(psi + dt * k3)
     
    psinew = psi + dt * 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return psinew

#Parameters here. We are using atomic units. 
L         = 8
Npoints   = 201
sigma     = 1./4.
x         = np.linspace(-L, L, Npoints)
y         = np.linspace(-L, L, Npoints)
dx        = x[1]-x[0]
time_unit = 2.4188843265857e-17
timestep  = 0.001
psi       = np.zeros((Npoints,Npoints), dtype=np.cdouble)
kx        = -1000.0
V         = np.zeros((Npoints,Npoints),dtype=np.cdouble)
A0        = 5/2
V[:,:]   = 0.0


for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = 0.7*Y(x[i],1.0,1.0,np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi/2,A0) + 0.7*Y(x[i],1.0,1.0,np.pi+np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi+np.pi/2,A0)
norm = ts.get_norm(psi,Npoints,dx)
#print(norm)
psi = psi/np.sqrt(norm)
#Initial particle position

# Set up figure.
fig, ax = plt.subplots()
line = ax.imshow(np.abs(psi)**2,cmap='Greys')
#line = ax.plot(np.imag(psi[]))3
#line2 = ax.plot(np.real(Y(x,1.0,1.0,np.pi,A0)))
#print(Y(0.5,1.0,1.0,np.pi,A0))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
textheight = abs(np.max(psi))**2
plt.title(r'Wave function')
rk4_steps_per_frame = 4
plt.show()

#Animate everything

def animate(i):
    global psi
    for q in range(rk4_steps_per_frame):
        psinew = update_psi_rk4(psi, timestep)
        psi = psinew

    #ax.patches = []
    currentnorm = ts.get_norm(psi,Npoints,dx)
    #If the norm changes from 1 significantly, the simulation is probably in trouble.
    print(i,currentnorm)
    plotted = abs(psi)**2 
    line.set_data(plotted)  # update the data
    line.set_clim(vmax=np.amax(np.abs(psi)**2))
    line.set_clim(vmin=0)
    return line

ss = w.Schrödinger(psi,V,Npoints,4*1500,L)
ss.simulate()
def animate_2(i):
    line = ss.animate(i)
    return line
    
def init():
    return line

#show or animate
ss.create_movie()
"""
ani = animation.FuncAnimation(fig, animate_2, np.arange(1, 1500), init_func=init,
                              interval=25, save_count=1500)
#plt.show()
FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
ani.save('psiresonance.mp4', writer = FFwriter)
"""
#plt.show()
