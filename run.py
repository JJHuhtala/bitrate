import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.abspath('src'))
import fastpotential as ts
import wavefunc as w
from utils import Y
from scipy import stats
from bohmian import BohmianSimulation
#If you have ffmpeg, you can use this and the two lines at the bottom to save an animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

#Parameters here. We are using atomic units. 
L         = 25
Npoints   = 628
sigma     = 1./4.
x         = np.linspace(-L, L, Npoints)
y         = np.linspace(-L, L, Npoints)
dx        = x[1]-x[0]
time_unit = 2.4188843265857e-17
timestep  = 0.001
psi       = np.zeros((Npoints,Npoints), dtype=np.cdouble)
V         = np.zeros((Npoints,Npoints),dtype=np.cdouble)
A0        = 5/2
V[:,100]  = 100.0
V[100,:]  = 100.0
num_basis_funcs = 40

"""for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = 0.7*Y(x[i],1.0,1.0,np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi/2,A0) + 0.7*Y(x[i],1.0,1.0,np.pi+np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi+np.pi/2,A0)

def psi_cont(x,y):
    return np.abs(0.7*Y(x,1.0,1.0,np.pi/2,A0)*Y(y,1.0,1.0,np.pi/2,A0) + 0.7*Y(x,1.0,1.0,np.pi+np.pi/2,A0)*Y(y,1.0,1.0,np.pi+np.pi/2,A0))**2
norm = ts.get_norm(psi,Npoints,dx)
psi = psi/np.sqrt(norm)
from scipy.stats import rv_continuous"""

#print(norm)

def integral(fun1,fun2):
    return np.sum(fun1 * fun2 * dx * dx)


def E(n1,n2):
    return (n1**2+n2**2)*(np.pi**2)/(2*(2*L)**2)

coeffs = np.zeros((num_basis_funcs,num_basis_funcs),dtype=complex)
print("Starting coeffs")
coeffs = np.loadtxt("coeffs_nowall.txt", dtype=complex)
coeffs = coeffs/np.sqrt((np.sum(np.abs(coeffs)**2)))

for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = ts.psi(coeffs,x[i],x[j],1.00,L,40)
    print(i)

print(np.sum(np.abs(psi)**2*dx*dx))
"""for i in range(num_basis_funcs):
    for j in range(num_basis_funcs):
        bs = ts.basis2d(x,L,i,j,Npoints)
        coeffs[i,j] = integral(bs,psi)
    print(i)
np.savetxt("coeffs_nowall.txt",coeffs)"""
"""
funbasis = np.zeros((Npoints,Npoints),dtype=complex)

for i in range(num_basis_funcs):
    for j in range(num_basis_funcs):
        funbasis += coeffs[i,j]*ts.basis2d(x,L,i,j,Npoints)*np.exp(-1.0j * E(i,j)* 3.0)
    
    print(i)


plt.imshow(np.abs(funbasis)**2)
plt.show()


#Initial particle position
custm = stats.rv_discrete(name='custm', values=(np.arange(Npoints*Npoints).reshape((Npoints,Npoints)), np.abs(psi)**2*dx*dx))
R = custm.rvs(size=1000)
Rx = np.array([np.count_nonzero(R==y) for y in np.arange(Npoints*Npoints)])
Rxs = Rx.reshape((Npoints,Npoints))
#plt.imshow(Rxs)
cc = 0"""

# Set up figure.
"""fig, ax = plt.subplots()
line = ax.imshow(np.abs(psi)**2,cmap='Greys')
#line = ax.plot(np.imag(psi[]))3
#line2 = ax.plot(np.real(Y(x,1.0,1.0,np.pi,A0)))
#print(Y(0.5,1.0,1.0,np.pi,A0))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
textheight = abs(np.max(psi))**2
plt.title(r'Wave function')
rk4_steps_per_frame = 4
plt.show()"""
#plt.clf()
#plt.cla()
#Animate everything

#bb = BohmianSimulation(psi, x, L, 1000, timestep, 100)
#bb.calculate_trajectories()



# Animation tools not used for now

"""
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
    return line"""
"""
ani = animation.FuncAnimation(fig, animate_2, np.arange(1, 1500), init_func=init,
                              interval=25, save_count=1500)
#plt.show()
FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
ani.save('psiresonance.mp4', writer = FFwriter)
"""
#plt.show()
