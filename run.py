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
Npoints   = 100
sigma     = 1./4.
x         = np.linspace(-L, L, Npoints)
y         = np.linspace(-L, L, Npoints)
dx        = x[1]-x[0]
time_unit = 2.4188843265857e-17
timestep  = 0.003
psi       = np.zeros((Npoints,Npoints), dtype=np.cdouble)
V         = np.zeros((Npoints,Npoints),dtype=np.cdouble)
A0        = 5/2
#V[:,100]  = 100.0
#V[100,:]  = 100.0
num_basis_funcs = 100

for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = 0.7*Y(x[i],1.0,1.0,np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi/2,A0) + 0.7*Y(x[i],1.0,1.0,np.pi+np.pi/2,A0)*Y(x[j],1.0,1.0,np.pi+np.pi/2,A0)

def psi_cont(x,y):
    return np.abs(0.7*Y(x,1.0,1.0,np.pi/2,A0)*Y(y,1.0,1.0,np.pi/2,A0) + 0.7*Y(x,1.0,1.0,np.pi+np.pi/2,A0)*Y(y,1.0,1.0,np.pi+np.pi/2,A0))**2
norm = ts.get_norm(psi,Npoints,dx)
psi = psi/np.sqrt(norm)
#from scipy.stats import rv_continuous

#print(norm)

def integral(fun1,fun2):
    return np.sum(fun1 * fun2 * dx * dx)


def E(n1,n2):
    return (n1**2+n2**2)*(np.pi**2)/(2*(2*L)**2)

coeffs = np.zeros((num_basis_funcs,num_basis_funcs),dtype=complex)
print("Starting coeffs")
coeffs = np.loadtxt("coeffs_nowall.txt", dtype=complex)
coeffs = coeffs/np.sqrt((np.sum(np.abs(coeffs)**2)))
"""
for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = ts.psi(coeffs,x[i],x[j],1.00,L,40)
    print(i)

print(np.sum(np.abs(psi)**2*dx*dx))"""
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


"""
# For getting 50k trajectories with 1000 timesteps:
#bb = BohmianSimulation(psi, x, L, 1000, timestep, Ntraj=50000, savelast=True)
#bb.calculate_trajectories()


print("Now calculating the eigenvectors and -values with a wall..")
dx = x[1]-x[0]
wall = np.zeros(Npoints)
wall[34] = 1000
wallm = np.diag(wall)
Id = np.eye(Npoints)
kin = np.diag(-2 * np.ones(Npoints)) + np.diag(np.ones(Npoints-1),1) + np.diag(np.ones(Npoints-1),-1)
kin = kin/(-2*dx*dx)
Hpot = np.kron(Id,wallm) + np.kron(wallm,Id)
Hkin = np.kron(Id,kin) + np.kron(kin,Id)

e,vec = np.linalg.eigh(Hpot+Hkin)
nvec = []
for i in range(num_basis_funcs*num_basis_funcs):
    nvec.append(vec[:,i].reshape((Npoints,Npoints)))
nvec = np.array(nvec)

# Calculating the coefficients for the wall case after 50 timesteps without a wall

"""print("Calculating psi to nstep = 50")
for i in range(Npoints):
    for j in range(Npoints):
        psi[i,j] = ts.psi(coeffs,x[i],x[j],50*timestep,L,40)
    print("Iterating over psi: ", i, " out of ", Npoints)

print("Calculated the eigenvectors and values, calculating")
coeffs = np.zeros((num_basis_funcs,num_basis_funcs))
for i in range(num_basis_funcs):
    for j in range(num_basis_funcs):
        bs = nvec[i*num_basis_funcs + j]
        bs = bs/np.sqrt(np.sum(bs**2*dx*dx))
        coeffs[i,j] = integral(bs,psi)
    print("Iterating over coefficients: ", i, " out of ", num_basis_funcs)
np.savetxt("coeffs_wall_n50.txt",coeffs)
norm = ts.get_norm(psi,Npoints,dx)
psi = psi/np.sqrt(norm)"""
"""coeffs = np.loadtxt("coeffs_wall_n50.txt", dtype=complex)
coeffs = coeffs/np.sqrt(np.sum(np.abs(coeffs)**2))"""
"""
for i in range(Npoints):
    print("comp iter, ", i)
    for j in range(Npoints):
        print("yo")
        psi[i,j] = ts.psiwall(nvec,e,coeffs,i,j,0,num_basis_funcs)
        print("eh")
    print("Computing psi, iteration ", i)
print("here we are")
plt.imshow(np.abs(psi)**2)
plt.show()"""


# For the appearing wall:

#bb = BohmianSimulation(psi, x, L, 50, timestep, Ntraj=50000,savelast=True)
bb = BohmianSimulation(psi, x, L, 950, timestep, t0=timestep*50, Ntraj=50000, coeff_file="coeffs_wall_n50.txt", start_from_previous=True, initpos="temp.npy", basis=nvec,energies=e)
bb.calculate_trajectories()