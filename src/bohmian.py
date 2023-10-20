from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from utils import find_nearest,x_deriv, y_deriv
class Trajectory():
    def __init__(self, psis, x, Nt, dt, startind, weight):
        self.weight = weight
        self.x = x
        self.pos = np.array([x[startind[0]],x[startind[1]]])
        self.dx = self.x[1]-self.x[0]
        self.dt = dt
        self.psis = psis
        self.traj = [self.pos]
        self.Nt = Nt
    def plot_xs(self):
        ntraj = np.array(self.traj)
        print(np.max(ntraj[:,0]))
        print(ntraj[:,0])
    
    def euler_step(self,step):
        xcoord = find_nearest(self.x,self.pos[0])
        ycoord = find_nearest(self.x,self.pos[1])
        psi = self.psis[:,:,step]
        k1 = np.nan_to_num((x_deriv(psi,xcoord,ycoord,self.dx)/psi[xcoord,ycoord]).imag )
        k2 = np.nan_to_num((y_deriv(psi,xcoord,ycoord,self.dx)/psi[xcoord,ycoord]).imag )
        return [self.pos[0] + self.dt * k1, self.pos[1] + self.dt * k2]

    def compute_trajectory(self):
        for i in range(self.Nt):
            self.pos = self.euler_step(i)
            self.traj.append(np.copy(self.pos))
    def get_trajectory(self):
        return np.array(self.traj)

    def get_weight(self):
        return self.weight

class BohmianSimulation():
    def __init__(self,psis,x, Np, Nt, dt, Ntraj = 1000):
        self.psis = psis
        self.Np = Np
        self.x = x
        self.dx = x[1]-x[0]
        self.Ntraj = Ntraj
        self.dt = dt
        self.Nt = Nt
    
    def generate_initial_distribution(self):
        psi_init = self.psis[:,:,0] # The first in the list corresponds to the initial condition.
        equilibrium = stats.rv_discrete(name='equilibrium', values=(np.arange(self.Np*self.Np).reshape((self.Np,self.Np)), np.abs(psi_init)**2*self.dx*self.dx))
        return equilibrium.rvs(size=self.Ntraj)

    def calculate_trajectories(self):
        R = self.generate_initial_distribution()
        Rx = np.array([np.count_nonzero(R==y) for y in np.arange(self.Np*self.Np)])
        Rxs = Rx.reshape((self.Np,self.Np)) # This now contains how many 
        trajectories = []
        for i in range(len(Rxs[0,:])):
            for j in range(len(Rxs[0,:])):
                if Rxs[i,j] != 0:
                    trajectories.append(Trajectory(self.psis, self.x, self.Nt, self.dt, (i,j),  Rxs[i,j]))

        print("Number of different trajectories: ", len(trajectories), ", starting calculation..")
        traj_xs = []
        for i in trajectories:
            i.compute_trajectory()
            for k in range(i.get_weight()):
                traj_xs.append(i.get_trajectory())
        
        np.save("testtraj.npy",np.array(traj_xs))