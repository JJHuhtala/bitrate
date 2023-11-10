from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import fastpotential as fp
from utils import find_nearest,x_deriv, y_deriv
class Trajectory():

    """A class describing a single Bohmian trajectory with a given weight.

    Parameters
    ----------
    psi0 : 2 dimensional complex ndarray
        The initial wave function to start the simulation from.
    x : 1 dimensional double ndarray
        The grid points for the box along one dimensional. The other dimension is forced to be the same
    L : double
        Half the length of the box.
    coeffs : 2 dimensional complex ndarray
        The coefficients for the basis expansion of psi.
    Nt : integer
        Number of timesteps to take
    dt : double
        Size of the timestep
    startind : integer tuple
        The starting position index from the x-array. 
    weight : integer
        How many trajectories start from the same index.
    
    Notes
    -----
    The class computes the trajectory only on command; by default, it contains some information about the weight of the trajectory
    (i.e. how many times the particle started in the initial position), the inital wave function and the coefficient required for
    the expansion. 
    """
    def __init__(self, psi0, x, L, coeffs, Nt, dt, startind, weight):
        self.weight = weight
        self.x = x
        self.pos = np.array([x[startind[0]],x[startind[1]]])
        self.dx = self.x[1]-self.x[0]
        self.dt = dt
        self.psi0 = psi0
        self.traj = [self.pos]
        self.Nt = Nt
        self.L = L
        self.coeffs = coeffs
        self.n_basis = len(coeffs[:,0])
    def plot_xs(self):
        """this is a description of the plot thing."""
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

    def euler_step_cython(self,step):
        """Take Euler method step in the trajectory calculation.

        Returns
        -------
        pos : tuple (double)
            The new coordinates of the particle in (x1,x2) space.

        """
        x1 = self.pos[0]
        x2 = self.pos[1]
        t = step*self.dt
        psi = fp.psi(self.coeffs, x1,x2,t,self.L, self.n_basis)
        k1 = np.nan_to_num( (fp.basis2d_x1deriv(self.coeffs, x1, x2, t, self.L, self.n_basis)/psi).imag  )
        k2 = np.nan_to_num( (fp.basis2d_x2deriv(self.coeffs, x1, x2, t, self.L, self.n_basis)/psi).imag )
        return [self.pos[0] + self.dt * k1, self.pos[1] + self.dt * k2]
    
    def compute_trajectory(self):
        """Computes the trajectory for Nt time steps.
        
        Notes
        -----
        This is an internal function that modifies the trajectory self.traj. That variable contains
        as a numpy array the positions of the particle at each timestamp.
        """
        for i in range(self.Nt):
            self.pos = self.euler_step_cython()
            self.traj.append(np.copy(self.pos))

    def get_trajectory(self):
        """Returns the trajectory as a numpy array.
        
        Returns
        -------
        traj : 2 dimensional ndarray
            The trajectory.
        """
        return np.array(self.traj)

    def get_weight(self):
        """Returns the weight of this trajectory
        
        Returns
        -------
        weight : integer
            The weight of the trajectory, i.e. how many times the particle started at this position.
        """
        return self.weight

class BohmianSimulation():
    """ A class containing a full Bohmian simulation. It simulates a given number of trajectories when given a coefficient file
    for the basis expansion of the wave function. The trajectories are then saved in the file specified by "output" as a .npy 
    file, loadable by numpy. The user should only call the constructor and calculate_trajectories.

    Parameters
    ----------
    psi0 : 2 dimensional complex ndarray
        The initial NORMALIZED wave function for the particles
    x : 1 dimensional double ndarray
        The grid points along one dimension, of length Np
    L : double
        Half the box length
    Nt : integer
        Number of time steps to take
    dt : double
        The length of the time step
    Ntraj : integer
        The number of trajectories to compute, given the initial distribution based on psi0
    coeff_file : string
        The .txt file containing the coefficients for the basis expansion of psi.
    output : string
        The output file, in to which the trajectories are dumped as a numpy array (which can be loaded with np.load).
    
    Notes
    -----
    This code draws from a discrete distribution of initial positions. This is basically equivalent
    to binning the end result distribution, so you might as well do it beforehand. This saves some computation,
    as trajectories starting at the same point need to only be calculated once.
    """
    def __init__(self,psi0, x, L, Nt, dt, Ntraj = 1000, coeff_file="coeffs_nowall.txt", output="trajs.npy"):
        self.psi0 = psi0
        self.Np = len(x)
        self.x = x
        self.dx = x[1]-x[0]
        self.Ntraj = Ntraj
        self.dt = dt
        self.Nt = Nt
        self.L = L
        self.coeffs = np.loadtxt(coeff_file, dtype=complex)
        self.output = output
    def generate_initial_distribution(self):
        """Generates the initial distribution of indices in the psi0 array. Looks a bit confusing, but easily understood
        by perusing the rv_discrete documentation
        
        Returns
        -------
        dist 
            The collection of indices randomly drawn based on the psi distribution, corresponding to initial starting positions.
        
        Notes
        -----
        Depends on the initial distribution being given, and thus shouldn't be called outside of the class.
        """
        equilibrium = stats.rv_discrete(name='equilibrium', values=(np.arange(self.Np*self.Np).reshape((self.Np,self.Np)), np.abs(self.psi0)**2*self.dx*self.dx))
        return equilibrium.rvs(size=self.Ntraj)

    def calculate_trajectories(self):
        """Calculates Nt trajectories, giving them weights according to how many particles start at the same position in the initial
        distribution.

        Returns
        -------
        output : .npy file
            The output is given in the form of a .npy file, which can then be later loaded with np.load.
            The file is indexed such that arr[0,:,:] contains the first trajectory, with arr[0,:,0] containing
            the sequence of x1 coordinates and arr[0,0,:] containing the x2 coordinates.
        """
        R = self.generate_initial_distribution()
        Rx = np.array([np.count_nonzero(R==y) for y in np.arange(self.Np*self.Np)])
        Rxs = Rx.reshape((self.Np,self.Np)) # This now contains how many 
        trajectories = []
        for i in range(len(Rxs[0,:])):
            for j in range(len(Rxs[0,:])):
                if Rxs[i,j] != 0:
                    trajectories.append(Trajectory(self.psi0, self.x,self.L, self.coeffs, self.Nt, self.dt, (i,j),  Rxs[i,j]))

        print("Number of different trajectories: ", len(trajectories), ", starting calculation..")
        traj_xs = []
        for i in trajectories:
            i.compute_trajectory()
            for k in range(i.get_weight()):
                traj_xs.append(i.get_trajectory())
            if i%10 == 0:
                print("Calculated trajectory: ", i)
        
        np.save(self.output,np.array(traj_xs))

if __name__=="__main__":
    pass