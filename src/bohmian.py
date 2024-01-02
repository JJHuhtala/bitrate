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
    def __init__(self, psi0, x, L, coeffs, Nt, dt, startind, weight, start_from_previous=False, initpos=[], t0=0, basis=[],energies=[]):
        self.weight = weight
        self.x = x
        if not start_from_previous:
            self.pos = np.array([x[startind[0]],x[startind[1]]])
        else:
            assert len(initpos)>0
            self.pos = np.array(initpos)
        self.dx = self.x[1]-self.x[0]
        self.dt = dt
        self.psi0 = psi0
        self.traj = [self.pos]
        self.Nt = Nt
        self.L = L
        self.coeffs = coeffs
        self.n_basis = len(coeffs[:,0])
        self.t0=t0
        self.basis = basis 
        self.energies=energies
        self.start_from_previous=start_from_previous
    
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
        t = step*self.dt#+self.t0
        if self.start_from_previous:
            x1ind = find_nearest(self.x,x1)
            x2ind = find_nearest(self.x,x2)
            x1pind = (x1ind+1)%len(self.x)
            x2pind = (x2ind+1)%len(self.x)

            psi = fp.psiwall(self.basis,self.energies,self.coeffs,x1ind,x2ind,t, self.n_basis)
            psix1 = fp.psiwall(self.basis,self.energies,self.coeffs,x1pind,x2ind,t, self.n_basis)
            psix2 = fp.psiwall(self.basis,self.energies,self.coeffs,x1ind,x2pind,t, self.n_basis)
            k1 = np.nan_to_num(((psix1-psi)/(self.dx*psi)).imag)
            k2 = np.nan_to_num(((psix2-psi)/(self.dx*psi)).imag)
            return [self.pos[0] + self.dt * k1, self.pos[1] + self.dt * k2]
        else:
            psi = fp.psi(self.coeffs, x1,x2,t,self.L, self.n_basis)
            k1 = np.nan_to_num( (fp.basis2d_x1deriv(self.coeffs, x1, x2, t, self.L, self.n_basis)/psi).imag )
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
            self.pos = self.euler_step_cython(i)
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
    start_from_previous : bool
        Do we start from a previously obtained position of particles, i.e. some collection of particle positions?
    savelast : bool
        Do we want to save the last positions of the particle for a future continuation?
    initpos : string
        File containing initial positions of particles in case start_from_previous is true.
    basis : 2 dimensional complex array
        Contains the basis functions, in case we are starting from a previous position.
    energies : 1 dimensional double array
        Contains the energy eigenvalues in case we are starting from a previous position.
    
    Notes
    -----
    This code draws from a discrete distribution of initial positions. This is basically equivalent
    to binning the end result distribution, so you might as well do it beforehand. This saves some computation,
    as trajectories starting at the same point need to only be calculated once.
    """
    def __init__(self,psi0, x, L, Nt, dt, t0=0, Ntraj = 1000, coeff_file="coeffs_nowall.txt", output="trajs.npy", start_from_previous=False, savelast=False, initpos="NONE",
                 basis=[],energies=[]):
        self.psi0 = psi0
        self.Np = len(x)
        self.x = x
        self.dx = x[1]-x[0]
        self.Ntraj = Ntraj
        self.dt = dt
        self.Nt = Nt
        self.L = L
        self.coeffs = np.loadtxt(coeff_file, dtype=complex)
        self.coeffs = self.coeffs/np.sqrt(np.sum(np.abs(self.coeffs)**2))
        self.output = output
        self.start_from_previous = start_from_previous
        self.initpos=initpos
        self.savelast=savelast
        self.t0=t0
        self.basis = basis
        self.energies = energies
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
            The file is indexed such that arr[0,:,0] contains the first trajectory, with arr[0,:,0] containing
            the sequence of x1 coordinates and arr[0,:,1] containing the x2 coordinates.
        """
        trajectories = []
        final_pos = []
        if not self.start_from_previous:
            print("Drawing from an initial psi distribution...")
            R = self.generate_initial_distribution()
            Rx = np.array([np.count_nonzero(R==y) for y in np.arange(self.Np*self.Np)])
            Rxs = Rx.reshape((self.Np,self.Np)) # This now contains how many 
            for i in range(len(Rxs[0,:])):
                for j in range(len(Rxs[0,:])):
                    if Rxs[i,j] != 0:
                        trajectories.append(Trajectory(self.psi0, self.x,self.L, self.coeffs, self.Nt, self.dt, (i,j),  Rxs[i,j]))
        else:
            ip = np.load(self.initpos)
            print("Starting from previous positions.. ")
            for i in range(len(ip)):
                print(ip[i,2])
                trajectories.append(Trajectory(self.psi0, self.x,self.L, self.coeffs, self.Nt, self.dt, (0,0),  ip[i,2],initpos=ip[i,:2],t0=self.t0,
                                               basis=self.basis,energies=self.energies,start_from_previous=True))

        print("Number of different trajectories: ", len(trajectories), ", starting calculation..")
        traj_xs = []
        counter = 0
        for i in trajectories:
            i.compute_trajectory()
            if self.savelast:
                final_pos.append([i.get_trajectory()[-1][0],i.get_trajectory()[-1][1],i.get_weight()])
            for k in range(int(i.get_weight())):
                traj_xs.append(i.get_trajectory())
                
            if counter%10 == 0:
                print("Calculated trajectory: ", counter)
            counter += 1
        np.save(self.output,np.array(traj_xs))
        if self.savelast:
            np.save("temp.npy",np.array(final_pos))

if __name__=="__main__":
    pass