from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import fastpotential as fp
from utils import find_nearest,x_deriv, y_deriv
class Trajectory():

    r"""A one-line summary that does not use variable names or the
    function name.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.
    type_without_description

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation
    """
    def __init__(self, psi0, x, L, coeffs, Nt, dt, startind, weight):
        """
        here's a random docs
        """
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
        x1 = self.pos[0]
        x2 = self.pos[1]
        psi = 0
        k1 = np.nan_to_num( (fp.basis2d_x1deriv(self.coeffs, x1, x2, self.L, self.n_basis)/psi).imag  )
        k2 = np.nan_to_num( (fp.basis2d_x2deriv(self.coeffs, x1, x2, self.L, self.n_basis)/psi).imag )
    def compute_trajectory(self):
        for i in range(self.Nt):
            self.pos = self.euler_step(i)
            self.traj.append(np.copy(self.pos))

    def get_trajectory(self):
        return np.array(self.traj)

    def get_weight(self):
        return self.weight

class BohmianSimulation():
    def __init__(self,psi0, x, L, Np, Nt, dt, Ntraj = 1000, coeff_file="coeffs_nowall.txt"):
        self.psi0 = psi0
        self.Np = Np
        self.x = x
        self.dx = x[1]-x[0]
        self.Ntraj = Ntraj
        self.dt = dt
        self.Nt = Nt
        self.L = L
        self.coeffs = np.loadtxt(coeff_file, dtype=complex)
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
                    trajectories.append(Trajectory(self.psi0, self.x, self.Nt, self.dt, (i,j),  Rxs[i,j]))

        print("Number of different trajectories: ", len(trajectories), ", starting calculation..")
        traj_xs = []
        for i in trajectories:
            i.compute_trajectory()
            for k in range(i.get_weight()):
                traj_xs.append(i.get_trajectory())
        
        np.save("testtraj.npy",np.array(traj_xs))

if __name__=="__main__":
    pass