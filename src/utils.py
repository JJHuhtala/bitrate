import numpy as np
import fastpotential as ts
#Find the nearest index in an array corresponding to a value.
#Used to convert particle position to grid point (i,j)
def find_nearest(array, value):
    """Finds the index of the element nearest to a given value.
    
    Parameters
    ----------
    array : list or array
        the array we look for the value in
    value : double
        the value we are looking for in the array

    Examples
    --------
    >>> array = [0,1,2]
    >>> value = 1.2
    >>> idx = find_nearest(array,value)
    >>> print(idx)
    1

    Returns
    -------
    idx : integer
        The array index for the element closest to "value".
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#Self explanatory
def x_deriv(psi,i,j,dx):
    """XDERIV"""
    return (psi[i+1,j]-psi[i,j])/dx

def y_deriv(psi,i,j,dx):
    """YDERIV"""
    return (psi[i,j+1]-psi[i,j])/dx


def xi(sigma, A0):
    """Helper function for an entangled state, not to be called by the user.
    
    Parameters
    ----------
    sigma : double
        controls the width of a wave packet
    A0 : double
        numerical constant controlling the shape of the wave packet
    
    
    Returns
    -------
    xi
        The value of xi.

    References
    ----------
    Tzemos, Contopoulos, Efthymiopoulos "Bohmian trajectories in an entangled two-qubit system" (2019) arxiv:1905.12619
    
    """
    return 0.5*np.abs(A0)**2*np.sin(-2*sigma)

def Y(x,m,omega,sigma,A0):
    """Helper function for creating entangled wave functions. Gives the value of Y in ref. [1] at x.

    Parameters
    ----------
    x : double
        Location in x-space
    m : double
        Mass of the particle
    omega : double
        Spread of the function
    A0 : double
        Numerical constant related to the shape of the function
    
    Returns
    -------
    Y
        The return of the Y-function defined in [1] at a given point.

    References
    ----------
    [1] Tzemos, Contopoulos, Efthymiopoulos "Bohmian trajectories in an entangled two-qubit system" (2019) arxiv:1905.12619
    """
    return (m*omega/(np.pi))**(1/4)*np.exp(-m*omega/2 * (x-np.sqrt(2/(m*omega))*np.abs(A0)*np.cos(sigma))**2 + 1j*(np.sqrt(2*m*omega)*np.abs(A0)*np.sin(sigma)*x + xi(sigma,A0)))