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

def create_wall(Npoints,x,y):
    """CREATEWALL"""
    V = np.zeros((Npoints,Npoints))
    Vdrop = np.zeros((Npoints,Npoints))
    sigmapot = 1/250
    factorpot = 50000
    yloc = np.linspace(-1,-0.13,40)
    yloc2 = np.linspace(-0.01,0.01,10)
    yloc3 = np.linspace(0.13,1,40)
    xloc = 0.3
    xdroploc = 0.9
    droploc = np.linspace(-1,1,100)
    for i in range(len(yloc)):
        V = V+ ts.create_gaussian(Npoints,x,y,xloc,yloc[i],factorpot,sigmapot)
    
    for i in range(len(yloc2)):
        V = V+ ts.create_gaussian(Npoints,x,y,xloc,yloc2[i],factorpot,sigmapot)
    for i in range(len(yloc3)):
        V = V+ ts.create_gaussian(Npoints,x,y,xloc,yloc3[i],factorpot,sigmapot)
    
    for i in range(len(droploc)):
        Vdrop = Vdrop + ts.create_gaussian(Npoints,x,y,xdroploc,droploc[i],-factorpot/10,sigmapot)
    return V+Vdrop


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
    
    References
    ----------
    [1] Tzemos, Contopoulos, Efthymiopoulos "Bohmian trajectories in an entangled two-qubit system" (2019) arxiv:1905.12619
    """
    return (m*omega/(np.pi))**(1/4)*np.exp(-m*omega/2 * (x-np.sqrt(2/(m*omega))*np.abs(A0)*np.cos(sigma))**2 + 1j*(np.sqrt(2*m*omega)*np.abs(A0)*np.sin(sigma)*x + xi(sigma,A0)))