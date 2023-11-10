import numpy as np
cimport numpy as np
from libc.math cimport abs,pow, sin, cos
# The functions marked "test" are dummies for using with pytest, as it's desirable to avoid
# cpdef functions for speed reasons.
cdef integral(complex[:,:] fun1, complex[:,:] fun2,Py_ssize_t N, double dx):
    """ Computes the numerical approximation to an integral using the simplest method ("midpoint") for
    a product of two functions.

    Parameters
    ----------
    fun1 : complex 2d ndarray
        Function 1 to be integrated over.
    fun2 : complex 2d ndarray
        Function 2 to be integrated over.
    N : integer
        Number of grid points in the 2d grid. Length of fun1[:,0], fun2[:,0]
    dx : double
        Grid spacing, i.e. L/N.
    
    Notes
    -----
    The function does not check whether the arrays have the same size or if that dimension is the same as N. That should be done
    somewhere else.

    Returns
    -------
    The integral int fun1*fun2*d^2x.
    """
    cdef Py_ssize_t i, j
    cdef complex sm
    sm = 0
    for i in range(N):
        for j in range(N):
            sm += fun1[i,j]*fun2[i,j]
    
    return sm*dx*dx

cpdef double get_norm(np.ndarray[complex, ndim=2] psi,int N, double dx):
    """Computes the norm of a wave function. 

    Parameters
    ----------
    psi : complex 2d ndarray
        Wave function for which to calculate normalization.
    N : integer
        Number of grid points. Must be the same as len(psi[:,0]).
    dx : double
        Grid spacing, i.e. L/N.
    
    Notes
    -----
    The size of the wave function array is not checked. It must be checked outside of the function.

    This function is useful for checking whether the norm of a wave function deviates from 1 (in which case there are probably numerical errors)
    or for normalizing the wave function initially.

    Returns
    -------
    The norm of the given wave function.
    """
    cdef int i,j
    cdef double norm
    norm = 0.0
    for i in range(N):
        for j in range(N):
            norm = norm + pow(abs(psi[i,j]),2)*dx*dx
    return norm

cdef basisfunc(double x1, double x2, double L, int n1, int n2):
    """ The value of a basis function indexed by n1,n2 at point x1,x2.

    Parameters
    ----------
    x1 : double
        Coordinate for particle 1
    x2 : double
        Coordinate for particle 2
    L : double
        HALF the length of the simulation box - the box is from (-L,L).
    n1 : integer
        First index for the basis function.
    n2 : integer
        Second index for the basis function.
    
    Notes
    -----
    Differs from the usual formula, because in this code the box is from -L to L, so that you have to insert "2L" for the
    usual formula where L generally indicates the total length of the box instead of half.

    Returns
    -------
    Basis function (n1,n2) at point (x1,x2)
    """
    cdef double k1,k2
    k1 = np.pi*n1/(2*L)
    k2 = np.pi*n2/(2*L)
    return 1/L * sin(k1*(x1+L))*sin(k2*(x2+L))

def basis2d(double[:] x, double L, int n1, int n2, int N):
    """A full representation of a basis function as a 2D array.
    
    Parameters
    ----------
    x : 1d ndarray
        The grid points from -L to L.
    L : double
        Half the length of the box.
    n1 : integer
        First index of the wave function.
    n2 : integer
        Second index of the wave function.
    N : integer
        Length of x.

    Notes
    -----
    Like other Cython functions here, lengths are NOT checked; the user must make sure they're correct.

    Returns
    -------
    bs
        2d representation as ndarray of the basis function indexed by (n1,n2).
    
    """
    cdef Py_ssize_t i, j
    cdef double k1,k2 
    k1 = np.pi*n1/(2*L)
    k2 = np.pi*n2/(2*L)
    bs = np.zeros((N,N),dtype=complex)
    for i in range(N):
        for j in range(N):
            bs[i,j] = basisfunc(x[i],x[j],L,n1,n2)
    return bs


def basis2d_x1deriv(complex[:,:] coeffs, double x1, double x2, double L, Py_ssize_t n_basis):
    """ Spatial derivative with respect to x1 excluding phase.

    Parameters
    ----------
    coeffs : double 2d ndarray.
        The coefficient for the basis expansion in terms of particle-in-a-box wave functions.
    x1 : double
        The x1 point at which to compute the derivative.
    x2 : double
        The x2 point at which to compute the derivative.
    L : double
        Half the length of the simulation box.
    n_basis : Py_ssize_t, positive integer
        The square root of the number of basis functions.
    
    Notes
    -----
    Used for trajectory calculations and specialized for 2d. Doesn't include the exponent factor (the phase),
    that is added in the trajectory; this is for the time-independent part.

    Returns
    -------
    bs : double
        Derivative of the wave function with respect to x1 at point x1,x2.
    """
    cdef Py_ssize_t i, j
    cdef double k1,k2
    cdef complex bs
    bs = 0
    for i in range(n_basis):
        for j in range(n_basis):
            k1 = np.pi*i/(2*L)
            k2 = np.pi*j/(2*L) 
            bs += coeffs[i,j] * 1/L * k1*cos(k1*(x1+L))*sin(k2*(x2+L))

    return bs

def basis2d_x2deriv(complex[:,:] coeffs, double x1, double x2, double L, Py_ssize_t n_basis):
    """ Spatial derivative with respect to x2 excluding phase.

    Parameters
    ----------
    coeffs : double 2d ndarray.
        The coefficient for the basis expansion in terms of particle-in-a-box wave functions.
    x1 : double
        The x1 point at which to compute the derivative.
    x2 : double
        The x2 point at which to compute the derivative.
    L : double
        Half the length of the simulation box.
    n_basis : Py_ssize_t, positive integer
        The square root of the number of basis functions.
    
    Notes
    -----
    Used for trajectory calculations and specialized for 2d. Doesn't include the exponent factor (the phase),
    that is added in the trajectory; this is for the time-independent part.

    Returns
    -------
    bs : double
        Derivative of the wave function with respect to x1 at point x1,x2.
    """
    cdef Py_ssize_t i, j
    cdef double k1,k2
    cdef complex bs
    bs = 0
    for i in range(n_basis):
        for j in range(n_basis):
            k1 = np.pi*i/(2*L)
            k2 = np.pi*j/(2*L) 
            bs += coeffs[i,j] * 1/L * k2*sin(k1*(x1+L))*cos(k2*(x2+L))
    return bs

def psi(complex[:,:] coeffs, double x1, double x2, double L, Py_ssize_t n_basis):
    """Find value of psi at (x1,x2) by expanding in terms of the basis functions given expansion coefficients.

    Parameters
    ----------
    x1 : double
        Coordinate for particle 1
    x2 : double
        Coordinate for particle 2
    L : double
        HALF the length of the simulation box - the box is from (-L,L).
    n_basis : Py_ssize_t, positive integer
        Number of basis functions per particle, i.e. square root of the total number of basis functions.

    Returns
    -------
    Value of psi at point (x1,x2) (which is a complex number!)
    """
    cdef complex ps
    cdef Py_ssize_t i,j
    ps = 0
    for i in range(n_basis):
        for j in range(n_basis):
            ps += coeffs[i,j]*basisfunc(x1,x2,L,i,j)

    return ps