import numpy as np
cimport numpy as np
from libc.math cimport abs,pow, sin
# The functions marked "test" are dummies for using with pytest, as it's desirable to avoid
# cpdef functions for speed reasons.
cdef integral(complex[:,:] fun1, complex[:,:] fun2,Py_ssize_t N, double dx):
    cdef Py_ssize_t i, j
    cdef complex sm
    sm = 0
    for i in range(N):
        for j in range(N):
            sm += fun1[i,j]*fun2[i,j]
    
    return sm*dx*dx

cpdef double get_norm(np.ndarray[complex, ndim=2] psi,int N, double dx):
    cdef int i,j
    cdef double norm
    norm = 0.0
    for i in range(N):
        for j in range(N):
            norm = norm + pow(abs(psi[i,j]),2)*dx*dx
    return norm
cdef basisfunc(double x, double y, double L, int n1, int n2):
    cdef double k1,k2
    k1 = np.pi*n1/(2*L)
    k2 = np.pi*n2/(2*L)
    return 2/L * sin(k1*(x+L))*sin(k2*(y+L))

def basis2d(double[:] x, double[:] y, double L, int n1, int n2, int N):
    cdef Py_ssize_t i, j
    cdef double k1,k2 
    k1 = np.pi*n1/(2*L)
    k2 = np.pi*n2/(2*L)
    bs = np.zeros((N,N),dtype=complex)
    for i in range(N):
        for j in range(N):
            bs[i,j] = basisfunc(x[i],x[j],L,n1,n2)
    return bs


def basis2d_xderiv(double x, double y, double L, int n1, int n2, int N):
    cdef Py_ssize_t i, j
    cdef double k1,k2, bs
    k1 = np.pi*n1/(2*L)
    k2 = np.pi*n2/(2*L)
    bs = 0
    """
    for i in range(N):
        for j in range(N):
            bs[i,j] = 2/L * sin(k1*(x[i]+L))*sin(k2*(x[j]+L))"""
    return bs

