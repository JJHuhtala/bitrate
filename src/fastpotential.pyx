import numpy as np
cimport numpy as np
from libc.math cimport abs,pow
# The functions marked "test" are dummies for using with pytest, as it's desirable to avoid
# cpdef functions for speed reasons.

cpdef double get_norm(np.ndarray[complex, ndim=2] psi,int N, double dx):
    cdef int i,j
    cdef double norm
    norm = 0.0
    for i in range(N):
        for j in range(N):
            norm = norm + pow(abs(psi[i,j]),2)*dx*dx
    return norm

def basis2d(double[:] x, double[:] y, double L, int n1, int n2, int N):
    cdef Py_ssize_t i, j
    cdef double k1,k2 
    k1 = np.pi*n1/L
    k2 = np.pi*n2/L
    bs = np.zeros((N,N),dtype=complex)
    for i in range(N):
        for j in range(N):
            bs[i,j] = 2/L * np.sin(k1*(x[i]+L/2))*np.sin(k2*(y[j]+L/2))
    return bs