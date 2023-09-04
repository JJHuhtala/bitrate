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

cpdef void multiply_by_constant_test(complex[:,:] a1,  double complex c, complex[:,:] res, Py_ssize_t N):
    multiply_by_constant(a1, c, res, N)
cdef void multiply_by_constant(complex[:,:] a1,  double complex c, complex[:,:] res, Py_ssize_t N):
    cdef Py_ssize_t i,j
    for i in range(N):
        for j in range(N):
            res[i,j] = c*a1[i,j]


cpdef void add_test(complex[:,:] a1, complex[:,:] a2, complex[:,:] result, Py_ssize_t N):
    add(a1,a2,result,N)
cdef void add(complex[:,:] a1, complex[:,:] a2, complex[:,:] result, Py_ssize_t N):
    cdef Py_ssize_t i, j
    for i in range(N):
        for j in range(N):
            result[i,j] = a1[i,j] + a2[i,j]
    
cdef void multiply(complex[:,:] a1, complex[:,:] a2, complex[:,:] result, Py_ssize_t N):
    cdef Py_ssize_t i, j
    for i in range(N):
        for j in range(N):
            result[i,j] = a1[i,j]*a2[i,j]

cpdef np.ndarray[complex,ndim=2] laplace(np.ndarray[complex,ndim=2] psi,int N, double dx):
    cdef np.ndarray[complex,ndim=2] laplace
    cdef int i,j
    laplace = np.zeros((N,N), dtype=complex)
    for i in range(1, N - 1):
        for j in range(1,N-1):
            laplace[i,j] = (psi[i+1,j] + psi[i-1,j] + psi[i,j-1] +\
                                psi[i,j+1] - 4. * psi[i,j]) / (dx * dx)
    laplace[0,:]         = 0
    laplace[N-1,:]       = 0  
    laplace[:,0]         = 0
    laplace[:,N-1]       = 0
    return laplace

cpdef void laplace_multi(complex[:,:] psi, complex[:,:] laplace, Py_ssize_t N, double dx):
    cdef Py_ssize_t i,j
    for i in range(1, N - 1):
        for j in range(1,N-1):
            laplace[i,j] = -(1. / (2.j)) * (psi[i+1,j] + psi[i-1,j] + psi[i,j-1] +\
                                psi[i,j+1] - 4. * psi[i,j]) / (dx * dx)


cpdef void rk4(complex[:,:,:] psis, complex[:,:] V, Py_ssize_t Nsteps, Py_ssize_t N, double dx, double dt):

    # This function is ugly as all hell since I've tried to explicitly avoid allocations to save time

    # Preliminary setup: I use numpy arrays since they're easy to create. Could use C arrays.
    halfstepar = np.zeros((N,N),dtype=np.cdouble)
    fullstepar = np.zeros((N,N),dtype=np.cdouble)
    laplacear = np.zeros((N,N), dtype=np.cdouble)

    k1ar = np.zeros((N,N),dtype=np.cdouble)
    k2ar = np.zeros((N,N),dtype=np.cdouble)
    k3ar = np.zeros((N,N),dtype=np.cdouble)
    k4ar = np.zeros((N,N),dtype=np.cdouble)

    Vjar =np.zeros((N,N),dtype=np.cdouble)
    Vjmultiar = np.zeros((N,N),dtype=np.cdouble)
    multikar = np.zeros((N,N), dtype=np.cdouble)
    cdef complex [:,:] halfstep = halfstepar
    cdef complex [:,:] fullstep = fullstepar
    cdef complex [:,:] laplace = laplacear
    cdef complex [:,:] k1 = k1ar
    cdef complex [:,:] k2 = k2ar
    cdef complex [:,:] k3 = k3ar
    cdef complex [:,:] k4 = k4ar
    cdef complex [:,:] Vj =  Vjar
    cdef complex [:,:] multik = multikar
    cdef complex [:,:] Vjmulti =  Vjmultiar
    cdef Py_ssize_t i
    cdef double halfstepconst
    halfstepconst = 0.5 * dt

    for i in range(1,Nsteps):
        laplace_multi(psis[:,:,i-1],laplace,N,dx)  # Calculating the k1 step
        multiply(psis[:,:,i-1],Vj,Vjmulti,N)
        add(laplace,Vjmulti,k1,N)  # Calculating the k1 step; need to add the potential properly here. (Currently assumes 0!)

        multiply_by_constant(k1,halfstepconst,multik,N) # Starting k2
        add(multik,psis[:,:,i-1],halfstep,N) # psi + 0.5*dt*k1
        laplace_multi(halfstep,laplace,N,dx) # laplacian at psi + 0.5*dt*k1
        multiply(halfstep,Vj,Vjmulti,N)
        add(laplace,Vjmulti,k2,N) # Finally k2
        
        multiply_by_constant(k2,halfstepconst,multik,N) # Starting k3
        add(multik,psis[:,:,i-1],halfstep,N) # psi + 0.5*dt*k2
        laplace_multi(halfstep,laplace,N,dx) # laplacian at psi + 0.5*dt*k2
        multiply(halfstep,Vj,Vjmulti, N)
        add(laplace,Vjmulti,k3,N) # Finally k3

        multiply_by_constant(k3,dt,multik,N) # Starting k4
        add(multik,psis[:,:,i-1],fullstep,N) # psi + dt*k3
        laplace_multi(fullstep,laplace,N,dx) # laplacian at psi + dt*k3
        multiply(fullstep,Vj,Vjmulti,N)
        add(laplace,Vjmulti,k4,N) # Finally k3

        #Reusing "laplace" and "halfstep" to store intermediate sums..
        multiply_by_constant(k1,1.0/6.0 * dt,halfstep,N)
        multiply_by_constant(k2,2.0/6.0 * dt,fullstep,N)
        multiply_by_constant(k3,2.0/6.0 * dt,multik,N)
        multiply_by_constant(k4,1.0/6.0 * dt,laplace,N)

        # Reusing k1,k2, etc. to store intermediate sums..

        add(halfstep,fullstep,k1,N)
        add(k1,multik,k2,N)
        add(k2,laplace,k3,N)
        add(k3,psis[:,:,i-1],psis[:,:,i],N)
        if i%10 == 0:
            print(i)


cpdef np.ndarray[complex,ndim=2] create_gaussian(int N, np.ndarray x, np.ndarray y, double centerx, double centery, double factor,double sigma):
    cdef int i,j 
    cdef np.ndarray[complex,ndim=2] gauss
    gauss = np.zeros((N,N),dtype=complex)
    for i in range(N):
        for j in range(N):
            gauss[i,j] = factor*np.exp(-(pow((x[i]-centerx),2)+pow((y[j]-centery),2)) / 2 / sigma / sigma)

    return gauss