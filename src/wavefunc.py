import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fastpotential as f
import gc
class Schrödinger():
    def __init__(self, initcond, V, Np, Nt, L):
        self.initcond = initcond
        self.V = V
        self.x1 = np.linspace(-L,L,Np)
        self.x2 = np.linspace(-L,L,Np)
        self.psis =np.zeros((Np,Np,Nt),dtype=np.cdouble)
        self.psis[:,:,0] = initcond
        self.Np = Np
        self.Nt = Nt
        self.L=L
        self.dt = 0.001
        self.dx = L/Np

        self.fig, self.ax = plt.subplots()
        self.line = self.ax.imshow(np.abs(self.psis[:,:,0])**2,cmap='Greys')
    def update_psi_rk4(self,psi, dt):
        k1 = -(1. / (2.j)) * f.laplace(psi,self.Np,self.dx) + (1. / 1.j)*self.V*(psi)
        k2 = -(1. / (2.j)) * f.laplace(psi + 0.5 * dt * k1,self.Np,self.dx)+ (1. / 1.j)*self.V*(psi + 0.5 * dt * k1)
        k3 = -(1. / (2.j)) * f.laplace(psi + 0.5 * dt * k2,self.Np,self.dx)+ (1. / 1.j)*self.V*(psi + 0.5 * dt * k2)
        k4 = -(1. / (2.j)) * f.laplace(psi + dt * k3,self.Np,self.dx)+ (1. / 1.j)*self.V*(psi + dt * k3)
        
        psinew = psi + dt * 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        return psinew
    def simulate(self):
        #f.rk4(self.psis,self.V,self.Nt,self.Np,self.L/self.Np, self.dt)
        for i in range(self.Nt-1):
            newpsi = self.update_psi_rk4(self.psis[:,:,i],self.dt)
            self.psis[:,:,i+1] = np.copy(newpsi)
            if i%10 == 0:
                gc.collect()
                print(i)
    def beg(self):
        return self.line
    def animate(self,i):
        ir = i*4
        if ir > self.Nt-1:
            print("No more frames available")
            return self.line
        plotted = abs(self.psis[:,:,ir])**2 
        self.line.set_data(plotted)  # update the data
        self.line.set_clim(vmax=np.amax(np.abs(self.psis[:,:,ir])**2))
        self.line.set_clim(vmin=0)
        return self.line
    
    def create_movie(self):
        ani = animation.FuncAnimation(self.fig, self.animate, np.arange(1, self.Nt//4), init_func=self.beg,
                              interval=25, save_count=self.Nt//4)
        FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
        ani.save('psiresonance.mp4', writer = FFwriter)
    
    def get_grid(self):
        return self.Nt, self.Np,self.L,self.x1,self.dt
    
    def get_psis(self):
        return self.psis