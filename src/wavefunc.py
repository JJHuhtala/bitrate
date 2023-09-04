import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fastpotential as f
class Schrödinger():
    def __init__(self, initcond, V, Np, Nt, L):
        self.initcond = initcond
        self.V = np.zeros((Np,Np),dtype=np.cdouble)
        self.x1 = np.linspace(-L,L,Np)
        self.x2 = np.linspace(-L,L,Np)
        self.psis =np.zeros((Np,Np,Nt),dtype=np.cdouble)
        self.psis[:,:,0] = initcond
        self.Np = Np
        self.Nt = Nt
        self.L=L
        self.dt = 0.001

        self.fig, self.ax = plt.subplots()
        self.line = self.ax.imshow(np.abs(self.psis[:,:,0])**2,cmap='Greys')
    def simulate(self):
        f.rk4(self.psis,self.V,self.Nt,self.Np,self.L/self.Np, self.dt)
        print(np.max(self.psis[:,:,0]-self.psis[:,:,300]))
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
        ani = animation.FuncAnimation(self.fig, self.animate, np.arange(1, 1500), init_func=self.beg,
                              interval=25, save_count=1500)
        FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
        ani.save('psiresonance.mp4', writer = FFwriter)
        