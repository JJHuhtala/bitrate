import numpy as np
class Schrödinger():
    def __init__(self, initcond, V, Npoints, L):
        self.initcond = initcond
        self.V = V
        self.x1 = np.linspace(-L,L,Npoints)
        self.x2 = np.linspace(-L,L,Npoints)
        self.psis = []
        self.psis.append(initcond)
