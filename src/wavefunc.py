import numpy as np
class Schrödinger():
    def __init__(self, initcond, V, Np, Nt, L):
        self.initcond = initcond
        self.V = V
        self.x1 = np.linspace(-L,L,Np)
        self.x2 = np.linspace(-L,L,Np)
        self.psis = []
        self.psis.append(initcond)
        self.Np = Np
        self.Nt = Nt