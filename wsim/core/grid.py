import numpy as np


class Grid:

    def __init__(self, N, d):
        self.N = tuple(int(n) for n in N)
        self.d = tuple(float(x) for x in d)
        self.dim = len(self.N)
        self.dx = float(self.d[0])
        self.dt = 0.0
        self.Nt = 0
        self.t = 0

    def makeTime(self, c_max, cfl=0.3, t_end=None, Nt=None):
        self.dt = float(cfl * self.dx / c_max)
        if Nt is not None:
            self.Nt = int(Nt)
        else:
            self.Nt = int(round(t_end / self.dt))
        self.t = self.dt * np.arange(0, self.Nt)
        return self.dt, self.Nt
