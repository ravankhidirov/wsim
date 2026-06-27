import numpy as np

from .backend import get_backend, to_host


def damping1d(L, lo, hi, APARA=0.015):
    w = np.ones((L,), dtype=np.float32)
    if lo > 0:
        d = np.arange(lo, 0, -1)
        w[0:lo] = np.exp(-((APARA * d) ** 2))
    if hi > 0:
        d = np.arange(1, hi + 1)
        w[L - hi:L] = np.exp(-((APARA * d) ** 2))
    return w


class EFITBase(object):

    def __init__(self, grid, medium, source, sensor, pml=20, device="cpu"):

        self.grid = grid
        self.medium = medium
        self.source = source
        self.sensor = sensor
        self.device = device
        self.xp = get_backend(device)
        self.dim = grid.dim
        self.n = 0

        self.pml = self.normalizePML(pml)

        self.dx = np.float32(grid.dx)
        self.dt = np.float32(grid.dt)
        self.dtx = np.float32(self.dt / self.dx)
        self.ddx = np.float32(1.0 / self.dx)
        self.dtdxx = self.dtx * self.ddx

        self.materialSetup()
        self.initFields()
        self.staggeredProp()
        self.applyBoundaries()
        self.sourceSetup()
        self.sensorSetup()
        self.transferToDevice()

    def normalizePML(self, pml):
        nfaces = 2 * self.dim
        if np.isscalar(pml):
            return [int(pml)] * nfaces
        pml = list(pml)
        if len(pml) == self.dim:
            out = []
            for v in pml:
                out += [int(v), int(v)]
            return out
        return [int(v) for v in pml]

    def toHost(self, a):
        return to_host(self.device, a)

    def run_all(self, return_fields=False, snapshot_every=0):
        T = int(self.grid.Nt)
        self.frames = []

        last_print = -1
        for n in range(T):
            self.n = n
            self.step(n)

            if snapshot_every and (n % snapshot_every == 0):
                self.frames.append(self.frame())

            p = int(100 * (n + 1) / T)
            if p != last_print:
                last_print = p
                print("\r[%s] Progress: %3d%%  (%d/%d)" % (self.device, p, n + 1, T), end="", flush=True)

        if self.device == "gpu":
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        print("\r[%s] Progress: 100%%  Done.            " % self.device)

        self.sensor_signals = self.toHost(self.sensor_signals)

        out = {"p": self.sensor_signals, "t": np.asarray(self.grid.t)}
        if snapshot_every:
            out["frames"] = np.asarray(self.frames)
        if return_fields:
            out.update(self.exportFields())
        return out
