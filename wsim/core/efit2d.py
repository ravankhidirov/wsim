import numpy as np

from .efitBase import EFITBase, damping1d
from . import kernels2d as Kernels


class EFIT2D(EFITBase):

    def materialSetup(self):
        m = self.medium
        self.Rho = m.Rho.copy()
        self.C11 = m.C11.copy()
        self.C12 = m.C12.copy()
        self.C22 = m.C22.copy()
        self.C44 = m.C44.copy()
        self.Eta_v = m.Eta_v.copy()
        self.Eta_s = m.Eta_s.copy()

        air = self.Rho < m.air_threshold
        self.Rho[air] = 10e23
        self.C11[air] = 1e-20
        self.C12[air] = 1e-20
        self.C22[air] = 1e-20
        self.C44[air] = 1e-20
        self.Eta_v[air] = 1e-20
        self.Eta_s[air] = 1e-20

        self.C44[self.C44 == 0] = 1e-30
        self.Eta_s[self.Eta_s == 0] = 1e-30

    def initFields(self):
        MRI, NRI = self.medium.shape
        self.MRI = MRI
        self.NRI = NRI
        self.source_signal = np.asarray(self.source.signal, dtype=np.float32)

        self.Vx = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.Vy = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.Txx = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.Txy = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.Tyy = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.DVx = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)
        self.DVy = self.xp.zeros((MRI, NRI), dtype=self.xp.float32)

    def staggeredProp(self):
        MRI, NRI = self.MRI, self.NRI

        BXtemp = 1.0 / self.Rho
        BYtemp = 1.0 / self.Rho
        self.BX = np.zeros((MRI, NRI), dtype=np.float32)
        self.BY = np.zeros((MRI, NRI), dtype=np.float32)

        self.BX[:-1, :] = 0.5 * (BXtemp[1:, :] + BXtemp[:-1, :])
        self.BY[:, :-1] = 0.5 * (BYtemp[:, 1:] + BYtemp[:, :-1])

        self.C44_Eff = np.zeros((MRI, NRI), dtype=np.float32)
        self.C44_Eff[:-1, :-1] = 4. / (
            (1. / self.C44[:-1, :-1]) + (1. / self.C44[1:, :-1]) +
            (1. / self.C44[:-1, 1:]) + (1. / self.C44[1:, 1:]))

        self.Eta_vs = (self.Eta_v + 2. * self.Eta_s).astype(np.float32)
        self.Eta_ss = np.zeros((MRI, NRI), dtype=np.float32)
        self.Eta_ss[:-1, :-1] = 4. / (
            (1. / self.Eta_s[:-1, :-1]) + (1. / self.Eta_s[1:, :-1]) +
            (1. / self.Eta_s[:-1, 1:]) + (1. / self.Eta_s[1:, 1:]))

    def applyBoundaries(self):
        MRI, NRI = self.MRI, self.NRI
        Top, Bottom, Left, Right = self.pml

        wx = damping1d(MRI, Top, Bottom)
        wy = damping1d(NRI, Left, Right)
        self.ABS_host = (wx[:, None] * wy[None, :]).astype(np.float32)

    def sourceSetup(self):
        kind = self.source.kind
        if kind == "pressure":
            self.is_pressure = True
            self.wave = np.int32(2)
        elif kind == "vx":
            self.is_pressure = False
            self.wave = np.int32(0)
        elif kind == "vy":
            self.is_pressure = False
            self.wave = np.int32(1)
        else:
            self.is_pressure = True
            self.wave = np.int32(2)

        pos = self.source.positions
        self.XL_tx = np.copy(np.int32(pos[:, 0]))
        self.YL_tx = np.copy(np.int32(pos[:, 1]))
        self.win_host = np.asarray(self.source.window, dtype=np.float32)

    def sensorSetup(self):
        pos = self.sensor.positions
        self.XL_rx = np.copy(np.int32(pos[:, 0]))
        self.YL_rx = np.copy(np.int32(pos[:, 1]))
        self.NRX = np.shape(pos)[0]

    def transferToDevice(self):
        xp = self.xp

        self.ABS = xp.asarray(self.ABS_host, dtype=xp.float32)
        self.BX = xp.asarray(self.BX, dtype=xp.float32)
        self.BY = xp.asarray(self.BY, dtype=xp.float32)
        self.C11 = xp.asarray(self.C11, dtype=xp.float32)
        self.C12 = xp.asarray(self.C12, dtype=xp.float32)
        self.C44_Eff = xp.asarray(self.C44_Eff, dtype=xp.float32)
        self.Eta_vs = xp.asarray(self.Eta_vs, dtype=xp.float32)
        self.Eta_s = xp.asarray(self.Eta_s, dtype=xp.float32)
        self.Eta_ss = xp.asarray(self.Eta_ss, dtype=xp.float32)

        self.XL_tx = xp.asarray(self.XL_tx, dtype=xp.int32)
        self.YL_tx = xp.asarray(self.YL_tx, dtype=xp.int32)
        self.win = xp.asarray(self.win_host, dtype=xp.float32)

        self.XL_rx = xp.asarray(self.XL_rx, dtype=xp.int32)
        self.YL_rx = xp.asarray(self.YL_rx, dtype=xp.int32)

        self.sensor_signals = xp.zeros((int(self.grid.Nt), self.NRX), dtype=xp.float32)

    def step(self, n):
        y = np.float32(self.source_signal[n])

        self.Vx, self.Vy, self.DVx, self.DVy = Kernels.velocityVoigt(
            self.Txx, self.Txy, self.Tyy,
            self.Vx, self.Vy, self.DVx, self.DVy,
            self.BX, self.BY, self.ABS, self.ddx, self.dt)

        if not self.is_pressure:
            self.Vx, self.Vy = Kernels.sourceVel(
                self.Vx, self.Vy, self.XL_tx, self.YL_tx,
                y, self.wave, self.win, self.dtdxx)

        self.Txx, self.Tyy, self.Txy = Kernels.stressVoigt(
            self.Txx, self.Txy, self.Tyy,
            self.Vx, self.Vy, self.DVx, self.DVy,
            self.C11, self.C12, self.C44_Eff,
            self.Eta_vs, self.Eta_s, self.Eta_ss, self.ABS, self.dtx)

        if self.is_pressure:
            self.Txx, self.Tyy = Kernels.sourceStress(
                self.Txx, self.Tyy, self.XL_tx, self.YL_tx,
                y, self.wave, self.win, self.dtdxx)

        self.sensor_signals[n, :] = -0.5 * (
            self.Txx[self.XL_rx, self.YL_rx] + self.Tyy[self.XL_rx, self.YL_rx])

    def frame(self):
        return self.toHost(-0.5 * (self.Txx + self.Tyy))

    def exportFields(self):
        return {
            "Vx": self.toHost(self.Vx),
            "Vy": self.toHost(self.Vy),
            "Txx": self.toHost(self.Txx),
            "Tyy": self.toHost(self.Tyy),
            "Txy": self.toHost(self.Txy),
        }
