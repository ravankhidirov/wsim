import numpy as np

from .efitBase import EFITBase, damping1d
from . import kernels3d as Kernels


class EFIT3D(EFITBase):

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
        MRI, NRI, PRI = self.medium.shape
        self.MRI = MRI
        self.NRI = NRI
        self.PRI = PRI
        self.source_signal = np.asarray(self.source.signal, dtype=np.float32)

        self.Vx = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Vy = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Vz = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Txx = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Tyy = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Tzz = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Txy = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Txz = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.Tyz = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.DVx = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.DVy = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)
        self.DVz = self.xp.zeros((MRI, NRI, PRI), dtype=self.xp.float32)

    def staggeredProp(self):
        MRI, NRI, PRI = self.MRI, self.NRI, self.PRI

        Btemp = 1.0 / self.Rho
        self.BX = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.BY = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.BZ = np.zeros((MRI, NRI, PRI), dtype=np.float32)

        self.BX[:-1, :, :] = 0.5 * (Btemp[1:, :, :] + Btemp[:-1, :, :])
        self.BY[:, :-1, :] = 0.5 * (Btemp[:, 1:, :] + Btemp[:, :-1, :])
        self.BZ[:, :, :-1] = 0.5 * (Btemp[:, :, 1:] + Btemp[:, :, :-1])

        self.C44xy = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.C44xz = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.C44yz = np.zeros((MRI, NRI, PRI), dtype=np.float32)

        self.C44xy[:-1, :-1, :] = 4. / (
            (1. / self.C44[:-1, :-1, :]) + (1. / self.C44[1:, :-1, :]) +
            (1. / self.C44[:-1, 1:, :]) + (1. / self.C44[1:, 1:, :]))
        self.C44xz[:-1, :, :-1] = 4. / (
            (1. / self.C44[:-1, :, :-1]) + (1. / self.C44[1:, :, :-1]) +
            (1. / self.C44[:-1, :, 1:]) + (1. / self.C44[1:, :, 1:]))
        self.C44yz[:, :-1, :-1] = 4. / (
            (1. / self.C44[:, :-1, :-1]) + (1. / self.C44[:, 1:, :-1]) +
            (1. / self.C44[:, :-1, 1:]) + (1. / self.C44[:, 1:, 1:]))

        self.Eta_vs = (self.Eta_v + 2. * self.Eta_s).astype(np.float32)

        self.Eta_xy = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.Eta_xz = np.zeros((MRI, NRI, PRI), dtype=np.float32)
        self.Eta_yz = np.zeros((MRI, NRI, PRI), dtype=np.float32)

        self.Eta_xy[:-1, :-1, :] = 4. / (
            (1. / self.Eta_s[:-1, :-1, :]) + (1. / self.Eta_s[1:, :-1, :]) +
            (1. / self.Eta_s[:-1, 1:, :]) + (1. / self.Eta_s[1:, 1:, :]))
        self.Eta_xz[:-1, :, :-1] = 4. / (
            (1. / self.Eta_s[:-1, :, :-1]) + (1. / self.Eta_s[1:, :, :-1]) +
            (1. / self.Eta_s[:-1, :, 1:]) + (1. / self.Eta_s[1:, :, 1:]))
        self.Eta_yz[:, :-1, :-1] = 4. / (
            (1. / self.Eta_s[:, :-1, :-1]) + (1. / self.Eta_s[:, 1:, :-1]) +
            (1. / self.Eta_s[:, :-1, 1:]) + (1. / self.Eta_s[:, 1:, 1:]))

    def applyBoundaries(self):
        MRI, NRI, PRI = self.MRI, self.NRI, self.PRI
        Top, Bottom, Left, Right, Front, Back = self.pml

        wx = damping1d(MRI, Top, Bottom)
        wy = damping1d(NRI, Left, Right)
        wz = damping1d(PRI, Front, Back)
        self.ABS_host = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32)

    def sourceSetup(self):
        kind = self.source.kind
        if kind == "pressure":
            self.is_pressure = True
            self.comp = np.int32(0)
        elif kind == "vx":
            self.is_pressure = False
            self.comp = np.int32(0)
        elif kind == "vy":
            self.is_pressure = False
            self.comp = np.int32(1)
        elif kind == "vz":
            self.is_pressure = False
            self.comp = np.int32(2)
        else:
            self.is_pressure = True
            self.comp = np.int32(0)

        pos = self.source.positions
        self.XL_tx = np.copy(np.int32(pos[:, 0]))
        self.YL_tx = np.copy(np.int32(pos[:, 1]))
        self.ZL_tx = np.copy(np.int32(pos[:, 2]))
        self.win_host = np.asarray(self.source.window, dtype=np.float32)

    def sensorSetup(self):
        pos = self.sensor.positions
        self.XL_rx = np.copy(np.int32(pos[:, 0]))
        self.YL_rx = np.copy(np.int32(pos[:, 1]))
        self.ZL_rx = np.copy(np.int32(pos[:, 2]))
        self.NRX = np.shape(pos)[0]

    def transferToDevice(self):
        xp = self.xp

        self.ABS = xp.asarray(self.ABS_host, dtype=xp.float32)
        self.BX = xp.asarray(self.BX, dtype=xp.float32)
        self.BY = xp.asarray(self.BY, dtype=xp.float32)
        self.BZ = xp.asarray(self.BZ, dtype=xp.float32)
        self.C11 = xp.asarray(self.C11, dtype=xp.float32)
        self.C12 = xp.asarray(self.C12, dtype=xp.float32)
        self.C44xy = xp.asarray(self.C44xy, dtype=xp.float32)
        self.C44xz = xp.asarray(self.C44xz, dtype=xp.float32)
        self.C44yz = xp.asarray(self.C44yz, dtype=xp.float32)
        self.Eta_vs = xp.asarray(self.Eta_vs, dtype=xp.float32)
        self.Eta_s = xp.asarray(self.Eta_s, dtype=xp.float32)
        self.Eta_xy = xp.asarray(self.Eta_xy, dtype=xp.float32)
        self.Eta_xz = xp.asarray(self.Eta_xz, dtype=xp.float32)
        self.Eta_yz = xp.asarray(self.Eta_yz, dtype=xp.float32)

        self.XL_tx = xp.asarray(self.XL_tx, dtype=xp.int32)
        self.YL_tx = xp.asarray(self.YL_tx, dtype=xp.int32)
        self.ZL_tx = xp.asarray(self.ZL_tx, dtype=xp.int32)
        self.win = xp.asarray(self.win_host, dtype=xp.float32)

        self.XL_rx = xp.asarray(self.XL_rx, dtype=xp.int32)
        self.YL_rx = xp.asarray(self.YL_rx, dtype=xp.int32)
        self.ZL_rx = xp.asarray(self.ZL_rx, dtype=xp.int32)

        self.sensor_signals = xp.zeros((int(self.grid.Nt), self.NRX), dtype=xp.float32)

    def step(self, n):
        y = np.float32(self.source_signal[n])

        self.Vx, self.Vy, self.Vz, self.DVx, self.DVy, self.DVz = Kernels.velocityVoigt3D(
            self.Txx, self.Tyy, self.Tzz, self.Txy, self.Txz, self.Tyz,
            self.Vx, self.Vy, self.Vz, self.DVx, self.DVy, self.DVz,
            self.BX, self.BY, self.BZ, self.ABS, self.ddx, self.dt)

        if not self.is_pressure:
            self.Vx, self.Vy, self.Vz = Kernels.sourceVel3D(
                self.Vx, self.Vy, self.Vz, self.XL_tx, self.YL_tx, self.ZL_tx,
                y, self.comp, self.win, self.dtdxx)

        self.Txx, self.Tyy, self.Tzz, self.Txy, self.Txz, self.Tyz = Kernels.stressVoigt3D(
            self.Txx, self.Tyy, self.Tzz, self.Txy, self.Txz, self.Tyz,
            self.Vx, self.Vy, self.Vz, self.DVx, self.DVy, self.DVz,
            self.C11, self.C12, self.C44xy, self.C44xz, self.C44yz,
            self.Eta_vs, self.Eta_s, self.Eta_xy, self.Eta_xz, self.Eta_yz, self.ABS, self.dtx)

        if self.is_pressure:
            self.Txx, self.Tyy, self.Tzz = Kernels.sourceStress3D(
                self.Txx, self.Tyy, self.Tzz, self.XL_tx, self.YL_tx, self.ZL_tx,
                y, self.win, self.dtdxx)

        self.sensor_signals[n, :] = -(1.0 / 3.0) * (
            self.Txx[self.XL_rx, self.YL_rx, self.ZL_rx] +
            self.Tyy[self.XL_rx, self.YL_rx, self.ZL_rx] +
            self.Tzz[self.XL_rx, self.YL_rx, self.ZL_rx])

    def frame(self):
        k = self.PRI // 2
        p = -(1.0 / 3.0) * (self.Txx[:, :, k] + self.Tyy[:, :, k] + self.Tzz[:, :, k])
        return self.toHost(p)

    def exportFields(self):
        return {
            "Vx": self.toHost(self.Vx),
            "Vy": self.toHost(self.Vy),
            "Vz": self.toHost(self.Vz),
            "Txx": self.toHost(self.Txx),
            "Tyy": self.toHost(self.Tyy),
            "Tzz": self.toHost(self.Tzz),
        }
