import numpy as np


class Medium:

    def __init__(self, rho, cl, ct, eta_v=None, eta_s=None, air_threshold=2.0):
        self.Rho = np.asarray(rho, dtype=np.float32)
        cl = np.asarray(cl, dtype=np.float32)
        ct = np.asarray(ct, dtype=np.float32)

        self.C11 = (self.Rho * cl * cl).astype(np.float32)
        self.C44 = (self.Rho * ct * ct).astype(np.float32)
        self.C12 = (self.C11 - 2.0 * self.C44).astype(np.float32)
        self.C22 = self.C11.copy()

        if eta_v is None:
            self.Eta_v = np.ones_like(self.Rho) * 1e-30
        else:
            self.Eta_v = np.asarray(eta_v, dtype=np.float32)

        if eta_s is None:
            self.Eta_s = np.ones_like(self.Rho) * 1e-30
        else:
            self.Eta_s = np.asarray(eta_s, dtype=np.float32)

        self.air_threshold = float(air_threshold)
        self.shape = self.Rho.shape

    def cmax(self):
        return float(np.sqrt(np.max(self.C11 / self.Rho)))
