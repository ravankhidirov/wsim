import numpy as np


class Source:

    def __init__(self, positions, signal, kind="pressure", window=None):
        positions = np.asarray(positions, dtype=np.int32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        self.positions = positions
        self.signal = np.asarray(signal, dtype=np.float32)
        self.kind = kind

        NX = np.shape(positions)[0]
        if window is None:
            self.window = np.ones((NX,), dtype=np.float32)
        else:
            self.window = np.asarray(window, dtype=np.float32)


class Sensor:

    def __init__(self, positions, record="pressure"):
        positions = np.asarray(positions, dtype=np.int32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        self.positions = positions
        self.record = record
