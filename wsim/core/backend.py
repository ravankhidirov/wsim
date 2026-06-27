import numpy as np


def get_backend(device):
    if device == "gpu":
        import cupy as xp
        return xp
    return np


def asarray(xp, a, dtype=np.float32):
    return xp.asarray(a, dtype=dtype)


def to_host(device, a):
    if device == "gpu":
        import cupy as cp
        return cp.asnumpy(a)
    return np.asarray(a)
