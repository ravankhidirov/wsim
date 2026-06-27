import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def plot_medium(field, filename, title="medium", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 5), layout="tight")
    im = ax.imshow(field.T, origin="lower", aspect="equal", cmap=cmap)
    ax.set_xlabel("x [grid]")
    ax.set_ylabel("y [grid]")
    fig.colorbar(im, ax=ax, label=title)
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def plot_traces(p, t, filename, title="sensor traces"):
    p = np.asarray(p)
    t_us = np.asarray(t) * 1e6
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout="tight")

    im = ax[0].imshow(p.T, aspect="auto", origin="lower",
                      extent=[t_us[0], t_us[-1], 0, p.shape[1]], cmap="seismic")
    ax[0].set_xlabel("time [us]")
    ax[0].set_ylabel("sensor index")
    ax[0].set_title(title)
    fig.colorbar(im, ax=ax[0])

    mid = p.shape[1] // 2
    ax[1].plot(t_us, p[:, 0], label="sensor 0")
    ax[1].plot(t_us, p[:, mid], label="sensor %d" % mid)
    ax[1].set_xlabel("time [us]")
    ax[1].set_ylabel("pressure")
    ax[1].legend()

    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def save_gif(frames, filename, cmap="seismic", fps=15, percentile=99.5):
    frames = np.asarray(frames, dtype=np.float32)
    vmax = np.percentile(np.abs(frames), percentile)
    if vmax <= 0:
        vmax = 1.0

    cmap_f = plt.get_cmap(cmap)
    images = []
    for fr in frames:
        norm = np.clip(fr / vmax, -1.0, 1.0) * 0.5 + 0.5
        rgba = (cmap_f(np.flipud(norm.T)) * 255).astype(np.uint8)
        images.append(Image.fromarray(rgba))

    images[0].save(filename, save_all=True, append_images=images[1:],
                   duration=int(1000.0 / fps), loop=0)
    return filename
