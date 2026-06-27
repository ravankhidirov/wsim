import os
import numpy as np

from wsim.core import Grid, Medium, Source, Sensor, Signals, EFIT2D, EFIT3D
from wsim.core import visualization as viz


c_air, rho_air = 343.0, 1.2
c_steel, rho_steel, ct_steel = 5900.0, 7850.0, 3200.0
c_water, rho_water = 1500.0, 1000.0

f0, amp = 2.0e6, 1.0e5


def example_2d(Nx=160, Ny=160, dx=1.0e-4, Nt=500, pml=20, device="cpu", outdir="results"):

    os.makedirs(outdir, exist_ok=True)
    grid = Grid([Nx, Ny], [dx, dx])

    N_air, N_steel = 6, 20
    rho = rho_water * np.ones((Nx, Ny), dtype=np.float32)
    cl = c_water * np.ones((Nx, Ny), dtype=np.float32)
    ct = np.zeros((Nx, Ny), dtype=np.float32)

    rho[:N_air, :] = rho_air
    cl[:N_air, :] = c_air
    rho[N_air:N_air + N_steel, :] = rho_steel
    cl[N_air:N_air + N_steel, :] = c_steel
    ct[N_air:N_air + N_steel, :] = ct_steel
    rho[Nx - (N_air + N_steel):Nx - N_air, :] = rho_steel
    cl[Nx - (N_air + N_steel):Nx - N_air, :] = c_steel
    ct[Nx - (N_air + N_steel):Nx - N_air, :] = ct_steel
    rho[Nx - N_air:, :] = rho_air
    cl[Nx - N_air:, :] = c_air

    medium = Medium(rho, cl, ct)
    grid.makeTime(c_steel, cfl=0.3, Nt=Nt)

    src_sig = Signals(Name="RaisedCosine", Amplitud=amp, Frequency=f0).generate(grid.t)
    x_src, y_src = N_air + N_steel + 6, Ny // 2
    source = Source([x_src, y_src], src_sig, kind="pressure")

    ys = np.linspace(int(0.1 * Ny), int(0.9 * Ny), 24).astype(int)
    xs = np.full_like(ys, x_src)
    sensor = Sensor(np.stack([xs, ys], axis=1), record="pressure")

    eng = EFIT2D(grid, medium, source, sensor, pml=pml, device=device)
    out = eng.run_all(snapshot_every=10)

    p = out["p"]
    print("2D  Im=", medium.shape, "Nt=", grid.Nt, "p.shape=", p.shape,
          "finite=", bool(np.all(np.isfinite(p))), "max|p|=%.3e" % float(np.max(np.abs(p))))

    viz.plot_medium(cl, os.path.join(outdir, "medium_2d.png"), title="c_l [m/s]")
    viz.plot_traces(p, out["t"], os.path.join(outdir, "traces_2d.png"), title="2D sensor traces")
    viz.save_gif(out["frames"], os.path.join(outdir, "wavefield_2d.gif"), fps=15)
    print("    saved medium_2d.png, traces_2d.png, wavefield_2d.gif")
    return out


def example_3d(Nx=56, Ny=56, Nz=56, dx=2.0e-4, Nt=160, pml=8, device="cpu", outdir="results"):

    os.makedirs(outdir, exist_ok=True)
    grid = Grid([Nx, Ny, Nz], [dx, dx, dx])

    N_air, N_steel = 4, 10
    rho = rho_water * np.ones((Nx, Ny, Nz), dtype=np.float32)
    cl = c_water * np.ones((Nx, Ny, Nz), dtype=np.float32)
    ct = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    rho[:N_air, :, :] = rho_air
    cl[:N_air, :, :] = c_air
    rho[N_air:N_air + N_steel, :, :] = rho_steel
    cl[N_air:N_air + N_steel, :, :] = c_steel
    ct[N_air:N_air + N_steel, :, :] = ct_steel
    rho[Nx - N_air:, :, :] = rho_air
    cl[Nx - N_air:, :, :] = c_air

    medium = Medium(rho, cl, ct)
    grid.makeTime(c_steel, cfl=0.3, Nt=Nt)

    src_sig = Signals(Name="RaisedCosine", Amplitud=amp, Frequency=f0).generate(grid.t)
    x_src, y_src, z_src = N_air + N_steel + 4, Ny // 2, Nz // 2
    source = Source([x_src, y_src, z_src], src_sig, kind="pressure")

    ys = np.linspace(int(0.2 * Ny), int(0.8 * Ny), 6).astype(int)
    zs = np.linspace(int(0.2 * Nz), int(0.8 * Nz), 6).astype(int)
    YY, ZZ = np.meshgrid(ys, zs)
    YY, ZZ = YY.ravel(), ZZ.ravel()
    XX = np.full_like(YY, x_src)
    sensor = Sensor(np.stack([XX, YY, ZZ], axis=1), record="pressure")

    eng = EFIT3D(grid, medium, source, sensor, pml=pml, device=device)
    out = eng.run_all(snapshot_every=4)

    p = out["p"]
    print("3D  Im=", medium.shape, "Nt=", grid.Nt, "p.shape=", p.shape,
          "finite=", bool(np.all(np.isfinite(p))), "max|p|=%.3e" % float(np.max(np.abs(p))))

    viz.plot_medium(cl[:, :, Nz // 2], os.path.join(outdir, "medium_3d.png"), title="c_l [m/s] (mid-z)")
    viz.plot_traces(p, out["t"], os.path.join(outdir, "traces_3d.png"), title="3D sensor traces")
    viz.save_gif(out["frames"], os.path.join(outdir, "wavefield_3d.gif"), fps=12)
    print("    saved medium_3d.png, traces_3d.png, wavefield_3d.gif (mid-z slice)")
    return out


if __name__ == "__main__":
    example_2d(device="cpu")
    example_3d(device="cpu")
