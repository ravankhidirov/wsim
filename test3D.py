import os
import numpy as np

from wsim.core import Grid, Medium, Source, Sensor, Signals, EFIT2D, EFIT3D
from wsim.core import visualization as viz


c_air, rho_air = 343.0, 1.2
c_steel, rho_steel, ct_steel = 5900.0, 7850.0, 3200.0
c_water, rho_water = 1500.0, 1000.0

f0, amp = 2.0e6, 1.0e5

Nx = 500
Ny = 500
Nz = 500
dx = 2.0e-4
Nt = 5000
pml = 12
device = "gpu"
outdir = "results"

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

src_sig = Signals(
    Name="RaisedCosine",
    Amplitud=amp,
    Frequency=f0
).generate(grid.t)

x_src = N_air + N_steel + 4
y_src = Ny // 2
z_src = Nz // 2

source = Source([x_src, y_src, z_src], src_sig, kind="pressure")

ys = np.linspace(int(0.2 * Ny), int(0.8 * Ny), 6).astype(int)
zs = np.linspace(int(0.2 * Nz), int(0.8 * Nz), 6).astype(int)

YY, ZZ = np.meshgrid(ys, zs)
YY = YY.ravel()
ZZ = ZZ.ravel()
XX = np.full_like(YY, x_src)

sensor = Sensor(
    np.stack([XX, YY, ZZ], axis=1),
    record="pressure"
)

eng = EFIT3D(grid, medium, source, sensor, pml=pml, device=device)
out = eng.run_all(snapshot_every=4)

p = out["p"]

print(
    "3D  Im=", medium.shape,
    "Nt=", grid.Nt,
    "p.shape=", p.shape,
    "finite=", bool(np.all(np.isfinite(p))),
    "max|p|=%.3e" % float(np.max(np.abs(p)))
)

viz.plot_medium(
    cl[:, :, Nz // 2],
    os.path.join(outdir, "medium_3d.png"),
    title="c_l [m/s] (mid-z)"
)

viz.plot_traces(
    p,
    out["t"],
    os.path.join(outdir, "traces_3d.png"),
    title="3D sensor traces"
)

viz.save_gif(
    out["frames"],
    os.path.join(outdir, "wavefield_3d.gif"),
    fps=12
)

print("    saved medium_3d.png, traces_3d.png, wavefield_3d.gif (mid-z slice)")