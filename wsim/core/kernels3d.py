def sourceVel3D(Vx, Vy, Vz, XL, YL, ZL, source, comp, win, dtdxx):
    if comp == 0:
        Vx[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]
    elif comp == 1:
        Vy[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]
    else:
        Vz[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]

    return (Vx, Vy, Vz)


def sourceStress3D(Txx, Tyy, Tzz, XL, YL, ZL, source, win, dtdxx):
    Txx[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]
    Tyy[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]
    Tzz[XL[:], YL[:], ZL[:]] -= source * dtdxx * win[:]

    return (Txx, Tyy, Tzz)


def velocityVoigt3D(Txx, Tyy, Tzz, Txy, Txz, Tyz, vx, vy, vz, d_vx, d_vy, d_vz, BX, BY, BZ, ABS, ddx, dt):
    d_vx[:-1, 1:, 1:] = (ddx * BX[:-1, 1:, 1:]) * (
        Txx[1:, 1:, 1:] - Txx[:-1, 1:, 1:] +
        Txy[:-1, 1:, 1:] - Txy[:-1, :-1, 1:] +
        Txz[:-1, 1:, 1:] - Txz[:-1, 1:, :-1])
    vx[:-1, 1:, 1:] += dt * d_vx[:-1, 1:, 1:]

    d_vy[1:, :-1, 1:] = (ddx * BY[1:, :-1, 1:]) * (
        Txy[1:, :-1, 1:] - Txy[:-1, :-1, 1:] +
        Tyy[1:, 1:, 1:] - Tyy[1:, :-1, 1:] +
        Tyz[1:, :-1, 1:] - Tyz[1:, :-1, :-1])
    vy[1:, :-1, 1:] += dt * d_vy[1:, :-1, 1:]

    d_vz[1:, 1:, :-1] = (ddx * BZ[1:, 1:, :-1]) * (
        Txz[1:, 1:, :-1] - Txz[:-1, 1:, :-1] +
        Tyz[1:, 1:, :-1] - Tyz[1:, :-1, :-1] +
        Tzz[1:, 1:, 1:] - Tzz[1:, 1:, :-1])
    vz[1:, 1:, :-1] += dt * d_vz[1:, 1:, :-1]

    vx[:, :, :] *= ABS[:, :, :]
    vy[:, :, :] *= ABS[:, :, :]
    vz[:, :, :] *= ABS[:, :, :]

    return (vx, vy, vz, d_vx, d_vy, d_vz)


def stressVoigt3D(Txx, Tyy, Tzz, Txy, Txz, Tyz, vx, vy, vz, d_vx, d_vy, d_vz,
                  C11, C12, C44xy, C44xz, C44yz, ETA_VS, ETA_S, ETA_xy, ETA_xz, ETA_yz, ABS, dtx):
    exx = vx[1:, 1:, 1:] - vx[:-1, 1:, 1:]
    eyy = vy[1:, 1:, 1:] - vy[1:, :-1, 1:]
    ezz = vz[1:, 1:, 1:] - vz[1:, 1:, :-1]

    dexx = d_vx[1:, 1:, 1:] - d_vx[:-1, 1:, 1:]
    deyy = d_vy[1:, 1:, 1:] - d_vy[1:, :-1, 1:]
    dezz = d_vz[1:, 1:, 1:] - d_vz[1:, 1:, :-1]

    Txx[1:, 1:, 1:] += ((dtx * C11[1:, 1:, 1:]) * exx +
                        (dtx * C12[1:, 1:, 1:]) * (eyy + ezz) +
                        (dtx * ETA_VS[1:, 1:, 1:]) * dexx +
                        (dtx * ETA_S[1:, 1:, 1:]) * (deyy + dezz))

    Tyy[1:, 1:, 1:] += ((dtx * C11[1:, 1:, 1:]) * eyy +
                        (dtx * C12[1:, 1:, 1:]) * (exx + ezz) +
                        (dtx * ETA_VS[1:, 1:, 1:]) * deyy +
                        (dtx * ETA_S[1:, 1:, 1:]) * (dexx + dezz))

    Tzz[1:, 1:, 1:] += ((dtx * C11[1:, 1:, 1:]) * ezz +
                        (dtx * C12[1:, 1:, 1:]) * (exx + eyy) +
                        (dtx * ETA_VS[1:, 1:, 1:]) * dezz +
                        (dtx * ETA_S[1:, 1:, 1:]) * (dexx + deyy))

    Txy[:-1, :-1, :] += ((dtx * C44xy[:-1, :-1, :]) *
                         (vx[:-1, 1:, :] - vx[:-1, :-1, :] + vy[1:, :-1, :] - vy[:-1, :-1, :]) +
                         (dtx * ETA_xy[:-1, :-1, :]) *
                         (d_vx[:-1, 1:, :] - d_vx[:-1, :-1, :] + d_vy[1:, :-1, :] - d_vy[:-1, :-1, :]))

    Txz[:-1, :, :-1] += ((dtx * C44xz[:-1, :, :-1]) *
                         (vx[:-1, :, 1:] - vx[:-1, :, :-1] + vz[1:, :, :-1] - vz[:-1, :, :-1]) +
                         (dtx * ETA_xz[:-1, :, :-1]) *
                         (d_vx[:-1, :, 1:] - d_vx[:-1, :, :-1] + d_vz[1:, :, :-1] - d_vz[:-1, :, :-1]))

    Tyz[:, :-1, :-1] += ((dtx * C44yz[:, :-1, :-1]) *
                         (vy[:, :-1, 1:] - vy[:, :-1, :-1] + vz[:, 1:, :-1] - vz[:, :-1, :-1]) +
                         (dtx * ETA_yz[:, :-1, :-1]) *
                         (d_vy[:, :-1, 1:] - d_vy[:, :-1, :-1] + d_vz[:, 1:, :-1] - d_vz[:, :-1, :-1]))

    Txx[:, :, :] *= ABS[:, :, :]
    Tyy[:, :, :] *= ABS[:, :, :]
    Tzz[:, :, :] *= ABS[:, :, :]
    Txy[:, :, :] *= ABS[:, :, :]
    Txz[:, :, :] *= ABS[:, :, :]
    Tyz[:, :, :] *= ABS[:, :, :]

    return (Txx, Tyy, Tzz, Txy, Txz, Tyz)
