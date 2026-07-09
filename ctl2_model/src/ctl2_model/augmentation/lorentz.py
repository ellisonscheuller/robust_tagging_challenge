import keras
import numpy as np


def levi_civita_3d(dtype="float32"):
    e = np.zeros((3, 3, 3), dtype=dtype)
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
    return keras.ops.cast(e, dtype)


def lorentz_gens(dtype="float32"):
    Lx = np.zeros((4, 4), dtype=dtype)
    Ly = np.zeros((4, 4), dtype=dtype)
    Lz = np.zeros((4, 4), dtype=dtype)
    Kx = np.zeros((4, 4), dtype=dtype)
    Ky = np.zeros((4, 4), dtype=dtype)
    Kz = np.zeros((4, 4), dtype=dtype)

    Lz[1, 2], Lz[2, 1] = 1, -1
    Ly[1, 3], Ly[3, 1] = -1, 1
    Lx[2, 3], Lx[3, 2] = 1, -1

    Kz[0, 3], Kz[3, 0] = 1, 1
    Ky[0, 2], Ky[2, 0] = 1, 1
    Kx[0, 1], Kx[1, 0] = 1, 1

    return keras.ops.stack([
        keras.ops.cast(Kx, dtype),
        keras.ops.cast(Ky, dtype),
        keras.ops.cast(Kz, dtype),
        keras.ops.cast(Lx, dtype),
        keras.ops.cast(Ly, dtype),
        keras.ops.cast(Lz, dtype),
    ])


def boost_3d(
    data,
    beta_max=0.95,
    seed=None,
):
    dtype = keras.ops.dtype(data)
    n = keras.ops.shape(data)[0]

    bf = keras.random.uniform((n,), minval=0, maxval=beta_max, seed=seed, dtype="float32")
    bf = bf ** (1.0 / 3.0)
    bf = keras.ops.cast(bf, dtype)

    b1 = keras.random.uniform((n,), minval=0, maxval=1, seed=seed, dtype=dtype)
    b2 = keras.random.uniform((n,), minval=0, maxval=1, seed=seed, dtype=dtype)
    theta = 2 * np.pi * b1
    phi = keras.ops.arccos(1 - 2 * b2)

    beta_x = keras.ops.sin(phi) * keras.ops.cos(theta)
    beta_y = keras.ops.sin(phi) * keras.ops.sin(theta)
    beta_z = keras.ops.cos(phi)

    beta = keras.ops.stack([bf * beta_x, bf * beta_y, bf * beta_z], axis=-1)
    beta_norm = keras.ops.sqrt(keras.ops.sum(beta ** 2, axis=-1) + 1e-10)
    gamma = 1.0 / keras.ops.sqrt(1.0 - beta_norm ** 2 + 1e-10)
    beta_sq = beta_norm ** 2

    L = keras.ops.zeros((n, 4, 4), dtype=dtype)
    L = keras.ops.slice_update(L, (0, 0, 0), gamma[:, None, None])
    L = keras.ops.slice_update(L, (0, 0, 1), (-gamma[:, None] * beta)[:, None, :])  
    L = keras.ops.slice_update(L, (0, 1, 0), (-gamma[:, None] * beta)[:, :, None])  

    outer_beta = keras.ops.einsum("...i,...j->...ij", beta, beta)
    factor = (gamma - 1)[:, None, None] / (beta_sq[:, None, None] + 1e-10)
    rot_part = outer_beta * factor
    diag = keras.ops.eye(3, dtype=dtype)
    rot_part = diag[None] + rot_part
    L = keras.ops.slice_update(L, (0, 1, 1), rot_part)

    L = keras.ops.cast(L, dtype)
    boosted = keras.ops.einsum("nij,n...j->n...i", L, keras.ops.cast(data, dtype))
    return boosted


def rotation_3d(
    data,
    seed=None,
):
    dtype = keras.ops.dtype(data)
    n = keras.ops.shape(data)[0]

    b1 = keras.random.uniform((n,), minval=0, maxval=1, seed=seed, dtype=dtype)
    b2 = keras.random.uniform((n,), minval=0, maxval=1, seed=seed, dtype=dtype)
    b3 = keras.random.uniform((n,), minval=0, maxval=1, seed=seed, dtype=dtype)

    theta = 2 * np.pi * b1
    phi = keras.ops.arccos(1 - 2 * b2)
    theta_rot = 2 * np.pi * b3

    u_x = keras.ops.sin(phi) * keras.ops.cos(theta)
    u_y = keras.ops.sin(phi) * keras.ops.sin(theta)
    u_z = keras.ops.cos(phi)
    u = keras.ops.stack([u_x, u_y, u_z], axis=-1)

    L = keras.ops.zeros((n, 4, 4), dtype=dtype)
    L = keras.ops.slice_update(L, (0, 0, 0), keras.ops.ones((n, 1, 1), dtype=dtype))

    eye = keras.ops.eye(3, dtype=dtype)[None]
    cos_t = keras.ops.cos(theta_rot)[:, None, None]
    sin_t = keras.ops.sin(theta_rot)[:, None, None]
    uxu = keras.ops.einsum("...i,...j->...ij", u, u)

    eps = levi_civita_3d(dtype)
    ucross = keras.ops.einsum("ijk,...k->...ij", eps, u)

    R = eye * cos_t + (1 - cos_t) * uxu + sin_t * ucross
    L = keras.ops.slice_update(L, (0, 1, 1), keras.ops.cast(R, dtype))

    L = keras.ops.cast(L, dtype)
    rotated = keras.ops.einsum("nij,n...j->n...i", L, keras.ops.cast(data, dtype))
    return rotated
