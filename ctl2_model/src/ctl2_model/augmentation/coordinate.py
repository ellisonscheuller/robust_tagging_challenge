import keras


def cylindrical_to_cartesian_p4(x, dtype="float32"):
    pt = keras.ops.cast(x[..., 0], dtype)
    eta = keras.ops.cast(x[..., 1], dtype)
    phi = keras.ops.cast(x[..., 2], dtype)

    px = pt * keras.ops.cos(phi)
    py = pt * keras.ops.sin(phi)
    pz = pt * keras.ops.sinh(eta)
    E = keras.ops.sqrt(px ** 2 + py ** 2 + pz ** 2)

    return keras.ops.stack([E, px, py, pz], axis=-1)


def cartesian_to_cylindrical_inplace(p4, x, eps=1e-8):
    dtype = keras.ops.dtype(x)
    p4 = keras.ops.cast(p4, dtype)

    E = p4[..., 0]
    px = p4[..., 1]
    py = p4[..., 2]
    pz = p4[..., 3]

    pt = keras.ops.sqrt(px ** 2 + py ** 2)
    phi = keras.ops.arctan2(py, px)
    eta = keras.ops.arcsinh(pz / (pt + keras.ops.cast(eps, dtype)))

    x_out = keras.ops.copy(x)
    x_out = keras.ops.slice_update(x_out, (0,) * (len(x_out.shape) - 1) + (0,), pt[..., None])
    x_out = keras.ops.slice_update(x_out, (0,) * (len(x_out.shape) - 1) + (1,), eta[..., None])
    x_out = keras.ops.slice_update(x_out, (0,) * (len(x_out.shape) - 1) + (2,), phi[..., None])
    return x_out
