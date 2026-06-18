import keras
from .coordinate import cylindrical_to_cartesian_p4, cartesian_to_cylindrical_inplace
from .lorentz import boost_3d, rotation_3d


class GSEAL(keras.layers.Layer):
    def __init__(
        self,
        rotate=True,
        boost=True,
        beta_max=0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rotate = rotate
        self.boost = boost
        self.beta_max = beta_max

    def call(self, x, training=None, seed=None):
        if not training:
            return x

        p4 = cylindrical_to_cartesian_p4(x, self.compute_dtype)
        p4_t = p4
        if self.boost:
            p4_t = boost_3d(p4_t, beta_max=self.beta_max, seed=seed)
        if self.rotate:
            p4_t = rotation_3d(p4_t, seed=seed)
        x_aug = cartesian_to_cylindrical_inplace(p4_t, x)
        return x_aug


class GSEALAugmentation:
    def __init__(self, rotate=True, boost=True, beta_max=0.95, batch_size=50000):
        self.rotate = rotate
        self.boost = boost
        self.beta_max = beta_max
        self.batch_size = batch_size

    def __call__(self, x):
        import numpy as np
        x = np.asarray(x, dtype=np.float32)
        parts = []
        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            p4 = cylindrical_to_cartesian_p4(batch)
            p4_t = p4
            if self.boost:
                p4_t = boost_3d(p4_t, beta_max=self.beta_max, seed=None)
            if self.rotate:
                p4_t = rotation_3d(p4_t, seed=None)
            parts.append(np.asarray(cartesian_to_cylindrical_inplace(p4_t, batch), dtype=np.float32))
        return np.concatenate(parts, axis=0)


def augment_pair(x, rotate=True, boost=True, beta_max=0.95):
    gseal = GSEALAugmentation(rotate=rotate, boost=boost, beta_max=beta_max)
    x_aug = gseal(x)
    return x, x_aug
