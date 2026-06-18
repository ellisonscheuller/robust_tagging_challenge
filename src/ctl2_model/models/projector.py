from __future__ import annotations

import keras


def build_projector(
    latent_dim: int = 12,
    proj_dim: int = 16,
    hidden_dim: int | None = None,
):
    hidden_dim = hidden_dim or proj_dim * 4
    return keras.Sequential([
        keras.layers.Dense(hidden_dim, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("gelu"),
        keras.layers.Dense(hidden_dim, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("gelu"),
        keras.layers.Dense(proj_dim, use_bias=False),
        keras.layers.Lambda(
            lambda x: x / (keras.ops.norm(x, axis=1, keepdims=True) + 1e-8)
        ),
    ], name="projector")
