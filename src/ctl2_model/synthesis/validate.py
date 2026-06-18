from __future__ import annotations

import numpy as np
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare_predictions(
    keras_model,
    comb_logic,
    x_test: np.ndarray,
    title: str = "Keras vs Combinational Logic",
    save_path: str | None = None,
):
    keras_out = keras_model.predict(x_test, verbose=0)
    comb_out = comb_logic.predict(x_test)

    n_dims = keras_out.shape[-1] if len(keras_out.shape) > 1 else 1
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for i in range(n_dims):
        k = keras_out[:, i] if n_dims > 1 else keras_out
        c = comb_out[:, i] if n_dims > 1 else comb_out

        axes[i].scatter(k, c, s=1, alpha=0.3)
        axes[i].plot([k.min(), k.max()], [k.min(), k.max()], "r--", lw=1)
        axes[i].set_xlabel("Keras")
        axes[i].set_ylabel("Comb Logic")
        axes[i].set_title(f"Dim {i}")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()


def compare_hls(
    keras_model,
    hls_model,
    x_test: np.ndarray,
    save_path: str | None = None,
):
    keras_out = keras_model.predict(x_test, verbose=0)
    hls_out = hls_model.predict(x_test)

    n_dims = keras_out.shape[-1] if len(keras_out.shape) > 1 else 1
    n_cols = min(4, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_dims > 1 else [axes]

    for i in range(n_dims):
        k = keras_out[:, i] if n_dims > 1 else keras_out
        h = hls_out[:, i] if n_dims > 1 else hls_out

        corr = np.corrcoef(k, h)[0, 1]
        mae = np.mean(np.abs(k - h))

        axes[i].scatter(k, h, s=1, alpha=0.3)
        axes[i].plot([k.min(), k.max()], [k.min(), k.max()], "r--", lw=1)
        axes[i].set_xlabel("Keras")
        axes[i].set_ylabel("HLS")
        axes[i].set_title(f"Dim {i}  r={corr:.4f}  MAE={mae:.4f}")

    plt.suptitle("Keras vs HLS Simulation")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()
