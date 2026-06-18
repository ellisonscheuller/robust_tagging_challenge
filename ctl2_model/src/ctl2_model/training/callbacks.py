from __future__ import annotations

import math
import numpy as np
from pathlib import Path

import keras
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
})

MINBIAS_COLOR = "#1B3A5C"
SIGNAL_COLOR = "#D62728"


class NCEWeightCallback(keras.callbacks.Callback):
    def __init__(
        self,
        total_epochs: int,
        nce_start: float = 0.05,
        nce_end: float = 0.5,
        nce_warmup: int = 100,
        nce_hold: int = 100,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.nce_start = nce_start
        self.nce_end = nce_end
        self.nce_warmup = nce_warmup
        self.nce_hold = nce_hold

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.nce_warmup:
            w = self.nce_start + (self.nce_end - self.nce_start) * (
                epoch / self.nce_warmup
            )
        elif epoch < self.nce_warmup + self.nce_hold:
            w = self.nce_end
        else:
            progress = (
                epoch - self.nce_warmup - self.nce_hold
            ) / max(self.total_epochs - self.nce_warmup - self.nce_hold, 1)
            w = (self.nce_end / 2) + (self.nce_end / 2) * (
                0.5 * (1 + math.cos(math.pi * progress))
            )
        self.model.nce_weight = float(w)


class LinearProbeCallback(keras.callbacks.Callback):
    def __init__(
        self,
        encoder,
        X_probe,
        y_probe,
        X_eval,
        y_eval,
        num_classes,
        batch_size=2048,
        steps_per_epoch=50,
    ):
        super().__init__()
        self.encoder = encoder
        self.X_probe = X_probe.astype(np.float32)
        self.y_probe = y_probe.astype(np.int32)
        self.X_eval = X_eval.astype(np.float32)
        self.y_eval = y_eval.astype(np.int32)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.probe = keras.layers.Dense(num_classes)
        self.probe_opt = keras.optimizers.Adam(1e-3)

    def on_epoch_end(self, epoch, logs=None):
        z_probe = tf.constant(
            self.encoder.predict(
                self.X_probe, batch_size=self.batch_size, verbose=0
            )["latent"]
        )
        z_eval = tf.constant(
            self.encoder.predict(
                self.X_eval, batch_size=self.batch_size, verbose=0
            )["latent"]
        )
        y_probe = tf.constant(self.y_probe)

        if not self.probe.built:
            self.probe(z_probe[:2])

        for _ in range(self.steps_per_epoch):
            with tf.GradientTape() as tape:
                logits = self.probe(z_probe, training=True)
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(
                        y_probe, logits, from_logits=True
                    )
                )
            grads = tape.gradient(loss, self.probe.trainable_weights)
            self.probe_opt.apply_gradients(
                zip(grads, self.probe.trainable_weights)
            )

        preds = tf.argmax(self.probe(z_eval, training=False), axis=1).numpy()
        logs["val_probe_acc"] = float(np.mean(preds == self.y_eval.numpy()))


class TriggerRateCallback(keras.callbacks.Callback):
    BUNCH_CROSSING_RATE = 40e3
    FILLED_BUNCHES = 2760 / 3564

    def __init__(
        self,
        encoder,
        X_val,
        y_val,
        minbias_label: int = 0,
        minbias_class_idx: int = 2,
        signal_label: int = 1,
        target_rate_hz: float = 16.0,
        n_thresholds: int = 10_000,
        batch_size: int = 2048,
        save_dir: str | None = None,
        save_every: int = 10,
        metric_key: str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.X_val = X_val.astype(np.float32)
        self.y_val = y_val.astype(np.int32)
        self.minbias_label = minbias_label
        self.minbias_class_idx = minbias_class_idx
        self.signal_label = signal_label
        self.target_rate_hz = target_rate_hz
        self.n_thresholds = n_thresholds
        self.batch_size = batch_size
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = save_every
        self._effective_rate = self.BUNCH_CROSSING_RATE * self.FILLED_BUNCHES
        self._metric_key = metric_key or f"val_eff_@{target_rate_hz:.0f}Hz"

    def _anomaly_scores(self):
        out = self.encoder.predict(
            self.X_val, batch_size=self.batch_size, verbose=0
        )
        probs = np.asarray(out["class_probs"])
        return 1.0 - probs[:, self.minbias_class_idx]

    @staticmethod
    def _rank_auc(scores, is_sig):
        n_pos, n_neg = is_sig.sum(), (~is_sig).sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return float(
            (ranks[is_sig].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        scores = self._anomaly_scores()
        bkg_mask = self.y_val == self.minbias_label
        sig_mask = self.y_val == self.signal_label
        scores_bkg = scores[bkg_mask]
        scores_sig = scores[sig_mask]

        if len(scores_bkg) == 0 or len(scores_sig) == 0:
            logs[self._metric_key] = float("nan")
            logs["val_cls_auc"] = float("nan")
            return

        thresholds = np.percentile(
            scores_bkg, np.linspace(0, 100, self.n_thresholds)
        )
        target_fpr = self.target_rate_hz / self._effective_rate
        bkg_rates = np.array(
            [(scores_bkg > t).mean() for t in thresholds]
        )
        closest_idx = np.argmin(np.abs(bkg_rates - target_fpr))
        eff = float((scores_sig > thresholds[closest_idx]).mean())

        logs[self._metric_key] = eff
        logs["val_cls_auc"] = self._rank_auc(scores, ~bkg_mask)

        if self.save_dir and (epoch % self.save_every == 0):
            self._save_plot(scores_bkg, scores_sig, thresholds, eff, epoch)

    def _save_plot(self, scores_bkg, scores_sig, thresholds, eff, epoch):
        bg_rate = (
            np.array([(scores_bkg > t).mean() for t in thresholds])
            * self._effective_rate
            / 1e3
        )
        sig_eff = np.array([(scores_sig > t).mean() for t in thresholds])
        order = np.argsort(bg_rate)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(bg_rate[order], sig_eff[order], color=SIGNAL_COLOR, lw=2)
        ax.axvline(
            self.target_rate_hz / 1e3,
            color=MINBIAS_COLOR,
            linestyle="--",
            lw=1,
            label=f"{self.target_rate_hz:.0f} Hz (eff={eff:.3f})",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Background Rate (kHz)")
        ax.set_ylabel("Signal Efficiency")
        ax.set_xlim(0.1, 40e3)
        ax.set_ylim(0, 1)
        ax.set_title(f"Epoch {epoch}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.save_dir / f"rate_curve_epoch{epoch:04d}.png", dpi=150)
        plt.close(fig)
