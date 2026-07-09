"""model_training pipeline nodes."""
import json
import re
from pathlib import Path

import keras
import mlflow
import numpy as np
from hgq.utils.sugar import BetaScheduler, ParetoFront, PBar, PieceWiseSchedule, FreeEBOPs, Dataset

from ctl2_model.models.linformer import get_linformer
from ctl2_model.models.contrastive_model import ContrastiveHGQEncoder
from ctl2_model.training.callbacks import NCEWeightCallback, TriggerRateCallback
from ctl2_model.utils.plotting import plot_training_history


def train_model(
    X_train: np.ndarray,
    X_aug: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_training: dict,
) -> tuple:
    latent_dim = model_training.get("latent_dim", 16)
    proj_dim = model_training.get("proj_dim", 16)
    ff_dim = model_training.get("ff_dim", 32)
    num_heads = model_training.get("num_heads", 2)
    proj_k = model_training.get("proj_k", 8)
    epochs = model_training.get("epochs", 400)
    batch_size = model_training.get("batch_size", 512)
    initial_lr = model_training.get("initial_lr", 0.001)
    nce_start = model_training.get("nce_start", 0.05)
    nce_end = model_training.get("nce_end", 0.7)
    nce_warmup = model_training.get("nce_warmup", 50)
    nce_hold = model_training.get("nce_hold", 100)
    temperature = model_training.get("temperature", 0.1)
    mse_weight = model_training.get("mse_weight", 0.1)
    target_rate_hz = model_training.get("target_rate_hz", 16.0)
    minbias_label = model_training.get("minbias_label", 2)
    minbias_class_idx = model_training.get("minbias_class_idx", 2)
    signal_label = model_training.get("signal_label", 1)
    beta_schedule = model_training.get("beta_schedule", [
        [0, 0, "constant"],
        [100, 0, "constant"],
        [150, 5.0e-8, "linear"],
        [200, 5.0e-7, "linear"],
        [250, 6.6e-7, "linear"],
        [400, 1.0e-5, "linear"],
    ])
    n_features = X_train.shape[-1]
    n_constituents = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    save_path = Path("data/04_models")
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_path / "ckpts"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    encoder = get_linformer(
        n_constituents=n_constituents,
        n_features=n_features,
        latent_dim=latent_dim,
        n_classes=n_classes,
        ff_dim=ff_dim,
        num_heads=num_heads,
        proj_k=proj_k,
    )

    model = ContrastiveHGQEncoder(
        encoder=encoder,
        proj_dim=proj_dim,
        mse_weight=mse_weight,
        temperature=temperature,
    )

    def cosine_decay_restarts(epoch):
        from math import cos, pi
        first_decay_steps = 30
        alpha = 1e-5
        m_mul = 0.8
        n_cycle, cycle_step, cycle_len = 1, epoch, first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            n_cycle += 1
        cycle_t = min(cycle_step / (cycle_len - 5), 1)
        return alpha + 0.5 * (initial_lr - alpha) * (1 + cos(pi * cycle_t)) * m_mul ** max(n_cycle - 1, 0)

    model.compile(optimizer=keras.optimizers.Adam())

    class CkptSaver(keras.callbacks.Callback):
        def __init__(self, encoder, ckpt_path, save_freq=5):
            super().__init__()
            self.encoder = encoder
            self.ckpt_path = Path(ckpt_path)
            self.save_freq = save_freq
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_freq == 0:
                eff = logs.get('val_eff_@16Hz', -1)
                eff_str = f"{eff:.4f}" if not np.isnan(eff) and eff >= 0 else "nan"
                fname = f"epoch={epoch+1:04d}-loss={logs.get('loss', 0):.4f}-ce={logs.get('train_ce', 0):.4f}-nce={logs.get('train_nce', 0):.4f}-eff={eff_str}-ebops={logs.get('ebops', 0):.0f}.keras"
                self.encoder.save(str(self.ckpt_path / fname))

    has_val = len(X_val) > 0
    pareto_front = None

    callbacks = [
        keras.callbacks.LearningRateScheduler(cosine_decay_restarts),
        NCEWeightCallback(epochs, nce_start, nce_end, nce_warmup, nce_hold),
        BetaScheduler(PieceWiseSchedule(beta_schedule)),
        FreeEBOPs(),
        keras.callbacks.TerminateOnNaN(),
    ]

    if has_val:
        callbacks.append(
            TriggerRateCallback(
                encoder, X_val, y_val,
                minbias_label=minbias_label, minbias_class_idx=minbias_class_idx,
                signal_label=signal_label, target_rate_hz=target_rate_hz,
                save_dir=str(ckpt_path / "rate_curves"), save_every=10,
            )
        )
        pareto_front = ParetoFront(
            path=str(ckpt_path / "pareto"),
            metrics=["val_eff_@16Hz", "ebops"],
            sides=[1, -1],
            fname_format="pareto_epoch={epoch}_eff={val_eff_@16Hz:.4f}_EBOPs={ebops}.keras",
            enable_if=lambda logs: logs.get("val_eff_@16Hz", 0) > 0.01,
        )
        callbacks.append(pareto_front)

    callbacks.append(
        PBar(metric='loss: {loss:.4f} - ce: {train_ce:.4f} - nce: {train_nce:.4f} - eff@16Hz: {val_eff_@16Hz:.4f} - auc: {val_cls_auc:.4f} - lr: {learning_rate:.2e} - EBOPs: {ebops}')
    )
    callbacks.append(CkptSaver(encoder, ckpt_path, save_freq=5))

    dataset_train = Dataset(
        (X_train.astype(np.float32), X_aug.astype(np.float32)),
        y_train.astype(np.int32),
        batch_size=batch_size,
        drop_last=True,
        device='gpu:0',
    )

    model.fit(
        dataset_train,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )

    encoder.save(str(save_path / 'trained_model.keras'))

    history = {}
    for k, v in model.history.history.items():
        history[k] = [float(x) if hasattr(x, 'item') else x for x in v]

    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f)

    # --- MLflow tracking ---
    report_dir = Path("data/05_validation")
    report_dir.mkdir(parents=True, exist_ok=True)

    for k, v in history.items():
        safe_k = k.replace("@", "_at_").replace(" ", "_").replace("/", "_per_")
        if v:
            mlflow.log_metric(f"final_{safe_k}", v[-1])
        for epoch_i, val in enumerate(v):
            mlflow.log_metric(safe_k, val, step=epoch_i)

    with open(report_dir / "per_epoch_history.json", "w") as f:
        json.dump(history, f, indent=2)
    mlflow.log_artifact(str(report_dir / "per_epoch_history.json"))
    mlflow.log_artifact(str(save_path / 'training_history.json'))

    hist_plot_path = plot_training_history(history)
    if hist_plot_path and Path(hist_plot_path).exists():
        mlflow.log_artifact(hist_plot_path)
    loss_plot_path = str(Path(hist_plot_path).with_suffix('') if hist_plot_path else "") + "_losses.png"
    if Path(loss_plot_path).exists():
        mlflow.log_artifact(loss_plot_path)

    try:
        mlflow.keras.log_model(
            encoder,
            artifact_path="encoder",
            registered_model_name="maglowac-ctl2-embedding-encoder",
        )
    except Exception as e:
        print(f"  (skipping mlflow model log: {e})")

    # --- Select best checkpoint ---
    ckpt_pattern = re.compile(
        r"epoch=(\d+)-loss=[\d.]+-ce=([\d.]+)-nce=([\d.]+)-eff=([\d.nan]+)-ebops=([\d.e+]+)\.keras"
    )
    candidates = []
    for f in sorted(ckpt_path.glob("epoch=*-loss=*-ce=*-nce=*-eff=*-ebops=*.keras")):
        m = ckpt_pattern.match(f.name)
        if m:
            ep = int(m.group(1))
            ce = float(m.group(2))
            nce = float(m.group(3))
            eff = float(m.group(4)) if m.group(4) != "nan" else -1.0
            eb = float(m.group(5))
            if eb < 2_000_000:
                candidates.append((ce + nce, -eff, ep, f))

    best_encoder = encoder
    selected_epoch = epochs
    selected_metrics = {"selected_from": -1.0}

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        cep, neff, ep, best_path = candidates[0]
        best_eff = -neff
        print(f"  Selected checkpoint: {best_path.name} (epoch={ep}, ce+nce={cep:.4f}, eff={best_eff:.4f})")
        best_encoder = keras.models.load_model(str(best_path), compile=False)
        selected_epoch = ep
        selected_metrics = {"ce_nce": cep, "val_eff": best_eff, "selected_epoch": ep, "selected_from": 1.0}
    else:
        print("  No checkpoint with ebops < 2M found, using final model")

    best_encoder.save(str(save_path / "trained_model.keras"))
    mlflow.log_artifact(str(save_path / "trained_model.keras"))
    mlflow.log_metrics(selected_metrics)

    return encoder, history
