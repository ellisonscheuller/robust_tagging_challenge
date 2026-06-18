"""data_processing pipeline nodes.

Takes the loaded DataFrame (from ctl2_modelLoader Kedro dataset),
extracts PF-candidate arrays, computes normalization, preprocesses,
splits into train/val/eval by sample name, and applies GSEAL augmentation.
"""
import mlflow
import numpy as np
import pandas as pd

from ctl2_model.data.preprocessing import preprocess_dataset, compute_normalization_constants, build_pid_mapping
from ctl2_model.augmentation.gseal import GSEALAugmentation
from ctl2_model.utils.plotting import plot_data_distributions, plot_pid_distributions


def preprocess_and_split(
    data: pd.DataFrame,
    data_processing: dict,
    load_data: dict,
) -> tuple:
    n_constituents = data_processing.get("n_constituents", 100)
    augment = data_processing.get("augment", True)
    rotate = data_processing.get("rotate", True)
    boost = data_processing.get("boost", True)
    beta_max = data_processing.get("beta_max", 0.95)

    # Direct numpy stack — no Python loops
    raw_tensor = np.stack([
        np.vstack(data["L1PuppiCands_pt"].values).astype(np.float32),
        np.vstack(data["L1PuppiCands_eta"].values).astype(np.float32),
        np.vstack(data["L1PuppiCands_phi"].values).astype(np.float32),
        np.vstack(data["L1PuppiCands_dxy"].values).astype(np.float32),
        np.vstack(data["L1PuppiCands_pdgId"].values).astype(np.float32),
    ], axis=-1)  # shape: (n_events, n_constituents, 5)

    y = data["y"].values.astype(np.int32)

    # Split first
    splits = load_data.get("splits", {})
    if splits:
        split_idx = _split_by_sample(data, splits, raw_tensor, y)
    else:
        from sklearn.model_selection import train_test_split
        val_size = data_processing.get("val_size", 0.15)
        eval_size = data_processing.get("eval_size", 0.15)
        random_state = data_processing.get("random_state", 42)
        all_idx = np.arange(len(data))
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=val_size + eval_size,
            random_state=random_state, stratify=y,
        )
        val_frac = val_size / (val_size + eval_size)
        val_idx, eval_idx = train_test_split(
            temp_idx, test_size=1 - val_frac,
            random_state=random_state, stratify=y[temp_idx],
        )
        split_idx = {"train": train_idx, "val": val_idx, "eval": eval_idx}

    train_idx = split_idx.get("train", np.array([], dtype=int))
    val_idx = split_idx.get("val", np.array([], dtype=int))
    eval_idx = split_idx.get("eval", np.array([], dtype=int))

    # Normalisation computed on train only — no leakage
    norm = compute_normalization_constants(raw_tensor[train_idx])
    pid_mapping = build_pid_mapping(raw_tensor[train_idx])

    X_all, pid_mapping_out = preprocess_dataset(
        raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True
    )

    X_train = X_all[train_idx] if len(train_idx) else np.empty((0, *X_all.shape[1:]))
    y_train = y[train_idx] if len(train_idx) else np.empty((0,), dtype=np.int32)
    X_val = X_all[val_idx] if len(val_idx) else np.empty((0, *X_all.shape[1:]))
    y_val = y[val_idx] if len(val_idx) else np.empty((0,), dtype=np.int32)
    X_eval = X_all[eval_idx] if len(eval_idx) else np.empty((0, *X_all.shape[1:]))
    y_eval = y[eval_idx] if len(eval_idx) else np.empty((0,), dtype=np.int32)

    if augment and len(train_idx) > 0:
        aug = GSEALAugmentation(rotate=rotate, boost=boost, beta_max=beta_max)
        X_aug_raw = aug(raw_tensor[train_idx])  # correct — augment actual train set
        X_aug, _ = preprocess_dataset(X_aug_raw, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
    else:
        X_aug = X_train.copy()

    mlflow.log_metrics({
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_eval": len(X_eval),
        "n_features": X_train.shape[-1] if len(X_train) else 0,
        "n_constituents": X_train.shape[1] if len(X_train) else 0,
    })
    dist_path = plot_data_distributions(X_train, y_train)
    mlflow.log_artifact(dist_path)
    pid_dist_path = plot_pid_distributions(X_train, pid_mapping)
    if pid_dist_path:
        mlflow.log_artifact(pid_dist_path)

    return X_train, X_aug, y_train, X_val, y_val, X_eval, y_eval