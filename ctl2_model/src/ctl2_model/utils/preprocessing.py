import numpy as np


NORM_CONSTANTS_DEFAULT = {
    "pt_min": 0.0,
    "pt_max": 8.3179,
    "eta_min": -5.0309,
    "eta_max": 5.0353,
    "phi_min": -np.pi,
    "phi_max": np.pi,
    "dxy_min": -3.1416,
    "dxy_max": 3.1406,
}


def compute_normalization_constants(data: np.ndarray) -> dict:
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    phi = data[:, :, 2]
    dxy = data[:, :, 3]

    pt_log = np.log1p(pt)
    return {
        "pt_min": pt_log.min(),
        "pt_max": pt_log.max(),
        "eta_min": eta.min(),
        "eta_max": eta.max(),
        "phi_min": -np.pi,
        "phi_max": np.pi,
        "dxy_min": dxy.min(),
        "dxy_max": dxy.max(),
    }


def normalize_particles(x: np.ndarray, norm: dict, eps_dxy: float = 1e-6) -> np.ndarray:
    pt = np.log1p(x[..., 0])
    pt = (pt - norm["pt_min"]) / (norm["pt_max"] - norm["pt_min"] + 1e-8)
    eta = (x[..., 1] - norm["eta_min"]) / (norm["eta_max"] - norm["eta_min"] + 1e-8)
    phi = (x[..., 2] + np.pi) / (2 * np.pi)

    mask_pad = x[..., 0] == 0
    pt[mask_pad] = 0.0
    eta[mask_pad] = 0.0
    phi[mask_pad] = 0.0

    out = np.stack([
        pt, eta, phi,
        x[..., 3].copy(),
        x[..., 4].copy(),
    ], axis=-1)
    out[mask_pad, :4] = 0.0
    return out


def build_pid_mapping(data: np.ndarray) -> dict:
    pid_tensor = data[:, :, 4]
    mask_valid = data[:, :, 0] > 0
    valid_pids = pid_tensor[mask_valid].astype(np.int32)
    unique_pids = np.unique(valid_pids)
    return {int(pid): idx for idx, pid in enumerate(unique_pids)}


def map_pid(pid: np.ndarray, pid_mapping: dict) -> np.ndarray:
    mapped = np.full_like(pid, -1, dtype=np.int32)
    for k, v in pid_mapping.items():
        mapped[pid == k] = v
    return mapped


def preprocess_dataset(
    data: np.ndarray,
    norm: dict | None = None,
    pid_mapping: dict | None = None,
    add_pid_ohe: bool = True,
) -> np.ndarray:
    if norm is None:
        norm = NORM_CONSTANTS_DEFAULT
    normed = normalize_particles(data, norm)
    kine = normed[:, :, :4]

    if add_pid_ohe:
        if pid_mapping is None:
            pid_mapping = build_pid_mapping(data)
        pid = normed[:, :, 4].astype(np.int32)
        pid_idx = map_pid(pid, pid_mapping)
        n_classes = len(pid_mapping)
        pid_ohe = np.zeros((*pid_idx.shape, n_classes), dtype=np.float32)
        valid = pid_idx >= 0
        pid_ohe[valid] = np.eye(n_classes, dtype=np.float32)[pid_idx[valid]]
        return np.concatenate([kine, pid_ohe], axis=-1), pid_mapping

    return normed, pid_mapping
