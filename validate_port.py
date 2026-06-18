"""Validate ported data_processing — test extraction + preprocessing format."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from ctl2_model.data.preprocessing import (
    preprocess_dataset, compute_normalization_constants, build_pid_mapping,
)

# Load cached parquet
print("Reading cached parquet...")
df = pd.read_parquet("data/02_loaded/ctl2_model_data.parquet")
print(f"Shape: {df.shape}")

# The data column is the first column (empty string name) containing numpy struct arrays
raw_col = df.columns[0]
raw = df[raw_col].values

# Extract fields from the structured arrays
pt = np.array([a["puppi_pt"] for a in raw], dtype=np.float32)
eta = np.array([a["puppi_eta"] for a in raw], dtype=np.float32)
phi = np.array([a["puppi_phi"] for a in raw], dtype=np.float32)
dxy = np.array([a["puppi_dxy"] for a in raw], dtype=np.float32)
pdgId = np.array([a["puppi_pdgId"] for a in raw], dtype=np.float32)

print(f"pt: {pt.shape}, pt range: [{pt.min():.4f}, {pt.max():.4f}]")
print(f"eta: {eta.shape}, eta range: [{eta.min():.4f}, {eta.max():.4f}]")
print(f"phi: {phi.shape}, phi range: [{phi.min():.4f}, {phi.max():.4f}]")
print(f"dxy: {dxy.shape}, dxy range: [{dxy.min():.4f}, {dxy.max():.4f}]")
print(f"pdgId: {pdgId.shape}, unique pdgId: {np.unique(pdgId).tolist()[:20]}")

# Stack into raw tensor (N, 100, 5): [pt, eta, phi, dxy, pdgId]
raw_tensor = np.stack([pt, eta, phi, dxy, pdgId], axis=-1)
print(f"\nraw_tensor: {raw_tensor.shape}, dtype={raw_tensor.dtype}")

# Sort by pt (descending) — like source pipeline
pt_sorted_idx = np.argsort(-raw_tensor[:, :, 0], axis=1)
batch_idx = np.arange(raw_tensor.shape[0])[:, None]
raw_tensor = raw_tensor[batch_idx, pt_sorted_idx]
raw_tensor = np.nan_to_num(raw_tensor, nan=0.0)
print("After sort_by_pt + nan_to_num")

# Compute normalization constants
norm = compute_normalization_constants(raw_tensor)
print(f"\nNormalization constants:")
print(f"  pt: [{norm['pt_min']:.4f}, {norm['pt_max']:.4f}]")
print(f"  eta: [{norm['eta_min']:.4f}, {norm['eta_max']:.4f}]")
print(f"  phi: [{norm['phi_min']:.4f}, {norm['phi_max']:.4f}]")
print(f"  dxy: [{norm['dxy_min']:.4f}, {norm['dxy_max']:.4f}]")

# Build PID mapping
pid_mapping = build_pid_mapping(raw_tensor)
print(f"pid_mapping: {pid_mapping}")

# Preprocess (normalize + PID OHE)
X_all, pid_out = preprocess_dataset(raw_tensor, norm=norm, pid_mapping=pid_mapping, add_pid_ohe=True)
print(f"\nPreprocessed X_all: {X_all.shape}, dtype={X_all.dtype}")
print(f"  PID OHE classes: {X_all.shape[-1] - 4}")

# Check kinematic features
print(f"  pt_norm range: [{X_all[:,:,0].min():.4f}, {X_all[:,:,0].max():.4f}]")
print(f"  eta_norm range: [{X_all[:,:,1].min():.4f}, {X_all[:,:,1].max():.4f}]")
print(f"  phi_norm range: [{X_all[:,:,2].min():.4f}, {X_all[:,:,2].max():.4f}]")
print(f"  dxy (raw) range: [{X_all[:,:,3].min():.4f}, {X_all[:,:,3].max():.4f}]")

# Check padding
pad_mask = raw_tensor[:, :, 0] == 0
print(f"  Padding fraction: {pad_mask.sum() / pad_mask.size:.4f}")
print(f"  Padded values zero (pt): {np.all(X_all[:,:,0][pad_mask] == 0.0)}")
print(f"  Padded values zero (eta): {np.all(X_all[:,:,1][pad_mask] == 0.0)}")
print(f"  Padded values zero (phi): {np.all(X_all[:,:,2][pad_mask] == 0.0)}")

# Check PID OHE
pid_ohe = X_all[:, :, 4:]
ohe_sum = pid_ohe.sum(axis=-1)
print(f"  PID OHE sum: range=[{ohe_sum.min()}, {ohe_sum.max()}]")

# Check for NaN/inf
assert not np.any(np.isnan(X_all)), "NaN in X_all!"
assert not np.any(np.isinf(X_all)), "Inf in X_all!"
assert X_all.dtype == np.float32, f"Expected float32, got {X_all.dtype}"
assert X_all.shape[1] == 100, f"Expected 100 constituents, got {X_all.shape[1]}"
assert X_all.shape[-1] == 4 + len(pid_mapping), f"Expected 4+{len(pid_mapping)} features, got {X_all.shape[-1]}"

print("\n✓ ALL FORMAT CHECKS PASSED!")
print(f"\nOutput: (N={X_all.shape[0]}, constituents={X_all.shape[1]}, features={X_all.shape[2]})")
