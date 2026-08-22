"""
Challenge 9 evaluation script.

What participants get out of this:
  - tsne_latents.png        latent space colored by class (nominal data)
  - auc_vs_severity.png     robustness curve: a linear probe trained on
                             nominal latents, evaluated on degraded latents
                             at increasing eta-phi dropout severity

Everything else from earlier eval scripts (PCA, KDE, Mahalanobis anomaly
score, Fisher pairwise ROC) has been dropped — not needed for this challenge.

Data contract:
  --data          nominal event tensor, [E, N, F+1] with the label in the
                  last channel of each event's row 0 (same convention as
                  embedding.utils.data_utils.load_data/clean_data).
  --degraded_dir  directory of degraded versions of the SAME events (same
                  order, same labels), one file per severity level, each in
                  the same [E, N, F+1] format. Files are matched by the
                  first integer found in their name (e.g. severity_10.pt,
                  sev10.pt, ..._10pct.pt all parse as severity 10).

Run with --diagnostics to additionally produce
tsne_zero_fraction_diagnostic.png — an internal sanity check, not something
participants need. Candidates dropped by degradation are zeroed the same
way padding rows already are (pt == 0), so this checks whether the model is
organizing its latent space around "how much of this event is padding"
rather than genuine class/physics content.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from embedding.models import TransformerEncoder, EvalMLP
from embedding.dataloader import PFCandsDataset, PUPPIDataset
from embedding.utils.data_utils import load_data, delta_r_from_normalized
from embedding.utils.cfg_handler import train_config, data_config


def build_preproc_and_encoder(cfg: train_config, checkpoint: dict, device: str):
    import importlib

    preproc_type = cfg.get_trdata_cfg("preproc_type", "PFPreProcessor")
    preproc_class = getattr(importlib.import_module("embedding.preprocs"), preproc_type)
    norm_constants = checkpoint["norm_constants"]

    preproc = preproc_class(norm_constants).to(device)
    preproc.load_state_dict(checkpoint["preproc"])
    preproc.eval()

    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=cfg.hp("embed_size", 128),
        latent_dim=cfg.hp("latent_dim", 6),
        num_heads=cfg.hp("num_heads", 8),
        num_layers=cfg.hp("num_layers", 4),
        linear_dim=cfg.hp("linear_dim", None),
        num_tokens=None,
        pairwise=cfg.get_trdata_cfg("pairwise", False),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    return preproc, encoder, norm_constants


@torch.no_grad()
def embed_dataset(preproc, encoder, feature_block, label_block, cfg_data, norm_constants, device, batch_size=1024):
    """
    Runs an [E, N, F] tensor through preproc -> encoder and returns per-event
    latents, labels, and the fraction of candidate slots that are zeroed
    (pt == 0 — padding and/or dropped-by-degradation, indistinguishable by
    design; see module docstring).
    """
    pfcands = cfg_data.get("pfcands", True)
    dataset_cls = PFCandsDataset if pfcands else PUPPIDataset
    dataset = dataset_cls(feature_block, label_block, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pairwise = cfg_data.get("pairwise", False)
    latents, labels, zero_frac = [], [], []

    for x, mask, y in loader:
        x = x.to(device)
        mask = mask.to(device)
        zero_frac.append(mask.float().mean(dim=1).cpu())

        cls_mask = torch.cat(
            [torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool), mask.bool()], dim=1
        )
        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        latent = encoder(preproc(x), delta_r, cls_mask)
        latents.append(latent.cpu())
        labels.append(y)

    return torch.cat(latents), torch.cat(labels), torch.cat(zero_frac)


def train_linear_probe(X_train, y_train, num_classes, device, epochs=20):
    probe = EvalMLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=512, shuffle=True
    )
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


@torch.no_grad()
def probe_auc(probe, X, y, num_classes, device):
    probs = torch.softmax(probe(X.to(device)), dim=1).cpu().numpy()
    y = y.numpy()
    if num_classes == 2:
        return roc_auc_score(y, probs[:, 1])
    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


def parse_severity(path: str) -> int:
    match = re.search(r"\d+", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse a severity level out of filename: {path}")
    return int(match.group())


def plot_tsne_by_class(latents, labels, label_name_map, outdir):
    coords = TSNE(n_components=2, init="random", random_state=42, perplexity=50).fit_transform(latents.numpy())

    plt.figure(figsize=(8, 6))
    for label, name in sorted(label_name_map.items()):
        sel = labels.numpy() == label
        plt.scatter(coords[sel, 0], coords[sel, 1], s=5, alpha=0.3, label=name, color=plt.cm.tab10(label % 10))
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_latents.png"), dpi=300)
    plt.close()


def plot_auc_vs_severity(severities, aucs, outdir):
    plt.figure(figsize=(7, 5))
    plt.plot(severities, aucs, marker="o")
    plt.xlabel("Dropout severity (% of eta-phi plane)")
    plt.ylabel("Macro AUC (linear probe on latents)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_vs_severity.png"), dpi=300)
    plt.close()

    area = np.trapz(aucs, severities) / (severities[-1] - severities[0])
    print(f"Robustness summary: area under AUC-vs-severity curve = {area:.4f} (1.0 = perfectly flat at AUC 1.0)")


def plot_tsne_zero_fraction(latents, zero_frac, outdir):
    coords = TSNE(n_components=2, init="random", random_state=42, perplexity=50).fit_transform(latents.numpy())

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], s=5, c=zero_frac.numpy(), cmap="viridis", alpha=0.5)
    plt.colorbar(sc, label="fraction of candidates zeroed (padding + dropped)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Diagnostic: does the latent space track zero-fraction?")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_zero_fraction_diagnostic.png"), dpi=300)
    plt.close()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    cfg = train_config(args.train_cfg)
    cfg_data = data_config(args.data_cfg)
    label_name_map = cfg_data.get_label_name_map()
    num_classes = len(label_name_map)

    checkpoint = torch.load(args.encoder, map_location=device)
    preproc, encoder, norm_constants = build_preproc_and_encoder(cfg, checkpoint, device)

    nominal_features, nominal_labels = load_data(args.data, map_location="cpu")
    nominal_latents, nominal_labels, nominal_zero_frac = embed_dataset(
        preproc, encoder, nominal_features, nominal_labels, cfg_data, norm_constants, device
    )

    plot_tsne_by_class(nominal_latents, nominal_labels, label_name_map, args.outdir)

    X_train, X_test, y_train, y_test = train_test_split(
        nominal_latents, nominal_labels, stratify=nominal_labels, test_size=0.2, random_state=42
    )
    probe = train_linear_probe(X_train, y_train, num_classes, device)

    severity_files = sorted(glob.glob(os.path.join(args.degraded_dir, "*.pt")), key=parse_severity)
    if not severity_files:
        raise FileNotFoundError(f"No .pt files found in --degraded_dir: {args.degraded_dir}")

    severities = [0]
    aucs = [probe_auc(probe, X_test, y_test, num_classes, device)]
    diag_latents = [nominal_latents]
    diag_zero_frac = [nominal_zero_frac]
    for path in severity_files:
        severity = parse_severity(path)
        deg_features, deg_labels = load_data(path, map_location="cpu")
        deg_latents, deg_labels, deg_zero_frac = embed_dataset(
            preproc, encoder, deg_features, deg_labels, cfg_data, norm_constants, device
        )
        severities.append(severity)
        aucs.append(probe_auc(probe, deg_latents, deg_labels, num_classes, device))
        diag_latents.append(deg_latents)
        diag_zero_frac.append(deg_zero_frac)

    plot_auc_vs_severity(severities, aucs, args.outdir)

    if args.diagnostics:
        plot_tsne_zero_fraction(torch.cat(diag_latents), torch.cat(diag_zero_frac), args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg", required=True, help="Same train config .yaml used to train the encoder")
    parser.add_argument("--data_cfg", required=True, help="Same data config .yaml used to train the encoder")
    parser.add_argument("--encoder", required=True, help="Path to a checkpoint saved by train.py")
    parser.add_argument("--data", required=True, help="Nominal (non-degraded) eval .pt file")
    parser.add_argument("--degraded_dir", required=True, help="Directory of per-severity degraded eval .pt files")
    parser.add_argument("--outdir", default="./evalPlots")
    parser.add_argument("--diagnostics", action="store_true", help="Also produce the zero-fraction latent-space diagnostic (organizer use, not needed by participants)")
    main(parser.parse_args())
