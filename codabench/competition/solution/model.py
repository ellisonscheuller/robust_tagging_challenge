import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.models import TransformerEncoder, Projector
from embedding.preprocs import PFPreProcessor
from embedding.loss import SupConLoss
from embedding.training import make_train_val_split, build_train_val_loaders
from embedding.utils.data_utils import load_data

NUM_BG_CLASSES = 4  # QCD, DY, TT, WJets
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# PASTE YOUR degrade FUNCTION HERE
# =====================================================================

def degrade(x: torch.Tensor) -> torch.Tensor:
    """Baseline: one random dead patch per event."""
    x = x.clone()
    B = x.size(0)
    eta_c = torch.empty(B, 1, device=x.device).uniform_(-2.0, 2.0)
    phi_c = torch.empty(B, 1, device=x.device).uniform_(-torch.pi, torch.pi)
    deta  = torch.empty(B, 1, device=x.device).uniform_(0.2, 1.5)
    dphi  = torch.empty(B, 1, device=x.device).uniform_(0.2, 1.5)
    eta, phi = x[..., 1], x[..., 2]
    dead = (
        (eta >= eta_c - deta / 2) & (eta < eta_c + deta / 2) &
        (phi >= phi_c - dphi / 2) & (phi < phi_c + dphi / 2) &
        (x[..., 0] > 0)
    )
    x[dead] = 0.0
    return x

# =====================================================================


class Model:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir

        self.preproc = PFPreProcessor(norm_constants={}).to(device)
        self.encoder = TransformerEncoder(
            num_features=self.preproc.num_features,
            embed_size=128,
            latent_dim=6,
            num_heads=8,
            num_layers=4,
        ).to(device)
        self.projector = Projector(6, 12, hidden_dim=48).to(device)
        self.classifier = nn.Linear(12, NUM_BG_CLASSES).to(device)

    @staticmethod
    def _cls_mask(x):
        m = x[..., 0] == 0
        return torch.cat([torch.zeros(m.size(0), 1, device=m.device, dtype=torch.bool), m], dim=1)

    def _embed(self, x):
        return F.normalize(self.projector(self.encoder(self.preproc(x), None, self._cls_mask(x))), dim=1)

    def fit(self):
        feature_block, label_block = load_data(
            os.path.join(self.data_dir, "REPLACE_ME"),  # train file
            map_location="cpu",
        )
        X_tr, y_tr, X_val, y_val, _, _ = make_train_val_split(feature_block, label_block, val_size=0.1)
        train_loader, val_loader = build_train_val_loaders(
            X_tr, y_tr, X_val, y_val, device=device, batch_size=256, pfcands=True
        )

        criterion  = SupConLoss(temperature=0.07)
        ce_loss_fn = nn.CrossEntropyLoss()
        optimizer  = torch.optim.Adam(
            list(self.preproc.parameters()) + list(self.encoder.parameters())
            + list(self.projector.parameters()) + list(self.classifier.parameters()),
            lr=1e-3,
        )

        for epoch in range(10):
            self.preproc.train(); self.encoder.train(); self.projector.train(); self.classifier.train()
            for x, _, labels in train_loader:
                x, labels = x.to(device), labels.to(device)
                x_aug = degrade(x)

                optimizer.zero_grad()
                emb_in  = self._embed(x)
                emb_aug = self._embed(x_aug)
                features = torch.stack([emb_in, emb_aug], dim=1)
                loss = criterion(features, labels) + ce_loss_fn(self.classifier(emb_in), labels)
                loss.backward()
                optimizer.step()

            self.preproc.eval(); self.encoder.eval(); self.projector.eval(); self.classifier.eval()
            val_loss = val_correct = 0
            with torch.no_grad():
                for x, _, labels in val_loader:
                    x, labels = x.to(device), labels.to(device)
                    emb    = self._embed(x)
                    logits = self.classifier(emb)
                    val_loss    += ce_loss_fn(logits, labels).item() * x.size(0)
                    val_correct += (logits.argmax(1) == labels).float().sum().item()
            N = len(val_loader.dataset)
            print(f"epoch {epoch+1}/10  val_loss={val_loss/N:.4f}  val_acc={val_correct/N:.4f}")

    @torch.no_grad()
    def _anomaly_scores(self, path):
        features = torch.load(path, map_location=device)
        self.preproc.eval(); self.encoder.eval(); self.projector.eval(); self.classifier.eval()
        emb      = self._embed(features)
        bg_probs = torch.softmax(self.classifier(emb), dim=1)
        anomaly  = 1.0 - bg_probs.max(dim=1).values
        return torch.stack([1.0 - anomaly, anomaly], dim=1).cpu().numpy()

    def predict(self):
        np.save(os.path.join(self.out_dir, "pred_nominal.npy"),
                self._anomaly_scores(os.path.join(self.data_dir, "REPLACE_ME")))  # eval nominal (private)

        for path in glob.glob(os.path.join(self.data_dir, "REPLACE_ME", "*.pt")):  # eval degraded (private)
            sev = re.search(r"\d+", os.path.basename(path)).group()
            np.save(os.path.join(self.out_dir, f"pred_severity_{sev}.npy"), self._anomaly_scores(path))
