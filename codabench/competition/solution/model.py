import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.models import TransformerEncoder, Projector
from embedding.preprocs import PFPreProcessor
from embedding.loss import InfoNCELoss
from embedding.training import make_train_val_split, build_train_val_loaders, train_epoch, validate_epoch
from embedding.utils.data_utils import load_data

NUM_CLASSES = 5  # must match the number of classes in the challenge data

# The user submission needs to implement the interface described in the challenge:
# Model(data_dir, out_dir), then .fit() then .predict(). This is the baseline.
class Model:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.preproc = PFPreProcessor(norm_constants={}).to(self.device)
        self.encoder = TransformerEncoder(
            num_features=self.preproc.num_features,
            embed_size=128,
            latent_dim=6,
            num_heads=8,
            num_layers=4,
        ).to(self.device)
        self.projector = Projector(6, 12, hidden_dim=48).to(self.device)
        self.classifier = nn.Linear(12, NUM_CLASSES).to(self.device)

    def fit(self):
        feature_block, label_block = load_data(os.path.join(self.data_dir, "train.pt"), map_location="cpu")

        X_tr, y_tr, X_val, y_val, idx_tr, idx_val = make_train_val_split(feature_block, label_block, val_size=0.1)

        train_loader, val_loader = build_train_val_loaders(
            X_tr, y_tr, X_val, y_val, device=self.device, batch_size=256, pfcands=True
        )

        criterion = InfoNCELoss(temperature=0.07)
        ce_loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.preproc.parameters()) + list(self.encoder.parameters())
            + list(self.projector.parameters()) + list(self.classifier.parameters()),
            lr=1e-3,
        )

        for epoch in range(10):
            tr = train_epoch(
                self.encoder, self.projector, self.classifier, ce_loss_fn, criterion,
                train_loader, {}, self.device, optimizer, self.preproc,
                contrastive_weight=0.05, num_classes=NUM_CLASSES,
            )
            va = validate_epoch(
                self.encoder, self.projector, self.classifier, ce_loss_fn, criterion,
                val_loader, {}, self.device, self.preproc,
                contrastive_weight=0.05, num_classes=NUM_CLASSES,
            )
            print(f"epoch {epoch}: train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")

    @torch.no_grad()
    def _predict_one(self, path):
        features = torch.load(path, map_location="cpu")
        mask = features[..., 0] == 0
        cls_mask = torch.cat([torch.zeros(mask.size(0), 1, dtype=torch.bool), mask], dim=1).to(self.device)
        latent = self.encoder(self.preproc(features.to(self.device)), None, cls_mask)
        embedding = F.normalize(self.projector(latent), dim=1)
        return torch.softmax(self.classifier(embedding), dim=1).cpu().numpy()

    def predict(self):
        self.preproc.eval(); self.encoder.eval(); self.projector.eval(); self.classifier.eval()

        preds = self._predict_one(os.path.join(self.data_dir, "eval_nominal.pt"))
        np.save(os.path.join(self.out_dir, "pred_nominal.npy"), preds)

        for path in glob.glob(os.path.join(self.data_dir, "eval_degraded", "*.pt")):
            severity = re.search(r"\d+", os.path.basename(path)).group()
            preds = self._predict_one(path)
            np.save(os.path.join(self.out_dir, f"pred_severity_{severity}.npy"), preds)
