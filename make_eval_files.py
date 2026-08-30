import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn


# Same grid as Degradation in embedding/degradation.py
N_ETA_BINS = 10
N_PHI_BINS = 10
ETA_EDGES = torch.linspace(-4.0, 4.0, N_ETA_BINS + 1)
PHI_EDGES = torch.linspace(-math.pi, math.pi, N_PHI_BINS + 1)


class FixedDegradation(nn.Module):
    """Kills a fixed set of eta-phi grid bins — same grid as Degradation, no randomness."""

    def __init__(self, dead_bins: list[tuple[int, int]]):
        super().__init__()
        regions = torch.tensor([
            [ETA_EDGES[i], ETA_EDGES[i + 1], PHI_EDGES[j], PHI_EDGES[j + 1]]
            for i, j in dead_bins
        ])  # [K, 4]
        self.register_buffer("regions", regions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta   = x[..., 1].unsqueeze(-1)   # [B, N, 1]
        phi   = x[..., 2].unsqueeze(-1)   # [B, N, 1]
        r     = self.regions               # [K, 4]
        in_bin = (eta >= r[:, 0]) & (eta < r[:, 1]) & (phi >= r[:, 2]) & (phi < r[:, 3])
        dead   = in_bin.any(-1) & (x[..., 0] > 0)
        if dead.any():
            x = x.clone()
            x[dead] = 0.0
        return x


# Severity N% = N bins dead out of 100 (10x10 grid).
# Bins are drawn from a fixed random permutation (seed=42) so the pattern is
# scattered across the detector rather than one big block — more realistic and
# harder to overfit to. Cumulative: severity 10% is severity 5% + 5 more bins.
_ALL_BINS = [(i, j) for i in range(N_ETA_BINS) for j in range(N_PHI_BINS)]
_rng  = torch.Generator().manual_seed(42)
_PERM = torch.randperm(len(_ALL_BINS), generator=_rng).tolist()

SEVERITIES: dict[int, FixedDegradation] = {
    sev: FixedDegradation([_ALL_BINS[k] for k in _PERM[:sev]])
    for sev in range(5, 55, 5)
}


def main(args):
    data = torch.load(args.data, map_location="cpu")  # [E, N, F+1], label in last col
    features  = data[..., :-1]   # [E, N, F] — degrade these only
    label_col = data[..., -1:]   # [E, N, 1] — preserved unchanged

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_orig = (features[..., 0] > 0).sum().item()

    for sev, deg in sorted(SEVERITIES.items()):
        deg_features = deg(features)
        torch.save(torch.cat([deg_features, label_col], dim=-1), out_dir / f"severity_{sev}.pt")

        killed = valid_orig - (deg_features[..., 0] > 0).sum().item()
        print(f"severity {sev:3d}%  {killed} candidates killed ({100*killed/valid_orig:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True,             help="Nominal labeled .pt file [E, N, F+1]")
    parser.add_argument("--out_dir", default="./eval_degraded", help="Output directory")
    main(parser.parse_args())
