import math
import torch
import torch.nn as nn

class Degradation(nn.Module):
    """Simulates detector dead zones by randomly dropping particles whose
    (eta, phi) falls in one of a handful of randomly chosen 'dead' bins.

    It just zeroes the entire feature row of any particle it drops.

    This layer has to come before the PreProcessor layer.

    Each event in the batch gets its own independently sampled set of dead
    bins. However, note that num_miss and p_miss are resampled once per forward call (i.e. once
    per batch) from their respective distributions.
    """

    def __init__(
        self,
        n_eta_bins: int = 10,
        n_phi_bins: int = 10,
        max_num_miss: int = 5,
        p_miss_mean: float = 0.5,
        p_miss_std: float = 0.1,
        eta_range: tuple = (-4.0, 4.0),
        phi_range: tuple = (-math.pi, math.pi),
    ):
        super().__init__()
        self.n_eta_bins = n_eta_bins
        self.n_phi_bins = n_phi_bins
        self.max_num_miss = max_num_miss
        self.p_miss_mean = p_miss_mean
        self.p_miss_std = p_miss_std
        eta_edges = torch.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)
        phi_edges = torch.linspace(phi_range[0], phi_range[1], n_phi_bins + 1)
        self.register_buffer("eta_edges", eta_edges)
        self.register_buffer("phi_edges", phi_edges)

    def _sample_dead_zone_prob_grid(self, B: int, device: torch.device) -> torch.Tensor:
        """Builds a histogram of detector defaults PER EVENT in the batch.

        num_miss is the number of bins that have a detector default.
        
        p_miss is the probability of a particle being dropped if it falls 
        inside a bin that has a detector default.
         
        Note that num_miss and p_miss is sampled once whole batch call,
        (so they are the same for the whole batch). The distributions they
        are sampled from are:

        num_miss ~ Uniform(0, max_num_miss), rounded to an int
        p_miss ~ Normal(p_miss_mean, p_miss_std), clamped to [0, 1]

        The function creates histograms with 0s everywhere except in the randomly
        chosen faulty bins. In the bad bins, the histograms have value p_miss.
        
        Returns tensor of size [B, n_eta_bins * n_phi_bins].
        """
        n_bins = self.n_eta_bins * self.n_phi_bins

        num_miss = int(torch.empty(1).uniform_(0, self.max_num_miss).round().item())
        num_miss = min(max(num_miss, 0), n_bins)
        p_miss = torch.empty(1).normal_(self.p_miss_mean, self.p_miss_std)
        p_miss = p_miss.clamp(0.0, 1.0).item()

        grid = torch.zeros(B, n_bins, device=device)
        if num_miss > 0:
            noise = torch.rand(B, n_bins, device=device)
            bad_idx = noise.argsort(dim=-1)[:, :num_miss]  # [B, num_miss]
            grid.scatter_(1, bad_idx, p_miss)
        return grid

    def _dead_zone_drop_mask(self, eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """For every particle, look up its (eta, phi) bin's miss-probability
        (own event's grid) and Bernoulli-sample whether it gets dropped.
        Returns bool [B, N].
        """
        B = eta.shape[0]
        eta_bin = torch.bucketize(eta, self.eta_edges, right=True) - 1
        eta_bin = eta_bin.clamp(0, self.n_eta_bins - 1)
        phi_bin = torch.bucketize(phi, self.phi_edges, right=True) - 1
        phi_bin = phi_bin.clamp(0, self.n_phi_bins - 1)

        flat_idx = eta_bin * self.n_phi_bins + phi_bin  # [B, N]
        prob_grid = self._sample_dead_zone_prob_grid(B, eta.device)  # [B, n_bins]
        miss_prob = torch.gather(prob_grid, 1, flat_idx)  # [B, N]
        return torch.bernoulli(miss_prob).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape [B, N, F] with features [pt, eta, phi, dxy, dxysig, is_pf, pdgId, ...]
        Returns x with the same shape but rows dropped by the dead-zone mask
        are zeroed out.
        """
        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]

        valid = pt_raw > 0  # [B, N] -- real particles vs. existing padding

        if valid.any():
            drop = self._dead_zone_drop_mask(eta_raw, phi_raw)
            drop = drop & valid  # only ever drop rows that were real to begin with

            if drop.any():
                x = x.clone()
                x[drop] = 0.0

        return x