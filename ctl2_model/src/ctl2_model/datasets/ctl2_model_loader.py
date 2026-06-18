import awkward as ak
import numpy as np
from .base_loader import BaseLoader


class ctl2_modelLoader(BaseLoader):

    def transform(self, events):
        try:
            puppi = events.L1PuppiCands
            pt = puppi.pt
            eta = puppi.eta
            phi = puppi.phi
            dxy = puppi.dxy
            pdgId = puppi.pdgId
        except (AttributeError, ValueError):
            return None

        # No standalone muon collection — skip muon merging
        N = 100
        pt_all = ak.fill_none(ak.pad_none(pt, N, axis=1, clip=True), 0.0)
        eta_all = ak.fill_none(ak.pad_none(eta, N, axis=1, clip=True), 0.0)
        phi_all = ak.fill_none(ak.pad_none(phi, N, axis=1, clip=True), 0.0)
        dxy_all = ak.fill_none(ak.pad_none(dxy, N, axis=1, clip=True), 0.0)
        pdgId_all = ak.fill_none(ak.pad_none(pdgId, N, axis=1, clip=True), 0)

        return ak.zip({
            "puppi_pt": ak.to_regular(pt_all),
            "puppi_eta": ak.to_regular(eta_all),
            "puppi_phi": ak.to_regular(phi_all),
            "puppi_dxy": ak.to_regular(dxy_all),
            "puppi_pdgId": ak.to_regular(pdgId_all),
        })
