import pandas as pd
import awkward as ak
import numpy as np
from .base_dataset import BaseDataset


class ctl2_modelDataset(BaseDataset):

    def get_branches_to_keep(self) -> list[str]:
        return [
            "L1PuppiCands_pt",
            "L1PuppiCands_eta",
            "L1PuppiCands_phi",
            "L1PuppiCands_dxy",
            "L1PuppiCands_pdgId",
            "StaMu_pt",
            "StaMu_eta",
            "StaMu_phi",
        ]

    def get_cut(self) -> str | None:
        return None

    def convert_to_pandas(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {k: ak.to_list(v) for k, v in data.items()}
        )
