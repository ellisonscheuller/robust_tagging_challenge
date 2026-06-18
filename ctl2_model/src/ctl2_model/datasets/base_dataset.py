import logging, uproot, json, os
import pandas as pd
import numpy as np
from abc import abstractmethod
from fnmatch import filter as fnmatch_filter
from kedro.io import AbstractDataset


class BaseDataset(AbstractDataset):
    """
    Abstract Base Class for loading data from ROOT files.

    Users must inherit from this class and implement the abstract methods.
    The core processing logic in `_load` is fixed and cannot be overridden.
    """

    def __init__(self, sample_info: str, sample_key: str):
        with open(sample_info, "r") as f:
            data = json.load(f)
        self._sample_info = data[sample_key]
        self._sample_key = sample_key

        # get logger for reporting
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing dataset: {self.__class__.__name__}")

    @abstractmethod
    def get_branches_to_keep(self) -> list[str]:
        """
        USER MUST IMPLEMENT: Return a list of branch names or patterns (with wildcards)
        to keep from the ROOT file.

        Example:
            return ["Jet_*", "PuppiMET_pt", "nJet"]
        """
        pass

    @abstractmethod
    def get_cut(self) -> str | None:
        """
        USER MUST IMPLEMENT: Return a string representing the cuts to apply to the data.
        """
        pass

    @abstractmethod
    def convert_to_pandas(self, data: dict) -> pd.DataFrame:
        """
        USER MUST IMPLEMENT: Convert the loaded data from a dictionary format to a pandas DataFrame.
        """
        pass

    def get_tree_name(self) -> str:
        return "Events"

    def _resolve_branches(self, all_branches: list) -> list[str]:
        """Internal method to resolve wildcard patterns."""
        selected = []
        for pattern in self.get_branches_to_keep():
            matched = fnmatch_filter(all_branches, pattern)
            if not matched:
                self.logger.warning(f"Pattern '{pattern}' did not match any branches.")
            selected.extend(matched)
        return sorted(list(set(selected)))

    def _load(self) -> pd.DataFrame:
        """
        CORE LOGIC (NOT OVERRIDABLE): Loads and processes a single ROOT file.
        """

        # Process all files in sample
        df = pd.DataFrame()

        all_root_files = []
        for key in self._sample_info.keys():
            files = os.listdir(self._sample_info[key]["folder"])
            cur_files = []
            for file_pattern in self._sample_info[key]["file_pattern"]:
                for f in fnmatch_filter(files, file_pattern):
                    cur_files.append(os.path.join(self._sample_info[key]["folder"], f))
            all_root_files.append(cur_files)

        is_signals = [
            self._sample_info[key]["is_signal"] for key in self._sample_info.keys()
        ]
        self.logger.info("Processing files")
        for root_files, is_signal in zip(all_root_files, is_signals):
            self.logger.info(f"Processing files: {root_files}")
            for root_file in root_files:
                if f"{root_file}" == "data/01_raw/samples_dummy.json":
                    n = 100
                    # generate dummy features
                    dummy_data = {}
                    for branch in self.get_branches_to_keep():
                        dummy_data[branch] = np.random.randn(n)
                    if is_signal:
                        dummy_data["is_signal"] = np.ones(n)
                    else:
                        dummy_data["is_signal"] = np.zeros(n)

                    cur_df = pd.DataFrame(dummy_data)

                    # generate a binary target (0/1)
                    cur_df["y"] = np.random.choice([0, 1], size=n)

                    df = pd.concat([df, cur_df])

                else:
                    with uproot.open(f"{root_file}") as f:
                        tree = f[self.get_tree_name()]
                        all_branches = tree.keys()
                        branches_to_load = self._resolve_branches(all_branches)

                        if not branches_to_load:
                            self.logger.warning(
                                f"No valid branches to load for {root_file}. Skipping."
                            )
                            continue

                        data = tree.arrays(branches_to_load, cut=self.get_cut())

                        cur_df = self.convert_to_pandas(data)

                        # set background or signal
                        if is_signal:
                            cur_df["is_signal"] = [1 for _ in range(len(cur_df))]
                        else:
                            cur_df["is_signal"] = [0 for _ in range(len(cur_df))]

                        df = pd.concat([df, cur_df])

        return df

    def _save(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _describe(self) -> dict:
        return {"output_sample_info": self._sample_info, "sample_key": self._sample_key}
