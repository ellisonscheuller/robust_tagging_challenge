import logging, json
from abc import abstractmethod
from kedro.io import AbstractDataset
from trigger_loader.loader import TriggerLoader
import pandas as pd
import numpy as np
from pathlib import Path


class BaseLoader(AbstractDataset):
    """
    Abstract Base Class for using the TriggerLoader.

    Users must inherit from this class and implement the abstract methods.
    The core processing logic in `_load` is fixed and cannot be overridden.
    """

    def __init__(self, sample_json: str, settings: str, config: str):
        self.sample_json = sample_json
        with open(settings, "r") as f:
            self.settings = json.load(f)
        with open(config, "r") as f:
            self.config = json.load(f)

        # get logger for reporting
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing loader: {self.__class__.__name__}")

    @abstractmethod
    def transform(self, events):
        """
        USER MUST IMPLEMENT.
        """
        pass

    def _load(self) -> pd.DataFrame:
        with open(self.sample_json) as f:
            raw_samples = json.load(f).get("samples", {})
        sample_y = {name: int(info.get("y", 0)) for name, info in raw_samples.items()}

        manifest_path = Path(self.settings["output_dir"]) / "manifest.json"
        dataset_keys = set(raw_samples.keys())

        # Read whatever is already in the manifest
        existing_records = {key: [] for key in dataset_keys}
        if manifest_path.exists():
            with manifest_path.open() as f:
                for line in f:
                    record = json.loads(line.strip())
                    if not line.strip():
                        continue
                    dataset = record.get("dataset")
                    parquet_file = Path(record.get("parquet_file", ""))
                    if dataset in existing_records and parquet_file.exists():
                        existing_records[dataset].append(record)

        missing_datasets = [k for k, v in existing_records.items() if not v]

        if missing_datasets:
            self.logger.info(f"Missing datasets, running TriggerLoader: {missing_datasets}")
            loader = TriggerLoader(
                sample_json=self.sample_json,
                transform=self.transform,
                output_path=self.settings["output_dir"]
            )
            # Only run for missing datasets — filter fileset
            loader.fileset = {k: v for k, v in loader.fileset.items() if k in missing_datasets}
            loader.meta_data = {k: v for k, v in loader.meta_data.items() if k in missing_datasets}

            if self.settings["run_local"]:
                loader.run_local(
                    num_workers=self.settings["num_workers"],
                    chunksize=self.settings["chunksize"]
                )
            else:
                loader.run_distributed(
                    cluster_type=self.settings["cluster_type"],
                    cluster_config=self.config,
                    chunksize=self.settings["chunksize"],
                    jobs=self.settings["jobs"]
                )

            # Re-read manifest to pick up newly written entries
            with manifest_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    dataset = record.get("dataset")
                    parquet_file = Path(record.get("parquet_file", ""))
                    if dataset in existing_records and parquet_file.exists() and record not in existing_records[dataset]:
                        existing_records[dataset].append(record)
        else:
            self.logger.info("All datasets found in manifest, skipping TriggerLoader run.")
            loader = TriggerLoader(  # still need meta_data for is_signal
                sample_json=self.sample_json,
                transform=self.transform,
                output_path=self.settings["output_dir"]
            )

        # Build final dataframe
        missing = [k for k, v in existing_records.items() if not v]
        if missing:
            raise ValueError(f"No manifest entries found after processing for: {missing}")

        final_dfs = []
        for dataset_key, records in existing_records.items():
            y_val = sample_y.get(dataset_key, 0)
            is_sig = bool(raw_samples[dataset_key].get("is_signal", y_val != 0))

            df = pd.concat(
                [pd.read_parquet(r["parquet_file"]) for r in records],
                ignore_index=True
            )
            df["y"] = np.full(len(df), y_val, dtype=int)
            df["is_signal"] = np.full(len(df), int(is_sig), dtype=int)
            df["sample_name"] = dataset_key
            final_dfs.append(df)

        return pd.concat(final_dfs, ignore_index=True)

    def _save(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _describe(self) -> dict:
        return {"sample_json": self.sample_json, "settings": self.settings, "config": self.config}
