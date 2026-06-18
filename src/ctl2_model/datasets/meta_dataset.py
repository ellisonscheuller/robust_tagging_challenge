"""Kedro dataset for meta data JSON files."""
import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class MetaDataset(AbstractDataset):
    def __init__(self, filepath: str, sample_key: str = "samples"):
        self._filepath = filepath
        self._sample_key = sample_key

    def _load(self) -> dict:
        with open(self._filepath) as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        Path(self._filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath, "sample_key": self._sample_key}
