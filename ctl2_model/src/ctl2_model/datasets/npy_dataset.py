from pathlib import Path
from typing import Any
import numpy as np
from kedro.io import AbstractDataset


class NumpyDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> np.ndarray:
        return np.load(self._filepath)

    def _save(self, data: np.ndarray) -> None:
        Path(self._filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(self._filepath, data)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath}
