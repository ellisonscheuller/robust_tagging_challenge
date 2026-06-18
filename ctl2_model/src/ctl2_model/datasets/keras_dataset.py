"""Kedro dataset for Keras .keras model files."""
from typing import Any

import keras
from kedro.io import AbstractDataset


class KerasDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> keras.Model:
        keras.config.enable_unsafe_deserialization()
        return keras.models.load_model(self._filepath)

    def _save(self, model: keras.Model) -> None:
        model.save(self._filepath)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath}