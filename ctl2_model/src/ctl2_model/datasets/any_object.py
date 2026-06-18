from kedro.io import AbstractDataset
from typing import Any


class AnyObject(AbstractDataset):
    """
    Abstract class which can be used for passing "Any" object
    """

    def __init__(self):
        pass

    def _load(self) -> None:
        pass

    def _save(self, data: Any) -> Any:
        return data

    def _describe(self) -> dict:
        return {}
