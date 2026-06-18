import inspect
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
from sklearn.base import BaseEstimator


class BaseModel(ABC, BaseEstimator):
    """
    Standard Wrapper for a model
    """

    def __init__(self, name: str, hps: dict):
        self.name = name
        # this will be overwritten after training
        self.model = None
        self.history = None
        self.callbacks = []
        self.hps = hps

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, hps: dict, **kwargs):
        """
        User code function.
        Args:
            X: features
            y: label
            hps: hyperparameters
            kwargs: anything else needed for training
        """
        pass

    @abstractmethod
    def build(self):
        """
        User code function to build the model.
        """
        pass

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates predictions of the model
        Args:
            X: features

        Returns:
            predictions
            (optional in user code) kwargs: anything else needed for predicting
        """
        y_pred = self.model.predict(X.astype("float32"))
        return pd.DataFrame(y_pred)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates proba predictions of the model
        Args:
            X: features

        Returns:
            predictions
            (optional in user code) kwargs: anything else needed for predicting
        """
        y_pred = self.model.predict_proba(X.astype("float32"))
        return pd.DataFrame(y_pred)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Same as train but get kwargs from __init__ for sklearn API
        Args:
            X: features
            y: label

        X can also contain optional inputs https://github.com/scikit-learn/scikit-learn/issues/2879.
        Which should be specified in the user code.
        For example when the train function needs additional inputs:
        ```python
            curX = X.copy()
            kwargs = {"S": curX["S"]}
            del curX["S"]
            self.train(curX, y, self.hps, **kwargs)
        ```
        """
        self.train(X, y, self.hps)

    def get_params(self, deep=True):
        """
        Get parameters for self.model and self.
        Args:
            deep : bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            params : dict
                Parameter names mapped to their values.
        """
        out = dict()
        # if self.hps is set return them and not the default values
        for key in self.hps:
            out[key] = self.hps[key]
        for key in get_param_names(self):
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        We overwrite the sklearn BaseEstimator and set params to self.hps
        Args:
            **params : dict
                Estimator parameters.

        Returns:
            self : estimator instance
                Estimator instance.
        """
        self.hps = params

        return self


def get_param_names(cls):
    """Get parameter names for the estimator"""
    # fetch the constructor or the original constructor before
    # deprecation wrapping if any
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    if init is object.__init__:
        # No explicit constructor to introspect
        return []

    # introspect the constructor arguments to find the model parameters
    # to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    parameters = [
        p
        for p in init_signature.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD
    ]
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError(
                "scikit-learn estimators should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s with constructor %s doesn't "
                " follow this convention." % (cls, init_signature)
            )
    # Extract and sort argument names excluding 'self'
    return sorted([p.name for p in parameters])
