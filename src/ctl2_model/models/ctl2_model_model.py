import pandas as pd
from .base_model import BaseModel
from sklearn.dummy import DummyClassifier


class ctl2_model(BaseModel):
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self.build()
        self.history = self.model.fit(X, y)

    def build(self):
        """Build the test Model.
        self.hps:
            - 
        """
        self.model = DummyClassifier()
