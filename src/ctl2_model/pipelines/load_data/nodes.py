"""load_data pipeline nodes.

Data loading uses the ctl2_modelLoader Kedro dataset which has
the transform() function — no additional node processing needed.
"""
import pandas as pd


def load_data(data: pd.DataFrame, meta_data: dict) -> tuple:
    return data, meta_data
