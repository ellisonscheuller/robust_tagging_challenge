"""load_data pipeline — uses ctl2_modelLoader (TriggerFlow transform pattern)."""
from kedro.pipeline import node, Pipeline, pipeline
from .nodes import load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_data,
            inputs=["ctl2_model_loader", "ctl2_model_meta_data"],
            outputs=["ctl2_model_data_loaded", "ctl2_model_meta_data_loaded"],
            name="load_data_node",
        ),
    ])
