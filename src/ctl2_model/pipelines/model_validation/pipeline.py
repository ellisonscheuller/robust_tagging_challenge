"""model_validation pipeline."""
from kedro.pipeline import node, Pipeline, pipeline
from .nodes import validate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_model,
            inputs=[
                "trained_model",
                "processed_X_train",
                "processed_y_train",
                "processed_X_val",
                "processed_y_val",
                "processed_X_eval",
                "processed_y_eval",
                "params:model_validation",
            ],
            outputs=[
                "val_metrics",
                "val_rate_curves",
                "val_pca_train",
                "val_pca_signals",
            ],
            name="validate_model_node",
        ),
    ])
