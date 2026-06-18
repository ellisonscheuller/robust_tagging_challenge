"""model_training pipeline."""
from kedro.pipeline import node, Pipeline, pipeline
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=[
                "processed_X_train",
                "processed_X_train_aug",
                "processed_y_train",
                "processed_X_val",
                "processed_y_val",
                "params:model_training",
            ],
            outputs=[
                "trained_model",
                "training_history",
            ],
            name="train_model_node",
        ),
    ])
