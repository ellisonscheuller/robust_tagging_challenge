"""compile pipeline."""
from kedro.pipeline import node, Pipeline, pipeline
from .nodes import compile_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compile_model,
            inputs=[
                "trained_model",
                "processed_X_train",
                "processed_y_train",
                "processed_X_val",
                "processed_y_val",
                "processed_X_eval",
                "processed_y_eval",
                "params:compile",
                "params:model_validation",
            ],
            outputs=[
                "synthesized_rtl_dir",
                "synthesized_hls_da4ml_dir",
                "synthesized_hls_hls4ml_dir",
                "synthesis_report",
                "compile_physics_metrics",
            ],
            name="compile_model_node",
        ),
    ])
