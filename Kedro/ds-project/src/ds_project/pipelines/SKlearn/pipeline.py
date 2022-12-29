"""
This is a boilerplate pipeline 'SKlearn'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, build_and_train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
    node(load_data, inputs=["train_dir", "val_dir"], outputs=["X_train", "y_train", "X_val", "y_val"]),
    node(build_and_train_model, inputs=["X_train", "y_train"], outputs="model"),
    node(evaluate_model, inputs=["model", "X_val", "y_val"], outputs="accuracy")
    ])
