"""
This is a boilerplate pipeline 'Pytorch'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_dataloaders, Net, train_model, evaluate_model, display_image

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                create_dataloaders,
                inputs=[
                    "image_size",
                    "batch_size",
                    "data_dir",
                    "train_dir",
                    "val_dir",
                ],
                outputs=[
                    "trainloader",
                    "valloader",
                ],
            ),
            node(
                Net,
                inputs=[],
                outputs="net",
            ),
            node(
                train_model,
                inputs=[
                    "net",
                    "device",
                    "trainloader",
                    "num_epochs",
                ],
                outputs=[],
            ),
            node(
                evaluate_model,
                inputs=[
                    "net",
                    "device",
                    "valloader",
                ],
                outputs=[],
            ),
            node(
                display_image,
                inputs=[
                    "trainloader",
                    "net",
                    "device",
                ],
                outputs=[],
            ),
        ]
    )
