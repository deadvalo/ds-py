"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from ds_project.pipelines import Pytorch as py
from ds_project.pipelines import Keras as ke
from ds_project.pipelines import SKlearn as sk


def register_pipelines() -> Dict[str, Pipeline]:
    pytorch_pipeline = py.create_pipeline()
    keras_pipeline = ke.create_pipeline()
    sklearn_pipeline = sk.create_pipeline()

    return {
        "default": pytorch_pipeline + keras_pipeline + sklearn_pipeline,
        "py": pytorch_pipeline,
        "ke": keras_pipeline,
        "sk": sklearn_pipeline
    }
    


