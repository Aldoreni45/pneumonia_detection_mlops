from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path:Path
    updated_base_model_path:Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_freeze_all: bool
    params_freeze_till: int | None

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    production_model_path: Path
    training_data: Path
    test_data: Path
    val_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    test_data: Path
    val_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
