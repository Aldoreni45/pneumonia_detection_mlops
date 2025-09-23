import os
from pathlib import Path
from pneumonia_detection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pneumonia_detection.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig,
                                                     TrainingConfig, EvaluationConfig)  
from pneumonia_detection.utils.common import read_yaml, create_directories 


class ConfigurationManager:
    def __init__(self,cpath: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(cpath)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config =self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES,
            params_freeze_all=params.FREEZE_ALL,
            params_freeze_till=params.FREEZE_TILL
        )
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params
        create_directories([config.root_dir])
        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            production_model_path=Path(config.production_model_path),
            training_data=Path(config.training_data),
            test_data=Path(config.test_data),
            val_data=Path(config.val_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        params = self.params
        evaluation_config = EvaluationConfig(
            path_of_model=Path(config.path_of_model),
            training_data=Path(config.training_data),
            test_data=Path(config.test_data),
            val_data=Path(config.val_data),
            all_params=params,
            mlflow_uri=config.mlflow_uri,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE
        )
        return evaluation_config