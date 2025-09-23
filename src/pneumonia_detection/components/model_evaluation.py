import tensorflow as tf
from pneumonia_detection.entity.config_entity import EvaluationConfig
from pneumonia_detection import logger
from pathlib import Path
import os
import mlflow
import mlflow.keras
from pneumonia_detection.utils.common import  save_json,create_directories,read_yaml
from urllib.parse import urlparse

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")



class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def train_valid_generator(self):
        """Create properly configured data generators."""
        
        # CRITICAL FIX: Add class_mode and other important parameters
        datagenerator_kwargs = dict(
            rescale=1.0 / 255.0,
            # Add normalization
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='binary',  # CRITICAL: Specify binary classification
            color_mode='rgb',     # CRITICAL: Specify RGB
            seed=42              # For reproducibility
        )
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.val_data,
            shuffle=True,
            **dataflow_kwargs
        )
        # CRITICAL: Log class information
        logger.info(f"Training classes: {self.train_generator.class_indices}")
        logger.info(f"Number of training samples: {self.train_generator.samples}")
        return self.train_generator
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluate_model(self):
        self.model = self.load_model(self.config.path_of_model)
        self.train_generator = self.train_valid_generator()
        self.score = self.model.evaluate(self.train_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.set_experiment("Pneumonia_Detection")
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("loss", self.score[0])
            mlflow.log_metric("accuracy", self.score[1])
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")


