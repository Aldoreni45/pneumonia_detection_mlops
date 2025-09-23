from pneumonia_detection.config.configuration import ConfigurationManager
from pneumonia_detection.components.training_model import Training
from pneumonia_detection import logger


STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        trainer = Training(config=training_config)

        # Step 1: Load model
        trainer.get_base_model()

        # Step 2: Prepare data generators
        trainer.train_valid_generator()

        # Step 3: Train the model
        history = trainer.train_model()


if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        