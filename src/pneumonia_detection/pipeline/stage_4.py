from  pneumonia_detection import logger
from pneumonia_detection.entity.config_entity import EvaluationConfig
from pneumonia_detection.config.configuration import ConfigurationManager
from pneumonia_detection.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()  
        model_evaluation = ModelEvaluation(config=evaluation_config)
        model_evaluation.evaluate_model()  
        model_evaluation.save_score()
        model_evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e