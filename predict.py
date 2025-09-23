from pneumonia_detection import logger
from pneumonia_detection.pipeline.stage_4 import EvaluationPipeline  # adjust path to where your pipeline class is

if __name__ == "__main__":
    STAGE_NAME = "Evaluation Stage"
    try:
        logger.info("*******************")
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()   # create pipeline object
        obj.main()                   # run evaluation
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
