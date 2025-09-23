from pneumonia_detection.entity.config_entity import DataIngestionConfig
import os
from pneumonia_detection.utils.common import get_size
import zipfile
from  pneumonia_detection import  logger

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def extract_zip_file(self):
        local_path=self.config.local_data_file
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        logger.info(f"Extracting data from {local_path} into directory {unzip_path}")
        with zipfile.ZipFile(local_path,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted data from {local_path} into directory {unzip_path} and the size of the data is {get_size(unzip_path)}")