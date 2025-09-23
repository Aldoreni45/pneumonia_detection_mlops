import os
from pathlib import Path
from box.exceptions import BoxValueError, BoxKeyError
from box import Box
from pneumonia_detection import logger
import json
from  ensure import ensure_annotations
from typing import Any
import base64
import joblib
import yaml

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> dict:
    try:
        from box import Box
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return Box(content)
    except BoxValueError as e:
        raise e
    except BoxKeyError as e:
        raise e
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> dict:  
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"JSON file loaded from: {path}")
    return data

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def encode_base64(data: bytes) -> str:
    encoded_data = base64.b64encode(data).decode('utf-8')
    return encoded_data   

@ensure_annotations
def decode_base64(data: str) -> bytes:
    decoded_data = base64.b64decode(data.encode('utf-8'))
    return decoded_data

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"