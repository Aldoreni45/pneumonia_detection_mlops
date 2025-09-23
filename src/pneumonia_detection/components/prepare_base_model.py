# import os
# from pneumonia_detection.entity.config_entity import PrepareBaseModelConfig
# from pneumonia_detection import logger
# from pathlib import Path
# from zipfile import ZipFile
# from pneumonia_detection.utils.common import get_size
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.metrics import Precision, Recall, Accuracy

# class PrepareBaseModel:
#     def __init__(self,config:PrepareBaseModelConfig):
#         self.config=config
    
#     def get_base_model(self):
#         self.model=tf.keras.applications.vgg16.VGG16(
#             input_shape=self.config.params_image_size,
#             weights=self.config.params_weights,
#             include_top=self.config.params_include_top)
#         self.model.save(self.config.base_model_path,self.model)
#         logger.info(f"Base model is saved at {self.config.base_model_path} with size {get_size(self.config.base_model_path)}")
    
#     @staticmethod
#     def _prepare_full_model(model:tf.keras.Model,classes:int,freeze_all:bool,freeze_till:int|None,learning_rate:float)->tf.keras.Model:
#         if freeze_all:
#             for layer in model.layers:
#                 layer.trainable=False
#         elif freeze_till is not None and freeze_till > 0:
#             # Freeze layers from start to freeze_till (not excluding last layers)
#             for i, layer in enumerate(model.layers):
#                 if i < len(model.layers) - freeze_till:
#                     layer.trainable = False
#                 else:
#                     layer.trainable = True


#         full_model = Sequential([
#             model,
#             GlobalAveragePooling2D(),  # Better than Flatten for feature maps
#             Dense(512, activation='relu'),  # Increased capacity
#             Dropout(0.5),
#             Dense(256, activation='relu'),  # Additional layer for better learning
#             Dropout(0.3),
#             Dense(1, activation='sigmoid')  # Binary classification
#         ])


#         full_model.compile(
#     optimizer=Adam(learning_rate=learning_rate),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

#         logger.info(f"Full model is compiled and the layers are freezed till {freeze_till} and freeze all is {freeze_all}")
#         return full_model
    

#     def update_base_model(self):
#         self.full_model=self._prepare_full_model(  
#             model=self.model,
#             classes=self.config.params_classes,
#             freeze_all=self.config.params_freeze_all,
#             freeze_till=self.config.params_freeze_till,
#             learning_rate=self.config.params_learning_rate
#         )

#         self.save_model(path=self.config.updated_base_model_path,model=self.full_model)
#         logger.info(f"Updated base model is saved at {self.config.updated_base_model_path} with size {get_size(self.config.updated_base_model_path)}")


#     @staticmethod
#     def save_model(path:Path,model:tf.keras.Model)->None:
#         model.save(path)
        



# prepare_base_model.py - FIXED VERSION
import os
from pneumonia_detection.entity.config_entity import PrepareBaseModelConfig
from pneumonia_detection import logger
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        # FIX: Remove the extra parameter in save()
        self.model.save(self.config.base_model_path)
        logger.info(f"Base model is saved at {self.config.base_model_path}")
        
    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model, 
        classes: int, 
        freeze_all: bool, 
        freeze_till: int | None, 
        learning_rate: float
    ) -> tf.keras.Model:
        
        # CRITICAL FIX: Proper layer freezing
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
            logger.info("All layers frozen")
        elif freeze_till is not None and freeze_till > 0:
            # Unfreeze the last 'freeze_till' layers
            for i, layer in enumerate(model.layers):
                if i >= len(model.layers) - freeze_till:
                    layer.trainable = True
                else:
                    layer.trainable = False
            logger.info(f"Unfrozen last {freeze_till} layers")
        else:
            # Unfreeze all layers
            for layer in model.layers:
                layer.trainable = True
            logger.info("All layers unfrozen")
        
        # Print trainable status
        trainable_count = sum(1 for layer in model.layers if layer.trainable)
        logger.info(f"Trainable layers: {trainable_count}/{len(model.layers)}")
        
        # Build the full model
        full_model = Sequential([
            model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', name='dense_1'),
            Dropout(0.5, name='dropout_1'),
            Dense(256, activation='relu', name='dense_2'),
            Dropout(0.3, name='dropout_2'),
            Dense(1, activation='sigmoid', name='predictions')
        ])
        
        # CRITICAL FIX: Better optimizer settings
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        full_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        
        # Print model summary
        full_model.summary()
        logger.info(f"Model compiled with learning_rate={learning_rate}")
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate
        )
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logger.info(f"Updated base model saved")
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        model.save(path)        