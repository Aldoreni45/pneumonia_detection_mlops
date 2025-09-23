# import  os
# from pathlib import Path
# from pneumonia_detection.entity.config_entity import TrainingConfig
# from pneumonia_detection import logger
# from pneumonia_detection.utils.common import create_directories
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# class Training:
#     def __init__(self,config:TrainingConfig):
#         self.config=config
#         create_directories([self.config.root_dir])
#         logger.info(f"Training directory is created at {self.config.root_dir}")

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)
#         logger.info(f"Model is saved at {path} with size {os.path.getsize(path)}")

#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(
#             self.config.trained_model_path
#         )

#     def train_model(self):

#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=5,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.2,
#                 patience=3,
#                 min_lr=1e-7,
#                 verbose=1
#             ),
#             ModelCheckpoint(
#                 filepath=str(self.config.production_model_path),
#                 monitor='val_accuracy',
#                 save_best_only=True,
#                 verbose=1
#             )
#         ]
        
#         # FIX 6: Better steps calculation
#         steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
#         validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)
        

#         self.model.fit(
#             self.train_generator,
#             validation_data=self.valid_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=steps_per_epoch,
#             validation_steps=validation_steps,
#             callbacks=callbacks,
#             verbose=1

#         )
#         self.save_model(path=self.config.production_model_path, model=self.model)

#     def train_valid_generator(self):
#         """
#         Create train, validation, and test generators from separate folders.
#         """
#         datagenerator_kwargs = dict(
#             rescale=1.0 / 255
#         )

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"
#         )

#         # ------------------- VALIDATION GENERATOR -------------------
#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.val_data,  # <- val folder
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         # ------------------- TRAIN GENERATOR -------------------
#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,  # <- train folder
#             shuffle=True,
#             **dataflow_kwargs
#         )

#         # ------------------- TEST GENERATOR (Optional) -------------------
#         test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.test_generator = test_datagenerator.flow_from_directory(
#             directory=self.config.test_data,  # <- test folder
#             shuffle=False,
#             **dataflow_kwargs
#         )










import os
from pathlib import Path
from pneumonia_detection.entity.config_entity import TrainingConfig
from pneumonia_detection import logger
from pneumonia_detection.utils.common import create_directories
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        create_directories([self.config.root_dir])
        logger.info(f"Training directory created at {self.config.root_dir}")
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info(f"Model saved at {path}")
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.trained_model_path)
        logger.info("Base model loaded successfully")
    
    def train_model(self):
        # Create model checkpoint directory
        checkpoint_dir = Path(self.config.production_model_path).parent
        create_directories([checkpoint_dir])
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Changed to val_accuracy
                patience=7,  # Increased patience
                restore_best_weights=True,
                verbose=1,
                mode='max'  # Maximize accuracy
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=4,  # Reduced patience
                min_lr=1e-8,
                verbose=1,
                cooldown=2
            ),
            ModelCheckpoint(
                filepath=str(self.config.production_model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max',
                save_weights_only=False
            )
        ]
        
        # Better steps calculation
        steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
        validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)
        
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.valid_generator.samples}")
        
        # CRITICAL: Use class weights if dataset is imbalanced
        history = self.model.fit(
            self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=getattr(self, 'class_weights', None),  # Use class weights if available
            verbose=1,
            workers=4,  # Parallel data loading
            use_multiprocessing=False  # Set to False to avoid issues
        )
        
        # Save final model
        self.save_model(path=self.config.production_model_path, model=self.model)
        logger.info("Training completed successfully")
        return history
    
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
        
        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.val_data,
            shuffle=False,
            **dataflow_kwargs
        )
        
        # Training generator with controlled augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,      # Reduced from 40
                horizontal_flip=True,
                width_shift_range=0.1,  # Reduced from 0.2
                height_shift_range=0.1, # Reduced from 0.2
                shear_range=0.1,       # Reduced from 0.2
                zoom_range=0.1,        # Reduced from 0.2
                brightness_range=[0.9, 1.1],  # NEW: Brightness augmentation
                fill_mode='nearest',
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            **dataflow_kwargs
        )
        
        # Test generator
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            shuffle=False,
            **dataflow_kwargs
        )
        
        # CRITICAL: Log class information
        logger.info(f"Training classes: {self.train_generator.class_indices}")
        logger.info(f"Number of training samples: {self.train_generator.samples}")
        logger.info(f"Number of validation samples: {self.valid_generator.samples}")
        logger.info(f"Number of test samples: {self.test_generator.samples}")
        
        # Calculate class distribution and weights
        self._calculate_class_weights()
        
        # CRITICAL: Verify data loading
        self._verify_data_loading()
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        # Get class distribution from training generator
        labels = self.train_generator.labels
        class_counts = np.bincount(labels)
        
        logger.info(f"Class distribution: {dict(zip(self.train_generator.class_indices.keys(), class_counts))}")
        
        # Calculate weights
        total_samples = len(labels)
        n_classes = len(class_counts)
        
        self.class_weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                self.class_weights[i] = total_samples / (n_classes * count)
            else:
                self.class_weights[i] = 1.0
        
        logger.info(f"Class weights: {self.class_weights}")
    
    def _verify_data_loading(self):
        """Verify that data is loading correctly."""
        try:
            # Get a batch from training generator
            batch_x, batch_y = next(self.train_generator)
            logger.info(f"Training batch shape: {batch_x.shape}, labels shape: {batch_y.shape}")
            logger.info(f"Training batch min/max: {batch_x.min():.3f}/{batch_x.max():.3f}")
            logger.info(f"Training labels unique values: {np.unique(batch_y)}")
            
            # Get a batch from validation generator
            val_batch_x, val_batch_y = next(self.valid_generator)
            logger.info(f"Validation batch shape: {val_batch_x.shape}, labels shape: {val_batch_y.shape}")
            logger.info(f"Validation labels unique values: {np.unique(val_batch_y)}")
            
        except Exception as e:
            logger.error(f"Error loading data batches: {e}")
            raise


# Additional debugging function
def debug_model_and_data(training_instance):
    """Debug function to check model and data."""
    logger.info("=== MODEL DEBUG INFO ===")
    
    # Check model architecture
    logger.info("Model summary:")
    training_instance.model.summary()
    
    # Check trainable parameters
    trainable_params = sum([tf.keras.utils.get_file.count_params(w) for w in training_instance.model.trainable_weights])
    total_params = sum([tf.keras.utils.get_file.count_params(w) for w in training_instance.model.weights])
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Test model on a batch
    try:
        batch_x, batch_y = next(training_instance.train_generator)
        predictions = training_instance.model.predict(batch_x[:1])
        logger.info(f"Test prediction shape: {predictions.shape}")
        logger.info(f"Test prediction value: {predictions[0]}")
    except Exception as e:
        logger.error(f"Error in model prediction test: {e}")
    
    logger.info("=== END DEBUG INFO ===")


# Usage example with debugging
def main():
    """Main training function with debugging."""
    from pneumonia_detection.config.configuration import ConfigurationManager
    
    config_manager = ConfigurationManager()
    training_config = config_manager.get_training_config()
    
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    
    # Debug before training
    debug_model_and_data(training)
    
    # Start training
    history = training.train_model()
    
    return history

 