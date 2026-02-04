
"""
Model architectures for Skin Cancer Classification
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, GlobalAveragePooling2D,
    BatchNormalization, Input
)
from tensorflow.keras.applications import ResNet152, EfficientNetB0
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

from config import config


class BaseSkinCancerModel:
    """Base class for skin cancer classification models"""
    
    def __init__(self, model_name: str = 'resnet152', input_shape: tuple = None):
        """
        Initialize base model
        
        Args:
            model_name: Name of the base model
            input_shape: Input shape for the model
        """
        self.model_name = model_name
        self.input_shape = input_shape or (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        self.model = None
        self.base_model = None
    
    def build_base_model(self):
        """Build the base model with frozen layers"""
        if self.model_name.lower() == 'resnet152':
            self.base_model = ResNet152(
                include_top=False,
                weights=config.BASE_MODEL_WEIGHTS,
                input_shape=self.input_shape
            )
        elif self.model_name.lower() == 'efficientnet':
            self.base_model = EfficientNetB0(
                include_top=False,
                weights=config.BASE_MODEL_WEIGHTS,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze base model layers
        self.base_model.trainable = False
        
        return self.base_model
    
    def add_custom_layers(self):
        """Add custom classification layers on top of base model"""
        # Get base model output
        x = self.base_model.output
        
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        
        # Fully connected layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE / 2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Output layer
        outputs = Dense(config.NUM_CLASSES, activation='softmax')(x)
        
        return outputs
    
    def build_model(self):
        """Build complete model"""
        # Build base model
        self.build_base_model()
        
        # Add custom layers
        outputs = self.add_custom_layers()
        
        # Create model
        self.model = Model(inputs=self.base_model.input, outputs=outputs)
        
        return self.model
    
    def compile_model(self, learning_rate: float = None):
        """Compile model with optimizer and loss"""
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def get_callbacks(self):
        """Get training callbacks"""
        # Model checkpoint
        checkpoint_path = config.MODEL_DIR / f"{self.model_name}_best.h5"
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.CHECKPOINT_MONITOR,
            mode=config.CHECKPOINT_MODE,
            save_best_only=True,
            verbose=config.CHECKPOINT_VERBOSE
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        )
        
        # TensorBoard logging
        tensorboard = TensorBoard(
            log_dir=str(config.LOGS_DIR / self.model_name),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        # CSV Logger
        csv_logger = CSVLogger(
            filename=str(config.LOGS_DIR / f"{self.model_name}_training.log"),
            separator=',',
            append=False
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]


class ResNet152Model(BaseSkinCancerModel):
    """ResNet152-based skin cancer classifier"""
    
    def __init__(self, input_shape: tuple = None):
        super().__init__(model_name='resnet152', input_shape=input_shape)
    
    def build_model(self):
        """Build ResNet152 model with custom classifier"""
        super().build_model()
        
        print(f"ResNet152 Model Summary:")
        print(f"  Total layers: {len(self.model.layers)}")
        print(f"  Trainable layers: {sum([layer.trainable for layer in self.model.layers])}")
        
        return self.model


class EfficientNetModel(BaseSkinCancerModel):
    """EfficientNet-based skin cancer classifier"""
    
    def __init__(self, input_shape: tuple = None):
        super().__init__(model_name='efficientnet', input_shape=input_shape)
    
    def build_model(self):
        """Build EfficientNet model with custom classifier"""
        super().build_model()
        
        print(f"EfficientNet Model Summary:")
        print(f"  Total layers: {len(self.model.layers)}")
        print(f"  Trainable layers: {sum([layer.trainable for layer in self.model.layers])}")
        
        return self.model


def create_model(model_name: str = 'resnet152', 
                input_shape: tuple = None,
                learning_rate: float = None) -> Model:
    """
    Factory function to create model
    
    Args:
        model_name: Name of model to create
        input_shape: Input shape for model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    if model_name.lower() == 'resnet152':
        model_class = ResNet152Model
    elif model_name.lower() == 'efficientnet':
        model_class = EfficientNetModel
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model
    model_builder = model_class(input_shape)
    model = model_builder.build_model()
    model = model_builder.compile_model(learning_rate)
    
    return model, model_builder.get_callbacks()
