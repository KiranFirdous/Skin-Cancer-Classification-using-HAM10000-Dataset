"""
Configuration file for Skin Cancer Classification project
"""

import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "saved_models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Create directories
    for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Dataset parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    CLASS_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Model parameters
    BASE_MODEL_WEIGHTS = 'imagenet'
    DROPOUT_RATE = 0.5
    
    # Data augmentation
    DATA_AUGMENTATION = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest'
    }
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Model checkpoint
    CHECKPOINT_MONITOR = 'val_accuracy'
    CHECKPOINT_MODE = 'max'
    CHECKPOINT_VERBOSE = 1
    
    # Colors for visualization
    CLASS_COLORS = {
        'AK': '#FF6B6B',
        'BCC': '#4ECDC4',
        'BKL': '#45B7D1',
        'DF': '#96CEB4',
        'MEL': '#FFEAA7',
        'NV': '#DDA0DD',
        'SCC': '#98D8C8',
        'VASC': '#F7DC6F'
    }

config = Config()
