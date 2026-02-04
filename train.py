
"""
Training script for Skin Cancer Classification
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_loader import SkinCancerDataLoader, load_dataset
from models import create_model
from config import config
from utils import setup_logging, plot_training_history


def train_model(model_name: str = 'resnet152',
               data_path: str = None,
               epochs: int = None,
               batch_size: int = None,
               learning_rate: float = None,
               use_augmentation: bool = True,
               save_model: bool = True):
    """
    Main training function
    
    Args:
        model_name: Name of model to train
        data_path: Path to dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        use_augmentation: Whether to use data augmentation
        save_model: Whether to save trained model
    """
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training with {model_name}")
    
    # Override config if provided
    if epochs:
        config.EPOCHS = epochs
    if batch_size:
        config.BATCH_SIZE = batch_size
    if learning_rate:
        config.LEARNING_RATE = learning_rate
    
    # Load dataset
    logger.info("Loading dataset...")
    loader = SkinCancerDataLoader(data_path)
    loader.load_from_folders()
    
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data()
    
    # Create data generators
    if use_augmentation:
        logger.info("Creating data generators with augmentation...")
        train_gen, val_gen = loader.create_data_generators(X_train, y_train, X_val, y_val)
        steps_per_epoch = len(X_train) // config.BATCH_SIZE
        validation_steps = len(X_val) // config.BATCH_SIZE
    else:
        logger.info("Training without data augmentation...")
        train_gen = (X_train, y_train)
        val_gen = (X_val, y_val)
        steps_per_epoch = None
        validation_steps = None
    
    # Create model
    logger.info(f"Creating {model_name} model...")
    model, callbacks = create_model(
        model_name=model_name,
        learning_rate=config.LEARNING_RATE
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    logger.info(f"Starting training for {config.EPOCHS} epochs...")
    
    if use_augmentation:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Save final model
    if save_model:
        model_path = config.MODEL_DIR / f"{model_name}_final.h5"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    logger.info("\n" + "="*50)
    logger.info("TEST SET PERFORMANCE:")
    logger.info(f"  Loss:      {test_loss:.4f}")
    logger.info(f"  Accuracy:  {test_accuracy:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  AUC:       {test_auc:.4f}")
    logger.info("="*50)
    
    return model, history


def main():
    """Main function for command-line training"""
    parser = argparse.ArgumentParser(description='Train skin cancer classification model')
    
    parser.add_argument('--model', type=str, default='resnet152',
                       choices=['resnet152', 'efficientnet'],
                       help='Model architecture to train')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset directory')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_name=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation,
        save_model=args.save_model
    )


if __name__ == "__main__":
    main()
