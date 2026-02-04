
"""
Evaluation script for Skin Cancer Classification
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from data_loader import SkinCancerDataLoader
from models import create_model
from config import config
from utils import plot_confusion_matrix, plot_classification_report


def evaluate_model(model_path: str, data_path: str = None,
                  model_name: str = None, batch_size: int = 32):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        model_name: Name of model architecture
        batch_size: Batch size for evaluation
    """
    print(f"Loading model from {model_path}")
    
    # Load model
    if model_name:
        # Create new model with same architecture
        model, _ = create_model(model_name=model_name)
        model.load_weights(model_path)
    else:
        # Load entire model
        model = tf.keras.models.load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    loader = SkinCancerDataLoader(data_path)
    loader.load_from_folders()
    
    _, _, X_test, _, _, y_test = loader.prepare_data()
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    
    # Get predictions
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(
        cm, 
        classes=config.CLASS_NAMES,
        title=f'Confusion Matrix - {Path(model_path).stem}'
    )
    
    # Calculate additional metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average=None
    )
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Class': config.CLASS_NAMES,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    print("\n" + "="*50)
    print("DETAILED METRICS BY CLASS")
    print("="*50)
    print(metrics_df.to_string(index=False))
    
    # Save metrics to CSV
    metrics_path = config.RESULTS_DIR / f"{Path(model_path).stem}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot classification report
    plot_classification_report(
        y_true_classes, y_pred_classes,
        classes=config.CLASS_NAMES,
        title=f'Classification Report - {Path(model_path).stem}'
    )
    
    return results, y_pred, y_true_classes


def compare_models(model_paths: list, model_names: list, data_path: str = None):
    """
    Compare multiple models
    
    Args:
        model_paths: List of model paths
        model_names: List of model names
        data_path: Path to dataset
    """
    comparison_results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print('='*60)
        
        results, y_pred, y_true = evaluate_model(
            model_path, data_path, model_name
        )
        
        comparison_results.append({
            'Model': model_name,
            'Accuracy': results[1],
            'Precision': results[2],
            'Recall': results[3],
            'AUC': results[4]
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.bar(comparison_df['Model'], comparison_df[metric])
        ax.set_title(f'{metric} Comparison')
       
