
"""
Data loading and preprocessing for Skin Cancer Classification
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import config


class SkinCancerDataLoader:
    """Load and preprocess HAM10000 dataset"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to dataset directory
        """
        self.data_path = Path(data_path) if data_path else config.DATA_DIR
        self.images = []
        self.labels = []
        self.label_map = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    
    def load_from_folders(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from folder structure
        
        Returns:
            Tuple of images and labels as numpy arrays
        """
        print("Loading images from folder structure...")
        
        for class_name, class_idx in self.label_map.items():
            class_path = self.data_path / class_name
            
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist. Skipping...")
                continue
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_path.glob(f"*{ext}")))
            
            print(f"Found {len(image_files)} images in {class_name}")
            
            # Load and preprocess images
            for img_path in image_files:
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, config.IMAGE_SIZE)
                    
                    # Normalize pixel values
                    img = img.astype('float32') / 255.0
                    
                    self.images.append(img)
                    self.labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Total images loaded: {len(self.images)}")
        print(f"Total labels loaded: {len(self.labels)}")
        
        # Print class distribution
        self._print_class_distribution()
        
        return self.images, self.labels
    
    def _print_class_distribution(self):
        """Print distribution of classes"""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass Distribution:")
        for label, count in zip(unique, counts):
            class_name = config.CLASS_NAMES[label]
            print(f"  {class_name}: {count} images ({count/len(self.labels)*100:.2f}%)")
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.2, 
                    random_state: int = 42) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Convert labels to categorical
        y_categorical = to_categorical(self.labels, num_classes=config.NUM_CLASSES)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, y_categorical,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        # Get labels for stratification
        temp_labels = np.argmax(y_temp, axis=1)
        
        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=temp_labels
        )
        
        print(f"\nDataset Split:")
        print(f"  Training set:   {X_train.shape[0]} images")
        print(f"  Validation set: {X_val.shape[0]} images")
        print(f"  Test set:       {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
        """
        Create data generators with augmentation
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Create augmentation generator for training
        train_datagen = ImageDataGenerator(
            rotation_range=config.DATA_AUGMENTATION['rotation_range'],
            width_shift_range=config.DATA_AUGMENTATION['width_shift_range'],
            height_shift_range=config.DATA_AUGMENTATION['height_shift_range'],
            shear_range=config.DATA_AUGMENTATION['shear_range'],
            zoom_range=config.DATA_AUGMENTATION['zoom_range'],
            horizontal_flip=config.DATA_AUGMENTATION['horizontal_flip'],
            vertical_flip=config.DATA_AUGMENTATION['vertical_flip'],
            fill_mode=config.DATA_AUGMENTATION['fill_mode']
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def load_from_csv(self, csv_path: str, images_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from CSV file (for HAM10000 metadata)
        
        Args:
            csv_path: Path to HAM10000_metadata.csv
            images_dir: Directory containing images
            
        Returns:
            Tuple of images and labels
        """
        print("Loading dataset from CSV...")
        
        # Load metadata
        metadata = pd.read_csv(csv_path)
        
        for idx, row in metadata.iterrows():
            try:
                # Construct image path
                img_filename = row['image_id'] + '.jpg'
                img_path = Path(images_dir) / img_filename
                
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue
                
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, config.IMAGE_SIZE)
                
                # Normalize
                img = img.astype('float32') / 255.0
                
                # Get label
                dx = row['dx']
                if dx in self.label_map:
                    label = self.label_map[dx]
                else:
                    # Map to class index if not in predefined names
                    if dx not in self.label_map:
                        self.label_map[dx] = len(self.label_map)
                    label = self.label_map[dx]
                
                self.images.append(img)
                self.labels.append(label)
                
                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1} images...")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\nTotal images loaded: {len(self.images)}")
        self._print_class_distribution()
        
        return self.images, self.labels


def load_dataset(data_path: str = None, from_csv: bool = False,
                csv_path: str = None, images_dir: str = None) -> Tuple:
    """
    Convenience function to load dataset
    
    Args:
        data_path: Path to dataset
        from_csv: Whether to load from CSV
        csv_path: Path to CSV file
        images_dir: Directory containing images
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    loader = SkinCancerDataLoader(data_path)
    
    if from_csv and csv_path and images_dir:
        loader.load_from_csv(csv_path, images_dir)
    else:
        loader.load_from_folders()
    
    return loader.prepare_data()
