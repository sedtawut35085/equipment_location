import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataLoader:
    def __init__(self, dataset_path, target_size=(128, 128), class_names=['fresh', 'rotten']):
        """
        Initialize DataLoader for image classification
        
        Args:
            dataset_path (str): Path to dataset directory
            target_size (tuple): Target size for images (height, width)
            class_names (list): List of class names. If None, will auto-detect from directory structure
        """
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.class_names = class_names
            
        self.class_indices = {name: idx for idx, name in enumerate(self.class_names)}
                    
    def load_images_from_folder(self, folder_path, class_label):
        """
        Load images from a specific folder
        
        Args:
            folder_path (str): Path to folder containing images
            class_label (int): Class label for the images
            
        Returns:
            tuple: (images, labels)
        """
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return np.array(images), np.array(labels)
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
        
        for filename in os.listdir(folder_path):
            if filename.endswith(supported_formats):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Load and preprocess image
                    img = load_img(img_path, target_size=self.target_size)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalize to [0, 1]
                    
                    images.append(img_array)
                    labels.append(class_label)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def load_dataset(self):
        """
        Load the complete dataset
        
        Returns:
            tuple: (X, y) where X is images and y is labels
        """
        all_images = []
        all_labels = []
        
        print("Loading dataset...")
        
        for class_name in self.class_names:
            class_folder = os.path.join(self.dataset_path, class_name)
            class_label = self.class_indices[class_name]
            
            images, labels = self.load_images_from_folder(class_folder, class_label)
            
            if len(images) > 0:
                all_images.extend(images)
                all_labels.extend(labels)
                print(f"Loaded {len(images)} images from {class_name} class")
            else:
                print(f"No images found in {class_name} class")
        
        if len(all_images) == 0:
            raise ValueError("No images found in dataset")
        
        X = np.array(all_images)
        y = np.array(all_labels)
        
        print(f"Total images loaded: {len(X)}")
        print(f"Image shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        return X, y
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set (from remaining data)
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """
        Create data augmentation pipeline
        
        Returns:
            tf.keras.Sequential: Data augmentation pipeline
        """
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        
        return data_augmentation
    
    def get_class_weights(self, y):
        """
        Calculate class weights for imbalanced dataset
        
        Args:
            y (np.array): Labels
            
        Returns:
            dict: Class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        return dict(zip(np.unique(y), class_weights))
    
    def preprocess_for_training(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline for training
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, class_weights)
        """
        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            X, y, test_size, val_size, random_state
        )
        
        # Calculate class weights
        class_weights = self.get_class_weights(y_train)
        
        print(f"Class weights: {class_weights}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights
