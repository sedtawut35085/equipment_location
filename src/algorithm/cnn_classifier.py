import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class CNNClassifier:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        """
        Initialize CNN Classifier for meat classification
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of classes (2 for fresh/rotten)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build CNN architecture
        """
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_callbacks(self):
        """
        Get training callbacks
        
        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, class_weight=None):
        """
        Train the CNN model
        
        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            X_val (np.array): Validation images
            y_val (np.array): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            class_weight (dict): Class weights for handling imbalanced data
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.compile_model()
            
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,  # Add class_weight parameter
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test (np.array): Test images
            y_test (np.array): Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Input images
            
        Returns:
            np.array: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(X)
    
    def predict_classes(self, X):
        """
        Predict classes
        
        Args:
            X (np.array): Input images
            
        Returns:
            np.array: Predicted classes
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        
        # Recompile the model to avoid the ABSL warning about unbuilt metrics
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model loaded and recompiled from {filepath}")
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()
