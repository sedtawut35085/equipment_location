import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    EfficientNetB3, EfficientNetB4, EfficientNetB5
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class TransferLearningClassifier:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2, base_model_name='VGG16'):
        """
        Initialize Transfer Learning Classifier
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of classes
            base_model_name (str): Name of the pre-trained model to use as base
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.model = None
        self.base_model = None
        self.history = None
        
        # Available pre-trained models
        self.available_models = {
            'VGG16': VGG16,
            'VGG19': VGG19,
            'ResNet50': ResNet50,
            'ResNet101': ResNet101,
            'ResNet152': ResNet152,
            'InceptionV3': InceptionV3,
            'InceptionResNetV2': InceptionResNetV2,
            'MobileNet': MobileNet,
            'MobileNetV2': MobileNetV2,
            'DenseNet121': DenseNet121,
            'DenseNet169': DenseNet169,
            'DenseNet201': DenseNet201,
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB1': EfficientNetB1,
            'EfficientNetB2': EfficientNetB2,
            'EfficientNetB3': EfficientNetB3,
            'EfficientNetB4': EfficientNetB4,
            'EfficientNetB5': EfficientNetB5
        }
        
    def build_model(self, freeze_base=True, fine_tune_layers=None):
        """
        Build transfer learning model
        
        Args:
            freeze_base (bool): Whether to freeze the base model layers
            fine_tune_layers (int): Number of top layers to unfreeze for fine-tuning
        """
        if self.base_model_name not in self.available_models:
            raise ValueError(f"Model {self.base_model_name} not available. "
                           f"Choose from: {list(self.available_models.keys())}")
        
        # Load pre-trained model
        base_model_class = self.available_models[self.base_model_name]
        
        # Handle different input requirements for different models
        if self.base_model_name in ['InceptionV3', 'InceptionResNetV2']:
            # These models require at least 75x75 input
            min_size = 75
            if self.input_shape[0] < min_size or self.input_shape[1] < min_size:
                print(f"Warning: {self.base_model_name} requires minimum {min_size}x{min_size} input. "
                      f"Consider using a larger input size for better performance.")
        
        self.base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers if specified
        if freeze_base:
            self.base_model.trainable = False
        else:
            # If fine_tune_layers is specified, freeze all layers except the top ones
            if fine_tune_layers is not None:
                for layer in self.base_model.layers[:-fine_tune_layers]:
                    layer.trainable = False
        
        # Add custom classification head
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Output layer
        if self.num_classes == 2:
            predictions = Dense(self.num_classes, activation='softmax')(x)
        else:
            predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, freeze_base=True):
        """
        Compile the model
        
        Args:
            learning_rate (float): Learning rate for optimizer
            freeze_base (bool): Whether base model is frozen (affects learning rate)
        """
        if self.model is None:
            self.build_model(freeze_base=freeze_base)
        
        # Use different learning rates for frozen vs unfrozen base models
        if freeze_base:
            lr = learning_rate
        else:
            lr = learning_rate / 10  # Lower learning rate for fine-tuning
            
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self, monitor='val_accuracy'):
        """
        Get training callbacks
        
        Args:
            monitor (str): Metric to monitor for callbacks
            
        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=7,
                min_lr=1e-8,
                verbose=1,
                mode='max' if 'accuracy' in monitor else 'min'
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
              class_weight=None, freeze_base=True):
        """
        Train the transfer learning model
        
        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            X_val (np.array): Validation images
            y_val (np.array): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            class_weight (dict): Class weights for handling imbalanced data
            freeze_base (bool): Whether to freeze base model layers
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.compile_model(freeze_base=freeze_base)
            
        callbacks = self.get_callbacks()
        
        print(f"Training {self.base_model_name} with base model "
              f"{'frozen' if freeze_base else 'unfrozen'}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=20, 
                  batch_size=16, fine_tune_layers=50, learning_rate=1e-5):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            X_val (np.array): Validation images
            y_val (np.array): Validation labels
            epochs (int): Number of fine-tuning epochs
            batch_size (int): Batch size (usually smaller for fine-tuning)
            fine_tune_layers (int): Number of top layers to unfreeze
            learning_rate (float): Learning rate for fine-tuning (usually lower)
        """
        if self.model is None:
            raise ValueError("Model must be trained first before fine-tuning")
        
        # Unfreeze top layers of the base model
        self.base_model.trainable = True
        
        # Freeze all layers except the top fine_tune_layers
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning: Unfreezing top {fine_tune_layers} layers of {self.base_model_name}")
        print(f"Total trainable parameters: {self.model.count_params()}")
        
        callbacks = self.get_callbacks()
        
        # Continue training
        fine_tune_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return fine_tune_history
    
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
        print(f"Transfer learning model ({self.base_model_name}) saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Transfer learning model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            self.build_model()
        
        print(f"Base Model: {self.base_model_name}")
        print(f"Total parameters: {self.model.count_params():,}")
        
        # Count trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {self.model.count_params() - trainable_params:,}")
        
        return self.model.summary()
    
    def compare_models(self, model_names, X_train, y_train, X_val, y_val, epochs=10):
        """
        Compare different pre-trained models
        
        Args:
            model_names (list): List of model names to compare
            X_train, y_train, X_val, y_val: Training and validation data
            epochs (int): Number of epochs for quick comparison
            
        Returns:
            dict: Results for each model
        """
        results = {}
        
        for model_name in model_names:
            print(f"\n{'='*20} Testing {model_name} {'='*20}")
            
            # Create new classifier with this model
            temp_classifier = TransferLearningClassifier(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                base_model_name=model_name
            )
            
            try:
                # Train for a few epochs
                temp_classifier.compile_model()
                temp_classifier.get_model_summary()
                
                history = temp_classifier.train(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs, batch_size=32
                )
                
                # Get final validation accuracy
                final_val_acc = max(history.history['val_accuracy'])
                results[model_name] = {
                    'val_accuracy': final_val_acc,
                    'history': history.history
                }
                
                print(f"{model_name} - Best Val Accuracy: {final_val_acc:.4f}")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results