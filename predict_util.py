#!/usr/bin/env python3
"""
Prediction Utilities Module
Contains shared functionality for image prediction
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = 'models/transfer_learning_mobilenetv2_model.h5'
TARGET_SIZE = (224, 224)  # Match the training input size
CLASS_NAMES = ['fresh', 'rotten']

def load_prediction_model(model_path):
    """Load the trained model for prediction"""
    try:
        model = load_model(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"Model input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def predict_single_image(model, img_path, target_size=TARGET_SIZE):
    """
    Predict the class of a single image
    
    Args:
        model: Trained Keras model
        img_path: Path to the image file
        target_size: Target size for image resizing
    
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    try:
        # Check if image file exists
        if not os.path.exists(img_path):
            return {'error': f"Image file not found: {img_path}", 'success': False}
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        confidence = float(np.max(prediction))
        pred_class = np.argmax(prediction, axis=1)[0]
        pred_class_name = CLASS_NAMES[pred_class]
        
        return {
            'image_path': img_path,
            'predicted_class': pred_class,
            'predicted_class_name': pred_class_name,
            'confidence': confidence,
            'probabilities': {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))},
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }