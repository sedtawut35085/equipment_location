import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import random
import tensorflow as tf

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

# Load the model
model = load_prediction_model(MODEL_PATH)
if model is None:
    print("Failed to load model. Please check if the model file exists.")
    exit(1)

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
            'predicted_class': pred_class,
            'predicted_class_name': pred_class_name,
            'confidence': confidence,
            'probabilities': prediction[0].tolist(),
            'success': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def collect_test_images(test_dir='dataset', num_samples=10):
    """
    Collect test images from dataset directories
    
    Args:
        test_dir: Directory containing class subdirectories
        num_samples: Number of samples to collect
    
    Returns:
        list: List of (image_path, true_class) tuples
    """
    test_images = []
    images_by_class = {class_name: [] for class_name in CLASS_NAMES}
    
    # Collect all images by class from CLASS_NAMES
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                    images_by_class[class_name].append((os.path.join(class_dir, fname), class_name))
    
    # Flatten all images and sample
    all_images = []
    for class_name in CLASS_NAMES:
        all_images.extend(images_by_class[class_name])
    
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    random.shuffle(selected_images)
    
    return selected_images

def predict_batch_images(model, test_images):
    """
    Predict multiple images and display results
    
    Args:
        model: Trained Keras model
        test_images: List of (image_path, true_class) tuples
    """
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, (img_path, true_class) in enumerate(test_images):
        print(f"\nSample {i+1}: {os.path.basename(img_path)}")
        print(f"True class: {true_class}")
        
        result = predict_single_image(model, img_path)
        
        if result['success']:
            pred_class_name = result['predicted_class_name']
            confidence = result['confidence']
            
            # Check if prediction is correct
            is_correct = pred_class_name == true_class
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"Predicted: {pred_class_name} (confidence: {confidence:.4f}) {status}")
            
            # Show probabilities for each class
            print("Probabilities:")
            for j, class_name in enumerate(CLASS_NAMES):
                prob = result['probabilities'][j]
                print(f"  {class_name}: {prob:.4f}")
        else:
            print(f"✗ Error: {result['error']}")
    
    # Summary
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n{'='*60}")
        print(f"SUMMARY: {correct_predictions}/{total_predictions} correct ({accuracy:.2%} accuracy)")
        print(f"{'='*60}")

def main():
    """Main prediction function"""
    print("Starting prediction...")
    
    # Collect test images
    test_images = collect_test_images(num_samples=10)
    
    if not test_images:
        print("No test images found. Please check your dataset directory.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Make predictions
    predict_batch_images(model, test_images)

if __name__ == "__main__":
    main()

    