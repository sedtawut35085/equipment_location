import os
import random
from predict_util import load_prediction_model, predict_single_image, MODEL_PATH, CLASS_NAMES

# Load the model
model = load_prediction_model(MODEL_PATH)
if model is None:
    print("Failed to load model. Please check if the model file exists.")
    exit(1)

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
            for class_name, prob in result['probabilities'].items():
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

    