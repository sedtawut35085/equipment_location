#!/usr/bin/env python3
"""
Single Image Prediction Script
Usage: python predict_single.py <image_path>
"""

import sys
import os
import argparse
from predict_util import load_prediction_model, predict_single_image, MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description='Predict the class of a single image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_prediction_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Make prediction
    print(f"Predicting image: {args.image_path}")
    result = predict_single_image(model, args.image_path)
    
    if result['success']:
        print(f"\n{'='*50}")
        print("PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted Class: {result['predicted_class_name']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        print(f"{'='*50}")
    else:
        print(f"✗ Prediction failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()