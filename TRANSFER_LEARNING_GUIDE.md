# Transfer Learning Guide for Equipment Classification

This guide explains how to use transfer learning with pre-trained models for equipment classification.

## Overview

Transfer learning allows you to leverage pre-trained models (trained on ImageNet) and adapt them for your specific classification task. This approach typically provides better accuracy with less training time compared to training from scratch.

## Available Models

### Recommended Models:

1. **EfficientNetB0** (Default)

   - Best balance of accuracy and efficiency
   - Good for general purposes
   - Fast training and inference

2. **ResNet50**

   - Well-established architecture
   - Good performance across various tasks
   - Extensive research backing

3. **MobileNetV2**

   - Lightweight for mobile deployment
   - Fast inference
   - Lower accuracy but very efficient

4. **VGG16**

   - Simple and interpretable
   - Good for educational purposes
   - Larger model size

5. **DenseNet121**
   - Efficient parameter usage
   - Good gradient flow
   - Memory intensive during training

## Usage Instructions

### Method 1: Interactive Mode

Run the main script and select your preferences:

```bash
python main.py
```

The script will ask you to:

1. Choose between Traditional CNN (1) or Transfer Learning (2)
2. Select a base model from available options

### Method 2: Direct Transfer Learning

Use the dedicated transfer learning script:

```bash
python main_transfer_learning.py
```

### Method 3: Programmatic Usage

```python
from src.algorithm.transfer_learning_classifier import TransferLearningClassifier

# Create classifier
model = TransferLearningClassifier(
    input_shape=(224, 224, 3),
    num_classes=2,  # Number of equipment classes
    base_model_name='EfficientNetB0'
)

# Train with frozen base model
model.compile_model(learning_rate=0.0001, freeze_base=True)
history = model.train(X_train, y_train, X_val, y_val, epochs=30)

# Optional: Fine-tune by unfreezing top layers
fine_tune_history = model.fine_tune(
    X_train, y_train, X_val, y_val,
    epochs=15, fine_tune_layers=50
)
```

## Configuration Parameters

### For Transfer Learning:

- **Image Size**: 224x224 (recommended for most pre-trained models)
- **Batch Size**: 16-32 (smaller than traditional CNN)
- **Learning Rate**: 0.0001 (lower than traditional CNN)
- **Epochs**: 20-30 for initial training, 10-15 for fine-tuning

### Model-Specific Requirements:

- **InceptionV3/InceptionResNetV2**: Minimum 75x75 input size
- **EfficientNet models**: Work well with 224x224 or higher
- **MobileNet models**: Optimized for smaller inputs, 224x224 works well

## Training Strategies

### 1. Two-Phase Training (Recommended)

**Phase 1: Frozen Base Model**

- Freeze pre-trained layers
- Train only the classification head
- Use higher learning rate (0.0001)
- Train for 20-30 epochs

**Phase 2: Fine-tuning**

- Unfreeze top layers of base model
- Use very low learning rate (0.00001)
- Train for 10-15 epochs
- Use smaller batch size

### 2. End-to-End Training

Train the entire model from the start with a very low learning rate.

## Performance Expectations

### Typical Results:

- **Traditional CNN**: 70-85% accuracy
- **Transfer Learning**: 85-95% accuracy
- **Transfer Learning + Fine-tuning**: 90-98% accuracy

### Training Time:

- **Traditional CNN**: 1-2 hours (50 epochs)
- **Transfer Learning**: 30-45 minutes (30 epochs)
- **Fine-tuning**: Additional 15-20 minutes

## File Structure

```
models/
├── evaluation/
│   ├── confusion_matrix_tl_1.png      # Transfer learning confusion matrices
│   ├── kfold_accuracies_tl_5_fold.png # K-fold results
│   └── model_comparison.png            # Model comparison plots
├── transfer_learning_efficientnetb0_model.h5  # Saved model
└── cnn_model.h5                        # Traditional CNN model
```

## Advanced Usage

### Model Comparison

Compare multiple models automatically:

```python
from src.evaluation.transfer_learning_utils import run_quick_comparison

# Compare multiple models
results = run_quick_comparison()
```

### Custom Model Architecture

Modify the classification head:

```python
# In transfer_learning_classifier.py, modify build_model():
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)  # Larger dense layer
x = Dropout(0.6)(x)                    # Higher dropout
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)
```

## Troubleshooting

### Common Issues:

1. **Out of Memory Error**

   - Reduce batch size
   - Use smaller input size
   - Choose lighter model (MobileNetV2)

2. **Poor Performance**

   - Increase input image size
   - Try different base model
   - Add data augmentation
   - Increase training epochs

3. **Overfitting**

   - Increase dropout rates
   - Add more regularization
   - Use smaller learning rate
   - Reduce model complexity

4. **Slow Training**
   - Use GPU acceleration
   - Reduce batch size
   - Choose efficient model (EfficientNet)

### Performance Optimization:

1. **For Better Accuracy**:

   - Use larger input size (384x384)
   - Try EfficientNetB3 or B4
   - Implement data augmentation
   - Use ensemble methods

2. **For Faster Inference**:
   - Use MobileNetV2
   - Quantize the model
   - Use smaller input size
   - Prune unnecessary layers

## Best Practices

1. **Data Preparation**:

   - Ensure images are properly normalized
   - Use appropriate image size for the model
   - Balance your dataset

2. **Training**:

   - Always start with frozen base model
   - Use learning rate scheduling
   - Monitor validation metrics
   - Save best model checkpoints

3. **Evaluation**:

   - Use K-fold cross-validation
   - Test on held-out dataset
   - Analyze misclassified examples
   - Generate confusion matrices

4. **Deployment**:
   - Convert to TensorFlow Lite for mobile
   - Use TensorFlow Serving for production
   - Monitor model performance
   - Plan for model updates

## Model Selection Guide

Choose your model based on requirements:

- **High Accuracy Needed**: EfficientNetB3/B4, ResNet101
- **Balanced Performance**: EfficientNetB0, ResNet50
- **Fast Inference**: MobileNetV2, EfficientNetB0
- **Mobile Deployment**: MobileNetV2, MobileNet
- **Learning/Research**: VGG16, ResNet50

## Results Interpretation

### Accuracy Thresholds:

- **> 95%**: Excellent, ready for production
- **90-95%**: Very good, consider fine-tuning
- **85-90%**: Good, may need more data or tuning
- **80-85%**: Acceptable for some applications
- **< 80%**: Needs improvement

### What to do with results:

- **High train, low val accuracy**: Overfitting - add regularization
- **Low train and val accuracy**: Underfitting - increase model capacity
- **Good val, poor test accuracy**: Data leakage or insufficient test data

Remember: Transfer learning is typically more effective than training from scratch, especially with limited data!
