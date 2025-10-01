"""
Transfer Learning Utilities and Examples
This module provides utility functions and examples for transfer learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def compare_models_performance(results_dict, save_path="models/evaluation/model_comparison.png"):
    """
    Compare performance of different transfer learning models
    
    Args:
        results_dict (dict): Dictionary with model names as keys and accuracy as values
        save_path (str): Path to save the comparison plot
    """
    if not results_dict:
        print("No results to compare")
        return
    
    # Filter out models with errors
    valid_results = {k: v for k, v in results_dict.items() 
                    if isinstance(v, dict) and 'val_accuracy' in v}
    
    if not valid_results:
        print("No valid results to compare")
        return
    
    models = list(valid_results.keys())
    accuracies = [valid_results[model]['val_accuracy'] for model in models]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, accuracies, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Transfer Learning Models Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    best_model = max(valid_results, key=lambda x: valid_results[x]['val_accuracy'])
    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {valid_results[best_model]['val_accuracy']:.4f}")

def plot_training_comparison(histories_dict, save_path="models/evaluation/training_comparison.png"):
    """
    Plot training history comparison for multiple models
    
    Args:
        histories_dict (dict): Dictionary with model names as keys and history objects as values
        save_path (str): Path to save the comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    for model_name, history in histories_dict.items():
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label=f'{model_name} Train')
            axes[0, 1].plot(history['val_accuracy'], label=f'{model_name} Val')
    
    axes[0, 0].set_title('Training Accuracy Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot loss
    for model_name, history in histories_dict.items():
        if 'loss' in history:
            axes[1, 0].plot(history['loss'], label=f'{model_name} Train')
            axes[1, 1].plot(history['val_loss'], label=f'{model_name} Val')
    
    axes[1, 0].set_title('Training Loss Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Loss Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_model_predictions(model, X_test, y_test, class_names, 
                            save_path="models/evaluation/prediction_analysis.png"):
    """
    Analyze model predictions and show misclassified examples
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        class_names: List of class names
        save_path: Path to save analysis plot
    """
    # Make predictions
    y_pred = model.predict_classes(X_test)
    y_pred_proba = model.predict(X_test)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Find misclassified examples
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    if len(misclassified_indices) > 0:
        # Plot some misclassified examples
        n_examples = min(8, len(misclassified_indices))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(n_examples):
            idx = misclassified_indices[i]
            
            axes[i].imshow(X_test[idx])
            axes[i].set_title(f'True: {class_names[y_test[idx]]}\n'
                            f'Pred: {class_names[y_pred[idx]]}\n'
                            f'Conf: {y_pred_proba[idx][y_pred[idx]]:.3f}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_examples, 8):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nTotal misclassified: {len(misclassified_indices)}/{len(y_test)} "
              f"({len(misclassified_indices)/len(y_test)*100:.2f}%)")
    else:
        print("No misclassified examples found!")

def get_model_recommendations():
    """
    Get recommendations for choosing transfer learning models
    """
    recommendations = {
        'EfficientNetB0': {
            'description': 'Good balance of accuracy and efficiency',
            'best_for': 'General purpose, limited computational resources',
            'pros': ['High accuracy', 'Efficient', 'Fast training'],
            'cons': ['Newer architecture, less studied']
        },
        'ResNet50': {
            'description': 'Popular and well-tested architecture',
            'best_for': 'General purpose, when stability is important',
            'pros': ['Well-established', 'Good performance', 'Lots of research'],
            'cons': ['Larger model size', 'More parameters']
        },
        'MobileNetV2': {
            'description': 'Lightweight model for mobile/edge deployment',
            'best_for': 'Mobile deployment, resource constraints',
            'pros': ['Very fast', 'Small size', 'Low memory usage'],
            'cons': ['Lower accuracy than larger models']
        },
        'VGG16': {
            'description': 'Simple and interpretable architecture',
            'best_for': 'Educational purposes, simple features',
            'pros': ['Simple architecture', 'Easy to understand'],
            'cons': ['Large model size', 'Slower training', 'Many parameters']
        },
        'DenseNet121': {
            'description': 'Efficient feature reuse architecture',
            'best_for': 'When feature reuse is important',
            'pros': ['Parameter efficient', 'Good gradient flow'],
            'cons': ['High memory usage during training']
        }
    }
    
    print("Transfer Learning Model Recommendations:")
    print("=" * 60)
    
    for model, info in recommendations.items():
        print(f"\n{model}:")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
    
    return recommendations

def create_transfer_learning_summary(model_name, history, test_accuracy, class_names,
                                   save_path="models/evaluation/transfer_learning_summary.txt"):
    """
    Create a summary report of transfer learning results
    
    Args:
        model_name (str): Name of the base model used
        history: Training history
        test_accuracy (float): Final test accuracy
        class_names (list): List of class names
        save_path (str): Path to save the summary
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Transfer Learning Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Base Model: {model_name}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n\n")
        
        if history:
            f.write("Training Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n")
            f.write(f"Training Epochs: {len(history.history['accuracy'])}\n\n")
        
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        
        # Add recommendations
        f.write("Recommendations:\n")
        f.write("-" * 20 + "\n")
        if test_accuracy >= 0.90:
            f.write("✓ Excellent performance! Model is ready for deployment.\n")
        elif test_accuracy >= 0.80:
            f.write("✓ Good performance. Consider fine-tuning for better results.\n")
        elif test_accuracy >= 0.70:
            f.write("⚠ Moderate performance. Try different base model or more training.\n")
        else:
            f.write("✗ Poor performance. Consider data augmentation or different approach.\n")
    
    print(f"Transfer learning summary saved to: {save_path}")

# Example usage functions
def run_quick_comparison():
    """
    Quick example of how to compare multiple transfer learning models
    """
    from src.preparation.data_loader import DataLoader
    from src.algorithm.transfer_learning_classifier import TransferLearningClassifier
    from sklearn.model_selection import train_test_split
    
    print("Running quick transfer learning model comparison...")
    
    # Load dataset
    data_loader = DataLoader("dataset", target_size=(224, 224))
    X, y = data_loader.load_dataset()
    class_names = data_loader.class_names
    
    # Use subset for quick comparison
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Models to compare
    models_to_test = ['EfficientNetB0', 'MobileNetV2', 'ResNet50']
    
    # Quick comparison
    classifier = TransferLearningClassifier(
        input_shape=(224, 224, 3),
        num_classes=len(class_names)
    )
    
    results = classifier.compare_models(
        models_to_test, X_train, y_train, X_val, y_val, epochs=5
    )
    
    # Visualize results
    compare_models_performance(results)
    
    return results

if __name__ == "__main__":
    # Show model recommendations
    get_model_recommendations()
    
    # Uncomment to run quick comparison
    # run_quick_comparison()