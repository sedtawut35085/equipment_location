import os
import numpy as np
import matplotlib.pyplot as plt
from src.algorithm.cnn_classifier import CNNClassifier
from src.preparation.data_loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, fold=None):
    """
    Plot confusion matrix and optionally save the plot.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
        save_path (str, optional): If provided, save the plot to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    title = f'Confusion Matrix'
    if fold is not None:
        title += f' Fold {fold}'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    plt.show()

def print_classification_metrics(y_true, y_pred, class_names):
    """
    Print detailed classification metrics
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_kfold_accuracies(accuracies, save_path=None, mean_accuracy=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'K-Fold Accuracies ({mean_accuracy if mean_accuracy else ""})')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(range(1, len(accuracies) + 1))
    # Annotate each point with its accuracy value
    for i, acc in enumerate(accuracies):
        plt.text(i + 1, acc - 0.05, f"({i+1}, {acc:.2f})", ha='center', va='top', fontsize=10, color='black')
    if save_path:
        plt.savefig(save_path)
        print(f"K-Fold accuracy plot saved to {save_path}")
    plt.show()

def main():
    """
    Main function to train and evaluate CNN on meat classification dataset
    """
    # Configuration
    DATASET_PATH = "dataset"  # Path to your dataset folder
    TARGET_SIZE = (128, 128)  # Image size for CNN
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print("="*60)
    print("MEAT CLASSIFICATION WITH CNN")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist!")
        print("Please make sure your dataset folder contains 'fresh' and 'rotten' subfolders")
        return
    
    # Initialize data loader
    data_loader = DataLoader(DATASET_PATH, target_size=TARGET_SIZE)
    
    try:
        # Load dataset
        print("Loading dataset...")
        X, y = data_loader.load_dataset()
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = data_loader.preprocess_for_training(
            X, y, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Initialize CNN model
        print("\nInitializing CNN model...")
        cnn_model = CNNClassifier(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=2)
        cnn_model.compile_model(learning_rate=LEARNING_RATE)
        
        # Print model summary
        print("\nModel Architecture:")
        print("-" * 40)
        cnn_model.get_model_summary()
        
        # Train model
        print(f"\nTraining model for {EPOCHS} epochs...")
        history = cnn_model.train(
            X_train, y_train, 
            X_val, y_val, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE
        )
        
        # Plot training history
        print("\nPlotting training history...")
        plot_training_history(history)
        
        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = cnn_model.predict_classes(X_test)
        
        # Print detailed metrics
        print_classification_metrics(y_test, y_pred, data_loader.class_names)
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(y_test, y_pred, data_loader.class_names)
        
        # Save model
        model_path = "results/meat_classifier_cnn.h5"
        os.makedirs("results", exist_ok=True)
        cnn_model.save_model(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Example predictions on a few test images
        print("\nSample predictions:")
        print("-" * 40)
        num_samples = min(20, len(X_test))
        sample_predictions = cnn_model.predict(X_test[:num_samples])
        
        for i in range(num_samples):
            true_class = data_loader.class_names[y_test[i]]
            pred_class = data_loader.class_names[np.argmax(sample_predictions[i])]
            confidence = np.max(sample_predictions[i])
            
            print(f"Sample {i+1}: True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.4f}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your dataset structure and file paths.")

if __name__ == "__main__":
    main()
