import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from src.evaluation.evaluation import plot_confusion_matrix, plot_kfold_accuracies, plot_training_history, print_classification_metrics
from src.preparation.data_loader import DataLoader
from src.algorithm.transfer_learning_classifier import TransferLearningClassifier

from config import CLASS_NAMES

def main():
    """
    Main function to train and evaluate Transfer Learning models on equipment classification dataset using K-Fold cross-validation
    """
    # Configuration
    DATASET_PATH = "dataset"
    TARGET_SIZE = (224, 224)  # Larger size for better transfer learning performance
    BATCH_SIZE = 16  # Smaller batch size for transfer learning
    EPOCHS = 30  # Fewer epochs needed with transfer learning
    LEARNING_RATE = 0.0001  # Lower learning rate for transfer learning
    N_SPLITS = 5  # Number of folds
    
    # Transfer Learning Configuration
    BASE_MODEL = 'MobileNetV2'  # You can change this to other models
    FINE_TUNE = True  # Whether to perform fine-tuning
    FINE_TUNE_EPOCHS = 15
    FINE_TUNE_LAYERS = 50
    
    # Available models to compare (uncomment to compare multiple models)
    MODELS_TO_COMPARE = [
        'EfficientNetB0',
        'MobileNetV2', 
        'ResNet50',
        'VGG16'
    ]

    print("="*70)
    print("EQUIPMENT CLASSIFICATION WITH TRANSFER LEARNING (K-FOLD)")
    print("="*70)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist!")
        print("Please make sure your dataset folder contains subfolders for each class")
        return

    data_loader = DataLoader(DATASET_PATH, target_size=TARGET_SIZE, class_names=CLASS_NAMES)
    results_dir = "models/evaluation"
    os.makedirs(results_dir, exist_ok=True)

    try:
        print("Loading dataset...")
        X, y = data_loader.load_dataset()
        class_names = data_loader.class_names
        print(f"Dataset loaded: {len(X)} images with {len(class_names)} classes")
        print(f"Classes: {class_names}")

        # Option 1: Train single model with K-Fold
        print(f"\n{'='*20} Training {BASE_MODEL} with K-Fold {'='*20}")
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold = 1
        all_fold_accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            print(f"\n{'='*20} Fold {fold} {'='*20}")

            X_trainval, X_test = X[train_idx], X[test_idx]
            y_trainval, y_test = y[train_idx], y[test_idx]

            # Split trainval into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.2, random_state=fold, stratify=y_trainval
            )

            # Create transfer learning model
            tl_model = TransferLearningClassifier(
                input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), 
                num_classes=len(class_names),
                base_model_name=BASE_MODEL
            )
            
            # Compile with frozen base model
            tl_model.compile_model(learning_rate=LEARNING_RATE, freeze_base=True)

            print("\nTransfer Learning Model Architecture:")
            print("-" * 50)
            tl_model.get_model_summary()

            print(f"\nTraining transfer learning model for {EPOCHS} epochs (frozen base)...")
            history = tl_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                freeze_base=True
            )

            # Fine-tuning phase
            if FINE_TUNE:
                print(f"\nFine-tuning for {FINE_TUNE_EPOCHS} epochs...")
                fine_tune_history = tl_model.fine_tune(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=FINE_TUNE_EPOCHS,
                    batch_size=BATCH_SIZE//2,  # Smaller batch size for fine-tuning
                    fine_tune_layers=FINE_TUNE_LAYERS,
                    learning_rate=LEARNING_RATE/10
                )

            print("\nPlotting training history...")
            plot_training_history(history)

            print("\nEvaluating model on test set...")
            test_loss, test_accuracy = tl_model.evaluate(X_test, y_test)
            all_fold_accuracies.append(test_accuracy)

            print(f"\nTest Results (Fold {fold}):")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            print("\nMaking predictions...")
            y_pred = tl_model.predict_classes(X_test)

            fold_result_path = os.path.join(results_dir, f"confusion_matrix_tl_{fold}.png")
            print_classification_metrics(y_test, y_pred, class_names)
            print("\nGenerating confusion matrix...")
            plot_confusion_matrix(y_test, y_pred, class_names, fold_result_path, fold)

            fold += 1

        print("\n" + "="*70)
        print("K-FOLD CROSS-VALIDATION RESULTS (TRANSFER LEARNING)")
        print("="*70)
        print(f"Base Model: {BASE_MODEL}")
        print(f"Accuracies for each fold: {all_fold_accuracies}")
        accuracy_result_path = os.path.join(results_dir, f"kfold_accuracies_tl_{N_SPLITS}_fold.png")
        mean_accuracy = np.mean(all_fold_accuracies)
        plot_kfold_accuracies(all_fold_accuracies, accuracy_result_path, f"{mean_accuracy:.4f}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Standard Deviation: {np.std(all_fold_accuracies):.4f}")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your dataset structure and file paths.")
        import traceback
        traceback.print_exc()

    # If mean accuracy across folds is >= 85%, retrain on full dataset and save model
    if np.mean(all_fold_accuracies) >= 0.85:
        print(f"\nMean accuracy >= 85%. Training final transfer learning model on full dataset...")

        # Split a small validation set from the full data for monitoring
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        final_model = TransferLearningClassifier(
            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), 
            num_classes=len(class_names),
            base_model_name=BASE_MODEL
        )
        
        final_model.compile_model(learning_rate=LEARNING_RATE, freeze_base=True)

        print("\nFinal Transfer Learning Model Architecture:")
        print("-" * 50)
        final_model.get_model_summary()

        print(f"\nTraining final model for {EPOCHS} epochs...")
        final_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            freeze_base=True
        )

        # Fine-tune final model
        if FINE_TUNE:
            print(f"\nFine-tuning final model for {FINE_TUNE_EPOCHS} epochs...")
            final_model.fine_tune(
                X_train, y_train,
                X_val, y_val,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=BATCH_SIZE//2,
                fine_tune_layers=FINE_TUNE_LAYERS,
                learning_rate=LEARNING_RATE/10
            )

        # Save the trained model
        model_save_path = f"models/transfer_learning_{BASE_MODEL.lower()}_model.h5"
        final_model.save_model(model_save_path)
        print(f"\nFinal transfer learning model saved to {model_save_path}")
    else:
        print(f"\nMean accuracy < 85%. Final model will not be created.")

def compare_models():
    """
    Compare different transfer learning models
    """
    print("="*70)
    print("COMPARING TRANSFER LEARNING MODELS")
    print("="*70)
    
    DATASET_PATH = "dataset"
    TARGET_SIZE = (224, 224)
    COMPARISON_EPOCHS = 10  # Quick comparison
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist!")
        return

    data_loader = DataLoader(DATASET_PATH, target_size=TARGET_SIZE, class_names=CLASS_NAMES)
    
    try:
        print("Loading dataset for model comparison...")
        X, y = data_loader.load_dataset()
        class_names = data_loader.class_names
        
        # Use a subset for quick comparison
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        models_to_test = ['EfficientNetB0', 'MobileNetV2', 'ResNet50', 'VGG16']
        
        comparison_classifier = TransferLearningClassifier(
            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
            num_classes=len(class_names)
        )
        
        results = comparison_classifier.compare_models(
            models_to_test, X_train, y_train, X_val, y_val, 
            epochs=COMPARISON_EPOCHS
        )
        
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"{model_name}: {result['val_accuracy']:.4f}")
            else:
                print(f"{model_name}: Error - {result['error']}")
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results, key=lambda x: valid_results[x]['val_accuracy'])
            print(f"\nBest performing model: {best_model} "
                  f"(Val Accuracy: {valid_results[best_model]['val_accuracy']:.4f})")
        
    except Exception as e:
        print(f"Error in model comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Uncomment the following line to run model comparison first
    # compare_models()
    
    # Run main transfer learning training
    main()