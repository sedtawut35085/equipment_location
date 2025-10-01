import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from src.evaluation.evaluation import plot_confusion_matrix, plot_kfold_accuracies, plot_training_history, print_classification_metrics
from src.preparation.data_loader import DataLoader
from src.algorithm.cnn_classifier import CNNClassifier

def main():
    """
    Main function to train and evaluate CNN on meat classification dataset using K-Fold cross-validation
    """
    # Configuration
    DATASET_PATH = "dataset"
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    N_SPLITS = 5  # Number of folds

    print("="*60)
    print("MEAT CLASSIFICATION WITH CNN (K-FOLD)")
    print("="*60)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist!")
        print("Please make sure your dataset folder contains 'fresh' and 'rotten' subfolders")
        return

    data_loader = DataLoader(DATASET_PATH, target_size=TARGET_SIZE)

    results_dir = "models/evaluation"
    os.makedirs(results_dir, exist_ok=True)

    try:
        print("Loading dataset...")
        X, y = data_loader.load_dataset()
    
        class_names = data_loader.class_names

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

            cnn_model = CNNClassifier(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=2)
            cnn_model.compile_model(learning_rate=LEARNING_RATE)

            print("\nModel Architecture:")
            print("-" * 40)
            cnn_model.get_model_summary()

            print(f"\nTraining model for {EPOCHS} epochs...")
            history = cnn_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            print("\nPlotting training history...")
            plot_training_history(history)

            print("\nEvaluating model on test set...")
            test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
            all_fold_accuracies.append(test_accuracy)

            print(f"\nTest Results (Fold {fold}):")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            print("\nMaking predictions...")
            y_pred = cnn_model.predict_classes(X_test)

            fold_result_path = os.path.join(results_dir, f"confusion_matrix_{fold}.png")
            print_classification_metrics(y_test, y_pred, class_names)
            print("\nGenerating confusion matrix...")
            plot_confusion_matrix(y_test, y_pred, class_names, fold_result_path, fold)

            fold += 1

        print("\n" + "="*60)
        print("K-FOLD CROSS-VALIDATION RESULTS")
        print("="*60)
        print(f"Accuracies for each fold: {all_fold_accuracies}")
        accuracy_result_path = os.path.join(results_dir, f"kfold_accuracies_{fold}_fold.png")
        mean_accuracy = f"{np.mean(all_fold_accuracies):.4f}"
        plot_kfold_accuracies(all_fold_accuracies, accuracy_result_path, mean_accuracy)
        print(f"Mean Accuracy: {mean_accuracy})")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your dataset structure and file paths.")

    # If mean accuracy across folds is >= 80%, retrain on full dataset and save model
    if np.mean(all_fold_accuracies) >= 0.80:
        print("\nMean accuracy >= 80%. Training final model on full dataset...")

        # Split a small validation set from the full data for monitoring
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        final_model = CNNClassifier(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=2)
        final_model.compile_model(learning_rate=LEARNING_RATE)

        print("\nFinal Model Architecture:")
        print("-" * 40)
        final_model.get_model_summary()

        print(f"\nTraining final model for {EPOCHS} epochs...")
        final_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        # Save the trained model
        model_save_path = "models/cnn_model.h5"
        final_model.save_model(model_save_path)
        print(f"\nFinal model saved to {model_save_path}")
    else:
        print("\nMean accuracy < 80%. Final model will not be created.")

if __name__ == "__main__":
    # results_dir = "models"
    # os.makedirs(results_dir, exist_ok=True)
    # accuracy_result_path = os.path.join(results_dir, f"kfold_accuracies-{5}-fold.png")
    # all_fold_accuracies = [0.7723214030265808, 0.9776785969734192, 0.8923766613006592, 0.7354260087013245, 0.9192824959754944]
    # mean_accuracy = f"{np.mean(all_fold_accuracies):.4f}"
    # plot_kfold_accuracies(all_fold_accuracies, accuracy_result_path, mean_accuracy)
    main()
