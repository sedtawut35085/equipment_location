import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone

class KFoldCrossValidator:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Initialize K-Fold Cross Validator
        
        Args:
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle data before splitting
            random_state (int): Random state for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    def cross_validate(self, model, X, y, scoring='accuracy'):
        """
        Perform k-fold cross-validation
        
        Args:
            model: Machine learning model with fit and predict methods
            X: Feature matrix
            y: Target vector
            scoring (str): Scoring metric ('accuracy' for classification, 'mse' for regression)
        
        Returns:
            dict: Results containing scores, mean, and std
        """
        scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model to avoid interference between folds
            model_clone = clone(model)
            
            # Train model
            model_clone.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model_clone.predict(X_val)
            
            # Calculate score
            if scoring == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif scoring == 'mse':
                score = mean_squared_error(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring}")
            
            scores.append(score)
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score
            })
            
            print(f"Fold {fold + 1}: {scoring} = {score:.4f}")
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_results': fold_results
        }
        
        return results
    
    def print_summary(self, results, scoring='accuracy'):
        """
        Print cross-validation summary
        
        Args:
            results (dict): Results from cross_validate method
            scoring (str): Scoring metric name
        """
        print(f"\n{'-'*50}")
        print(f"K-Fold Cross-Validation Summary ({self.n_splits} folds)")
        print(f"{'-'*50}")
        print(f"Mean {scoring}: {results['mean_score']:.4f}")
        print(f"Standard Deviation: {results['std_score']:.4f}")
        print(f"95% Confidence Interval: [{results['mean_score'] - 2*results['std_score']:.4f}, "
              f"{results['mean_score'] + 2*results['std_score']:.4f}]")

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Example 1: Classification
    print("Classification Example:")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_clf_scaled = scaler.fit_transform(X_clf)
    
    # Initialize model and cross-validator
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = KFoldCrossValidator(n_splits=5, random_state=42)
    
    # Perform cross-validation
    clf_results = cv.cross_validate(clf_model, X_clf_scaled, y_clf, scoring='accuracy')
    cv.print_summary(clf_results, 'accuracy')
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Regression
    print("Regression Example:")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    # Standardize features
    X_reg_scaled = scaler.fit_transform(X_reg)
    
    # Initialize model and cross-validator
    reg_model = LinearRegression()
    
    # Perform cross-validation
    reg_results = cv.cross_validate(reg_model, X_reg_scaled, y_reg, scoring='mse')
    cv.print_summary(reg_results, 'mse')