"""
Model trainer for the Customer Churn Prediction project.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import get_config
from src.utils.helpers import setup_logging, calculate_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb


class ModelTrainer:
    """Model trainer class for churn prediction."""
    
    def __init__(self, config=None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = setup_logging(self.config)
        self.config.ensure_directories()
        
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.model_thresholds = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize all models with their configurations."""
        self.logger.info("Initializing models...")
        
        # Logistic Regression
        lr_params = self.config.get_model_params("logistic_regression")
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.config.get("models.random_state", 42),
            **lr_params
        )
        
        # Random Forest
        rf_params = self.config.get_model_params("random_forest")
        self.models['random_forest'] = RandomForestClassifier(
            random_state=self.config.get("models.random_state", 42),
            **rf_params
        )
        
        # XGBoost
        xgb_params = self.config.get_model_params("xgboost")
        self.models['xgboost'] = xgb.XGBClassifier(
            random_state=self.config.get("models.random_state", 42),
            **xgb_params
        )
        
        # Support Vector Machine
        svm_params = self.config.get_model_params("svm")
        self.models['svm'] = SVC(
            random_state=self.config.get("models.random_state", 42),
            probability=True,
            **svm_params
        )
        
        self.logger.info(f"Initialized {len(self.models)} models")
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of balanced features and target
        """
        if not self.config.get("models.use_smote", True):
            return X_train, y_train
        
        self.logger.info("Applying SMOTE to handle class imbalance...")
        
        # Check class distribution before SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        self.logger.info(f"Class distribution before SMOTE: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        smote = SMOTE(
            random_state=self.config.get("models.random_state", 42),
            sampling_strategy=self.config.get("models.smote_sampling_strategy", "auto")
        )
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Check class distribution after SMOTE
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        self.logger.info(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")
        
        return X_train_balanced, y_train_balanced
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            
        Returns:
            Dictionary with model performance metrics
        """
        self.initialize_models()
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred, y_prob)

                # Threshold optimization (optional)
                best_threshold = 0.5
                if y_prob is not None and self.config.get("evaluation.optimize_threshold", True):
                    best_threshold = self._find_best_threshold(
                        y_test, y_prob,
                        optimize_for=self.config.get("evaluation.optimize_for", "f1"),
                        threshold_range=self.config.get("evaluation.threshold_range", [0.1, 0.9]),
                        threshold_step=self.config.get("evaluation.threshold_step", 0.05)
                    )
                    self.model_thresholds[model_name] = best_threshold
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_balanced, y_train_balanced,
                    cv=self.config.get("models.cv_folds", 5),
                    scoring='f1'
                )
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'best_threshold': best_threshold,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
                
                self.trained_models[model_name] = model
                self.model_scores[model_name] = metrics['f1_score']
                
                self.logger.info(f"{model_name} - F1 Score: {metrics['f1_score']:.4f}, CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {
                    'model': None,
                    'metrics': {},
                    'cv_mean': 0,
                    'cv_std': 0,
                    'predictions': None,
                    'probabilities': None,
                    'error': str(e)
                }
        
        # Find best model
        self._find_best_model()
        
        return results

    def _find_best_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                             optimize_for: str = "f1",
                             threshold_range=None, threshold_step: float = 0.05) -> float:
        if threshold_range is None:
            threshold_range = [0.1, 0.9]
        thresholds = np.arange(threshold_range[0], threshold_range[1] + 1e-9, threshold_step)
        best_thr = 0.5
        best_score = -1.0
        for thr in thresholds:
            y_pred_thr = (y_prob >= thr).astype(int)
            if optimize_for == "accuracy":
                score = accuracy_score(y_true, y_pred_thr)
            elif optimize_for == "recall":
                score = recall_score(y_true, y_pred_thr, zero_division=0)
            elif optimize_for == "precision":
                score = precision_score(y_true, y_pred_thr, zero_division=0)
            else:
                score = f1_score(y_true, y_pred_thr, zero_division=0)
            if score > best_score:
                best_score = score
                best_thr = thr
        return float(best_thr)
    
    def _find_best_model(self):
        """Find the best performing model based on F1 score."""
        if not self.model_scores:
            return
        
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.trained_models[self.best_model_name]
        
        self.logger.info(f"Best model: {self.best_model_name} (F1 Score: {self.model_scores[self.best_model_name]:.4f})")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             model_name: str = 'random_forest') -> Dict:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to tune
            
        Returns:
            Dictionary with tuning results
        """
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }
        }
        
        if model_name not in param_grids:
            self.logger.error(f"No parameter grid defined for {model_name}")
            return {}
        
        # Initialize base model
        if model_name == 'logistic_regression':
            base_model = LogisticRegression(random_state=self.config.get("models.random_state", 42))
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=self.config.get("models.random_state", 42))
        elif model_name == 'xgboost':
            base_model = xgb.XGBClassifier(random_state=self.config.get("models.random_state", 42))
        elif model_name == 'svm':
            base_model = SVC(probability=True, random_state=self.config.get("models.random_state", 42))
        else:
            self.logger.error(f"Unknown model: {model_name}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[model_name],
            cv=self.config.get("models.cv_folds", 5),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Handle class imbalance for tuning
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Fit grid search
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        # Get results
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def save_models(self, filepath: Optional[str] = None) -> Dict[str, str]:
        """
        Save all trained models to files.
        
        Args:
            filepath: Base path for saving models
            
        Returns:
            Dictionary mapping model names to their file paths
        """
        def _dataset_models_path(base_path: str) -> str:
            raw_filename = self.config.get("data.raw_filename", "default")
            dataset_key = os.path.splitext(os.path.basename(raw_filename))[0].replace(" ", "_")
            if self.config.get("auto_mode", True):
                return os.path.join(base_path, dataset_key)
            return base_path

        if filepath is None:
            models_path = _dataset_models_path(self.config.get_model_path())
        else:
            models_path = _dataset_models_path(filepath)
        
        os.makedirs(models_path, exist_ok=True)
        
        saved_models = {}
        
        for model_name, model in self.trained_models.items():
            if model is not None:
                model_file = os.path.join(models_path, f"{model_name}.joblib")
                joblib.dump(model, model_file)
                saved_models[model_name] = model_file
                self.logger.info(f"Saved {model_name} to {model_file}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_file = os.path.join(models_path, "best_model.joblib")
            joblib.dump(self.best_model, best_model_file)
            saved_models['best_model'] = best_model_file
            self.logger.info(f"Saved best model ({self.best_model_name}) to {best_model_file}")

        # Save thresholds
        try:
            thresholds_path = os.path.join(models_path, "thresholds.json")
            payload = dict(self.model_thresholds)
            if self.best_model_name in self.model_thresholds:
                payload['best_model'] = self.model_thresholds[self.best_model_name]
            with open(thresholds_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            self.logger.info(f"Saved thresholds to {thresholds_path}")
        except Exception as e:
            self.logger.warning(f"Could not save thresholds: {e}")
        
        return saved_models
    
    def load_models(self, models_path: Optional[str] = None) -> Dict[str, object]:
        """
        Load trained models from files.
        
        Args:
            models_path: Path to the models directory
            
        Returns:
            Dictionary mapping model names to loaded models
        """
        def _dataset_models_path(base_path: str) -> str:
            raw_filename = self.config.get("data.raw_filename", "default")
            dataset_key = os.path.splitext(os.path.basename(raw_filename))[0].replace(" ", "_")
            if self.config.get("auto_mode", True):
                return os.path.join(base_path, dataset_key)
            return base_path

        if models_path is None:
            models_path = _dataset_models_path(self.config.get_model_path())
        
        loaded_models = {}
        
        # Load individual models
        for model_name in self.models.keys():
            model_file = os.path.join(models_path, f"{model_name}.joblib")
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                loaded_models[model_name] = model
                self.trained_models[model_name] = model
                self.logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load best model
        best_model_file = os.path.join(models_path, "best_model.joblib")
        if os.path.exists(best_model_file):
            best_model = joblib.load(best_model_file)
            loaded_models['best_model'] = best_model
            self.best_model = best_model
            self.logger.info(f"Loaded best model from {best_model_file}")

        # Load thresholds if present
        thresholds_path = os.path.join(models_path, "thresholds.json")
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r', encoding='utf-8') as f:
                    self.model_thresholds = json.load(f)
                self.logger.info(f"Loaded thresholds from {thresholds_path}")
            except Exception as e:
                self.logger.warning(f"Could not load thresholds: {e}")
        
        return loaded_models
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all trained models.
        
        Returns:
            Dictionary with model summaries
        """
        summary = {
            'total_models': len(self.trained_models),
            'best_model': self.best_model_name,
            'model_performance': {},
            'training_summary': {}
        }
        
        for model_name, model in self.trained_models.items():
            if model is not None:
                summary['model_performance'][model_name] = {
                    'f1_score': self.model_scores.get(model_name, 0),
                    'model_type': type(model).__name__
                }
        
        return summary
    
    def predict_with_best_model(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of predictions and probabilities
        """
        if self.best_model is None:
            raise ValueError("No best model available. Train models first.")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        return predictions, probabilities


def main():
    """Main function to run model training process."""
    config = get_config()
    trainer = ModelTrainer(config)
    
    # Load preprocessed data
    from src.features.feature_engineering import FeatureEngineer
    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader(config)
    processed_data_path = config.get_data_path("processed")
    train_file = os.path.join(processed_data_path, config.get("data.train_file", "train_data.csv"))
    test_file = os.path.join(processed_data_path, config.get("data.test_file", "test_data.csv"))
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Processed data files not found. Running data loading and feature engineering...")
        data = loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data(data)
        loader.save_processed_data(X_train, X_test, y_train, y_test)
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer(config)
        X_train_transformed, X_test_transformed = feature_engineer.fit_transform(
            X_train, X_test, y_train
        )
        feature_engineer.save_pipeline()
    else:
        # Load preprocessed data
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']
        
        # Load feature engineering pipeline
        feature_engineer = FeatureEngineer(config)
        feature_engineer.load_pipeline()
        
        # Transform data
        X_train_transformed = feature_engineer.transform_new_data(X_train)
        X_test_transformed = feature_engineer.transform_new_data(X_test)
    
    # Train models
    results = trainer.train_models(X_train_transformed, y_train, X_test_transformed, y_test)
    
    # Print results
    print("\nModel Training Results:")
    print("=" * 50)
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  CV F1-Score: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        else:
            print(f"\n{model_name.upper()}: ERROR - {result['error']}")
    
    # Save models
    trainer.save_models()
    
    # Get model summary
    summary = trainer.get_model_summary()
    print(f"\nTraining Summary:")
    print(f"Total models trained: {summary['total_models']}")
    print(f"Best model: {summary['best_model']}")
    
    print("\nModel training process completed successfully!")


if __name__ == "__main__":
    main()
