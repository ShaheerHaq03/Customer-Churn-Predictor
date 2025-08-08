"""
Feature engineering for the Customer Churn Prediction project.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import sys
import joblib

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import get_config
from src.utils.helpers import setup_logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    """Feature engineering class for telecom customer data."""
    
    def __init__(self, config=None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = setup_logging(self.config)
        self.config.ensure_directories()
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.preprocessing_pipeline = None
        
        # Get column configurations; when auto_mode, infer from data dynamically at fit time
        self.categorical_columns = self.config.get_feature_columns("categorical")
        self.numerical_columns = self.config.get_feature_columns("numerical")
        
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit transformers and transform training and test data.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target (for feature selection)
            
        Returns:
            Tuple of transformed training and test arrays
        """
        self.logger.info("Starting feature engineering process...")
        
        # Auto-infer schema if enabled, or if configured columns are missing in the current dataset
        if self.config.get("auto_mode", True):
            inferred_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
            inferred_cat = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            self.numerical_columns = inferred_num
            self.categorical_columns = inferred_cat
            self.logger.info(
                f"Schema set for dataset - numerical: {len(inferred_num)}, categorical: {len(inferred_cat)}"
            )

        # Create preprocessing pipeline
        self._create_preprocessing_pipeline()
        
        # Fit and transform training data (without y parameter for LabelEncoder)
        X_train_transformed = self.preprocessing_pipeline.fit_transform(X_train)
        
        # Transform test data
        X_test_transformed = self.preprocessing_pipeline.transform(X_test)
        
        # Apply feature selection if needed
        feature_selection_threshold = self.config.get("features.feature_selection_threshold", 0.0)
        if feature_selection_threshold and feature_selection_threshold > 0:
            self.feature_selector = SelectKBest(score_func=f_classif, k='all')
            X_train_transformed = self.feature_selector.fit_transform(X_train_transformed, y_train)
            X_test_transformed = self.feature_selector.transform(X_test_transformed)
        
        self.logger.info(f"Feature engineering completed. Final shapes:")
        self.logger.info(f"  Training: {X_train_transformed.shape}")
        self.logger.info(f"  Testing: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed
    
    def _create_preprocessing_pipeline(self):
        """Create the preprocessing pipeline."""
        # Numerical features preprocessing
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical features preprocessing (use most_frequent to avoid type mixing)
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self._get_encoder())
        ]) if self.categorical_columns else None
        
        # Combine transformers
        preprocessors = []
        
        if self.numerical_columns:
            preprocessors.append(('num', numerical_transformer, self.numerical_columns))
        
        if self.categorical_columns:
            preprocessors.append(('cat', categorical_transformer, self.categorical_columns))
        
        # Create column transformer
        column_transformer = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'
        )
        
        # Build final pipeline without feature selection first
        self.preprocessing_pipeline = Pipeline([
            ('preprocessor', column_transformer)
        ])
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        scaling_method = self.config.get("features.scaling_method", "standard")
        
        if scaling_method == "standard":
            return StandardScaler()
        elif scaling_method == "minmax":
            return MinMaxScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_method}. Using StandardScaler.")
            return StandardScaler()
    
    def _get_encoder(self):
        """Get the appropriate encoder based on configuration."""
        encoding_method = self.config.get("features.encoding_method", "label")
        
        if encoding_method == "label":
            # For ColumnTransformer, use OneHotEncoder for robust handling; LabelEncoder is not column-wise.
            return OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        elif encoding_method == "onehot":
            return OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        else:
            self.logger.warning(f"Unknown encoding method: {encoding_method}. Using LabelEncoder.")
            return OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the final features after preprocessing.
        
        Returns:
            List of feature names
        """
        if self.preprocessing_pipeline is None:
            return []
        
        # Get feature names from the pipeline
        feature_names = []
        
        # Add numerical features
        if self.numerical_columns:
            feature_names.extend(self.numerical_columns)
        
        # Add categorical features
        if self.categorical_columns:
            try:
                encoder = self.preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                
                if hasattr(encoder, 'get_feature_names_out'):
                    # OneHotEncoder
                    cat_feature_names = encoder.get_feature_names_out(self.categorical_columns)
                    feature_names.extend(cat_feature_names)
                else:
                    # LabelEncoder - one feature per categorical column
                    feature_names.extend(self.categorical_columns)
            except KeyError:
                # If categorical transformer is not found, just add the original column names
                feature_names.extend(self.categorical_columns)
        
        return feature_names
    
    def get_feature_importance_scores(self) -> np.ndarray:
        """
        Get feature importance scores from the feature selector.
        
        Returns:
            Array of feature importance scores
        """
        if self.feature_selector is None:
            return np.array([])
        
        return self.feature_selector.scores_
    
    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted pipeline.
        
        Args:
            X: New data to transform
            
        Returns:
            Transformed data array
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        # Transform using preprocessing pipeline
        X_transformed = self.preprocessing_pipeline.transform(X)
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        return X_transformed
    
    def save_pipeline(self, filepath: Optional[str] = None) -> str:
        """
        Save the preprocessing pipeline to file.
        
        Args:
            filepath: Path to save the pipeline
            
        Returns:
            Path where pipeline was saved
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("No pipeline to save. Call fit_transform first.")
        
        def _dataset_models_path(base_path: str) -> str:
            raw_filename = self.config.get("data.raw_filename", "default")
            dataset_key = os.path.splitext(os.path.basename(raw_filename))[0].replace(" ", "_")
            if self.config.get("auto_mode", True):
                return os.path.join(base_path, dataset_key)
            return base_path

        if filepath is None:
            models_path = _dataset_models_path(self.config.get_model_path())
            filepath = os.path.join(models_path, "preprocessing_pipeline.joblib")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save both preprocessing pipeline and feature selector
        pipeline_data = {
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'feature_selector': self.feature_selector
        }
        joblib.dump(pipeline_data, filepath)
        
        self.logger.info(f"Preprocessing pipeline saved to {filepath}")
        return filepath
    
    def load_pipeline(self, filepath: Optional[str] = None) -> None:
        """
        Load the preprocessing pipeline from file.
        
        Args:
            filepath: Path to the pipeline file
        """
        def _dataset_models_path(base_path: str) -> str:
            raw_filename = self.config.get("data.raw_filename", "default")
            dataset_key = os.path.splitext(os.path.basename(raw_filename))[0].replace(" ", "_")
            if self.config.get("auto_mode", True):
                return os.path.join(base_path, dataset_key)
            return base_path

        if filepath is None:
            models_path = _dataset_models_path(self.config.get_model_path())
            filepath = os.path.join(models_path, "preprocessing_pipeline.joblib")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        
        # Handle both new format (dict) and old format (single pipeline)
        if isinstance(pipeline_data, dict):
            self.preprocessing_pipeline = pipeline_data['preprocessing_pipeline']
            self.feature_selector = pipeline_data.get('feature_selector')
        else:
            # Backward compatibility with old format
            self.preprocessing_pipeline = pipeline_data
            self.feature_selector = None
        self.logger.info(f"Preprocessing pipeline loaded from {filepath}")
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of the preprocessing steps.
        
        Returns:
            Dictionary with preprocessing summary
        """
        if self.preprocessing_pipeline is None:
            return {}
        
        summary = {
            'numerical_features': self.numerical_columns,
            'categorical_features': self.categorical_columns,
            'scaling_method': self.config.get("features.scaling_method", "standard"),
            'encoding_method': self.config.get("features.encoding_method", "label"),
            'final_feature_count': len(self.get_feature_names()),
            'feature_importance_scores': self.get_feature_importance_scores().tolist()
        }
        
        return summary
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between numerical variables.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        X_interactions = X.copy()
        
        # Create interaction features for numerical columns
        numerical_cols = [col for col in self.numerical_columns if col in X.columns]
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                X_interactions[interaction_name] = X[col1] * X[col2]
                
                # Ratio interaction (avoid division by zero)
                ratio_name = f"{col1}_div_{col2}"
                X_interactions[ratio_name] = np.where(
                    X[col2] != 0, X[col1] / X[col2], 0
                )
        
        self.logger.info(f"Created {len(X_interactions.columns) - len(X.columns)} interaction features")
        return X_interactions
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        X_cleaned = X.copy()
        
        # Handle numerical missing values
        for col in self.numerical_columns:
            if col in X.columns and X[col].isnull().any():
                median_val = X[col].median()
                X_cleaned[col].fillna(median_val, inplace=True)
                self.logger.info(f"Filled {X[col].isnull().sum()} missing values in {col} with median")
        
        # Handle categorical missing values
        for col in self.categorical_columns:
            if col in X.columns and X[col].isnull().any():
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X_cleaned[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Filled {X[col].isnull().sum()} missing values in {col} with mode")
        
        return X_cleaned


def main():
    """Main function to run feature engineering process."""
    config = get_config()
    feature_engineer = FeatureEngineer(config)
    
    # Load data
    from src.data.data_loader import DataLoader
    loader = DataLoader(config)
    
    # Load processed data
    processed_data_path = config.get_data_path("processed")
    train_file = os.path.join(processed_data_path, config.get("data.train_file", "train_data.csv"))
    test_file = os.path.join(processed_data_path, config.get("data.test_file", "test_data.csv"))
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Processed data files not found. Running data loading first...")
        data = loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data(data)
        loader.save_processed_data(X_train, X_test, y_train, y_test)
    else:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']
    
    # Apply feature engineering
    X_train_transformed, X_test_transformed = feature_engineer.fit_transform(
        X_train, X_test, y_train
    )
    
    # Get preprocessing summary
    summary = feature_engineer.get_preprocessing_summary()
    print("\nFeature Engineering Summary:")
    print(f"Numerical features: {len(summary.get('numerical_features', []))}")
    print(f"Categorical features: {len(summary.get('categorical_features', []))}")
    print(f"Final feature count: {summary.get('final_feature_count', 0)}")
    print(f"Scaling method: {summary.get('scaling_method', 'N/A')}")
    print(f"Encoding method: {summary.get('encoding_method', 'N/A')}")
    
    # Save pipeline
    feature_engineer.save_pipeline()
    
    print("\nFeature engineering process completed successfully!")


if __name__ == "__main__":
    main()
