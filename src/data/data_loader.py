"""
Data loader for the Customer Churn Prediction project.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import get_config
from src.utils.helpers import generate_sample_data, setup_logging


class DataLoader:
    """Data loader class for handling telecom customer data."""
    
    def __init__(self, config=None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = setup_logging(self.config)
        self.config.ensure_directories()
        
    def _discover_raw_csv(self) -> Optional[str]:
        """
        Discover a CSV file in the raw data directory if configured filename is missing.
        Picks the most recently modified CSV when multiple exist.
        """
        raw_data_path = self.config.get_data_path("raw")
        # Honor configured filename first
        configured = self.config.get("data.raw_filename")
        if configured:
            candidate = os.path.join(raw_data_path, configured)
            if os.path.exists(candidate):
                return candidate

        # Fallback to discovering any CSV in raw dir
        if not os.path.isdir(raw_data_path):
            return None

        csv_files = [
            os.path.join(raw_data_path, f)
            for f in os.listdir(raw_data_path)
            if f.lower().endswith(".csv")
        ]
        if not csv_files:
            return None

        # Pick most recent CSV
        csv_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return csv_files[0]

    def _detect_target_column(self, data: pd.DataFrame) -> str:
        """
        Detect the target column heuristically for churn datasets.
        Preference order: known names, binary columns with typical names, last binary column, else raise.
        """
        # Common churn-like column names
        candidate_names = [
            "churn", "exited", "target", "label", "churn_flag", "is_churn"
        ]
        lower_cols = {c.lower(): c for c in data.columns}
        for name in candidate_names:
            if name in lower_cols:
                return lower_cols[name]

        # Find binary columns (exactly two unique values ignoring NaN)
        binary_cols = []
        for col in data.columns:
            uniques = data[col].dropna().unique()
            if len(uniques) == 2:
                binary_cols.append(col)

        # Prefer binary columns whose names hint churn/exit
        hint_substrings = ["churn", "exit", "left", "quit", "cancel", "resign", "churned"]
        for col in binary_cols:
            if any(hint in col.lower() for hint in hint_substrings):
                return col

        # Fallback: take the last binary column if exists
        if binary_cols:
            return binary_cols[-1]

        raise ValueError(
            "Unable to detect target column automatically. "
            "Please specify 'features.target_column' in config.yaml."
        )

    def generate_sample_dataset(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate sample telecom customer dataset.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample data
        """
        if n_samples is None:
            n_samples = self.config.get("data.sample_size", 10000)
            
        self.logger.info(f"Generating {n_samples} sample records...")
        
        data = generate_sample_data(n_samples)
        
        # Save to raw data directory
        raw_data_path = self.config.get_data_path("raw")
        raw_filename = self.config.get("data.raw_filename", "telecom_customer_data.csv")
        filepath = os.path.join(raw_data_path, raw_filename)
        data.to_csv(filepath, index=False)
        
        self.logger.info(f"Sample dataset saved to {filepath}")
        self.logger.info(f"Dataset shape: {data.shape}")
        self.logger.info(f"Churn distribution:\n{data['churn'].value_counts()}")
        
        return data
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            DataFrame with loaded data
        """
        if filepath is None:
            # Try configured filename; else discover
            discovered = self._discover_raw_csv()
            filepath = discovered if discovered is not None else os.path.join(
                self.config.get_data_path("raw"),
                self.config.get("data.raw_filename", "telecom_customer_data.csv")
            )
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Data file not found: {filepath}")
            self.logger.info("Generating sample dataset...")
            return self.generate_sample_dataset()
        
        self.logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        
        # Normalize dtypes: cast categorical-like columns to string to avoid mixed encodings
        for col in data.columns:
            # If column contains both strings and numbers, cast to string uniformly
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str)
            else:
                # If mixed types detected (object-like numerics), coerce to string
                try:
                    # Heuristic: if there are any non-numeric values in a numeric-looking column
                    if data[col].apply(lambda v: isinstance(v, str)).any():
                        data[col] = data[col].astype(str)
                except Exception:
                    pass
        
        self.logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    
    def split_data(self, data: pd.DataFrame,
                   target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            data: DataFrame to split
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Drop obvious identifier columns
        columns_to_drop = []
        for col in list(data.columns):
            lower = col.lower()
            if lower in {"rownumber", "customerid", "surname"} or "id" in lower:
                columns_to_drop.append(col)
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop, axis=1, errors='ignore')

        if target_column is None:
            if self.config.get("auto_mode", True):
                target_column = self._detect_target_column(data)
            else:
                target_column = self.config.get_target_column()
        
        # Ensure target exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Convert target to binary (handle both 'Yes'/'No', True/False, and 0/1 formats)
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
        elif y.dtype == 'bool':
            y = y.astype(int)
        else:
            y = y.astype(int)
        
        # Split data
        test_size = self.config.get("models.test_size", 0.2)
        random_state = self.config.get("models.random_state", 42)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        self.logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        self.logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> None:
        """
        Save processed data to files.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        """
        processed_data_path = self.config.get_data_path("processed")
        
        # Save training data
        train_data = X_train.copy()
        train_data['churn'] = y_train
        train_file = os.path.join(processed_data_path, self.config.get("data.train_file", "train_data.csv"))
        train_data.to_csv(train_file, index=False)
        
        # Save testing data
        test_data = X_test.copy()
        test_data['churn'] = y_test
        test_file = os.path.join(processed_data_path, self.config.get("data.test_file", "test_data.csv"))
        test_data.to_csv(test_file, index=False)
        
        self.logger.info(f"Processed data saved:")
        self.logger.info(f"  Training data: {train_file}")
        self.logger.info(f"  Testing data: {test_file}")
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numerical_summary': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical columns summary
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            summary['categorical_summary'][col] = data[col].value_counts().to_dict()
        
        return summary
    
    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """
        Validate data quality and return issues.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        issues = {
            'missing_values': {},
            'duplicates': 0,
            'outliers': {},
            'data_types': {}
        }
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        issues['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for duplicates
        issues['duplicates'] = data.duplicated().sum()
        
        # Check for outliers in numerical columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                issues['outliers'][col] = outliers
        
        # Check data types
        for col in data.columns:
            issues['data_types'][col] = str(data[col].dtype)
        
        return issues


def main():
    """Main function to run data loading process."""
    config = get_config()
    loader = DataLoader(config)
    
    # Generate or load data
    data = loader.load_data()
    
    # Get data summary
    summary = loader.get_data_summary(data)
    print("\nData Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Missing values: {summary['missing_values']}")
    
    # Validate data quality
    issues = loader.validate_data_quality(data)
    print("\nData Quality Issues:")
    print(f"Duplicates: {issues['duplicates']}")
    print(f"Outliers: {issues['outliers']}")
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(data)
    
    # Save processed data
    loader.save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\nData loading process completed successfully!")


if __name__ == "__main__":
    main()
