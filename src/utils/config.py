"""
Configuration management utilities for the Customer Churn Prediction project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class to manage project settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_path(self, path_type: str = "raw") -> str:
        """
        Get data path by type.
        
        Args:
            path_type: Type of data path ('raw', 'processed', 'external')
            
        Returns:
            Full path to the data directory
        """
        base_path = self.get(f"data.{path_type}_data_path", f"data/{path_type}/")
        return os.path.join(os.getcwd(), base_path)
    
    def get_model_path(self) -> str:
        """Get model storage path."""
        models_path = self.get("output.models_path", "models/")
        return os.path.join(os.getcwd(), models_path)
    
    def get_reports_path(self) -> str:
        """Get reports storage path."""
        reports_path = self.get("output.reports_path", "reports/")
        return os.path.join(os.getcwd(), reports_path)
    
    def get_plots_path(self) -> str:
        """Get plots storage path."""
        plots_path = self.get("output.plots_path", "reports/figures/")
        return os.path.join(os.getcwd(), plots_path)
    
    def get_logs_path(self) -> str:
        """Get logs storage path."""
        logs_file = self.get("logging.file", "logs/app.log")
        return os.path.join(os.getcwd(), logs_file)
    
    def ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.get_data_path("raw"),
            self.get_data_path("processed"),
            self.get_data_path("external"),
            self.get_model_path(),
            self.get_reports_path(),
            self.get_plots_path(),
            os.path.dirname(self.get_logs_path())
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameters by model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of model parameters
        """
        return self.get(f"models.{model_name}", {})
    
    def get_feature_columns(self, column_type: str = "categorical") -> list:
        """
        Get feature columns by type.
        
        Args:
            column_type: Type of columns ('categorical' or 'numerical')
            
        Returns:
            List of column names
        """
        return self.get(f"features.{column_type}_columns", [])
    
    def get_target_column(self) -> str:
        """Get target column name."""
        return self.get("features.target_column", "churn")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config
