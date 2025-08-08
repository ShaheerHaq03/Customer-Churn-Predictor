"""
Helper utilities for the Customer Churn Prediction project.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def setup_logging(config) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger
    """
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format", 
                           "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get_logs_path()
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that data contains required columns.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes, False otherwise
    """
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    return True


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Confusion Matrix") -> go.Figure:
    """
    Create confusion matrix plot using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=400
    )
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   title: str = "ROC Curve") -> go.Figure:
    """
    Create ROC curve plot using Plotly.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=400
    )
    
    return fig


def plot_feature_importance(feature_names: List[str], 
                           importance_scores: np.ndarray,
                           title: str = "Feature Importance") -> go.Figure:
    """
    Create feature importance plot using Plotly.
    
    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]
    
    fig = go.Figure(data=go.Bar(
        x=sorted_scores,
        y=sorted_features,
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        width=600,
        height=max(400, len(feature_names) * 20)
    )
    
    return fig


def plot_class_distribution(data: pd.DataFrame, target_column: str,
                           title: str = "Class Distribution") -> go.Figure:
    """
    Create class distribution plot using Plotly.
    
    Args:
        data: DataFrame containing the data
        target_column: Name of the target column
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    class_counts = data[target_column].value_counts()
    
    fig = go.Figure(data=go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.3
    ))
    
    fig.update_layout(
        title=title,
        width=400,
        height=400
    )
    
    return fig


def plot_correlation_matrix(data: pd.DataFrame, 
                           title: str = "Correlation Matrix") -> go.Figure:
    """
    Create correlation matrix heatmap using Plotly.
    
    Args:
        data: DataFrame containing numerical data
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        width=600,
        height=500
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, config) -> str:
    """
    Save plot to file.
    
    Args:
        fig: Plotly figure object
        filename: Name of the file to save
        config: Configuration object
        
    Returns:
        Path to saved file
    """
    plots_path = config.get_plots_path()
    os.makedirs(plots_path, exist_ok=True)
    
    filepath = os.path.join(plots_path, filename)
    
    if filename.endswith('.html'):
        fig.write_html(filepath)
    elif filename.endswith('.png'):
        fig.write_image(filepath)
    else:
        fig.write_html(filepath + '.html')
    
    return filepath


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def get_model_performance_summary(metrics: Dict[str, float]) -> str:
    """
    Create a summary string of model performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        
    Returns:
        Formatted summary string
    """
    summary = f"""
    **Model Performance Summary:**
    
    - **Accuracy**: {format_percentage(metrics.get('accuracy', 0))}
    - **Precision**: {format_percentage(metrics.get('precision', 0))}
    - **Recall**: {format_percentage(metrics.get('recall', 0))}
    - **F1-Score**: {format_percentage(metrics.get('f1_score', 0))}
    """
    
    if 'roc_auc' in metrics:
        summary += f"- **ROC-AUC**: {metrics['roc_auc']:.3f}\n"
    
    return summary


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample telecom customer data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'tenure': np.random.exponential(30, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 25, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure age is within reasonable bounds
    df['age'] = df['age'].clip(18, 90)
    
    # Ensure tenure is positive
    df['tenure'] = df['tenure'].clip(1, 72)
    
    # Ensure charges are positive
    df['monthly_charges'] = df['monthly_charges'].clip(20, 150)
    df['total_charges'] = df['total_charges'].clip(100, 10000)
    
    # Generate churn based on some business logic
    churn_prob = (
        (df['tenure'] < 12) * 0.3 +
        (df['monthly_charges'] > 80) * 0.2 +
        (df['contract_type'] == 'Month-to-month') * 0.25 +
        (df['internet_service'] == 'Fiber optic') * 0.15 +
        (df['payment_method'] == 'Electronic check') * 0.1
    )
    
    df['churn'] = np.random.binomial(1, churn_prob)
    df['churn'] = df['churn'].map({0: 'No', 1: 'Yes'})
    
    return df
