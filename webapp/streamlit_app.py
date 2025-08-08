"""
Streamlit web application for Customer Churn Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import joblib
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import get_config
from src.utils.helpers import (
    generate_sample_data, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_class_distribution, plot_correlation_matrix,
    calculate_metrics, format_percentage, get_model_performance_summary
)
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from run_pipeline import run_data_pipeline, run_feature_engineering, run_model_training


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .prediction-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(selected_filename: str):
    """Load and cache the dataset for a specific selected file."""
    config = get_config()
    # Point config to the selected dataset
    config.config.setdefault('data', {})
    config.config['data']['raw_filename'] = selected_filename
    loader = DataLoader(config)
    data = loader.load_data()
    return data


@st.cache_resource
def load_models(selected_filename: str):
    """Load and cache the trained models."""
    config = get_config()
    # Point config to the selected dataset
    config.config.setdefault('data', {})
    config.config['data']['raw_filename'] = selected_filename
    trainer = ModelTrainer(config)
    models = trainer.load_models()
    # Auto-train if models not present
    if 'best_model' not in models or models.get('best_model') is None:
        run_data_pipeline()
        run_feature_engineering()
        run_model_training()
        models = trainer.load_models()
    return models


@st.cache_resource
def load_feature_engineer(selected_filename: str):
    """Load and cache the feature engineering pipeline."""
    config = get_config()
    # Point config to the selected dataset
    config.config.setdefault('data', {})
    config.config['data']['raw_filename'] = selected_filename
    feature_engineer = FeatureEngineer(config)
    try:
        feature_engineer.load_pipeline()
    except FileNotFoundError:
        # Auto-generate pipeline then load
        run_data_pipeline()
        run_feature_engineering()
        feature_engineer.load_pipeline()
    return feature_engineer


def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("📊 Customer Churn Prediction")
    
    # Dataset selection
    config = get_config()
    raw_dir = config.get_data_path("raw")
    csv_files = []
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith('.csv'):
                csv_files.append(f)
    csv_files = sorted(csv_files)
    if not csv_files:
        st.sidebar.warning("No CSV files found in data/raw/. Add your dataset to proceed.")
        selected_file = config.get("data.raw_filename", "")
    else:
        # Default to the most recent file by mtime
        files_with_mtime = [(f, os.path.getmtime(os.path.join(raw_dir, f))) for f in csv_files]
        files_with_mtime.sort(key=lambda x: x[1], reverse=True)
        default_index = csv_files.index(files_with_mtime[0][0])
        selected_file = st.sidebar.selectbox("Dataset (from data/raw)", options=csv_files, index=default_index)
        # Progress indicator when user switches dataset
        st.sidebar.progress(0, text="Ready")
        st.sidebar.caption(f"Using dataset: {selected_file}")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📈 Data Exploration", "🤖 Model Performance", "🔮 Predictions", "📊 Feature Analysis"]
    )
    
    # Load data and models with visible progress
    try:
        with st.spinner("Loading dataset..."):
            data = load_data(selected_file)
        st.toast(f"Dataset '{selected_file}' loaded", icon="✅")
        with st.spinner("Preparing preprocessing pipeline..."):
            feature_engineer = load_feature_engineer(selected_file)
        st.toast("Preprocessing ready", icon="⚙️")
        with st.spinner("Loading/Training models (first run may take a few minutes)..."):
            models = load_models(selected_file)
        st.toast("Models ready", icon="🤖")
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.info("Tip: Place a CSV in data/raw/ and run the app again. The pipeline will auto-detect schema and train if needed: `python run_pipeline.py --step all`.")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📈 Data Exploration":
        show_data_exploration_page(data)
    elif page == "🤖 Model Performance":
        show_model_performance_page(data, models, feature_engineer)
    elif page == "🔮 Predictions":
        show_predictions_page(data, models, feature_engineer)
    elif page == "📊 Feature Analysis":
        show_feature_analysis_page(data, models, feature_engineer)


def show_home_page(data):
    """Display the home page."""
    st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{len(data):,}")
    
    with col2:
        # Handle different target column names
        target_col = detect_target_column(data)
        if data[target_col].dtype == 'object':
            churn_rate = (data[target_col] == 'Yes').mean()
        else:
            churn_rate = data[target_col].mean()
        st.metric("Churn Rate", format_percentage(churn_rate))
    
    with col3:
        st.metric("Features", len(data.columns) - 1)  # Exclude target
    
    # Overview
    st.subheader("📋 Project Overview")
    st.write("""
    This application helps companies predict customer churn using machine learning. 
    The system analyzes customer behavior patterns and demographic information to identify 
    customers who are likely to leave the service.
    
    **Key Features:**
    - 🔍 **Data Exploration**: Interactive visualizations of customer data
    - 🤖 **Model Performance**: Compare different machine learning algorithms
    - 🔮 **Predictions**: Get real-time churn predictions for new customers
    - 📊 **Feature Analysis**: Understand which factors drive churn
    """)
    
    # Quick stats
    st.subheader("📊 Quick Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Class distribution
        target_col = detect_target_column(data)
        fig = plot_class_distribution(data, target_col)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Numerical features summary
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        # Check if columns exist in the data
        available_numerical_cols = [col for col in numerical_cols if col in data.columns]
        if available_numerical_cols:
            summary_stats = data[available_numerical_cols].describe()
            st.dataframe(summary_stats, use_container_width=True)
        else:
            st.write("No numerical columns found in the dataset.")


def show_data_exploration_page(data):
    """Display the data exploration page."""
    st.title("📈 Data Exploration")
    
    # Data overview
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Dataset Shape:** {data.shape}")
        st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
        st.write(f"**Duplicate Rows:** {data.duplicated().sum()}")
    
    with col2:
        st.write("**Data Types:**")
        st.write(data.dtypes.value_counts())
    
    # Correlation matrix
    st.subheader("Correlation Analysis")
    numerical_data = data.select_dtypes(include=[np.number])
    if len(numerical_data.columns) > 1:
        fig = plot_correlation_matrix(numerical_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    target_col = detect_target_column(data)
    # Categorical features
    categorical_cols = [c for c in data.select_dtypes(include=['object']).columns if c != target_col]
    if len(categorical_cols) > 0:
        selected_cat = st.selectbox("Select categorical feature:", categorical_cols)
        if selected_cat:
            fig = px.histogram(data, x=selected_cat, color=target_col, 
                             title=f"Distribution of {selected_cat} by Churn")
            st.plotly_chart(fig, use_container_width=True)
    
    # Numerical features
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        selected_num = st.selectbox("Select numerical feature:", numerical_cols)
        if selected_num:
            fig = px.histogram(data, x=selected_num, color=target_col, 
                             title=f"Distribution of {selected_num} by Churn")
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive filters
    st.subheader("Interactive Data Filters")

    # Build dynamic filters for categorical columns (excluding target)
    filterable_cols = [
        c for c in data.select_dtypes(include=['object', 'category']).columns
        if c != target_col
    ]

    selected_values = {}
    if len(filterable_cols) == 0:
        st.info("No categorical columns available for filtering.")
        filtered_data = data
    else:
        # Limit the number of unique options to avoid overwhelming the UI
        max_unique_options = 50
        columns_per_row = 2 if len(filterable_cols) > 1 else 1
        for i, col in enumerate(filterable_cols):
            if i % columns_per_row == 0:
                cols = st.columns(columns_per_row)
            ui_col = cols[i % columns_per_row]
            uniques = data[col].dropna().astype(str).unique().tolist()
            if len(uniques) > max_unique_options:
                # Skip very high-cardinality columns
                continue
            with ui_col:
                selected = st.multiselect(
                    f"Filter by {col}", options=sorted(uniques), default=sorted(uniques)
                )
                selected_values[col] = set(selected)

        # Apply filters
        mask = pd.Series([True] * len(data))
        for col, values in selected_values.items():
            if values:
                mask &= data[col].astype(str).isin(values)
        filtered_data = data[mask]

    st.write(f"**Filtered Rows:** {len(filtered_data)}")

    # Show filtered data
    if st.checkbox("Show filtered data"):
        st.dataframe(filtered_data.head(100), use_container_width=True)


def show_model_performance_page(data, models, feature_engineer):
    """Display the model performance page."""
    st.title("🤖 Model Performance")
    
    if 'best_model' not in models:
        st.error("No trained models found. Please train models first.")
        return
    
    # Load test data for evaluation
    config = get_config()
    processed_data_path = config.get_data_path("processed")
    test_file = os.path.join(processed_data_path, config.get("data.test_file", "test_data.csv"))
    
    if not os.path.exists(test_file):
        st.error("Test data not found. Please run the data processing pipeline first.")
        return
    
    test_data = pd.read_csv(test_file)
    target_col = detect_target_column(test_data)
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    # Ensure y_test is binary (0/1)
    if y_test.dtype == 'object':
        y_test = (y_test == 'Yes').astype(int)
    elif y_test.dtype == 'bool':
        y_test = y_test.astype(int)
    
    # Transform test data
    X_test_transformed = feature_engineer.transform_new_data(X_test)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Calculate metrics for all models
    model_metrics = {}
    
    for model_name, model in models.items():
        if model_name != 'best_model' and model is not None:
            try:
                y_pred = model.predict(X_test_transformed)
                y_prob = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, 'predict_proba') else None
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                model_metrics[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
            except Exception as e:
                st.warning(f"Error evaluating {model_name}: {str(e)}")
    
    # Display metrics comparison
    if model_metrics:
        metrics_df = pd.DataFrame({
            model_name: metrics['metrics'] 
            for model_name, metrics in model_metrics.items()
        }).T
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Metrics visualization
        st.subheader("Performance Metrics Comparison")
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'roc_auc' in metrics_df.columns:
            metrics_to_plot.append('roc_auc')
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            if metric in metrics_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=list(model_metrics.keys()),
                    y=metrics_df[metric],
                    text=[f"{val:.3f}" for val in metrics_df[metric]],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        best_model_name = metrics_df['f1_score'].idxmax()
        best_model_metrics = model_metrics[best_model_name]
        
        st.subheader(f"Best Model: {best_model_name.replace('_', ' ').title()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            fig = plot_confusion_matrix(y_test, best_model_metrics['predictions'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC curve
            if best_model_metrics['probabilities'] is not None:
                fig = plot_roc_curve(y_test, best_model_metrics['probabilities'])
                st.plotly_chart(fig, use_container_width=True)


def show_predictions_page(data, models, feature_engineer):
    """Display the predictions page."""
    st.title("🔮 Churn Predictions")
    
    if 'best_model' not in models:
        st.error("No trained models found. Please train models first.")
        return
    
    best_model = models['best_model']
    
    # Helpers
    def get_expected_input_columns(fe: FeatureEngineer, fallback: list[str]) -> list[str]:
        cols = []
        try:
            preprocessor = fe.preprocessing_pipeline.named_steps.get('preprocessor')
            if preprocessor is not None and hasattr(preprocessor, 'transformers'):
                for _, _, c in preprocessor.transformers:
                    if c is not None:
                        cols.extend(list(c))
            cols = list(dict.fromkeys(cols))
            return cols if cols else fallback
        except Exception:
            return fallback

    def align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
        safe_df = df.copy()
        for col in expected:
            if col not in safe_df.columns:
                safe_df[col] = np.nan
        return safe_df.reindex(columns=expected, fill_value=np.nan)

    # Prediction interface
    st.subheader("Make Predictions")

    target_col = detect_target_column(data)
    feature_cols = [c for c in data.columns if c != target_col]
    expected_cols = get_expected_input_columns(feature_engineer, feature_cols)

    st.write("Select a row from your dataset to predict:")
    row_index = st.number_input("Row index", min_value=0, max_value=max(0, len(data) - 1), value=0)
    if st.button("Predict selected row"):
        input_row = data.iloc[[row_index]][feature_cols]
        safe_input = align_to_expected(input_row, expected_cols)
        input_transformed = feature_engineer.transform_new_data(safe_input)
        prediction = best_model.predict(input_transformed)[0]
        probability = best_model.predict_proba(input_transformed)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                st.write("**Prediction: HIGH RISK**")
                st.write("This customer is likely to churn.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                st.write("**Prediction: LOW RISK**")
                st.write("This customer is likely to stay.")
                st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.metric("Confidence", f"{max(probability, 1-probability):.2%}")

    st.divider()

    # Row-by-row predictions table
    st.subheader("Row-by-Row Predictions")
    with st.spinner("Scoring dataset..."):
        X = data[feature_cols]
        X_safe = align_to_expected(X, expected_cols)
        X_transformed = feature_engineer.transform_new_data(X_safe)
        y_prob = best_model.predict_proba(X_transformed)[:, 1]
        # Use optimized threshold if available
        thr = None
        try:
            thr = models.get('thresholds', None)
        except Exception:
            thr = None
        if thr is None:
            # Try to get from trainer state
            from src.models.model_trainer import ModelTrainer as _MT
            _trainer = _MT(get_config())
            _ = _trainer.load_models()  # will load thresholds if saved
            thr = _trainer.model_thresholds.get('best_model') if _trainer.model_thresholds else None
        if thr is None:
            thr = 0.5
        y_pred = (y_prob >= float(thr)).astype(int)

    preds_df = pd.DataFrame({
        'row_index': np.arange(len(data)),
        'prediction': y_pred,
        'probability': y_prob
    })
    preds_df['risk'] = np.where(preds_df['prediction'] == 1, 'HIGH', 'LOW')
    st.dataframe(preds_df.head(200), use_container_width=True)
    st.download_button(
        label="Download predictions (CSV)",
        data=preds_df.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv'
    )

    st.divider()

    # Dataset-level prediction summary
    st.subheader("Dataset Prediction Summary")
    total = len(preds_df)
    churn_count = int((preds_df['prediction'] == 1).sum())
    churn_rate = churn_count / total if total else 0.0
    colA, colB, colC = st.columns(3)
    colA.metric("Total Rows Scored", f"{total:,}")
    colB.metric("Predicted Churn", f"{churn_count:,}")
    colC.metric("Predicted Churn Rate", f"{churn_rate:.2%}")

    # Optional: compare to true labels if target exists
    if target_col in data.columns:
        y_true = data[target_col]
        if y_true.dtype == 'object':
            y_true = (y_true == 'Yes').astype(int)
        elif y_true.dtype == 'bool':
            y_true = y_true.astype(int)
        metrics = calculate_metrics(y_true.values, y_pred, y_prob)
        st.write(get_model_performance_summary(metrics))


def show_feature_analysis_page(data, models, feature_engineer):
    """Display the feature analysis page."""
    st.title("📊 Feature Analysis")
    
    if 'best_model' not in models:
        st.error("No trained models found. Please train models first.")
        return
    
    best_model = models['best_model']
    
    # Feature importance
    st.subheader("Feature Importance")
    
    try:
        # Get feature names
        feature_names = feature_engineer.get_feature_names()
        
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based model
            importance_scores = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            # Linear model
            importance_scores = np.abs(best_model.coef_[0])
        else:
            st.warning("Feature importance not available for this model type.")
            return
        
        # Plot feature importance
        fig = plot_feature_importance(feature_names, importance_scores)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")
    
    # Feature correlation with churn
    st.subheader("Feature Correlation with Churn")
    
    # Convert churn to numeric for correlation
    data_numeric = data.copy()
    target_col = detect_target_column(data_numeric)
    if data_numeric[target_col].dtype == 'object':
        data_numeric['churn_numeric'] = (data_numeric[target_col] == 'Yes').astype(int)
    else:
        data_numeric['churn_numeric'] = data_numeric[target_col].astype(int)
    
    # Calculate correlations
    numerical_cols = data_numeric.select_dtypes(include=[np.number]).columns
    if 'churn_numeric' in numerical_cols and len(numerical_cols) > 1:
        correlations = data_numeric[numerical_cols].corr()['churn_numeric'].sort_values(ascending=False)
    else:
        st.warning("Not enough numerical columns to calculate correlations.")
        return
    
    # Plot correlations
    fig = px.bar(
        x=correlations.index,
        y=correlations.values,
        title="Feature Correlation with Churn",
        labels={'x': 'Features', 'y': 'Correlation'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation table
    st.dataframe(correlations.to_frame('Correlation'), use_container_width=True)


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column using DataLoader's heuristic."""
    try:
        loader = DataLoader(get_config())
        return loader._detect_target_column(df)
    except Exception:
        # Fallbacks for common names
        if 'Exited' in df.columns:
            return 'Exited'
        if 'churn' in df.columns:
            return 'churn'
        # Default to last column
        return df.columns[-1]


if __name__ == "__main__":
    main()
