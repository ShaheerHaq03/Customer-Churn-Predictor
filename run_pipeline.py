#!/usr/bin/env python3
"""
Main script to run the complete Customer Churn Prediction pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import get_config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer


def run_data_pipeline():
    """Run the data loading and preprocessing pipeline."""
    print("=" * 60)
    print("📊 DATA PIPELINE")
    print("=" * 60)
    
    config = get_config()
    loader = DataLoader(config)
    
    # Generate or load data
    print("1. Loading/Generating data...")
    data = loader.load_data()
    
    # Split data
    print("2. Splitting data...")
    X_train, X_test, y_train, y_test = loader.split_data(data)
    
    # Save processed data
    print("3. Saving processed data...")
    loader.save_processed_data(X_train, X_test, y_train, y_test)
    
    print("✅ Data pipeline completed successfully!")
    return X_train, X_test, y_train, y_test


def run_feature_engineering():
    """Run the feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 60)
    
    config = get_config()
    feature_engineer = FeatureEngineer(config)
    
    # Load processed data
    processed_data_path = config.get_data_path("processed")
    train_file = os.path.join(processed_data_path, config.get("data.train_file", "train_data.csv"))
    test_file = os.path.join(processed_data_path, config.get("data.test_file", "test_data.csv"))
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("❌ Processed data files not found. Running data pipeline first...")
        run_data_pipeline()
    
    # Load data
    print("1. Loading processed data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Detect target dynamically if in auto mode
    target_col = config.get_target_column() if not config.get("auto_mode", True) else DataLoader(config)._detect_target_column(train_data)
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    # Apply feature engineering
    print("2. Applying feature engineering...")
    X_train_transformed, X_test_transformed = feature_engineer.fit_transform(
        X_train, X_test, y_train
    )
    
    # Save pipeline
    print("3. Saving preprocessing pipeline...")
    feature_engineer.save_pipeline()
    
    print("✅ Feature engineering completed successfully!")
    return X_train_transformed, X_test_transformed, y_train, y_test


def run_model_training():
    """Run the model training pipeline."""
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING")
    print("=" * 60)
    
    config = get_config()
    trainer = ModelTrainer(config)
    
    # Check if feature engineering has been done
    models_path = config.get_model_path()
    pipeline_file = os.path.join(models_path, "preprocessing_pipeline.joblib")
    
    if not os.path.exists(pipeline_file):
        print("❌ Preprocessing pipeline not found. Running feature engineering first...")
        run_feature_engineering()
    
    # Load preprocessed data
    processed_data_path = config.get_data_path("processed")
    train_file = os.path.join(processed_data_path, config.get("data.train_file", "train_data.csv"))
    test_file = os.path.join(processed_data_path, config.get("data.test_file", "test_data.csv"))
    
    print("1. Loading preprocessed data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    target_col = config.get_target_column() if not config.get("auto_mode", True) else DataLoader(config)._detect_target_column(train_data)
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    # Load feature engineering pipeline
    print("2. Loading preprocessing pipeline...")
    feature_engineer = FeatureEngineer(config)
    feature_engineer.load_pipeline()
    
    # Transform data
    print("3. Transforming data...")
    X_train_transformed = feature_engineer.transform_new_data(X_train)
    X_test_transformed = feature_engineer.transform_new_data(X_test)
    
    # Train models
    print("4. Training models...")
    results = trainer.train_models(X_train_transformed, y_train, X_test_transformed, y_test)
    
    # Save models
    print("5. Saving models...")
    trainer.save_models()
    
    # Print results
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    print("-" * 40)
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"{model_name.upper()}:")
            print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print()
    
    print("✅ Model training completed successfully!")
    return results


def run_webapp():
    """Run the Streamlit web application."""
    print("\n" + "=" * 60)
    print("🌐 WEB APPLICATION")
    print("=" * 60)
    
    print("Starting Streamlit web application...")
    print("The application will open in your default web browser.")
    print("To stop the application, press Ctrl+C in the terminal.")
    
    import subprocess
    import sys
    
    # Run streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "webapp", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=["data", "features", "models", "webapp", "all"],
        default="all",
        help="Pipeline step to run (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if files exist"
    )
    
    args = parser.parse_args()
    
    print("🚀 Customer Churn Prediction Pipeline")
    print("=" * 60)
    
    try:
        if args.step == "data" or args.step == "all":
            run_data_pipeline()
        
        if args.step == "features" or args.step == "all":
            run_feature_engineering()
        
        if args.step == "models" or args.step == "all":
            run_model_training()
        
        if args.step == "webapp":
            run_webapp()
        
        if args.step == "all":
            print("\n" + "=" * 60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Run the web application: python run_pipeline.py --step webapp")
            print("2. Or run: streamlit run webapp/streamlit_app.py")
            print("3. Open your browser and navigate to the provided URL")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()
