def test_imports_and_config():
    from src.utils.config import get_config
    from src.data.data_loader import DataLoader
    from src.features.feature_engineering import FeatureEngineer
    from src.models.model_trainer import ModelTrainer

    config = get_config()
    assert config.get_target_column() in ["Exited", "churn"]

    # Instantiate classes without running heavy workloads
    loader = DataLoader(config)
    fe = FeatureEngineer(config)
    trainer = ModelTrainer(config)

    assert loader is not None
    assert fe is not None
    assert trainer is not None
