"""
TRAINING GUIDE - Person 2's Step-by-Step Training Guide
=====================================================

This script takes Person 1's data and trains a complete pipeline for 
classification and forecasting models.

Usage: python train_all_models.py
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path

# Import libraries
from src.model_prep import DataPreprocessor, create_lag_features
from src.classification import LogisticRegressionClassifier, NeuralNetworkClassifier
from src.forecasting import LinearForecaster, ARIMAForecaster, ProphetForecaster
from src.evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator
from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_person1_data(data_path: str) -> pd.DataFrame:
    """
    Load data from Person 1.
    
    Expected format:
    - CSV file
    - Columns: feature_1, feature_2, ..., label (RS/RP)
    - Optional: timestamp (for time series)
    
    Args:
        data_path: CSV file path
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {data_path} | Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        return df
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise


def train_classification_models(X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               models_dir: str = 'models') -> dict:
    """
    Train and compare classification models.
    
    Models to train:
    1. Logistic Regression (Baseline)
    2. Neural Network (Deep Learning)
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        models_dir: Model save directory
        
    Returns:
        dict: Information about all models
    """
    logger.info("="*60)
    logger.info("CLASSIFICATION MODELS TRAINING STARTING")
    logger.info("="*60)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. LOGISTIC REGRESSION
    logger.info("\n[1/2] Training Logistic Regression...")
    try:
        lr_clf = LogisticRegressionClassifier(C=1.0, solver='lbfgs', max_iter=1000)
        lr_clf.train(X_train, y_train)
        
        lr_eval = ClassificationEvaluator()
        lr_metrics = lr_eval.evaluate(y_test, lr_clf.predict(X_test))
        
        # Save model
        lr_model_path = models_path / 'classification_logistic_regression.pkl'
        lr_clf.save(str(lr_model_path))
        
        results['logistic_regression'] = {
            'model_path': str(lr_model_path),
            'metrics': lr_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Logistic Regression trained successfully")
        logger.info(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {lr_metrics['precision']:.4f}")
        logger.info(f"  Recall: {lr_metrics['recall']:.4f}")
        logger.info(f"  F1: {lr_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Logistic Regression training error: {e}")
        results['logistic_regression'] = {'trained': False, 'error': str(e)}
    
    # 2. NEURAL NETWORK
    logger.info("\n[2/2] Training Neural Network...")
    try:
        nn_clf = NeuralNetworkClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001
        )
        
        # Validation split
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train_nn = X_train[:split_idx]
        y_train_nn = y_train[:split_idx]
        
        nn_clf.train(X_train_nn, y_train_nn, epochs=50, batch_size=32,
                    X_val=X_val, y_val=y_val)
        
        nn_eval = ClassificationEvaluator()
        nn_metrics = nn_eval.evaluate(y_test, nn_clf.predict(X_test))
        
        # Save model
        nn_model_path = models_path / 'classification_neural_network.pt'
        nn_clf.save(str(nn_model_path))
        
        results['neural_network'] = {
            'model_path': str(nn_model_path),
            'metrics': nn_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Neural Network trained successfully")
        logger.info(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {nn_metrics['precision']:.4f}")
        logger.info(f"  Recall: {nn_metrics['recall']:.4f}")
        logger.info(f"  F1: {nn_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Neural Network training error: {e}")
        results['neural_network'] = {'trained': False, 'error': str(e)}
    
    return results


def train_forecasting_models(timeseries_data: np.ndarray,
                            test_size: float = 0.2,
                            models_dir: str = 'models') -> dict:
    """
    Train and compare forecasting models.
    
    Models to train:
    1. Linear Forecaster
    2. ARIMA
    3. Prophet
    
    Args:
        timeseries_data: 1D time series
        test_size: Test set ratio
        models_dir: Model save directory
        
    Returns:
        dict: Information about all models
    """
    logger.info("\n" + "="*60)
    logger.info("FORECASTING MODELS TRAINING STARTING")
    logger.info("="*60)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    # Train/Test split
    split_idx = int(len(timeseries_data) * (1 - test_size))
    y_train = timeseries_data[:split_idx]
    y_test = timeseries_data[split_idx:]
    
    results = {}
    
    # 1. LINEAR FORECASTER
    logger.info("\n[1/3] Training Linear Forecaster...")
    try:
        lf = LinearForecaster(lookback=24, degree=1)
        lf.fit(y_train)
        
        # Make predictions
        forecast_lf = lf.forecast(steps=len(y_test), last_sequence=y_train[-24:])
        
        # Evaluate
        lf_eval = ForecastingEvaluator()
        lf_metrics = lf_eval.evaluate(y_test, forecast_lf[:len(y_test)])
        
        # Save model
        lf_model_path = models_path / 'forecasting_linear.pkl'
        lf.save(str(lf_model_path))
        
        results['linear'] = {
            'model_path': str(lf_model_path),
            'metrics': lf_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Linear Forecaster trained successfully")
        logger.info(f"  MAE: {lf_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {lf_metrics['rmse']:.4f}")
        logger.info(f"  R2: {lf_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Linear Forecaster training error: {e}")
        results['linear'] = {'trained': False, 'error': str(e)}
    
    # 2. ARIMA
    logger.info("\n[2/3] Training ARIMA...")
    try:
        af = ARIMAForecaster(order=(1, 1, 1))
        af.fit(y_train)
        
        # Make predictions
        forecast_af = af.forecast(steps=len(y_test))
        
        # Evaluate
        af_eval = ForecastingEvaluator()
        af_metrics = af_eval.evaluate(y_test, forecast_af[:len(y_test)])
        
        # Save model
        af_model_path = models_path / 'forecasting_arima.pkl'
        af.save(str(af_model_path))
        
        results['arima'] = {
            'model_path': str(af_model_path),
            'metrics': af_metrics,
            'trained': True
        }
        
        logger.info(f"✓ ARIMA trained successfully")
        logger.info(f"  MAE: {af_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {af_metrics['rmse']:.4f}")
        logger.info(f"  R2: {af_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"ARIMA training error: {e}")
        results['arima'] = {'trained': False, 'error': str(e)}
    
    # 3. PROPHET
    logger.info("\n[3/3] Training Prophet...")
    try:
        df_prophet = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=len(y_train), freq='h'),
            'y': y_train
        })
        
        pf = ProphetForecaster(interval_width=0.95, seasonality_mode='additive')
        pf.fit(df_prophet)
        
        # Make predictions
        forecast_pf, lower_pf, upper_pf = pf.forecast(steps=len(y_test), freq='h')
        
        # Evaluate
        pf_eval = ForecastingEvaluator()
        pf_metrics = pf_eval.evaluate(y_test, forecast_pf[:len(y_test)])
        
        # Save model
        pf_model_path = models_path / 'forecasting_prophet.pkl'
        pf.save(str(pf_model_path))
        
        results['prophet'] = {
            'model_path': str(pf_model_path),
            'metrics': pf_metrics,
            'trained': True
        }
        
        logger.info(f"✓ Prophet trained successfully")
        logger.info(f"  MAE: {pf_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {pf_metrics['rmse']:.4f}")
        logger.info(f"  R2: {pf_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Prophet training error: {e}")
        results['prophet'] = {'trained': False, 'error': str(e)}
    
    return results


def generate_training_report(clf_results: dict, fcst_results: dict,
                            output_file: str = 'training_report.json'):
    """
    Generate training report.
    
    Args:
        clf_results: Classification results
        fcst_results: Forecasting results
        output_file: Report file name
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING TRAINING REPORT")
    logger.info("="*60)
    
    report = {
        'training_date': pd.Timestamp.now().isoformat(),
        'classification_models': clf_results,
        'forecasting_models': fcst_results,
        'summary': {
            'total_classification_models': len([m for m in clf_results.values() if m.get('trained', False)]),
            'total_forecasting_models': len([m for m in fcst_results.values() if m.get('trained', False)])
        }
    }
    
    # Save as JSON
    with open(output_file, 'w') as f:
        # Make numpy arrays in metrics serializable
        report_str = json.dumps(report, indent=2, default=str)
        f.write(report_str)
    
    logger.info(f"✓ Report saved: {output_file}")
    
    # Print to console
    logger.info("\nSUMMARY:")
    logger.info(f"  Classification Models: {report['summary']['total_classification_models']}")
    logger.info(f"  Forecasting Models: {report['summary']['total_forecasting_models']}")


def main():
    """Main training function"""
    logger.info("╔" + "="*58 + "╗")
    logger.info("║  PERSON 2 - COMPLETE TRAINING PIPELINE              ║")
    logger.info("╚" + "="*58 + "╝")
    
    try:
        # 1. LOAD DATA FROM PERSON 1 (Member 1)
        logger.info("\n[STEP 1] Loading data from labeled_customers.csv...")
        
        # Load the labeled data from Person 1
        data_path = 'data/labeled_customers.csv'
        clf_df = pd.read_csv(data_path)
        logger.info(f"✓ Data loaded: {data_path}")
        logger.info(f"  Shape: {clf_df.shape}")
        logger.info(f"  Columns: {list(clf_df.columns)}")
        logger.info(f"  Label distribution:\n{clf_df['label'].value_counts()}")
        
        # Use feature columns (exclude customer_id and label)
        exclude_cols = ['customer_id', 'label']
        feature_cols = [col for col in clf_df.columns if col not in exclude_cols]
        
        # Extract features and labels
        X_clf_data = clf_df[feature_cols].values
        y_clf_data = clf_df['label'].values
        
        # Create dataframe with features and labels for preprocessing
        clf_df_processed = pd.DataFrame(X_clf_data, columns=feature_cols)
        clf_df_processed['label'] = y_clf_data
        
        # Use mean daily kwh as time series for forecasting
        timeseries = clf_df['mean_daily_kwh'].values
        
        logger.info("✓ Real data from Person 1 loaded and prepared")
        
        # 2. DATA PREPARATION
        logger.info("\n[STEP 2] Preparing data...")
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        
        # Balance the dataset
        clf_df_balanced = preprocessor.balance_dataset(clf_df_processed, label_col='label')
        logger.info(f"  Balanced dataset shape: {clf_df_balanced.shape}")
        
        # Train/Test split (without time column since we don't have it)
        X = clf_df_balanced.drop('label', axis=1).values
        y = clf_df_balanced['label'].values
        
        # Simple train/test split
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        # Normalization
        X_train_norm, X_test_norm = preprocessor.normalize_features(X_train, X_test)
        
        logger.info("✓ Data prepared")
        
        # 3. CLASSIFICATION MODELS
        clf_results = train_classification_models(X_train_norm, X_test_norm, y_train, y_test)
        
        # 4. FORECASTING MODELS
        fcst_results = train_forecasting_models(timeseries, test_size=0.2)
        
        # 5. GENERATE REPORT
        generate_training_report(clf_results, fcst_results)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 TRAINING COMPLETED")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
