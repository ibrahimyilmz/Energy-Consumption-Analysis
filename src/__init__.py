"""
Data Science Project - Person 2 (Model Architect) Module
=========================================================

Bu paket Kişi 2'nin (Model Mimarı) tüm modüllerini içerir.

Modüller:
    - model_prep: Veri hazırlığı ve dengeleme
    - classification: Sınıflandırma modelleri (RS/RP)
    - forecasting: Tahminleme modelleri (Linear, ARIMA, Prophet)
    - evaluator: Model değerlendirme araçları
    - integration_logic: Entegrasyon katmanı
"""

__version__ = "1.0.0"
__author__ = "Person 2 - Model Architect"

from src.model_prep import DataPreprocessor, create_lag_features
from src.classification import BaseClassifier, LogisticRegressionClassifier, NeuralNetworkClassifier
from src.forecasting import BaseForecaster, LinearForecaster, ARIMAForecaster, ProphetForecaster
from src.evaluator import ClassificationEvaluator, ForecastingEvaluator, ModelComparator
from src.integration_logic import ModelIntegrator, DataPipeline, ResultsFormatter

__all__ = [
    # Data Preparation
    'DataPreprocessor',
    'create_lag_features',
    
    # Classification
    'BaseClassifier',
    'LogisticRegressionClassifier',
    'NeuralNetworkClassifier',
    
    # Forecasting
    'BaseForecaster',
    'LinearForecaster',
    'ARIMAForecaster',
    'ProphetForecaster',
    
    # Evaluation
    'ClassificationEvaluator',
    'ForecastingEvaluator',
    'ModelComparator',
    
    # Integration
    'ModelIntegrator',
    'DataPipeline',
    'ResultsFormatter',
]
