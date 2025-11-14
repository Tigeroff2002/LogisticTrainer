import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from typing import Tuple, Dict, Any

class RouteTimePredictor:
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка фич и таргета"""
        self.logger.info("Starting feature preparation")
        
        # Копируем данные
        data = df.copy()
        
        # Целевая переменная
        y = data['duration_seconds']
        
        # Признаки
        feature_columns = ['start_fav_area_id', 'end_fav_area_id', 'month', 
                          'time_of_day', 'day_of_week', 'number_of_rides']
        
        X = data[feature_columns]
        
        # Кодируем категориальные переменные
        categorical_columns = ['time_of_day']
        for col in categorical_columns:
            if col in X.columns:
                self.logger.info(f"Encoding categorical column: {col}")
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
        
        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Target shape: {y.shape}")
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Обучение модели"""
        self.logger.info("Starting model training")
        
        # Подготовка данных
        X, y = self.prepare_features(df)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Создание и обучение модели
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Обучение с логированием прогресса
        self.logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
        # Предсказание и метрики
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        self.logger.info(f"Training metrics: {metrics}")
        
        return metrics
    
    def get_model_metadata(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Метаданные обученной модели"""
        return {
            'model_type': self.model_type,
            'training_date': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'feature_columns': list(self.label_encoders.keys()),
            'model_class': self.model.__class__.__name__
        }