import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime

class RouteTimePredictor:
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.training_date = None
        self.metrics = None
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка фич и таргета"""
        self.logger.info("Starting feature preparation")
        
        # Копируем данные
        data = df.copy()
        
        # Целевая переменная
        y = data['duration_seconds']
        
        self.feature_columns = ['user_id', 'start_fav_area_id', 'end_fav_area_id', 
                                'month_of_year', 'time_of_day', 'day_of_week']
        
        X = data[self.feature_columns]
        
        categorical_columns = ['time_of_day', 'month_of_year', 'day_of_week']
        for col in categorical_columns:
            if col in X.columns:
                self.logger.info(f"Encoding categorical column: {col}")
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
        
        self.logger.info(f"Features shape: {X.shape}, columns: {X.columns.tolist()}")
        self.logger.info(f"Target shape: {y.shape}")
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Обучение модели с опциональным сохранением"""
        self.logger.info("Starting model training")
        self.training_date = datetime.now()
        
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
        
        # Сохраняем feature_names для будущих предсказаний
        self.feature_columns = X.columns.tolist()
        self.logger.info(f"Saved feature columns: {self.feature_columns}")
        
        # Предсказание и метрики
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        self.logger.info(f"Training metrics: {self.metrics}")
        
        return self.metrics
    
    def get_model_metadata(self, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Метаданные обученной модели"""
        if metrics is None:
            metrics = self.metrics
            
        metadata = {
            'model_type': self.model_type,
            'training_date': self.training_date.isoformat() if self.training_date else datetime.now().isoformat(),
            'random_state': self.random_state,
            'feature_columns': self.feature_columns,
            'categorical_columns': list(self.label_encoders.keys()),
            'model_class': self.model.__class__.__name__ if self.model else None,
            'has_scaler': self.scaler is not None,
            'metrics': metrics
        }
        
        # Добавляем информацию о label encoders
        for col, encoder in self.label_encoders.items():
            metadata[f'{col}_classes'] = encoder.classes_.tolist()
            metadata[f'{col}_classes_count'] = len(encoder.classes_)
            
        return metadata
    
    def predict_single(self, input_data: Dict[str, Any]) -> float:
        """Предсказание для одного примера"""
        try:
            self.logger.info(f"Starting prediction for input data: {input_data}")
            
            # Создаем DataFrame из входных данных
            input_df = pd.DataFrame([input_data])
            self.logger.debug(f"Created DataFrame with columns: {input_df.columns.tolist()}")
            
            # Преобразуем категориальные переменные
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns:
                    original_value = input_df[col].iloc[0]
                    self.logger.debug(f"Processing categorical column '{col}' with value: {original_value}")
                    
                    # Проверяем, есть ли значение в кодировщике
                    if original_value in encoder.classes_:
                        encoded_value = encoder.transform([original_value])[0]
                        input_df.loc[:, col] = encoded_value
                        self.logger.debug(f"Encoded '{original_value}' to {encoded_value} for column '{col}'")
                    else:
                        # Если значение новое, используем "unknown" класс
                        self.logger.warning(f"Unknown value '{original_value}' for column '{col}'. Available classes: {encoder.classes_.tolist()}")
                        # Используем наиболее частый класс
                        if len(encoder.classes_) > 0:
                            default_value = encoder.transform([encoder.classes_[0]])[0]
                        else:
                            default_value = 0
                        input_df.loc[:, col] = default_value
                        self.logger.info(f"Using default value {default_value} for unknown category in column '{col}'")
            
            # Убедимся, что все нужные колонки присутствуют
            if self.feature_columns:
                self.logger.debug(f"Expected feature columns: {self.feature_columns}")
                self.logger.debug(f"Current DataFrame columns: {input_df.columns.tolist()}")
                
                # Добавляем отсутствующие колонки
                missing_cols = [col for col in self.feature_columns if col not in input_df.columns]
                if missing_cols:
                    self.logger.warning(f"Missing columns in input data: {missing_cols}. Adding with default value 0")
                    for col in missing_cols:
                        input_df[col] = 0
                
                # Выбираем только нужные фичи в правильном порядке
                X = input_df[self.feature_columns]
            else:
                self.logger.warning("feature_columns is empty, using all available columns")
                X = input_df
            
            self.logger.debug(f"Final feature matrix shape: {X.shape}")
            self.logger.debug(f"Feature matrix columns: {X.columns.tolist()}")
            
            # Проверяем модель
            if self.model is None:
                self.logger.error("Model is not loaded or initialized")
                raise ValueError("Model is not loaded or initialized")
            
            # Делаем предсказание
            self.logger.info("Making prediction...")
            prediction = self.model.predict(X)[0]
            self.logger.info(f"Prediction result: {prediction}")
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            self.logger.error(f"Input data that caused error: {input_data}")
            self.logger.error(f"DataFrame columns: {locals().get('input_df', pd.DataFrame()).columns.tolist() if 'input_df' in locals() else 'DataFrame not created'}")
            self.logger.error(f"Feature columns from model: {self.feature_columns}")
            raise