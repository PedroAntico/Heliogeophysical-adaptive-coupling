import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class HeliogeophysicalPredictor:
    """
    Modelo preditivo para eventos heliofísicos
    Prevê eventos com 1-6 horas de antecedência baseado em condições atuais
    """
    
    def __init__(self, model_path: str = "models/heliogeophysical_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Parâmetros do modelo
        self.prediction_horizon = 3  # horas para previsão
        self.target_event_types = ['strong_negative_bz', 'high_speed_stream']
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para o modelo preditivo
        """
        if df.empty:
            return df
        
        # Features básicas
        features = df.copy()
        
        # Features temporais
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_year'] = features['timestamp'].dt.dayofyear
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        features['month'] = features['timestamp'].dt.month
        
        # Features de tendência (primeira derivada)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['hour', 'day_of_year', 'day_of_week', 'month']]
        
        for col in numeric_cols:
            # Taxa de variação
            features[f'{col}_change_rate'] = features[col].diff() / features[col].shift(1)
            # Aceleração (segunda derivada)
            features[f'{col}_acceleration'] = features[f'{col}_change_rate'].diff()
            
            # Features rolling para tendências
            features[f'{col}_rolling_mean_1h'] = features[col].rolling(window=12, min_periods=1).mean()
            features[f'{col}_rolling_std_1h'] = features[col].rolling(window=12, min_periods=1).std()
            features[f'{col}_rolling_trend_1h'] = features[col].rolling(window=12, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        # Features de interação
        if all(col in features.columns for col in ['speed', 'density']):
            features['momentum_flux'] = features['speed'] * features['density']
        
        if all(col in features.columns for col in ['bx_gse', 'by_gse', 'bz_gse']):
            features['magnetic_turbulence'] = (
                features['bx_gse'].rolling(window=6).std() +
                features['by_gse'].rolling(window=6).std() +
                features['bz_gse'].rolling(window=6).std()
            ) / 3
        
        # Preenche valores NaN
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def create_targets(self, df: pd.DataFrame, events: list) -> pd.Series:
        """
        Cria variável target para previsão de eventos futuros
        """
        # Inicializa targets como 0 (sem evento)
        targets = pd.Series(0, index=df.index)
        
        # Cria mapeamento timestamp -> eventos
        event_times = {}
        for event in events:
            event_time = pd.to_datetime(event['timestamp'])
            event_type = event['type']
            if event_type in self.target_event_types:
                # Marca eventos que ocorreram dentro do horizonte de previsão
                event_window_start = event_time - timedelta(hours=self.prediction_horizon)
                event_window_end = event_time
                
                # Encontra índices dentro da janela de previsão
                mask = (df['timestamp'] >= event_window_start) & (df['timestamp'] < event_window_end)
                targets[mask] = 1
        
        return targets
    
    def train(self, features_df: pd.DataFrame, events: list, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Treina o modelo preditivo
        """
        logger.info("Iniciando treinamento do modelo preditivo...")
        
        try:
            # Prepara features
            features_df = self.prepare_features(features_df)
            
            # Cria targets
            y = self.create_targets(features_df, events)
            
            if y.sum() == 0:
                logger.warning("Nenhum evento encontrado para treinamento")
                return {"status": "no_events", "accuracy": 0}
            
            # Seleciona features numéricas
            X = features_df.select_dtypes(include=[np.number])
            self.feature_names = X.columns.tolist()
            
            # Remove colunas com variância zero
            X = X.loc[:, X.std() > 0]
            
            if X.empty:
                logger.error("Nenhuma feature válida para treinamento")
                return {"status": "no_features", "accuracy": 0}
            
            # Split temporal (não random para séries temporais)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Balanceamento de classes
            from sklearn.utils import class_weight
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            
            # Escala features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treina modelo (Random Forest para lidar bem com features não-lineares)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliação
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Salva modelo
            self._save_model()
            self.is_trained = True
            
            logger.info(f"Modelo treinado com sucesso! Acurácia: {accuracy:.3f}")
            logger.info(f"Top features: {top_features}")
            
            return {
                "status": "success",
                "accuracy": accuracy,
                "top_features": top_features,
                "n_events": int(y.sum()),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return {"status": "error", "error": str(e)}
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Faz previsões de eventos futuros
        """
        if not self.is_trained:
            self._load_model()
        
        if self.model is None:
            logger.error("Modelo não disponível para previsão")
            return pd.DataFrame(), pd.Series()
        
        try:
            # Prepara features
            features_df = self.prepare_features(features_df)
            X = features_df.select_dtypes(include=[np.number])
            
            # Mantém apenas features conhecidas
            available_features = [f for f in self.feature_names if f in X.columns]
            X = X[available_features]
            
            if X.empty:
                logger.warning("Nenhuma feature disponível para previsão")
                return pd.DataFrame(), pd.Series()
            
            # Adiciona features faltantes como zero
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            X = X[self.feature_names]
            
            # Escala e faz previsão
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probabilidade da classe positiva
            predictions = (probabilities > 0.5).astype(int)
            
            # Cria DataFrame de resultados
            results = features_df[['timestamp']].copy()
            results['event_probability'] = probabilities
            results['predicted_event'] = predictions
            results['prediction_horizon_hours'] = self.prediction_horizon
            
            logger.info(f"Previsões realizadas: {predictions.sum()} eventos previstos")
            
            return results, probabilities
            
        except Exception as e:
            logger.error(f"Erro na previsão: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _save_model(self):
        """Salva modelo e scaler"""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Modelo salvo em: {self.model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
    
    def _load_model(self):
        """Carrega modelo e scaler"""
        try:
            if Path(self.model_path).exists():
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = model_data['is_trained']
                logger.info(f"Modelo carregado: {self.model_path}")
            else:
                logger.warning("Modelo não encontrado, precisa ser treinado")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")

def create_predictor(model_path: str = "models/heliogeophysical_model.pkl") -> HeliogeophysicalPredictor:
    """Factory function para criar o predictor"""
    return HeliogeophysicalPredictor(model_path)
