#!/usr/bin/env python3
"""
Expansão do Projeto Heliogeophysical - ML e Novas Fontes de Dados
"""

import os
from pathlib import Path

def create_ml_module():
    """Cria módulo de machine learning preditivo"""
    
    Path("src/model/__init__.py").touch()
    
    # model/predictive_model.py
    ml_content = '''import logging
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
'''
    
    with open("src/model/predictive_model.py", "w") as f:
        f.write(ml_content)
    print("✓ model/predictive_model.py criado")

def create_advanced_fetchers():
    """Cria fetchers avançados para novas fontes de dados"""
    
    # fetchers/nasa_cdaweb_fetcher.py
    nasa_content = '''import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.retries import retry_with_exponential_backoff

logger = logging.getLogger(__name__)

class NASACDAWebFetcher:
    """
    Fetcher avançado para dados da NASA CDAWeb
    Acessa dados de múltiplos satélites e experimentos
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://cdaweb.gsfc.nasa.gov')
        self.datasets = config.get('datasets', {})
        self.cdas_client = None
    
    def _initialize_cdas_client(self) -> bool:
        """Inicializa cliente CDAWeb"""
        try:
            from cdasws import CdasWs
            self.cdas_client = CdasWs()
            return True
        except ImportError:
            logger.warning("cdasws não instalado. Use: pip install cdasws")
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar CDAWeb: {e}")
            return False
    
    @retry_with_exponential_backoff(max_retries=2, base_delay=10.0)
    def fetch_dataset(self, dataset_name: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Busca dados de um dataset específico do CDAWeb"""
        if not self.cdas_client and not self._initialize_cdas_client():
            return None
        
        dataset_id = self.datasets.get(dataset_name)
        if not dataset_id:
            logger.error(f"Dataset não configurado: {dataset_name}")
            return None
        
        logger.info(f"Buscando dados CDAWeb: {dataset_name} ({dataset_id})")
        
        try:
            # Define variáveis baseadas no dataset
            variable_mapping = {
                'dscovr_swepam': ['Np', 'Vp', 'Tp'],
                'dscovr_mag': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'omni_1min': ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'V', 'N', 'T', 'KP'],
                'ace_mag': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'ace_swepam': ['Np', 'Vp', 'Tp'],
                'wind_mfi': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'wind_swe': ['Np', 'Vp', 'Tp']
            }
            
            variables = variable_mapping.get(dataset_name, ['Np', 'Vp', 'Tp'])
            
            # Período de busca
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Busca dados
            status, data = self.cdas_client.get_data(
                dataset_id,
                variables,
                start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            if status != 200 or not data:
                logger.warning(f"CDAWeb retornou status {status} sem dados")
                return None
            
            # Processa dados
            df = self._process_cdas_data(data, dataset_name)
            logger.info(f"Dados {dataset_name} processados: {len(df)} linhas")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar {dataset_name}: {e}")
            return None
    
    def _process_cdas_data(self, data: dict, dataset_name: str) -> pd.DataFrame:
        """Processa dados brutos do CDAWeb"""
        df = pd.DataFrame()
        
        # Mapeamento de variáveis para nomes padronizados
        standard_mapping = {
            'Np': 'density', 'Vp': 'speed', 'Tp': 'temperature',
            'B1GSE': 'bx_gse', 'B2GSE': 'by_gse', 'B3GSE': 'bz_gse', 'Bt': 'bt',
            'BX_GSE': 'bx_gse', 'BY_GSE': 'by_gse', 'BZ_GSE': 'bz_gse',
            'V': 'speed', 'N': 'density', 'T': 'temperature', 'KP': 'kp_index'
        }
        
        for key, values in data.items():
            if key.lower() == 'epoch':
                df['timestamp'] = pd.to_datetime(values, utc=True)
            else:
                standard_name = standard_mapping.get(key, key)
                df[standard_name] = values
        
        # Converte para numérico
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicatas de timestamp
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        return df.reset_index(drop=True)
    
    def fetch_multiple_datasets(self, dataset_names: list, days: int = 7) -> Dict[str, pd.DataFrame]:
        """Busca múltiplos datasets e retorna dicionário com DataFrames"""
        results = {}
        
        for dataset_name in dataset_names:
            df = self.fetch_dataset(dataset_name, days)
            if df is not None and not df.empty:
                results[dataset_name] = df
        
        logger.info(f"CDAWeb: {len(results)} datasets coletados com sucesso")
        return results

def create_nasa_fetcher(config: Dict[str, Any]) -> NASACDAWebFetcher:
    """Factory function para NASA CDAWeb Fetcher"""
    return NASACDAWebFetcher(config)
'''
    
    with open("src/fetchers/nasa_cdaweb_fetcher.py", "w") as f:
        f.write(nasa_content)
    print("✓ fetchers/nasa_cdaweb_fetcher.py criado")
    
    # fetchers/ensemble_fetcher.py
    ensemble_content = '''import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class EnsembleDataFetcher:
    """
    Fetcher de ensemble que combina múltiplas fontes de dados
    para criar dataset robusto e completo
    """
    
    def __init__(self, noaa_fetcher, nasa_fetcher):
        self.noaa_fetcher = noaa_fetcher
        self.nasa_fetcher = nasa_fetcher
        self.data_quality_threshold = 0.6  % mínimo de dados válidos
    
    def fetch_ensemble_data(self, days: int = 3) -> pd.DataFrame:
        """
        Busca dados de múltiplas fontes e combina em dataset único
        """
        logger.info("Coletando dados de ensemble...")
        
        all_datasets = []
        
        # 1. Dados NOAA (prioridade máxima - tempo real)
        noaa_data = self._fetch_noaa_data(days)
        if noaa_data is not None:
            all_datasets.append(("NOAA", noaa_data))
        
        # 2. Dados NASA CDAWeb (dados complementares)
        nasa_data = self._fetch_nasa_data(days)
        if nasa_data:
            all_datasets.extend(nasa_data)
        
        # 3. Combina todos os datasets
        ensemble_df = self._merge_ensemble_data(all_datasets)
        
        # 4. Avalia qualidade dos dados
        quality_report = self._assess_data_quality(ensemble_df)
        
        logger.info(f"Ensemble criado: {len(ensemble_df)} linhas, Qualidade: {quality_report['overall_quality']:.1%}")
        
        return ensemble_df
    
    def _fetch_noaa_data(self, days: int) -> Optional[pd.DataFrame]:
        """Busca dados NOAA"""
        try:
            datasets = []
            
            plasma = self.noaa_fetcher.fetch_plasma_data(days)
            if plasma is not None:
                datasets.append(plasma)
            
            mag = self.noaa_fetcher.fetch_magnetic_data(days)
            if mag is not None:
                datasets.append(mag)
            
            if datasets:
                # Combina dados NOAA
                from ..processing.preprocessor import DataPreprocessor
                preprocessor = DataPreprocessor()
                combined = preprocessor._merge_datasets(datasets)
                return combined
                
        except Exception as e:
            logger.error(f"Erro ao buscar dados NOAA: {e}")
        
        return None
    
    def _fetch_nasa_data(self, days: int) -> List[tuple]:
        """Busca dados NASA CDAWeb"""
        nasa_datasets = []
        
        try:
            # Datasets prioritários da NASA
            priority_datasets = ['dscovr_swepam', 'dscovr_mag', 'omni_1min']
            
            nasa_results = self.nasa_fetcher.fetch_multiple_datasets(priority_datasets, days)
            
            for name, df in nasa_results.items():
                if df is not None and not df.empty:
                    nasa_datasets.append((f"NASA_{name}", df))
                    
        except Exception as e:
            logger.error(f"Erro ao buscar dados NASA: {e}")
        
        return nasa_datasets
    
    def _merge_ensemble_data(self, datasets: List[tuple]) -> pd.DataFrame:
        """Combina dados de múltiplas fontes usando estratégia de ensemble"""
        if not datasets:
            return pd.DataFrame()
        
        # Ordena por prioridade (NOAA primeiro)
        datasets.sort(key=lambda x: 0 if x[0].startswith('NOAA') else 1)
        
        base_df = datasets[0][1].copy()
        
        for source_name, df in datasets[1:]:
            if df is None or df.empty:
                continue
            
            # Encontra colunas únicas para merge
            new_columns = [col for col in df.columns if col not in base_df.columns and col != 'timestamp']
            
            if not new_columns:
                continue
            
            # Merge com tolerância temporal
            merge_df = df[['timestamp'] + new_columns].sort_values('timestamp')
            
            base_df = pd.merge_asof(
                base_df.sort_values('timestamp'),
                merge_df,
                on='timestamp',
                tolerance=pd.Timedelta('15min'),
                direction='nearest'
            )
            
            logger.info(f"Merge com {source_name}: +{len(new_columns)} colunas")
        
        # Preenche gaps estratégicamente
        base_df = self._smart_fill_missing_values(base_df)
        
        return base_df.sort_values('timestamp').reset_index(drop=True)
    
    def _smart_fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche valores faltantes usando estratégias inteligentes"""
        if df.empty:
            return df
        
        # Para cada coluna, escolhe a melhor estratégia de preenchimento
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isna().sum() > 0:
                # Se menos de 20% missing, usa interpolação
                if df[column].isna().mean() < 0.2:
                    df[column] = df[column].interpolate(method='linear', limit=10)
                
                # Preenche restantes com rolling mean
                df[column] = df[column].fillna(df[column].rolling(window=6, min_periods=1).mean())
                
                # Último recurso: forward/backward fill
                df[column] = df[column].ffill().bfill()
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Avalia qualidade dos dados do ensemble"""
        if df.empty:
            return {"overall_quality": 0.0}
        
        quality_metrics = {}
        
        # Completude temporal
        if len(df) > 1:
            time_diff = df['timestamp'].diff().dt.total_seconds().mean()
            expected_interval = 300  # 5 minutos
            temporal_completeness = max(0, 1 - (abs(time_diff - expected_interval) / expected_interval))
            quality_metrics['temporal_completeness'] = temporal_completeness
        
        # Completude de dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_completeness = df[numeric_cols].notna().mean().mean()
        quality_metrics['data_completeness'] = data_completeness
        
        # Variabilidade (dados não constantes)
        variability_scores = []
        for col in numeric_cols:
            if df[col].std() > 0:
                variability_scores.append(1.0)
            else:
                variability_scores.append(0.0)
        quality_metrics['variability'] = np.mean(variability_scores) if variability_scores else 0.0
        
        # Qualidade geral (média ponderada)
        weights = {'temporal_completeness': 0.3, 'data_completeness': 0.4, 'variability': 0.3}
        overall_quality = sum(quality_metrics.get(k, 0) * weights.get(k, 0) for k in weights.keys())
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics

def create_ensemble_fetcher(noaa_fetcher, nasa_fetcher) -> EnsembleDataFetcher:
    """Factory function para Ensemble Fetcher"""
    return EnsembleDataFetcher(noaa_fetcher, nasa_fetcher)
'''
    
    with open("src/fetchers/ensemble_fetcher.py", "w") as f:
        f.write(ensemble_content)
    print("✓ fetchers/ensemble_fetcher.py criado")

def create_advanced_detection():
    """Cria módulos avançados de detecção"""
    
    # detection/advanced_detector.py
    advanced_content = '''import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedEventDetector:
    """
    Detector avançado de eventos usando:
    - Análise estatística
    - Detecção de anomalias
    - Clusterização temporal
    - Padrões complexos
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        
        # Parâmetros avançados
        self.anomaly_z_threshold = 3.0
        self.cluster_eps_minutes = 30
        self.pattern_lookback = 12  # pontos para análise de padrões
    
    def detect_advanced_events(self, df: pd.DataFrame, predictions: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Detecção avançada combinando múltiplas técnicas
        """
        events = []
        
        if df is None or df.empty:
            return events
        
        # 1. Detecção baseada em thresholds (básica)
        basic_events = self._detect_threshold_events(df)
        events.extend(basic_events)
        
        # 2. Detecção de anomalias estatísticas
        anomaly_events = self._detect_statistical_anomalies(df)
        events.extend(anomaly_events)
        
        # 3. Detecção de clusters temporais
        cluster_events = self._detect_temporal_clusters(df, events)
        events.extend(cluster_events)
        
        # 4. Detecção de padrões complexos
        pattern_events = self._detect_complex_patterns(df)
        events.extend(pattern_events)
        
        # 5. Integração com previsões de ML
        if predictions is not None and not predictions.empty:
            ml_events = self._integrate_ml_predictions(df, predictions)
            events.extend(ml_events)
        
        # Remove duplicatas e eventos muito próximos
        events = self._deduplicate_events(events)
        
        logger.info(f"Detecção avançada: {len(events)} eventos identificados")
        return events
    
    def _detect_threshold_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecção baseada em thresholds (herdada do detector básico)"""
        events = []
        
        # Bz negativo forte
        if 'bz_gse' in df.columns:
            strong_bz = df[df['bz_gse'] <= self.thresholds.get('strong_negative_bz', -10.0)]
            for _, row in strong_bz.iterrows():
                events.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'type': 'strong_negative_bz',
                    'value': float(row['bz_gse']),
                    'severity': self._classify_bz_severity(row['bz_gse']),
                    'detection_method': 'threshold',
                    'description': f'Bz negativo forte: {row["bz_gse"]:.1f} nT'
                })
        
        # Alta velocidade
        if 'speed' in df.columns:
            high_speed = df[df['speed'] >= self.thresholds.get('high_speed_stream', 600.0)]
            for _, row in high_speed.iterrows():
                events.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'type': 'high_speed_stream',
                    'value': float(row['speed']),
                    'severity': self._classify_speed_severity(row['speed']),
                    'detection_method': 'threshold',
                    'description': f'Alta velocidade: {row["speed"]:.1f} km/s'
                })
        
        return events
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta anomalias usando métodos estatísticos"""
        events = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['hour', 'day_of_year', 'day_of_week', 'month']:
                continue
            
            # Remove outliers extremos para cálculo robusto
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Encontra anomalias
            anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            for _, row in anomalies.iterrows():
                z_score = abs((row[col] - data.mean()) / data.std()) if data.std() > 0 else 0
                
                events.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'type': f'anomaly_{col}',
                    'value': float(row[col]),
                    'severity': 'high' if z_score > 4 else 'medium',
                    'detection_method': 'statistical',
                    'z_score': float(z_score),
                    'description': f'Anomalia em {col}: {row[col]:.1f} (Z-score: {z_score:.1f})'
                })
        
        return events
    
    def _detect_temporal_clusters(self, df: pd.DataFrame, existing_events: List[Dict]) -> List[Dict[str, Any]]:
        """Agrupa eventos próximos temporalmente em clusters"""
        if not existing_events:
            return []
        
        # Converte eventos para DataFrame temporal
        event_times = [pd.to_datetime(event['timestamp']) for event in existing_events]
        event_df = pd.DataFrame({
            'timestamp': event_times,
            'event_index': range(len(event_times))
        }).sort_values('timestamp')
        
        # Converte para minutos desde o primeiro evento
        base_time = event_df['timestamp'].min()
        event_df['minutes_from_start'] = (event_df['timestamp'] - base_time).dt.total_seconds() / 60
        
        # Clusterização temporal
        if len(event_df) > 1:
            clustering = DBSCAN(eps=self.cluster_eps_minutes, min_samples=2)
            event_df['cluster'] = clustering.fit_predict(event_df[['minutes_from_start']])
            
            # Identifica clusters
            cluster_events = []
            for cluster_id in event_df['cluster'].unique():
                if cluster_id == -1:  # Noise
                    continue
                
                cluster_events_idx = event_df[event_df['cluster'] == cluster_id]['event_index']
                cluster_events_list = [existing_events[i] for i in cluster_events_idx]
                
                if len(cluster_events_list) >= 2:
                    # Cria evento de cluster
                    cluster_times = [pd.to_datetime(e['timestamp']) for e in cluster_events_list]
                    cluster_start = min(cluster_times)
                    cluster_end = max(cluster_times)
                    
                    cluster_events.append({
                        'timestamp': cluster_start.isoformat(),
                        'type': 'event_cluster',
                        'value': len(cluster_events_list),
                        'severity': 'high',
                        'detection_method': 'clustering',
                        'duration_minutes': (cluster_end - cluster_start).total_seconds() / 60,
                        'event_count': len(cluster_events_list),
                        'description': f'Cluster de {len(cluster_events_list)} eventos em {(cluster_end - cluster_start).total_seconds() / 60:.1f} minutos'
                    })
            
            return cluster_events
        
        return []
    
    def _detect_complex_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta padrões complexos em múltiplas variáveis"""
        events = []
        
        # Padrão: Bz negativo prolongado + alta velocidade
        if all(col in df.columns for col in ['bz_gse', 'speed']):
            # Janela deslizante para padrões
            window_size = 6  # 30 minutos para dados de 5min
            
            for i in range(len(df) - window_size + 1):
                window = df.iloc[i:i + window_size]
                
                # Condições para padrão de tempestade
                bz_negative = (window['bz_gse'] < -5).all()
                high_speed = (window['speed'] > 500).mean() > 0.8  # 80% da janela
                
                if bz_negative and high_speed:
                    center_time = window.iloc[window_size // 2]['timestamp']
                    
                    events.append({
                        'timestamp': center_time.isoformat(),
                        'type': 'storm_pattern',
                        'value': window['bz_gse'].mean(),
                        'severity': 'high',
                        'detection_method': 'pattern',
                        'description': 'Padrão de tempestade: Bz negativo prolongado com alta velocidade'
                    })
        
        # Padrão: Oscilações rápidas do campo magnético
        if 'bt' in df.columns:
            if len(df) > 10:
                # Calcula variabilidade em janela deslizante
                df['bt_variability'] = df['bt'].rolling(window=6).std()
                
                high_variability = df[df['bt_variability'] > df['bt_variability'].quantile(0.9)]
                
                for _, row in high_variability.iterrows():
                    events.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'type': 'magnetic_oscillation',
                        'value': float(row['bt_variability']),
                        'severity': 'medium',
                        'detection_method': 'pattern',
                        'description': f'Oscilações magnéticas: variabilidade {row["bt_variability"]:.1f} nT'
                    })
        
        return events
    
    def _integrate_ml_predictions(self, df: pd.DataFrame, predictions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Integra previsões do modelo de ML"""
        events = []
        
        if 'predicted_event' not in predictions.columns:
            return events
        
        # Encontra previsões positivas
        ml_events = predictions[predictions['predicted_event'] == 1]
        
        for _, row in ml_events.iterrows():
            events.append({
                'timestamp': row['timestamp'].isoformat(),
                'type': 'predicted_event',
                'value': float(row.get('event_probability', 0.5)),
                'severity': 'medium',
                'detection_method': 'ml_prediction',
                'confidence': float(row.get('event_probability', 0.5)),
                'prediction_horizon': int(row.get('prediction_horizon_hours', 3)),
                'description': f'Evento previsto pelo ML (confiança: {row.get("event_probability", 0.5):.1%})'
            })
        
        return events
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Remove eventos duplicados ou muito próximos"""
        if not events:
            return events
        
        # Ordena por timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        deduplicated = []
        last_time = None
        time_threshold = pd.Timedelta('10 minutes')  # Janela de deduplicação
        
        for event in events:
            current_time = pd.to_datetime(event['timestamp'])
            
            if last_time is None or (current_time - last_time) > time_threshold:
                deduplicated.append(event)
                last_time = current_time
            else:
                # Evento próximo - mantém o de maior severidade
                existing = deduplicated[-1]
                if self._get_severity_score(event) > self._get_severity_score(existing):
                    deduplicated[-1] = event
        
        return deduplicated
    
    def _get_severity_score(self, event: Dict) -> int:
        """Calcula score de severidade para comparação"""
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return severity_scores.get(event.get('severity', 'low'), 1)
    
    def _classify_bz_severity(self, bz_value: float) -> str:
        if bz_value <= -15:
            return 'critical'
        elif bz_value <= -10:
            return 'high'
        else:
            return 'medium'
    
    def _classify_speed_severity(self, speed_value: float) -> str:
        if speed_value >= 800:
            return 'critical'
        elif speed_value >= 600:
            return 'high'
        else:
            return 'medium'

def create_advanced_detector(config: Dict[str, Any]) -> AdvancedEventDetector:
    """Factory function para Advanced Event Detector"""
    return AdvancedEventDetector(config)
'''
    
    with open("src/detection/advanced_detector.py", "w") as f:
        f.write(advanced_content)
    print("✓ detection/advanced_detector.py criado")

def create_enhanced_main():
    """Cria versão aprimorada do main.py com ML e novas fontes"""
    
    enhanced_main = '''#!/usr/bin/env python3
"""
Pipeline Principal Aprimorado - Heliogeophysical Adaptive Coupling v3.0
Inclui ML Preditivo, Múltiplas Fontes de Dados e Detecção Avançada
"""
import logging
import yaml
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Importações internas
from src.utils.logger import setup_logging
from src.fetchers.noaa_fetcher import create_noaa_fetcher
from src.fetchers.nasa_cdaweb_fetcher import create_nasa_fetcher
from src.fetchers.ensemble_fetcher import create_ensemble_fetcher
from src.processing.preprocessor import DataPreprocessor
from src.detection.advanced_detector import create_advanced_detector
from src.model.predictive_model import create_predictor

class EnhancedHeliogeophysicalPipeline:
    """Pipeline aprimorado com ML e múltiplas fontes"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logging(
            self.config['logging']['file'],
            self.config['logging']['level']
        )
        
        # Inicializa componentes avançados
        self.noaa_fetcher = create_noaa_fetcher(self.config['data_sources']['noaa'])
        self.nasa_fetcher = create_nasa_fetcher(self.config['data_sources'].get('nasa_cdaweb', {}))
        self.ensemble_fetcher = create_ensemble_fetcher(self.noaa_fetcher, self.nasa_fetcher)
        self.preprocessor = DataPreprocessor(
            self.config['processing']['resample_frequency']
        )
        self.advanced_detector = create_advanced_detector(self.config['detection'])
        self.predictor = create_predictor()
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return {}
    
    def run_enhanced_pipeline(self) -> dict:
        """Executa pipeline aprimorado completo"""
        self.logger.info("🚀 Iniciando pipeline HELIOGEOPHYSICAL 3.0")
        
        try:
            # Fase 1: Coleta de Dados com Ensemble
            self.logger.info("📡 Coletando dados de múltiplas fontes...")
            ensemble_data = self._fetch_ensemble_data()
            
            if ensemble_data.empty:
                self.logger.warning("Nenhum dado coletado do ensemble")
                return {"events": [], "status": "no_data"}
            
            # Fase 2: Processamento Avançado
            self.logger.info("⚙️ Processamento avançado de dados...")
            processed_data = self._advanced_processing(ensemble_data)
            
            # Fase 3: Detecção de Eventos (Modo Básico)
            self.logger.info("🔍 Detecção básica de eventos...")
            basic_events = self._basic_event_detection(processed_data)
            
            # Fase 4: Treinamento/Atualização do Modelo Preditivo
            self.logger.info("🧠 Operações de Machine Learning...")
            ml_results = self._ml_operations(processed_data, basic_events)
            
            # Fase 5: Detecção Avançada com ML
            self.logger.info("🎯 Detecção avançada com ML...")
            final_events = self._advanced_event_detection(processed_data, ml_results.get('predictions'))
            
            # Fase 6: Análise e Relatórios
            self.logger.info("📊 Gerando relatórios avançados...")
            analysis_report = self._generate_analysis_report(
                processed_data, final_events, ml_results
            )
            
            # Fase 7: Persistência de Resultados
            self._save_enhanced_results(processed_data, final_events, ml_results, analysis_report)
            
            self.logger.info(f"✅ Pipeline concluído — {len(final_events)} eventos detectados")
            
            return {
                "events": final_events,
                "ml_results": ml_results,
                "analysis": analysis_report,
                "processed_records": len(processed_data),
                "status": "success",
                "pipeline_version": "3.0",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline: {e}")
            return {
                "events": [],
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _fetch_ensemble_data(self) -> pd.DataFrame:
        """Coleta dados usando estratégia de ensemble"""
        return self.ensemble_fetcher.fetch_ensemble_data(days=3)
    
    def _advanced_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processamento avançado com features para ML"""
        processed = self.preprocessor.preprocess_data([data])
        
        if processed is not None:
            # Adiciona features específicas para ML
            processed = self.predictor.prepare_features(processed)
        
        return processed if processed is not None else pd.DataFrame()
    
    def _basic_event_detection(self, data: pd.DataFrame) -> list:
        """Detecção básica de eventos para treinamento do ML"""
        from src.detection.simple_detector import detect_events
        return detect_events(data)
    
    def _ml_operations(self, data: pd.DataFrame, events: list) -> dict:
        """Operações de Machine Learning"""
        ml_results = {
            "training_status": "skipped",
            "prediction_status": "skipped",
            "predictions": pd.DataFrame()
        }
        
        try:
            # Verifica se há dados suficientes para treinamento
            if len(data) >= 100 and len(events) >= 5:
                self.logger.info("🔄 Treinando/Atualizando modelo preditivo...")
                
                # Treina o modelo
                training_result = self.predictor.train(data, events)
                ml_results["training_status"] = training_result.get("status", "unknown")
                ml_results["training_metrics"] = training_result
                
                if training_result.get("status") == "success":
                    self.logger.info(f"✅ Modelo treinado - Acurácia: {training_result.get('accuracy', 0):.3f}")
            
            # Faz previsões com o modelo
            self.logger.info("🔮 Fazendo previsões com ML...")
            predictions, probabilities = self.predictor.predict(data)
            
            if not predictions.empty:
                ml_results["prediction_status"] = "success"
                ml_results["predictions"] = predictions
                ml_results["prediction_confidence"] = probabilities.tolist()
                self.logger.info(f"📈 Previsões ML: {predictions['predicted_event'].sum()} eventos previstos")
            
        except Exception as e:
            self.logger.error(f"❌ Erro nas operações de ML: {e}")
            ml_results["error"] = str(e)
        
        return ml_results
    
    def _advanced_event_detection(self, data: pd.DataFrame, ml_predictions: pd.DataFrame) -> list:
        """Detecção avançada integrando ML"""
        return self.advanced_detector.detect_advanced_events(data, ml_predictions)
    
    def _generate_analysis_report(self, data: pd.DataFrame, events: list, ml_results: dict) -> dict:
        """Gera relatório analítico avançado"""
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_quality": {
                "total_records": len(data),
                "data_completeness": data.notna().mean().mean(),
                "temporal_coverage_hours": self._calculate_temporal_coverage(data),
                "variables_available": list(data.select_dtypes(include=[np.number]).columns)
            },
            "events_summary": {
                "total_events": len(events),
                "by_type": self._count_events_by_type(events),
                "by_severity": self._count_events_by_severity(events),
                "by_detection_method": self._count_events_by_method(events)
            },
            "ml_insights": {
                "model_trained": ml_results.get("training_status") == "success",
                "predictions_made": ml_results.get("prediction_status") == "success",
                "predicted_events": int(ml_results.get("predictions", pd.DataFrame()).get("predicted_event", pd.Series()).sum()),
                "average_confidence": np.mean(ml_results.get("prediction_confidence", [])) if ml_results.get("prediction_confidence") else 0
            },
            "recommendations": self._generate_recommendations(events, ml_results)
        }
        
        return report
    
    def _calculate_temporal_coverage(self, data: pd.DataFrame) -> float:
        """Calcula cobertura temporal em horas"""
        if len(data) < 2:
            return 0
        time_span = data['timestamp'].max() - data['timestamp'].min()
        return time_span.total_seconds() / 3600
    
    def _count_events_by_type(self, events: list) -> dict:
        """Conta eventos por tipo"""
        from collections import Counter
        return dict(Counter(event['type'] for event in events))
    
    def _count_events_by_severity(self, events: list) -> dict:
        """Conta eventos por severidade"""
        from collections import Counter
        return dict(Counter(event['severity'] for event in events))
    
    def _count_events_by_method(self, events: list) -> dict:
        """Conta eventos por método de detecção"""
        from collections import Counter
        return dict(Counter(event.get('detection_method', 'unknown') for event in events))
    
    def _generate_recommendations(self, events: list, ml_results: dict) -> list:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações baseadas em eventos
        critical_events = [e for e in events if e.get('severity') in ['high', 'critical']]
        if critical_events:
            recommendations.append("⚠️ Eventos críticos detectados - Monitoramento intensivo recomendado")
        
        # Recomendações baseadas em ML
        if ml_results.get("training_status") == "success":
            accuracy = ml_results.get("training_metrics", {}).get("accuracy", 0)
            if accuracy < 0.7:
                recommendations.append("🤖 Modelo ML com acurácia baixa - Considere retreinamento com mais dados")
        
        if not recommendations:
            recommendations.append("✅ Situação normal - Continue monitoramento de rotina")
        
        return recommendations
    
    def _save_enhanced_results(self, data: pd.DataFrame, events: list, ml_results: dict, analysis: dict):
        """Salva resultados aprimorados"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Salva dados processados
        data_path = f"data/processed/helio_enhanced_{timestamp}.csv"
        data.to_csv(data_path, index=False)
        
        # Salva eventos
        if events:
            events_path = f"data/processed/events_enhanced_{timestamp}.json"
            with open(events_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
        
        # Salva resultados ML
        if not ml_results.get("predictions", pd.DataFrame()).empty:
            ml_path = f"data/processed/ml_predictions_{timestamp}.csv"
            ml_results["predictions"].to_csv(ml_path, index=False)
        
        # Salva relatório de análise
        report_path = f"data/processed/analysis_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"💾 Resultados salvos com prefixo: {timestamp}")

def main():
    """Função principal"""
    pipeline = EnhancedHeliogeophysicalPipeline()
    result = pipeline.run_enhanced_pipeline()
    
    # Relatório Executivo Expandido
    print(f"\\n{'='*60}")
    print("🌌 RELATÓRIO EXECUTIVO - HELIOGEOPHYSICAL 3.0")
    print(f"{'='*60}")
    print(f"📅 Timestamp: {result['timestamp']}")
    print(f"🔄 Status: {result['status']}")
    print(f"📊 Registros processados: {result.get('processed_records', 0)}")
    print(f"🚨 Eventos detectados: {len(result['events'])}")
    print(f"🤖 Status ML: {result.get('ml_results', {}).get('training_status', 'N/A')}")
    
    # Análise de Eventos
    if result['events']:
        print(f"\\n📈 ANÁLISE DE EVENTOS:")
        events_by_type = result.get('analysis', {}).get('events_summary', {}).get('by_type', {})
        for event_type, count in events_by_type.items():
            print(f"   • {event_type}: {count} eventos")
    
    # Recomendações
    recommendations = result.get('analysis', {}).get('recommendations', [])
    if recommendations:
        print(f"\\n💡 RECOMENDAÇÕES:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print(f"{'='*60}")
    print("🎯 Sistema heliogeofísico avançado operacional!")

if __name__ == "__main__":
    main()
'''
    
    with open("src/enhanced_main.py", "w") as f:
        f.write(enhanced_main)
    print("✓ enhanced_main.py criado")

def update_config():
    """Atualiza configuração para incluir novas opções"""
    
    updated_config = '''# Configuração do Projeto Heliogeophysical - Versão 3.0
project:
  name: "Heliogeophysical Adaptive Coupling"
  version: "3.0.0"
  description: "Sistema avançado com ML preditivo e múltiplas fontes"

data_sources:
  noaa:
    base_url: "https://services.swpc.noaa.gov"
    endpoints:
      plasma: "/products/solar-wind/plasma-5-minute.json"
      mag: "/products/solar-wind/mag-5-minute.json"
      dscovr: "/products/solar-wind/dscovr_1m.json"
      alerts: "/products/alerts.json"
    timeout: 30
    retry_attempts: 3

  nasa_cdaweb:
    base_url: "https://cdaweb.gsfc.nasa.gov"
    datasets:
      dscovr_swepam: "DSCOVR_H1_SWEPAM"
      dscovr_mag: "DSCOVR_H1_MAG"
      omni_1min: "OMNI_HRO_1MIN"
      ace_mag: "AC_H0_MFI"
      ace_swepam: "AC_H0_SWE"
      wind_mfi: "WI_H0_MFI"
      wind_swe: "WI_H0_SWE"
    timeout: 60

processing:
  resample_frequency: "5T"
  interpolation_method: "linear"
  rolling_window: "1H"
  features:
    - rolling_mean
    - rolling_std
    - temporal_features
    - derived_parameters
    - ml_features

detection:
  thresholds:
    strong_negative_bz: -10.0
    high_speed_stream: 600.0
    density_spike: 20.0
    temperature_anomaly: 100000.0
  min_event_duration: "5 minutes"
  advanced_methods:
    anomaly_detection: true
    temporal_clustering: true
    pattern_recognition: true
    ml_integration: true

machine_learning:
  prediction_horizon_hours: 3
  model_path: "models/heliogeophysical_model.pkl"
  retrain_interval_hours: 24
  min_training_samples: 100
  min_events_for_training: 5
  features:
    - basic_parameters
    - temporal_features
    - rolling_statistics
    - derived_parameters
    - interaction_terms

storage:
  raw_data: "data/raw"
  processed_data: "data/processed"
  live_data: "data/live"
  model_storage: "models"
  backup:
    enabled: true
    keep_days: 7

logging:
  level: "INFO"
  file: "logs/heliogeophysical.log"
  max_size_mb: 100
  backup_count: 5
  format: "%(asctime)s | %(levelname)-8s | %(message)s"

monitoring:
  enable_health_checks: true
  metrics_port: 9090
  alert_on_failures: true
  performance_tracking: true
'''
    
    with open("src/config/config.yaml", "w") as f:
        f.write(updated_config)
    print("✓ config.yaml atualizado")

def create_requirements_enhanced():
    """Cria requirements atualizado"""
    
    enhanced_req = '''# Core dependencies
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
urllib3>=2.0.0

# Machine Learning
scikit-learn>=1.3.0
joblib>=1.3.0
imbalanced-learn>=0.10.0

# Data science
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0

# NASA CDAWeb access
cdasws>=1.7.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
schedule>=1.2.0

# Testing
pytest>=7.0.0
pytest-mock>=3.10.0

# Optional: Advanced visualization
plotly>=5.14.0
dash>=2.14.0
'''
    
    with open("requirements_enhanced.txt", "w") as f:
        f.write(enhanced_req)
    print("✓ requirements_enhanced.txt criado")

def main():
    print("🚀 EXPANDINDO PROJETO HELIOGEOPHYSICAL...")
    print("Adicionando ML Preditivo e Novas Fontes de Dados")
    print("=" * 60)
    
    create_ml_module()
    create_advanced_fetchers()
    create_advanced_detection()
    create_enhanced_main()
    update_config()
    create_requirements_enhanced()
    
    print("=" * 60)
    print("🎉 EXPANSÃO CONCLUÍDA!")
    print("\\n🆕 NOVAS FUNCIONALIDADES:")
    print("  • 🤖 Modelo de ML preditivo (3h de antecedência)")
    print("  • 🌐 Múltiplas fontes: NOAA + NASA CDAWeb")
    print("  • 🔄 Fetcher de ensemble para dados robustos")
    print("  • 🎯 Detecção avançada com estatística e clustering")
    print("  • 📊 Relatórios analíticos automáticos")
    print("\\n📋 PRÓXIMOS PASSOS:")
    print("1. Instale novas dependências:")
    print("   pip install -r requirements_enhanced.txt")
    print("2. Execute a versão avançada:")
    print("   python src/enhanced_main.py")
    print("3. Para CDAWeb (NASA):")
    print("   pip install cdasws")
    print("\\n💡 DICA: O sistema agora aprende com os dados e melhora com o tempo!")

if __name__ == "__main__":
    main()
