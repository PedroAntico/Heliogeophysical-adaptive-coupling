import logging
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
