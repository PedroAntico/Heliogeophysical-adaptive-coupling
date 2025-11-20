import logging
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
