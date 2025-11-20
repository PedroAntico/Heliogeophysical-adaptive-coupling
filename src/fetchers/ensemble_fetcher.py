import logging
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class EnsembleDataFetcher:
    """
    Fetcher de ensemble que combina múltiplas fontes de dados
    (NOAA + NASA CDAWeb) em um dataset robusto.
    """

    def __init__(self, noaa_fetcher, nasa_fetcher):
        self.noaa_fetcher = noaa_fetcher
        self.nasa_fetcher = nasa_fetcher
        
        # Corrigido: antes estava usando "%", o que causava erro.
        self.data_quality_threshold = 0.60  # mínimo de qualidade aceitável

    def fetch_ensemble_data(self, days: int = 3) -> pd.DataFrame:
        """
        Busca dados de múltiplas fontes e combina em dataset único.
        """
        logger.info("Coletando dados de ensemble...")

        all_datasets = []

        # 1. NOAA (prioridade 1)
        noaa_data = self._fetch_noaa_data(days)
        if noaa_data is not None and not noaa_data.empty:
            all_datasets.append(("NOAA", noaa_data))

        # 2. NASA CDAWeb
        nasa_data = self._fetch_nasa_data(days)
        if nasa_data:
            all_datasets.extend(nasa_data)

        # 3. Merge de todos os datasets
        ensemble_df = self._merge_ensemble_data(all_datasets)

        # 4. Avaliação de qualidade
        quality_report = self._assess_data_quality(ensemble_df)

        logger.info(
            f"Ensemble criado: {len(ensemble_df)} linhas — "
            f"Qualidade final: {quality_report['overall_quality']:.1%}"
        )

        return ensemble_df

    # ---------------------------------------------------------
    # -------------- COLETA NOAA ------------------------------
    # ---------------------------------------------------------

    def _fetch_noaa_data(self, days: int) -> Optional[pd.DataFrame]:
        """Busca dados NOAA combinando plasma + magnetômetro"""

        try:
            datasets = []

            plasma = self.noaa_fetcher.fetch_plasma_data(days)
            if plasma is not None and not plasma.empty:
                datasets.append(plasma)

            mag = self.noaa_fetcher.fetch_magnetic_data(days)
            if mag is not None and not mag.empty:
                datasets.append(mag)

            if datasets:
                from ..processing.preprocessor import DataPreprocessor
                preprocessor = DataPreprocessor()

                combined = preprocessor._merge_datasets(datasets)
                return combined

        except Exception as e:
            logger.error(f"Erro ao buscar dados NOAA: {e}")

        return None

    # ---------------------------------------------------------
    # -------------- COLETA NASA CDAWEB -----------------------
    # ---------------------------------------------------------

    def _fetch_nasa_data(self, days: int) -> List[tuple]:
        """Busca múltiplos datasets da NASA CDAWeb"""
        nasa_datasets = []

        try:
            priority_datasets = ['dscovr_swepam', 'dscovr_mag', 'omni_1min']

            nasa_results = self.nasa_fetcher.fetch_multiple_datasets(
                priority_datasets, days
            )

            for name, df in nasa_results.items():
                if df is not None and not df.empty:
                    nasa_datasets.append((f"NASA_{name}", df))

        except Exception as e:
            logger.error(f"Erro ao buscar dados NASA: {e}")

        return nasa_datasets

    # ---------------------------------------------------------
    # -------------- MERGE ENSEMBLE ---------------------------
    # ---------------------------------------------------------

    def _merge_ensemble_data(self, datasets: List[tuple]) -> pd.DataFrame:
        """Combina datasets múltiplos com merge_asof (tolerância 15 min)"""

        if not datasets:
            return pd.DataFrame()

        # Ordena por prioridade: NOAA antes
        datasets.sort(key=lambda x: 0 if x[0].startswith("NOAA") else 1)

        base_df = datasets[0][1].copy()

        for source_name, df in datasets[1:]:
            if df is None or df.empty:
                continue

            # Colunas que DF possui e base_df não
            new_columns = [
                col for col in df.columns
                if col not in base_df.columns and col != "timestamp"
            ]

            if not new_columns:
                continue

            merge_df = df[['timestamp'] + new_columns].sort_values("timestamp")

            base_df = pd.merge_asof(
                base_df.sort_values("timestamp"),
                merge_df,
                on="timestamp",
                tolerance=pd.Timedelta("15min"),
                direction="nearest"
            )

            logger.info(f"Merge com {source_name}: adicionadas {len(new_columns)} colunas")

        base_df = self._smart_fill_missing_values(base_df)
        return base_df.sort_values("timestamp").reset_index(drop=True)

    # ---------------------------------------------------------
    # -------------- FILL DE VALORES FALTANTES ---------------
    # ---------------------------------------------------------

    def _smart_fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usa estratégias inteligentes para preencher missing values"""

        if df.empty:
            return df

        for column in df.select_dtypes(include=[np.number]).columns:

            # 1. Interpolação se poucos missing
            if df[column].isna().mean() < 0.2:
                df[column] = df[column].interpolate(
                    method="linear", limit=10
                )

            # 2. Rolling mean
            df[column] = df[column].fillna(
                df[column].rolling(window=6, min_periods=1).mean()
            )

            # 3. Forward/backward fill
            df[column] = df[column].ffill().bfill()

        return df

    # ---------------------------------------------------------
    # -------------- MÉTRICAS DE QUALIDADE -------------------
    # ---------------------------------------------------------

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Avalia completude, variabilidade e consistência temporal"""

        if df.empty:
            return {"overall_quality": 0.0}

        quality = {}

        # 1. Completude temporal
        if len(df) > 1:
            time_diff = df["timestamp"].diff().dt.total_seconds().mean()
            expected_interval = 300  # 5 min
            temporal_completeness = max(
                0, 1 - abs(time_diff - expected_interval) / expected_interval
            )
            quality["temporal_completeness"] = temporal_completeness

        # 2. Completude de dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_completeness = df[numeric_cols].notna().mean().mean()
        quality["data_completeness"] = data_completeness

        # 3. Variabilidade (não ser constante)
        var_scores = [1 if df[col].std() > 0 else 0 for col in numeric_cols]
        quality["variability"] = np.mean(var_scores)

        # 4. Score final ponderado
        weights = {
            "temporal_completeness": 0.3,
            "data_completeness": 0.4,
            "variability": 0.3,
        }

        quality["overall_quality"] = sum(
            quality.get(k, 0) * weights[k] for k in weights
        )

        return quality


def create_ensemble_fetcher(noaa_fetcher, nasa_fetcher) -> EnsembleDataFetcher:
    """Factory function"""
    return EnsembleDataFetcher(noaa_fetcher, nasa_fetcher)
