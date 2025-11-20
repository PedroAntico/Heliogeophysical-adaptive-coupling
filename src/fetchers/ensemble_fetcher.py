import logging
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional
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
        self.data_quality_threshold = 0.60  # mínimo de qualidade aceitável

    def fetch_ensemble_data(self, days: int = 3) -> pd.DataFrame:
        logger.info("Coletando dados de ensemble...")

        all_datasets = []

        # 1. NOAA
        noaa_data = self._fetch_noaa_data(days)
        if noaa_data is not None and not noaa_data.empty:
            all_datasets.append(("NOAA", noaa_data))

        # 2. NASA
        nasa_data = self._fetch_nasa_data(days)
        if nasa_data:
            all_datasets.extend(nasa_data)

        # 3. Merge final
        ensemble_df = self._merge_ensemble_data(all_datasets)

        # 4. Qualidade dos dados
        quality_report = self._assess_data_quality(ensemble_df)

        logger.info(
            f"Ensemble criado: {len(ensemble_df)} linhas — "
            f"Qualidade final: {quality_report['overall_quality']:.1%}"
        )

        return ensemble_df

    # ---------------------- NOAA ----------------------

    def _fetch_noaa_data(self, days: int) -> Optional[pd.DataFrame]:
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
                return preprocessor._merge_datasets(datasets)

        except Exception as e:
            logger.error(f"Erro ao buscar dados NOAA: {e}")

        return None

    # ---------------------- NASA ----------------------

    def _fetch_nasa_data(self, days: int) -> List[tuple]:
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

    # ---------------------- MERGE ----------------------

    def _merge_ensemble_data(self, datasets: List[tuple]) -> pd.DataFrame:
        if not datasets:
            return pd.DataFrame()

        datasets.sort(key=lambda x: 0 if x[0].startswith("NOAA") else 1)

        base_df = datasets[0][1].copy()

        for source_name, df in datasets[1:]:
            if df is None or df.empty:
                continue

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

    # ---------------------- FILL ----------------------

    def _smart_fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        for column in df.select_dtypes(include=[np.number]).columns:

            if df[column].isna().mean() < 0.2:
                df[column] = df[column].interpolate(
                    method="linear", limit=10
                )

            df[column] = df[column].fillna(
                df[column].rolling(window=6, min_periods=1).mean()
            )

            df[column] = df[column].ffill().bfill()

        return df

    # ---------------------- QUALIDADE ----------------------

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"overall_quality": 0.0}

        quality = {}

        # Completude temporal
        if len(df) > 1:
            time_diff = df["timestamp"].diff().dt.total_seconds().mean()
            expected_interval = 300
            temporal = max(0, 1 - abs(time_diff - expected_interval) / expected_interval)
            quality["temporal_completeness"] = temporal

        # Completude de valores
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        completeness = df[numeric_cols].notna().mean().mean()
        quality["data_completeness"] = completeness

        # Variabilidade
        var_scores = [1 if df[col].std() > 0 else 0 for col in numeric_cols]
        quality["variability"] = float(np.mean(var_scores))

        # Score final
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
    return EnsembleDataFetcher(noaa_fetcher, nasa_fetcher)
