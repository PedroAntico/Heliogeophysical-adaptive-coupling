"""
src/data_fetcher_REAL.py
Coletor DEFINITIVO com variáveis e datasets 100% REAIS (NASA + NOAA)
Compatível com data_sources_REAL.py atualizado
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Importa configurações corretas
from config.data_sources_REAL import (
    NASA_REAL_SOURCES,
    NOAA_REAL_SOURCES,
    VARIABLE_MAPPING_REAL,
    VALIDATION_CONFIG,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_fetcher_real.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("data_fetcher_real")


# =====================================================================================
# COLETOR PRINCIPAL
# =====================================================================================

class RealDataCollector:
    """Coleta dados REAIS da NASA + NOAA com datasets que EXISTEM e estão ATIVOS"""

    def __init__(self):
        self.data_source = None

    # =============================================================================
    # NASA CDAWEB — datasets REAIS
    # =============================================================================

    def fetch_nasa(self, key: str, days: int = 7):
        """
        Busca dados NASA usando dataset_id REAL definido em data_sources_REAL.py
        """
        try:
            if key not in NASA_REAL_SOURCES:
                logger.error(f"❌ Dataset NASA inexistente: {key}")
                return None

            config = NASA_REAL_SOURCES[key]
            dataset_id = config.dataset_id
            variables = config.variables

            logger.info(f"🌐 Conectando à NASA CDAWeb")
            logger.info(f"➡ Dataset: {dataset_id}")
            logger.info(f"➡ Variáveis: {variables}")

            from cdasws import CdasWs
            cdas = CdasWs()

            end = datetime.utcnow()
            start = end - timedelta(days=days)

            status, data = cdas.get_data(
                dataset_id,
                variables,
                start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

            if status != 200 or not data:
                logger.error(f"❌ NASA retornou status {status}")
                return None

            df = self._process_nasa_data(key, data)

            if len(df) < VALIDATION_CONFIG["min_data_points"]:
                logger.warning(f"⚠ Dados insuficientes ({len(df)} registros)")
                return None

            logger.info(f"✅ NASA {key}: {len(df)} registros REAIS")
            return df

        except Exception as e:
            logger.error(f"❌ Erro NASA ({key}): {e}")
            return None

    def _process_nasa_data(self, key: str, data: dict):
        """Processa dados NASA conforme mapeamento REAL"""

        df = pd.DataFrame()

        # --- Timestamp ---
        if "Epoch" in data:
            df["time_tag"] = pd.to_datetime(data["Epoch"], utc=True)
        else:
            logger.error("❌ NASA sem coluna Epoch")
            return pd.DataFrame()

        # --- Mapeamento para nomes HAC ---
        mapping = VARIABLE_MAPPING_REAL[key]

        for nasa_var, hac_var in mapping.items():
            if nasa_var in data:
                df[hac_var] = pd.to_numeric(data[nasa_var], errors="coerce")

        # --- Calcular Bt se necessário ---
        if all(c in df.columns for c in ["bx_gse", "by_gse", "bz_gse"]) and "bt" not in df.columns:
            df["bt"] = np.sqrt(df["bx_gse"]**2 + df["by_gse"]**2 + df["bz_gse"]**2)

        df = df.dropna(subset=["time_tag"]).sort_values("time_tag")

        return df.reset_index(drop=True)

    # =============================================================================
    # NOAA JSON — plasma e magnetômetro em 5 min
    # =============================================================================

    def fetch_noaa(self, key: str):
        """
        Busca NOAA com JSON estruturado
        """
        try:
            if key not in NOAA_REAL_SOURCES:
                logger.error(f"❌ Tipo NOAA inexistente: {key}")
                return None

            config = NOAA_REAL_SOURCES[key]

            url = (
                "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
                if key == "PLASMA_5MIN"
                else "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
            )

            logger.info(f"📡 Buscando NOAA: {url}")

            response = requests.get(url, timeout=20)

            if response.status_code != 200:
                logger.error(f"❌ NOAA HTTP {response.status_code}")
                return None

            json_data = response.json()

            if len(json_data) <= 1:
                logger.error("❌ NOAA retornou JSON vazio")
                return None

            columns = json_data[0]
            rows = json_data[1:]

            df = pd.DataFrame(rows, columns=columns)

            return self._process_noaa_data(df)

        except Exception as e:
            logger.error(f"❌ Erro NOAA ({key}): {e}")
            return None

    def _process_noaa_data(self, df):
        """Converte NOAA para formato HAC"""

        if "time_tag" not in df.columns:
            logger.error("❌ NOAA sem coluna time_tag")
            return pd.DataFrame()

        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")

        for col in ["density", "speed", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["time_tag"]).sort_values("time_tag")

        # Cortar últimos 30 dias
        cutoff = datetime.utcnow() - timedelta(days=30)
        df = df[df["time_tag"] >= cutoff]

        return df.reset_index(drop=True)

    # =============================================================================
    # COMBINAR E VALIDAR
    # =============================================================================

    def collect_real(self):
        """Estratégia: NASA primeiro, depois NOAA"""

        logger.info("🚀 INICIANDO COLETA 100% REAL")

        sources = []
        dfs = []

        # Ordem de tentativa (correta)
        sources_try = [
            ("nasa", "DSCOVR_SWEPAM"),
            ("nasa", "DSCOVR_MAG"),
            ("nasa", "OMNI_1MIN"),
            ("noaa", "PLASMA_5MIN"),
            ("noaa", "MAG_5MIN"),
        ]

        for source_type, key in sources_try:
            logger.info(f"🔄 Tentando {source_type.upper()}: {key}")

            df = (
                self.fetch_nasa(key) if source_type == "nasa"
                else self.fetch_noaa(key)
            )

            if df is not None and len(df) >= VALIDATION_CONFIG["min_data_points"]:
                logger.info(f"✅ Fonte adicionada: {key} ({len(df)} registros)")
                sources.append(key)
                dfs.append(df)

            if len(dfs) >= 2:  # suficiente
                break

        if not dfs:
            logger.error("❌ Todas as fontes falharam")
            return None, None

        # Combinar múltiplas fontes
        final = dfs[0]
        for other in dfs[1:]:
            final = pd.merge_asof(
                final.sort_values("time_tag"),
                other.sort_values("time_tag"),
                on="time_tag",
                tolerance=pd.Timedelta("10min"),
                direction="nearest",
            )

        final = final.drop_duplicates("time_tag").sort_values("time_tag")

        return final.reset_index(drop=True), sources


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    collector = RealDataCollector()

    df, used = collector.collect_real()

    if df is None:
        print("\n❌ COLETA REAL FALHOU")
        sys.exit(1)

    os.makedirs("data_real", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = f"data_real/solar_real_{ts}.csv"

    df.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("✅ COLETA 100% REAL CONCLUÍDA")
    print("=" * 70)
    print(f"📡 Fontes utilizadas: {used}")
    print(f"📊 Registros: {len(df)}")
    print(f"💾 Arquivo salvo: {path}")


if __name__ == "__main__":
    main()
