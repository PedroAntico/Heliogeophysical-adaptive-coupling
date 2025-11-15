import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOAA_URL = "https://services.swpc.noaa.gov/json/omni/omni_lite.json"

def fetch_noaa():
    try:
        logger.info("🌐 Coletando dados da NOAA (OMNI Lite)...")
        response = requests.get(NOAA_URL, timeout=20)

        if response.status_code != 200:
            logger.error(f"NOAA retornou erro HTTP {response.status_code}")
            return None
        
        data = response.json()

        if not data or len(data) < 10:
            logger.error("NOAA retornou lista vazia")
            return None
        
        df = pd.DataFrame(data)

        # Normalização básica de colunas
        rename_map = {
            "density": "proton_density",
            "speed": "speed_km_s",
            "bt": "bt_nT",
            "bx_gse": "bx_nT",
            "by_gse": "by_nT",
            "bz_gse": "bz_nT"
        }
        df.rename(columns=rename_map, inplace=True)

        # Converter timestamps
        if "time_tag" in df.columns:
            df["datetime"] = pd.to_datetime(df["time_tag"])

        logger.info("✅ NOAA OK — dados reais obtidos")
        return df

    except Exception as e:
        logger.error(f"Falha NOAA: {e}")
        return None


def fallback_synthetic():
    """Gera dados sintéticos caso tudo falhe"""
    logger.warning("⚠️ ATENÇÃO: usando dados sintéticos fallback!")

    now = datetime.utcnow()
    times = [now - timedelta(hours=i) for i in range(24)]
    times.reverse()

    df = pd.DataFrame({
        "datetime": times,
        "proton_density": np.random.uniform(3, 12, 24),
        "speed_km_s": np.random.uniform(280, 500, 24),
        "bt_nT": np.random.uniform(2, 8, 24),
        "bx_nT": np.random.uniform(-5, 5, 24),
        "by_nT": np.random.uniform(-5, 5, 24),
        "bz_nT": np.random.uniform(-6, 6, 24),
        "synthetic": True
    })

    return df


def save(df):
    os.makedirs("data/observational", exist_ok=True)
    path = "data/observational/heliospheric_observational.csv"
    df.to_csv(path, index=False)
    logger.info(f"💾 Arquivo salvo em: {path}")


def run():
    logger.info("=== INICIANDO COLETA OBSERVACIONAL ===")

    df = fetch_noaa()

    if df is None:
        logger.error("❌ NOAA falhou — usando fallback sintético")
        df = fallback_synthetic()

    save(df)

    logger.info("🏁 Processo finalizado.")


if __name__ == "__main__":
    run()
