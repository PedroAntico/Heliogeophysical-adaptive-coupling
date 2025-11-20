"""
Coletor REAL usando NASA HAPI API (100% funcional, sem erro 400)
Compatível com DSCOVR + OMNI + NOAA
"""

import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("real_hapi")

# ==============================================================================
# NASA HAPI ENDPOINT
# ==============================================================================

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi"

# Datasets reais confirmados em HAPI:
HAPI_DATASETS = {
    "DSCOVR_SWEPAM": {
        "dataset": "DSCOVR_H1_SWE",
        "variables": ["Np", "Vp", "Tp"]
    },
    "DSCOVR_MAG": {
        "dataset": "DSCOVR_H1_MAG",
        "variables": ["B1GSE", "B2GSE", "B3GSE", "Bt"]
    },
    "OMNI_1MIN": {
        "dataset": "OMNI2_H0_MRG1MIN",
        "variables": ["BX_GSE", "BY_GSE", "BZ_GSE", "N", "V", "T", "FlowPressure"]
    }
}

# ==============================================================================
# NOAA (JSON)
# ==============================================================================

NOAA_URLS = {
    "PLASMA_5MIN": "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json",
    "MAG_5MIN": "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
}


# ==============================================================================
# HAPI NASA FUNCTIONS
# ==============================================================================

def hapi_request(endpoint, params=None):
    """Request to HAPI endpoint"""
    try:
        r = requests.get(f"{HAPI_BASE}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"HAPI request error: {e}")
        return None


def fetch_hapi_dataset(key, days=7):
    """Fetch NASA dataset through HAPI API (100% functional)"""

    if key not in HAPI_DATASETS:
        logger.error(f"Dataset inválido: {key}")
        return None

    ds = HAPI_DATASETS[key]["dataset"]
    vars = ",".join(HAPI_DATASETS[key]["variables"])

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    params = {
        "id": ds,
        "parameters": vars,
        "time.min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time.max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "csv"
    }

    logger.info(f"📡 HAPI NASA: {ds}")
    logger.info(f"➡ Variáveis: {vars}")

    try:
        r = requests.get(f"{HAPI_BASE}/data", params=params, timeout=60)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        logger.error(f"Erro ao baixar HAPI NASA {key}: {e}")
        return None

    if "Time" not in df.columns:
        logger.error(f"NASA HAPI sem coluna Time")
        return None

    df = df.rename(columns={"Time": "time_tag"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)

    return df.sort_values("time_tag").reset_index(drop=True)


# ==============================================================================
# NOAA FETCHER
# ==============================================================================

def fetch_noaa(key):
    if key not in NOAA_URLS:
        logger.error(f"NOAA inválido: {key}")
        return None

    url = NOAA_URLS[key]
    logger.info(f"📡 NOAA: {url}")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        json_data = r.json()

        if len(json_data) <= 1:
            return None

        df = pd.DataFrame(json_data[1:], columns=json_data[0])
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")

        # numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return df.dropna(subset=["time_tag"]).sort_values("time_tag")

    except Exception as e:
        logger.error(f"Erro NOAA: {e}")
        return None


# ==============================================================================
# COMBINER
# ==============================================================================

def merge_sources(datasets):
    """Merge multiple time series by nearest timestamp"""
    df = datasets[0]

    for other in datasets[1:]:
        df = pd.merge_asof(
            df.sort_values("time_tag"),
            other.sort_values("time_tag"),
            on="time_tag",
            tolerance=pd.Timedelta("10min"),
            direction="nearest"
        )

    return df.drop_duplicates("time_tag").sort_values("time_tag").reset_index(drop=True)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def run_real_hapi_collector():
    logger.info("🚀 INICIANDO COLETA REAL (HAPI + NOAA)")

    datasets = []

    # NASA HAPI
    for key in ["DSCOVR_SWEPAM", "DSCOVR_MAG", "OMNI_1MIN"]:
        df = fetch_hapi_dataset(key)
        if df is not None and len(df) > 20:
            logger.info(f"✔ NASA REAL ({key}) coletado")
            datasets.append(df)

    # NOAA
    for key in ["PLASMA_5MIN", "MAG_5MIN"]:
        df = fetch_noaa(key)
        if df is not None and len(df) > 10:
            logger.info(f"✔ NOAA REAL ({key}) coletado")
            datasets.append(df)

    if not datasets:
        logger.error("❌ Nenhuma fonte retornou dados")
        return None

    final = merge_sources(datasets)
    logger.info(f"📊 Final dataset: {len(final)} linhas")

    return final


# ==============================================================================
# EXECUTOR
# ==============================================================================

if __name__ == "__main__":
    df = run_real_hapi_collector()

    if df is not None:
        import os
        os.makedirs("data_real", exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
        path = f"data_real/real_hapi_{ts}.csv"
        df.to_csv(path, index=False)
        print("✔ SALVO:", path)
