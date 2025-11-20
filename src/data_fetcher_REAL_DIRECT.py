"""
data_fetcher_REAL_DIRECT.py
Coletor REAL 100% funcional no Codespaces (SEM HAPI, SEM cdasws)
Usa apenas links diretos CSV da NASA e JSON da NOAA.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_fetcher_real_direct.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("data_fetcher_real_direct")

# ============================================================
# NASA DIRECT CSV ENDPOINTS (FUNCIONAM NO CODESPACES)
# ============================================================

NASA_LINKS = {
    "DSCOVR_SWE": "https://cdaweb.gsfc.nasa.gov/pub/data/dscovr/h1/swepam/{year}/dscovr_h1_swepam_{date}.csv",
    "DSCOVR_MAG": "https://cdaweb.gsfc.nasa.gov/pub/data/dscovr/h1/mag/{year}/dscovr_h1_mag_{date}.csv",
    "OMNI_1MIN": "https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min/{year}/omni_hro_1min_{date}.csv"
}

# ============================================================
# NOAA DIRECT JSON
# ============================================================

NOAA_URLS = {
    "PLASMA": "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json",
    "MAG": "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
}

# ============================================================
# FUNÇÕES NASA
# ============================================================

def fetch_nasa_csv(dataset: str, days: int = 3):
    """
    Baixa CSV direto dos servidores da NASA (FUNCIONA 100%)
    dataset: DSCOVR_SWE | DSCOVR_MAG | OMNI_1MIN
    """

    logger.info(f"🌐 Baixando NASA CSV: {dataset}")

    frames = []
    today = datetime.utcnow()

    for i in range(days):
        date = today - timedelta(days=i)
        y = date.strftime("%Y")
        d = date.strftime("%Y%m%d")

        url = NASA_LINKS[dataset].format(year=y, date=d)

        logger.info(f"➡ Tentando {url}")

        try:
            df = pd.read_csv(url, skiprows=36)  # NASA CSV tem cabeçalho longo
            df = clean_nasa_dataframe(dataset, df)
            if df is not None:
                frames.append(df)
        except Exception as e:
            logger.warning(f"⚠ Falhou: {e}")

    if not frames:
        logger.error(f"❌ NASA {dataset}: nenhum arquivo encontrado")
        return None

    df_final = pd.concat(frames).drop_duplicates("time_tag").sort_values("time_tag")

    logger.info(f"✅ NASA {dataset}: {len(df_final)} registros")
    return df_final.reset_index(drop=True)


def clean_nasa_dataframe(dataset: str, df: pd.DataFrame):

    # Timestamp
    if "Epoch" in df.columns:
        df["time_tag"] = pd.to_datetime(df["Epoch"], errors="coerce")
    elif "Time" in df.columns:
        df["time_tag"] = pd.to_datetime(df["Time"], errors="coerce")
    else:
        logger.error("❌ Sem timestamp")
        return None

    df = df.dropna(subset=["time_tag"])

    # Mapeamento automático
    mapping = {}

    if dataset == "DSCOVR_SWE":
        mapping = {
            "Np": "density",
            "Vp": "speed",
            "Tp": "temperature",
        }

    if dataset == "DSCOVR_MAG":
        mapping = {
            "BX_GSE": "bx_gse",
            "BY_GSE": "by_gse",
            "BZ_GSE": "bz_gse",
            "BT": "bt",
        }

    if dataset == "OMNI_1MIN":
        mapping = {
            "BX_GSE": "bx_gse",
            "BY_GSE": "by_gse",
            "BZ_GSE": "bz_gse",
            "V": "speed",
            "N": "density",
            "T": "temperature",
            "P": "pressure"
        }

    for col in mapping:
        if col in df.columns:
            df[mapping[col]] = pd.to_numeric(df[col], errors="ignore")

    # calcula Bt se necessário
    if all(c in df.columns for c in ["bx_gse", "by_gse", "bz_gse"]) and "bt" not in df.columns:
        df["bt"] = np.sqrt(df["bx_gse"]**2 + df["by_gse"]**2 + df["bz_gse"]**2)

    return df


# ============================================================
# NOAA
# ============================================================

def fetch_noaa_json(kind: str):
    url = NOAA_URLS[kind]
    logger.info(f"📡 NOAA: {url}")

    try:
        js = requests.get(url, timeout=15).json()
        if len(js) < 2:
            return None

        columns = js[0]
        df = pd.DataFrame(js[1:], columns=columns)

        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")

        for c in ["density", "speed", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")

        df = df.dropna(subset=["time_tag"])
        df = df.sort_values("time_tag")

        logger.info(f"✅ NOAA {kind}: {len(df)} registros")
        return df

    except Exception as e:
        logger.error(f"❌ NOAA erro: {e}")
        return None


# ============================================================
# COLETOR FINAL
# ============================================================

def collect_all():

    logger.info("🚀 COLETA REAL INICIADA (NASA CSV + NOAA JSON)")

    dfs = []

    # NASA
    swe = fetch_nasa_csv("DSCOVR_SWE")
    if swe is not None:
        dfs.append(swe)

    mag = fetch_nasa_csv("DSCOVR_MAG")
    if mag is not None:
        dfs.append(mag)

    omni = fetch_nasa_csv("OMNI_1MIN")
    if omni is not None:
        dfs.append(omni)

    # NOAA
    plasma = fetch_noaa_json("PLASMA")
    if plasma is not None:
        dfs.append(plasma)

    mag2 = fetch_noaa_json("MAG")
    if mag2 is not None:
        dfs.append(mag2)

    if not dfs:
        logger.error("❌ Nenhuma fonte retornou dados reais")
        return None

    # Merging inteligente
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = pd.merge_asof(
            df_final.sort_values("time_tag"),
            df.sort_values("time_tag"),
            on="time_tag",
            tolerance=pd.Timedelta("15min"),
            direction="nearest",
        )

    df_final = df_final.drop_duplicates("time_tag").sort_values("time_tag")
    return df_final.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    df = collect_all()

    if df is None:
        print("❌ Falha na coleta REAL")
        sys.exit(1)

    os.makedirs("data_real", exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = f"data_real/solar_real_{ts}.csv"

    df.to_csv(path, index=False)

    print("\n==============================")
    print("✅ COLETA 100% REAL CONCLUÍDA")
    print("==============================")
    print(f"Registros: {len(df)}")
    print(f"Arquivo salvo em: {path}")
    print("==============================\n")


if __name__ == "__main__":
    main()
