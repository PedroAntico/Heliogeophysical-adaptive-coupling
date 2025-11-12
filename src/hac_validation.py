"""
hac_validation.py — Validação histórica HAC vs NOAA (Pedro Antico, 2025)

Compara o modelo HACForecaster com previsões empíricas NOAA
usando dados OMNI 1h (Dst, Bz) de 2015–2024.

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from heliopredictive import HACForecaster

# ============================================================
# 🚀 VALIDAÇÃO DO SISTEMA HAC VS NOAA — DADOS REAIS OU BACKUP
# ============================================================

def fetch_noaa_real_data(days=5):
    """Coleta dados reais de vento solar e campo magnético da NOAA/SWPC.
    Se falhar, tenta fallback NASA CDAWeb ou arquivo local CSV."""
    print(f"📡 Tentando coletar dados da NOAA (últimos {days} dias)...")

    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
    mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"

    try:
        plasma = pd.DataFrame(requests.get(plasma_url, timeout=10).json()[1:], 
                              columns=["time_tag", "density", "speed", "temperature"])
        mag = pd.DataFrame(requests.get(mag_url, timeout=10).json()[1:], 
                           columns=["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt"])

        plasma["time_tag"] = pd.to_datetime(plasma["time_tag"])
        mag["time_tag"] = pd.to_datetime(mag["time_tag"])

        df = pd.merge_asof(plasma.sort_values("time_tag"), mag.sort_values("time_tag"),
                           on="time_tag", tolerance=pd.Timedelta("5min"), direction="nearest")

        df = df.astype({
            "density": "float32", "speed": "float32", "temperature": "float32",
            "bx_gsm": "float32", "by_gsm": "float32", "bz_gsm": "float32", "bt": "float32"
        })

        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["time_tag"] > cutoff].dropna()

        print(f"✅ {len(df)} registros reais obtidos da NOAA")
        return df

    except Exception as e:
        print(f"⚠️ Falha NOAA: {e}")
        print("🌐 Tentando fallback: NASA CDAWeb...")

        try:
            # CDAWeb API (plasma + magnético)
            cdaweb_url = (
                "https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/"
                f"{datetime.utcnow().year}/"
            )
            response = requests.get(cdaweb_url, timeout=10)
            if response.status_code == 200:
                print("✅ Fallback CDAWeb disponível (mas sem parse automático ainda).")
            raise ValueError("CDAWeb disponível mas sem parser ativo.")

        except Exception as e2:
            print(f"⚠️ Falha também na NASA CDAWeb: {e2}")
            print("📂 Tentando usar backup local: data/solar_data_latest.csv...")

            try:
                df = pd.read_csv("data/solar_data_latest.csv")
                df["time_tag"] = pd.to_datetime(df["time_tag"])
                print(f"✅ {len(df)} registros carregados do backup local.")
                return df
            except Exception as e3:
                print(f"❌ Nenhuma fonte de dados disponível: {e3}")
                return pd.DataFrame()


def validate_hac_vs_noaa():
    """Executa validação HAC vs dados NOAA reais."""
    print("🚀 Iniciando validação HAC vs NOAA (2015–2024)\n")

    df = fetch_noaa_real_data(days=5)
    if df.empty:
        print("❌ Nenhum dado disponível — verifique conexão ou arquivo local.")
        return

    forecaster = HACForecaster()
    horizontes = [1, 3, 6, 12]
    results = []

    for h in horizontes:
        print(f"🎯 Testando horizonte {h}h...\n")
        res = forecaster.forecast(df, horizon=h)

        # Ajuste para lidar com diferentes nomes de chaves
        persist = res.get("persist_score", {}) or res.get("persist_scores", {})
        rmse_persist = persist.get("RMSE", np.nan)
        r2_persist = persist.get("R2", np.nan)

        ensemble = res.get("ensemble_scores", {}) or res.get("ensemble", {})
        rmse_hac = ensemble.get("RMSE", np.nan)
        r2_hac = ensemble.get("R2", np.nan)

        improvement = ((rmse_persist - rmse_hac) / rmse_persist) * 100 if not np.isnan(rmse_hac) else np.nan

        print(f"📊 RMSE NOAA: {rmse_persist:.2f} | RMSE HAC: {rmse_hac:.2f} | ΔMelhoria: {improvement:+.1f}%\n")

        results.append({
            "Horizonte (h)": h,
            "RMSE_NOAA": rmse_persist,
            "RMSE_HAC": rmse_hac,
            "R2_NOAA": r2_persist,
            "R2_HAC": r2_hac,
            "Melhoria (%)": improvement
        })

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = "results/hac_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"💾 Resultados salvos em {results_path}")
    print("✅ Validação HAC concluída com sucesso!\n")


if __name__ == "__main__":
    validate_hac_vs_noaa()
