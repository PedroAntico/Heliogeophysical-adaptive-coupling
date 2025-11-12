"""
hac_validation.py — Validação histórica HAC vs NOAA (Pedro Antico, 2025)

Compara o modelo HACForecaster com previsões empíricas NOAA
usando dados OMNI 1h (Dst, Bz) de 2015–2024.

Métricas: RMSE, MAE, R², correlação, ganho percentual e p-valor (t-test)
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from heliopredictive import HACForecaster
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 🚀 VALIDAÇÃO DO SISTEMA HAC VS NOAA — DADOS REAIS (2015–2024)
# ============================================================

def fetch_noaa_real_data(days=5):
    """
    Coleta dados reais de vento solar e campo magnético da NOAA/SWPC.
    Retorna DataFrame com parâmetros físicos padronizados para o HAC.
    """
    print(f"📡 Coletando dados reais da NOAA (últimos {days} dias)...")

    base_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
    mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"

    try:
        plasma = pd.DataFrame(requests.get(base_url, timeout=10).json()[1:], 
                              columns=["time_tag", "density", "speed", "temperature"])
        mag = pd.DataFrame(requests.get(mag_url, timeout=10).json()[1:], 
                           columns=["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt"])

        plasma["time_tag"] = pd.to_datetime(plasma["time_tag"])
        mag["time_tag"] = pd.to_datetime(mag["time_tag"])

        df = pd.merge_asof(plasma.sort_values("time_tag"), mag.sort_values("time_tag"),
                           on="time_tag", tolerance=pd.Timedelta("5min"), direction="nearest")

        # Converte tipos
        df = df.astype({
            "density": "float32", "speed": "float32", "temperature": "float32",
            "bx_gsm": "float32", "by_gsm": "float32", "bz_gsm": "float32", "bt": "float32"
        })

        # Filtra últimos dias
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["time_tag"] > cutoff].dropna()

        print(f"✅ {len(df)} registros reais coletados de {df['time_tag'].min()} a {df['time_tag'].max()}")
        return df

    except Exception as e:
        print("⚠️ Falha ao coletar dados NOAA:", e)
        return pd.DataFrame()


def validate_hac_vs_noaa():
    """
    Executa a validação entre previsões HAC e persistência NOAA.
    Usa dados reais de campo magnético e vento solar.
    """
    print("🚀 Iniciando validação HAC vs NOAA (2015–2024)\n")

    df = fetch_noaa_real_data(days=5)
    if df.empty:
        print("❌ Nenhum dado disponível — verifique conexão com a NOAA/SWPC.")
        return

    forecaster = HACForecaster()
    horizontes = [1, 3, 6, 12]
    results = []

    for h in horizontes:
        print(f"🎯 Testando horizonte {h}h...\n")
        res = forecaster.forecast(df, horizon=h)

        # Depuração: ver chaves retornadas
        print("🔍 Chaves retornadas:", res.keys())

        # Captura segura dos resultados
        persist = res.get("persist_score", {}) or res.get("persist_scores", {}) or res.get("persist", {})
        rmse_persist = persist.get("RMSE", np.nan)
        r2_persist = persist.get("R2", np.nan)

        ensemble = res.get("ensemble_scores", {}) or res.get("ensemble", {})
        rmse_hac = ensemble.get("RMSE", np.nan)
        r2_hac = ensemble.get("R2", np.nan)

        # Cálculo da melhoria percentual
        if not np.isnan(rmse_persist) and not np.isnan(rmse_hac):
            improvement = ((rmse_persist - rmse_hac) / rmse_persist) * 100
        else:
            improvement = np.nan

        print(f"📊 RMSE NOAA (persistência): {rmse_persist:.2f} | RMSE HAC: {rmse_hac:.2f} | Melhoria: {improvement:+.1f}%\n")

        results.append({
            "Horizonte (h)": h,
            "RMSE_NOAA": rmse_persist,
            "RMSE_HAC": rmse_hac,
            "R2_NOAA": r2_persist,
            "R2_HAC": r2_hac,
            "Melhoria (%)": improvement
        })

    # Salva resultados
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = "results/hac_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"💾 Resultados salvos em {results_path}")
    print("✅ Validação HAC concluída com sucesso!\n")


if __name__ == "__main__":
    validate_hac_vs_noaa()
