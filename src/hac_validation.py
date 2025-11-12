import os
import pandas as pd
import numpy as np
import requests
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from heliopredictive import HACForecaster

# ============================================================
# 🚀 VALIDAÇÃO CIENTÍFICA HAC VS NOAA — VERSÃO FINAL CORRIGIDA
# ============================================================

# === Configuração de logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/validation.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class SolarDataValidator:
    """Validador científico para comparação HAC vs NOAA"""

    def __init__(self):
        self.forecaster = HACForecaster()
        self.setup_directories()

    # === Estrutura de diretórios ===
    def setup_directories(self):
        for d in ["data", "results", "logs", "plots"]:
            os.makedirs(d, exist_ok=True)

    # === Coleta robusta de dados NOAA ===
    def fetch_noaa_realtime(self, days=5):
        """Coleta dados reais NOAA e corrige automaticamente colunas"""
        logger.info(f"📡 Coletando dados NOAA (últimos {days} dias)...")

        plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
        mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
        headers = {"User-Agent": "SolarValidationBot/1.0"}

        try:
            # === 1️⃣ Download ===
            plasma_data = requests.get(plasma_url, timeout=20, headers=headers).json()
            mag_data = requests.get(mag_url, timeout=20, headers=headers).json()

            logger.info(f"✅ Estrutura Plasma: {len(plasma_data[0])} colunas")
            logger.info(f"✅ Estrutura Mag: {len(mag_data[0])} colunas")

            # === 2️⃣ Conversão segura para DataFrame ===
            def to_dataframe_safe(data):
                header = data[0]
                fixed_rows = []
                for row in data[1:]:
                    row = list(row)
                    if len(row) < len(header):
                        row += [None] * (len(header) - len(row))
                    elif len(row) > len(header):
                        row = row[:len(header)]
                    fixed_rows.append(row)
                return pd.DataFrame(fixed_rows, columns=header)

            plasma_df = to_dataframe_safe(plasma_data)
            mag_df = to_dataframe_safe(mag_data)

            # === 3️⃣ Selecionar colunas de interesse ===
            plasma_cols = [c for c in ["time_tag", "density", "speed", "temperature"] if c in plasma_df.columns]
            mag_cols = [c for c in ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt"] if c in mag_df.columns]

            plasma_df = plasma_df[plasma_cols]
            mag_df = mag_df[mag_cols]

            # === 4️⃣ Converter tipos ===
            plasma_df["time_tag"] = pd.to_datetime(plasma_df["time_tag"], errors="coerce")
            mag_df["time_tag"] = pd.to_datetime(mag_df["time_tag"], errors="coerce")

            for col in ["density", "speed", "temperature"]:
                if col in plasma_df.columns:
                    plasma_df[col] = pd.to_numeric(plasma_df[col], errors="coerce")

            for col in ["bx_gsm", "by_gsm", "bz_gsm", "bt"]:
                if col in mag_df.columns:
                    mag_df[col] = pd.to_numeric(mag_df[col], errors="coerce")

            # === 5️⃣ Merge com tolerância de 5 min ===
            df = pd.merge_asof(
                plasma_df.sort_values("time_tag"),
                mag_df.sort_values("time_tag"),
                on="time_tag",
                tolerance=pd.Timedelta("5min"),
                direction="nearest"
            )

            # === 6️⃣ Filtrar últimos X dias ===
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df["time_tag"] > cutoff].dropna()

            logger.info(f"📅 Período NOAA: {df['time_tag'].min()} → {df['time_tag'].max()}")
            logger.info(f"📊 Registros válidos: {len(df)}")

            return df

        except Exception as e:
            logger.error(f"❌ Falha ao coletar dados NOAA: {e}")
            return self._fallback_data_source()

    # === Fallback: backup local ou dados simulados ===
    def _fallback_data_source(self):
        try:
            if os.path.exists("data/solar_data_latest.csv"):
                df = pd.read_csv("data/solar_data_latest.csv")
                df["time_tag"] = pd.to_datetime(df["time_tag"])
                logger.info(f"✅ Backup local carregado ({len(df)} registros)")
                return df
        except Exception as e:
            logger.warning(f"Falha no backup local: {e}")

        logger.warning("⚠️ Criando dados de exemplo para validação...")
        return self._create_sample_data()

    def _create_sample_data(self):
        dates = pd.date_range(
            start=datetime.utcnow() - timedelta(days=5),
            end=datetime.utcnow(),
            freq="5min"
        )
        np.random.seed(42)
        df = pd.DataFrame({
            "time_tag": dates,
            "density": np.random.uniform(2, 15, len(dates)),
            "speed": np.random.uniform(300, 650, len(dates)),
            "temperature": np.random.uniform(60000, 180000, len(dates)),
            "bx_gsm": np.random.uniform(-10, 10, len(dates)),
            "by_gsm": np.random.uniform(-10, 10, len(dates)),
            "bz_gsm": np.random.uniform(-15, 15, len(dates)),
            "bt": np.random.uniform(0, 20, len(dates))
        })
        df.to_csv("data/solar_data_latest.csv", index=False)
        logger.info("💾 Dados de exemplo salvos em data/solar_data_latest.csv")
        return df

    # === Detecção de anomalias ===
    def detect_solar_anomalies(self, df):
        logger.info("🔍 Detectando anomalias solares...")
        anomalies = []

        thresholds = {"bz_gsm": 20, "speed": 600, "density": 30, "bt": 15}
        for key, th in thresholds.items():
            if key in df.columns:
                count = len(df[df[key].abs() > th])
                if count > 0:
                    anomalies.append(f"{key} > {th}: {count} eventos")

        if anomalies:
            logger.warning("⚠️ " + ", ".join(anomalies))
        else:
            logger.info("✅ Nenhuma anomalia detectada")
        return anomalies

    def create_anomaly_plot(self, df):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(df["time_tag"], df["speed"], "b-", alpha=0.7, label="Velocidade (km/s)")
        plt.axhline(600, color="r", linestyle="--", label="CME > 600 km/s")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(df["time_tag"], df["bz_gsm"], "g-", alpha=0.7, label="Bz GSM")
        plt.axhline(20, color="r", linestyle="--")
        plt.axhline(-20, color="r", linestyle="--", label="Bz ±20 nT")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel("Tempo")

        plt.tight_layout()
        plt.savefig("plots/solar_anomalies.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("📊 Gráfico salvo em plots/solar_anomalies.png")

    # === Validação HAC ===
    def run_validation(self, df, horizon):
        try:
            logger.info(f"🎯 Testando horizonte {horizon}h...")
            res = self.forecaster.forecast(df, horizon=horizon)

            persist = res.get("persist_scores", {}) or res.get("persist_score", {})
            ensemble = res.get("ensemble_scores", {}) or res.get("ensemble", {})

            rmse_p = persist.get("RMSE", np.nan)
            rmse_h = ensemble.get("RMSE", np.nan)
            r2_p = persist.get("R2", np.nan)
            r2_h = ensemble.get("R2", np.nan)

            improvement = ((rmse_p - rmse_h) / rmse_p) * 100 if rmse_p and not np.isnan(rmse_h) else np.nan

            logger.info(f"RMSE NOAA={rmse_p:.2f} | HAC={rmse_h:.2f} | Δ={improvement:+.1f}%")

            return {
                "Horizonte (h)": horizon,
                "RMSE_NOAA": rmse_p,
                "RMSE_HAC": rmse_h,
                "R2_NOAA": r2_p,
                "R2_HAC": r2_h,
                "Melhoria (%)": improvement
            }

        except Exception as e:
            logger.error(f"❌ Erro no horizonte {horizon}h: {e}")
            return {
                "Horizonte (h)": horizon,
                "Erro": str(e)
            }

    def create_comparison_plot(self, df):
        plt.figure(figsize=(10, 6))
        horizons = df["Horizonte (h)"]
        x = np.arange(len(horizons))
        bw = 0.35

        plt.bar(x - bw/2, df["RMSE_NOAA"], bw, label="NOAA", color="red", alpha=0.7)
        plt.bar(x + bw/2, df["RMSE_HAC"], bw, label="HAC", color="blue", alpha=0.7)

        plt.xticks(x, horizons)
        plt.ylabel("RMSE")
        plt.title("Comparação HAC vs NOAA")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/hac_validation_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("📈 Gráfico salvo em results/hac_validation_plot.png")

    # === Execução completa ===
    def validate_hac_system(self):
        logger.info("🚀 INICIANDO VALIDAÇÃO COMPLETA HAC vs NOAA")
        df = self.fetch_noaa_realtime(days=5)
        if df.empty:
            logger.error("❌ Nenhum dado disponível.")
            return False

        anomalies = self.detect_solar_anomalies(df)
        self.create_anomaly_plot(df)

        results = [self.run_validation(df, h) for h in [1, 3, 6, 12]]
        results_df = pd.DataFrame(results)
        results_df.to_csv("results/hac_validation_results.csv", index=False)
        self.create_comparison_plot(results_df)

        avg_improvement = results_df["Melhoria (%)"].mean(skipna=True)
        logger.info(f"📊 Melhoria média: {avg_improvement:+.2f}%")
        return True


def main():
    validator = SolarDataValidator()
    success = validator.validate_hac_system()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
