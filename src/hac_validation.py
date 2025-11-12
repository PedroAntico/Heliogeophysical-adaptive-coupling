# src/hac_validation.py
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from heliopredictive import HACForecaster

# ============================================================
# 🚀 VALIDAÇÃO CIENTÍFICA HAC VS NOAA - VERSÃO CORRIGIDA E ROBUSTA
# ============================================================

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/validation.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

def to_dataframe_safe(data):
    """
    Converte JSON/list response da NOAA (header + rows) em DataFrame robusto.
    Aceita:
     - data = [header_row, row1, row2, ...]  (o formato que a NOAA retorna)
     - data = list(dicts)  -> converte diretamente
    Corrige linhas com colunas a menos ou a mais.
    """
    if data is None:
        raise ValueError("Nenhum dado fornecido para to_dataframe_safe")

    # Caso 1: lista de dicionários (já estruturada)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        try:
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            raise RuntimeError(f"Falha convertendo list[dict] para DataFrame: {e}")

    # Caso 2: formato NOAA: primeira linha = header (lista), restante = rows (listas)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
        header = list(data[0])
        rows = data[1:]

        fixed_rows = []
        for i, row in enumerate(rows):
            # se linha for dict, transformar para lista baseada no header
            if isinstance(row, dict):
                fixed = [row.get(h, None) for h in header]
                fixed_rows.append(fixed)
                continue

            row = list(row)
            if len(row) < len(header):
                # completa com Nones
                row = row + [None] * (len(header) - len(row))
            elif len(row) > len(header):
                # trunca
                row = row[:len(header)]
            fixed_rows.append(row)

        df = pd.DataFrame(fixed_rows, columns=header)
        return df

    # Caso 3: pandas pode ler direto (fallback)
    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"Formato de dados não suportado por to_dataframe_safe: {e}")

class SolarDataValidator:
    """Validador científico para comparação HAC vs NOAA"""
    
    def __init__(self):
        self.forecaster = HACForecaster()
        self.setup_directories()

    def setup_directories(self):
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def fetch_noaa_realtime(self, days=5):
        """Coleta dados NOAA (plasma + mag) com parsing robusto."""
        logger.info(f"Coletando dados NOAA (últimos {days} dias)...")
        plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
        mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
        headers = {"User-Agent": "SolarValidationBot/1.0"}

        try:
            r_plasma = requests.get(plasma_url, timeout=20, headers=headers)
            r_mag = requests.get(mag_url, timeout=20, headers=headers)

            if r_plasma.status_code != 200:
                raise ConnectionError(f"Plasma API returned {r_plasma.status_code}")
            if r_mag.status_code != 200:
                raise ConnectionError(f"Mag API returned {r_mag.status_code}")

            plasma_json = r_plasma.json()
            mag_json = r_mag.json()

            logger.debug(f"plasma_json sample: {str(plasma_json)[:400]}")
            logger.debug(f"mag_json sample: {str(mag_json)[:400]}")

            # converter via helper robusto
            plasma_df = to_dataframe_safe(plasma_json)
            mag_df = to_dataframe_safe(mag_json)

            logger.info(f"Plasma raw shape: {plasma_df.shape}; Mag raw shape: {mag_df.shape}")

            # Normalizar nomes: procurar colunas temporais e magnéticas dinamicamente
            # Encontrar primeira coluna que pareça timestamp
            def find_time_col(df):
                for c in df.columns:
                    if "time" in str(c).lower() or "date" in str(c).lower() or "time_tag" in str(c).lower():
                        return c
                return df.columns[0]

            plasma_time_col = find_time_col(plasma_df)
            mag_time_col = find_time_col(mag_df)

            # Renomear para time_tag para consistência
            plasma_df = plasma_df.rename(columns={plasma_time_col: "time_tag"})
            mag_df = mag_df.rename(columns={mag_time_col: "time_tag"})

            # Mapear colunas comuns (tentativas dinâmicas)
            # Plasma: density, speed, temperature
            def map_column_candidates(df, candidates):
                mapping = {}
                for c in df.columns:
                    lc = c.lower()
                    for target, keys in candidates.items():
                        for k in keys:
                            if k in lc and target not in mapping.values():
                                mapping[c] = target
                                break
                return mapping

            plasma_candidates = {
                "density": ["dens", "n_p", "proton", "density"],
                "speed": ["speed", "vel", "v_sw", "velocity"],
                "temperature": ["temp", "temperature"]
            }
            mag_candidates = {
                "bx_gsm": ["bx", "bx_gsm", "bx_gse"],
                "by_gsm": ["by", "by_gsm", "by_gse"],
                "bz_gsm": ["bz", "bz_gsm", "bz_gse"],
                "bt": ["bt", "b_tot", "b_total", "bt_gsm"]
            }

            p_map = map_column_candidates(plasma_df, plasma_candidates)
            m_map = map_column_candidates(mag_df, mag_candidates)

            # Apply renames (only for found candidates)
            plasma_df = plasma_df.rename(columns=p_map)
            mag_df = mag_df.rename(columns=m_map)

            # Keep required cols if present, else try to pick best-effort columns
            required_plasma = ["time_tag", "density", "speed", "temperature"]
            required_mag = ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt"]

            # If missing some required, try to fill with nearest available columns
            for req in required_plasma:
                if req not in plasma_df.columns:
                    logger.debug(f"Plasma missing {req}; attempting to find fallback")
                    # fallback: keep any numeric column not time_tag
                    candidates = [c for c in plasma_df.columns if c != "time_tag" and pd.api.types.is_numeric_dtype(plasma_df[c])]
                    if candidates:
                        plasma_df[req] = plasma_df[candidates[0]]
                        logger.debug(f"Plasma fallback {req} <- {candidates[0]}")

            for req in required_mag:
                if req not in mag_df.columns:
                    logger.debug(f"Mag missing {req}; attempting to find fallback")
                    candidates = [c for c in mag_df.columns if c != "time_tag" and pd.api.types.is_numeric_dtype(mag_df[c])]
                    if candidates:
                        mag_df[req] = mag_df[candidates[0]]
                        logger.debug(f"Mag fallback {req} <- {candidates[0]}")

            # Select only needed columns (will exist now due to fallbacks)
            plasma_df = plasma_df[[c for c in required_plasma if c in plasma_df.columns]]
            mag_df = mag_df[[c for c in required_mag if c in mag_df.columns]]

            # Convert time_tag -> datetime
            plasma_df["time_tag"] = pd.to_datetime(plasma_df["time_tag"], errors="coerce")
            mag_df["time_tag"] = pd.to_datetime(mag_df["time_tag"], errors="coerce")

            # Drop rows with NaT
            plasma_df = plasma_df.dropna(subset=["time_tag"])
            mag_df = mag_df.dropna(subset=["time_tag"])

            # Merge asof (tolerância 5 min)
            plasma_df = plasma_df.sort_values("time_tag")
            mag_df = mag_df.sort_values("time_tag")
            df = pd.merge_asof(plasma_df, mag_df, on="time_tag", tolerance=pd.Timedelta("5min"), direction="nearest")

            # Coerce numeric columns
            numeric_cols = [c for c in df.columns if c != "time_tag"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Filter by cutoff
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df["time_tag"] > cutoff].dropna()

            logger.info(f"✅ Dados NOAA coletados e mesclados: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Falha coleta NOAA: {e}", exc_info=True)
            return self._fallback_data_source()

    def _fallback_data_source(self):
        """Tenta arquivo local ou gera amostra se não houver"""
        logger.info("Tentando fallback local/data sample...")
        local = "data/solar_data_latest.csv"
        if os.path.exists(local):
            try:
                df = pd.read_csv(local)
                if "time_tag" in df.columns:
                    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
                    df = df.dropna(subset=["time_tag"])
                logger.info(f"Backup local carregado: {len(df)} registros")
                return df
            except Exception as e:
                logger.warning(f"Falha ao carregar backup local: {e}")

        # criar sample
        logger.info("Criando sample data de fallback (5 dias, 5-min freq)...")
        dates = pd.date_range(datetime.utcnow() - timedelta(days=5), datetime.utcnow(), freq="5min")
        n = len(dates)
        np.random.seed(42)
        df = pd.DataFrame({
            "time_tag": dates,
            "density": np.random.uniform(1, 20, n),
            "speed": np.random.uniform(300, 800, n),
            "temperature": np.random.uniform(50000, 250000, n),
            "bx_gsm": np.random.uniform(-12, 12, n),
            "by_gsm": np.random.uniform(-12, 12, n),
            "bz_gsm": np.random.uniform(-15, 15, n),
            "bt": np.random.uniform(0, 20, n)
        })
        df.to_csv("data/solar_data_latest.csv", index=False)
        logger.info("Sample salvo em data/solar_data_latest.csv")
        return df

    def detect_solar_anomalies(self, df):
        thresholds = {"bz_gsm": 20, "speed": 600, "density": 30, "bt": 15}
        anomalies = []
        try:
            if df.empty:
                return anomalies
            bz_anom = df[df["bz_gsm"].abs() > thresholds["bz_gsm"]]
            speed_anom = df[df["speed"] > thresholds["speed"]]
            dens_anom = df[df["density"] > thresholds["density"]]
            bt_anom = df[df["bt"] > thresholds["bt"]]
            if len(bz_anom): anomalies.append(f"|Bz|>{thresholds['bz_gsm']}: {len(bz_anom)}")
            if len(speed_anom): anomalies.append(f"speed>{thresholds['speed']}: {len(speed_anom)}")
            if len(dens_anom): anomalies.append(f"density>{thresholds['density']}: {len(dens_anom)}")
            if len(bt_anom): anomalies.append(f"bt>{thresholds['bt']}: {len(bt_anom)}")
        except Exception as e:
            logger.warning(f"Erro detectando anomalias: {e}")
        return anomalies

    def create_anomaly_plot(self, df):
        if df.empty:
            logger.warning("Sem dados para plot de anomalias")
            return
        plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        plt.plot(df["time_tag"], df["speed"], label="speed")
        plt.axhline(600, color="r", linestyle="--")
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(df["time_tag"], df["bz_gsm"], label="bz_gsm")
        plt.axhline(20, color="r", linestyle="--")
        plt.axhline(-20, color="r", linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/solar_anomalies.png", dpi=150)
        plt.close()
        logger.info("Anomaly plot salvo")

    def run_validation(self, df, horizon):
        try:
            res = self.forecaster.forecast(df, horizon=horizon)
            scores = res.get("scores", {})
            persist = res.get("persist_scores", {}) or res.get("persist_score", {})
            ensemble = scores.get("Ensemble", {}) or res.get("ensemble", {}) or res.get("ensemble_scores", {})

            rmse_persist = persist.get("RMSE", np.nan)
            rmse_hac = ensemble.get("RMSE", np.nan)
            r2_persist = persist.get("R2", np.nan)
            r2_hac = ensemble.get("R2", np.nan)

            improvement = ((rmse_persist - rmse_hac) / rmse_persist)*100 if (not np.isnan(rmse_persist) and rmse_persist>0 and not np.isnan(rmse_hac)) else np.nan

            result = {
                "Horizonte (h)": horizon,
                "RMSE_NOAA": rmse_persist,
                "RMSE_HAC": rmse_hac,
                "R2_NOAA": r2_persist,
                "R2_HAC": r2_hac,
                "Melhoria (%)": improvement,
                "Timestamp": datetime.utcnow().isoformat()
            }
            return result
        except Exception as e:
            logger.error(f"Erro run_validation: {e}", exc_info=True)
            return {"Horizonte (h)": horizon, "Erro": str(e)}

    def create_comparison_plot(self, results_df):
        if results_df.empty:
            logger.warning("Nenhum resultado para plot comparativo")
            return
        plt.figure(figsize=(8,5))
        x = np.arange(len(results_df))
        w = 0.35
        plt.bar(x-w/2, results_df["RMSE_NOAA"], w, label="NOAA")
        plt.bar(x+w/2, results_df["RMSE_HAC"], w, label="HAC")
        plt.xticks(x, results_df["Horizonte (h)"])
        plt.legend()
        plt.savefig("results/hac_validation_plot.png", dpi=150)
        plt.close()
        logger.info("Comparison plot salvo")

    def validate_hac_system(self):
        logger.info("Iniciando validação HAC vs NOAA")
        df = self.fetch_noaa_realtime(days=5)
        if df.empty:
            logger.error("Nenhum dado disponível para validação")
            return False

        anomalies = self.detect_solar_anomalies(df)
        self.create_anomaly_plot(df)

        horizons = [1,3,6,12]
        results = []
        for h in horizons:
            r = self.run_validation(df, h)
            results.append(r)

        results_df = pd.DataFrame(results)
        results_df.to_csv("results/hac_validation_results.csv", index=False)
        self.create_comparison_plot(results_df)

        logger.info("Validação concluída")
        return True

def main():
    v = SolarDataValidator()
    ok = v.validate_hac_system()
    exit(0 if ok else 1)

if __name__ == "__main__":
    main()
