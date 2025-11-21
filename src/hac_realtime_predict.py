"""
hac_realtime_predict.py
Previsor em tempo real + Alertas de Tempestades Solares
Versão FINAL — compatível com modelos .keras (HAC Deep Learning)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_RT - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HAC_RT")

# ============================================================
# UTILIDADES
# ============================================================

def safe_timestamp(obj):
    """Converte datetime/Timestamp para string JSON-friendly."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return obj

def json_safe(data):
    """Converte recursivamente todos os tipos para JSON-safe."""
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [json_safe(v) for v in data]
    return safe_timestamp(data)

# ============================================================
# CARREGAR DADOS
# ============================================================

def load_latest_solar_data():
    folders = ["data_real", "data", "data/raw", "data/processed"]

    for folder in folders:
        if not os.path.exists(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)
        if files:
            latest = os.path.join(folder, files[0])
            logger.info(f"📥 Carregando dados: {latest}")
            df = pd.read_csv(latest)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            return df

    raise FileNotFoundError("Nenhum CSV encontrado nas pastas de dados.")

# ============================================================
# CRIAR SEQUÊNCIA PARA PREVISÃO
# ============================================================

def prepare_sequence(df, features, lookback):
    seq = df[features].tail(lookback).values

    scaler = StandardScaler()
    seq_scaled = scaler.fit_transform(seq)

    return seq_scaled.reshape(1, lookback, len(features)), scaler

# ============================================================
# CARREGAMENTO CORRIGIDO DOS MODELOS
# ============================================================

def load_models():
    model_dir = "models/deep_hac"
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Nenhum modelo encontrado em models/deep_hac/")

    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".keras"):   # ← agora correto!
            path = os.path.join(model_dir, fname)
            logger.info(f"📦 Carregando modelo: {fname}")

            parts = fname.split("_")
            model_type = parts[0]
            horizon = int(parts[1].replace("h", ""))

            if model_type not in models:
                models[model_type] = {}

            # evitar erro de métricas → compile=False
            model = load_model(path, compile=False)
            model.compile(optimizer="adam", loss="mse")

            models[model_type][horizon] = model

    return models

# ============================================================
# PREVISÃO E ALERTAS
# ============================================================

def classify_risk(speed, bz):
    score = 0
    if speed > 550: score += 1
    if speed > 700: score += 2
    if bz < -5: score += 1
    if bz < -10: score += 2
    return score

def risk_level(score):
    if score <= 1: return "Baixo"
    if score <= 3: return "Moderado"
    return "Alto"

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def run_realtime_hac():
    try:
        logger.info("🚀 HAC 3.0 — Previsor em Tempo Real iniciado")

        df = load_latest_solar_data()

        possible_features = [
            "speed", "density", "temperature",
            "bx_gse", "by_gse", "bz_gse", "bt"
        ]
        features = [f for f in possible_features if f in df.columns]

        if len(features) < 2:
            raise ValueError("Poucas features disponíveis para prever.")

        models = load_models()

        lookback = 36
        seq, scaler = prepare_sequence(df, features, lookback)

        predictions = {}

        for model_type in models:
            predictions[model_type] = {}
            for horizon_h, model in models[model_type].items():
                pred_scaled = model.predict(seq, verbose=0)[0][0]
                pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0][0]
                predictions[model_type][horizon_h] = float(pred_unscaled)

        current = {
            "timestamp": safe_timestamp(df["timestamp"].iloc[-1]) 
                if "timestamp" in df.columns else None,
            "speed": float(df["speed"].iloc[-1]) if "speed" in df.columns else None,
            "bz": float(df["bz_gse"].iloc[-1]) if "bz_gse" in df.columns else None,
        }

        score = classify_risk(current["speed"], current["bz"])
        risk = risk_level(score)

        output = json_safe({
            "generated_at": datetime.utcnow(),
            "current_conditions": current,
            "risk_score": score,
            "risk_level": risk,
            "predictions": predictions
        })

        os.makedirs("results/realtime", exist_ok=True)
        out_file = f"results/realtime/hac_rt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("\n🌞 Condição Atual:")
        logger.info(f"Velocidade: {current['speed']} km/s | Bz: {current['bz']} nT")
        logger.info(f"\n⚠️ Risco geomagnético: {risk} (score={score})")
        logger.info("\n🔮 Previsões:")

        for model_type in predictions:
            for horizon, value in predictions[model_type].items():
                logger.info(f"{model_type.upper()} H{horizon}: {value:.2f}")

        logger.info(f"\n💾 Resultado salvo em: {out_file}\n")

    except Exception as e:
        logger.error(f"❌ Erro no previsor em tempo real: {e}")
        raise

if __name__ == "__main__":
    run_realtime_hac()
