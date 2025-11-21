"""
deep_hac_forecast.py
HAC 3.0 — Previsão Deep Learning (LSTM + GRU)
VERSÃO FINAL — Compatível com Keras 3.0 (.keras) 
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Deep - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HAC_Deep")

tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# CARREGAR DADOS REAIS
# ============================================================

def load_and_validate_real_data():
    """Carrega o arquivo CSV mais recente da pasta data_real/ ou similares."""
    search_folders = ["data_real", "data/raw", "data/processed", "data"]

    for folder in search_folders:
        if not os.path.exists(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)
        if files:
            path = os.path.join(folder, files[0])
            logger.info(f"📥 Carregando dados de: {path}")
            df = pd.read_csv(path)

            if df.empty:
                logger.warning(f"{path} está vazio.")
                continue

            # Conversão opcional
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="ignore")

            logger.info(f"✅ Dados carregados: {len(df)} registros.")
            return df

    raise FileNotFoundError("Nenhum arquivo CSV encontrado para treino.")

# ============================================================
# GERAR SEQUÊNCIAS
# ============================================================

def create_sequences(df, features, target_col, lookback, horizon):
    """Cria janelas para seq-to-one (previsão 1 valor daqui H horas)."""
    X, y = [], []

    data = df[features].values
    target = df[target_col].values

    for i in range(lookback, len(df) - horizon):
        X.append(data[i - lookback:i])
        y.append(target[i + horizon])

    if len(X) == 0:
        raise ValueError("Dados insuficientes para criar janelas.")

    return np.array(X), np.array(y)

# ============================================================
# MODELOS LSTM/GRU
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        LSTM(32),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        GRU(32),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return model

# ============================================================
# TREINAMENTO
# ============================================================

def train_model(df, features, horizon, model_type):
    logger.info(f"\n🎯 Treinando {model_type.upper()} para horizonte {horizon}h...")

    lookback = min(36, len(df) // 3)
    if lookback < 12:
        lookback = 12

    target_col = features[0]  # usamos speed normalmente

    X, y = create_sequences(df, features, target_col, lookback, horizon)

    # Normalização
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_reshaped = X.reshape(-1, len(features))
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    input_shape = (lookback, len(features))
    model = build_lstm(input_shape) if model_type == "lstm" else build_gru(input_shape)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=0
    )

    # Avaliação
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    logger.info(f"📊 {model_type.upper()} H{horizon}h -> RMSE={rmse:.3f}, MAE={mae:.3f}")

    return model, scaler_X, scaler_y, lookback, rmse, mae

# ============================================================
# SALVAR MODELOS EM FORMATO .keras
# ============================================================

def save_model(model, model_type, horizon, timestamp):
    os.makedirs("models/deep_hac", exist_ok=True)
    path = f"models/deep_hac/{model_type}_h{horizon}_{timestamp}.keras"
    model.save(path, save_format="keras")
    logger.info(f"💾 Modelo salvo: {path}")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def run_deep_hac():
    logger.info("🚀 Iniciando HAC Deep Learning 3.0")

    df = load_and_validate_real_data()

    possible = ["speed", "density", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
    features = [f for f in possible if f in df.columns][:3]

    if len(features) < 2:
        raise ValueError("Features insuficientes para treino.")

    df = df[features].dropna()
    logger.info(f"📊 Dados utilizados: {len(df)} registros | Features: {features}")

    horizons = [1, 3, 6]
    model_types = ["lstm", "gru"]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for model_type in model_types:
        for h in horizons:
            model, scX, scY, lookback, rmse, mae = train_model(df, features, h, model_type)
            save_model(model, model_type, h, timestamp)

    logger.info("\n✅ HAC Deep Learning concluído com sucesso!")

if __name__ == "__main__":
    run_deep_hac()
