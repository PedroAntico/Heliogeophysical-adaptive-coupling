"""
deep_hac_forecast.py
HAC 3.0 — Previsão Deep Learning (LSTM + GRU)
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Deep - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HAC_Deep")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_real_data():
    """Carrega o dataset real mais recente de data_real/"""
    folder = "data_real"
    if not os.path.exists(folder):
        raise FileNotFoundError("A pasta data_real/ não existe. Execute o coletor real primeiro.")

    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".csv")],
        reverse=True
    )

    if not files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado em data_real/")

    latest = os.path.join(folder, files[0])
    logger.info(f"📥 Carregando dados de: {latest}")

    df = pd.read_csv(latest)
    df = df.dropna()

    return df


def create_sequences(df, features, horizon_hours=1, lookback=48):
    """
    Constrói sequências para treino.
    """
    logger.info(f"➡ Construindo sequências (lookback={lookback}, horizon={horizon_hours}h)")

    X, y = [], []
    data = df[features].values
    speed = df["speed"].values

    for i in range(lookback, len(df) - horizon_hours):
        X.append(data[i - lookback:i])
        y.append(speed[i + horizon_hours])

    X = np.array(X)
    y = np.array(y)

    return X, y


# ============================================================
# MODELOS
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ============================================================
# TREINAMENTO
# ============================================================

def train_deep_for_horizon(df, features, horizon_h):
    """
    Treina modelos LSTM e GRU para um horizonte específico.
    """

    logger.info("")
    logger.info("========================================================")
    logger.info(f"🌙 Previsão Deep HAC — Horizonte {horizon_h}h")
    logger.info("========================================================")

    lookback = 48  # 4h de histórico

    # Criar sequências
    X, y = create_sequences(df, features, horizon_hours=horizon_h, lookback=lookback)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    logger.info(f"Treinando LSTM...")
    lstm = build_lstm((lookback, len(features)))
    lstm.fit(X_train, y_train, epochs=12, batch_size=32, verbose=0)

    y_pred_lstm = lstm.predict(X_test).flatten()

    # RMSE corrigido (sem 'squared')
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    logger.info(f"📘 LSTM | RMSE: {rmse_lstm:.3f}")

    logger.info(f"Treinando GRU...")
    gru = build_gru((lookback, len(features)))
    gru.fit(X_train, y_train, epochs=12, batch_size=32, verbose=0)

    y_pred_gru = gru.predict(X_test).flatten()

    # RMSE corrigido
    rmse_gru = np.sqrt(mean_squared_error(y_test, y_pred_gru))
    logger.info(f"📗 GRU  | RMSE: {rmse_gru:.3f}")

    return {
        "horizon": horizon_h,
        "rmse_lstm": rmse_lstm,
        "rmse_gru": rmse_gru
    }


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def run_deep_hac():
    logger.info("🚀 Sistema Deep HAC iniciado")
    df = load_real_data()

    features = ["speed", "density", "temperature", "bx_gse", "by_gse", "bz_gse"]
    df = df.dropna(subset=features)

    results = []

    for h in [1, 3, 6, 12]:
        res = train_deep_for_horizon(df, features, h)
        results.append(res)

    logger.info("\n===== RESULTADOS GERAIS =====")
    for r in results:
        logger.info(
            f"H{r['horizon']}h -> LSTM: {r['rmse_lstm']:.3f} | GRU: {r['rmse_gru']:.3f}"
        )


if __name__ == "__main__":
    run_deep_hac()
