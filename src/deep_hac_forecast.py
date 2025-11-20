"""
src/deep_hac_forecast.py

HAC Deep Forecast - LSTM e GRU
Fase 2: modelos de Deep Learning para previsões de 1h, 3h, 6h e 12h
usando dados reais do vento solar (NOAA / NASA / HAC).

Não mexe em nada do pipeline atual. É um módulo de pesquisa.
"""

import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Tentativa de importar TensorFlow/Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HAC_Deep")


# ============================================================
# 1) Carregamento de dados
# ============================================================

def load_solar_data():
    """
    Tenta carregar dados reais de arquivos já gerados pelo HAC.
    Prioridade:
      1) data/solar_data_latest.csv
      2) data/solar_data_history.csv (se existir)
    """
    candidates = [
        "data/solar_data_latest.csv",
        "data/solar_data_history.csv"
    ]

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"📂 Carregando dados de: {path}")
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "Nenhum arquivo de dados encontrado. "
            "Esperado: data/solar_data_latest.csv ou data/solar_data_history.csv"
        )

    # Normalizar coluna de tempo
    time_col = None
    for col in ["time_tag", "timestamp", "date"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        raise ValueError("Nenhuma coluna temporal encontrada (time_tag/timestamp/date).")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df.rename(columns={time_col: "time_tag"})

    # Selecionar features principais, se existirem
    feature_candidates = [
        "speed", "density", "temperature",
        "bx_gse", "by_gse", "bz_gse", "bt"
    ]
    features = [c for c in feature_candidates if c in df.columns]

    if "speed" not in features:
        raise ValueError("Coluna 'speed' é obrigatória para prever. Não encontrada no dataset.")

    logger.info(f"✅ Dados carregados: {len(df)} registros")
    logger.info(f"📊 Features disponíveis: {features}")

    # Remover linhas com muitos NaNs nas features
    df = df[["time_tag"] + features].dropna().reset_index(drop=True)

    return df, features


# ============================================================
# 2) Construção de janelas temporais (sequências)
# ============================================================

def make_supervised_sequences(df, feature_cols, target_col, lookback_steps, horizon_steps):
    """
    Constrói dataset supervisonado para seq2one:
        X: janelas de [t-lookback+1 ... t]
        y: valor em t + horizon
    """

    values = df[feature_cols].values
    target = df[target_col].shift(-horizon_steps).values
    times = df["time_tag"].values

    X, y, t_out = [], [], []

    # i = índice do "tempo atual" (t)
    for i in range(lookback_steps, len(df) - horizon_steps):
        X.append(values[i - lookback_steps:i])
        y.append(target[i])
        t_out.append(times[i])

    X = np.array(X)
    y = np.array(y)
    t_out = np.array(t_out)

    return X, y, t_out


# ============================================================
# 3) Criação dos modelos LSTM e GRU
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ============================================================
# 4) Treino e avaliação para um horizonte
# ============================================================

def train_deep_for_horizon(df, features, horizon_hours, freq_minutes=5, lookback_hours=3):
    """
    Treina LSTM e GRU para um determinado horizonte de previsão.
    - df: dataframe com colunas ['time_tag'] + features
    - features: lista de colunas usadas como entrada
    - horizon_hours: horizonte em horas (1, 3, 6, 12...)
    - freq_minutes: resolução temporal (5 min padrão)
    - lookback_hours: histórico usado como entrada (padrão: 3h)
    """

    if not TF_AVAILABLE:
        logger.warning("TensorFlow/Keras não disponível. Pulei DL.")
        return None

    logger.info("==============================================")
    logger.info(f"🕒 Horizonte: {horizon_hours}h")
    logger.info("==============================================")

    steps_per_hour = int(60 / freq_minutes)
    horizon_steps = horizon_hours * steps_per_hour
    lookback_steps = lookback_hours * steps_per_hour

    if len(df) < (lookback_steps + horizon_steps + 10):
        logger.warning("Dados insuficientes para este horizonte. Pulando.")
        return None

    # Construir sequências
    X, y, t_out = make_supervised_sequences(
        df, feature_cols=features, target_col="speed",
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps
    )

    logger.info(f"📦 Sequências construídas: {X.shape[0]} amostras, "
                f"lookback={lookback_hours}h, horizon={horizon_hours}h")

    # Split temporal: 80% treino, 20% teste
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    t_test = t_out[split_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Callbacks
    cb = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    results = {}

    # ----------------- LSTM -----------------
    logger.info("🔧 Treinando LSTM...")
    lstm = build_lstm(input_shape)
    lstm.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=cb,
        verbose=0
    )

    y_pred_lstm = lstm.predict(X_test, verbose=0).flatten()
    rmse_lstm = mean_squared_error(y_test, y_pred_lstm, squared=False)
    r2_lstm = r2_score(y_test, y_pred_lstm)

    logger.info(f"📈 LSTM  RMSE={rmse_lstm:.3f}  R²={r2_lstm:.3f}")

    results["LSTM"] = {
        "rmse": rmse_lstm,
        "r2": r2_lstm,
        "y_true": y_test,
        "y_pred": y_pred_lstm,
        "time_tag": t_test,
    }

    # ----------------- GRU -----------------
    logger.info("🔧 Treinando GRU...")
    gru = build_gru(input_shape)
    gru.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=cb,
        verbose=0
    )

    y_pred_gru = gru.predict(X_test, verbose=0).flatten()
    rmse_gru = mean_squared_error(y_test, y_pred_gru, squared=False)
    r2_gru = r2_score(y_test, y_pred_gru)

    logger.info(f"📈 GRU   RMSE={rmse_gru:.3f}  R²={r2_gru:.3f}")

    results["GRU"] = {
        "rmse": rmse_gru,
        "r2": r2_gru,
        "y_true": y_test,
        "y_pred": y_pred_gru,
        "time_tag": t_test,
    }

    return results


# ============================================================
# 5) Runner principal
# ============================================================

def run_deep_hac():
    if not TF_AVAILABLE:
        print("❌ TensorFlow/Keras não está disponível neste ambiente.")
        print("   Instale com: pip install tensorflow-cpu")
        return

    df, features = load_solar_data()

    horizons = [1, 3, 6, 12]  # horas
    all_results = {}

    for h in horizons:
        res = train_deep_for_horizon(df, features, horizon_hours=h)
        if res is not None:
            all_results[h] = res

    if not all_results:
        print("⚠ Nenhum horizonte foi treinado (dados insuficientes ou erro).")
        return

    print("\n============================================")
    print(" RESUMO – HAC DEEP FORECAST (LSTM/GRU)")
    print("============================================\n")

    for h, models in all_results.items():
        print(f"🕒 Horizonte {h}h:")
        for name, r in models.items():
            print(f"  • {name:4s} | RMSE: {r['rmse']:.3f} | R²: {r['r2']:.3f}")
        print()

    print("✅ Deep Learning executado com sucesso.")


if __name__ == "__main__":
    run_deep_hac()
