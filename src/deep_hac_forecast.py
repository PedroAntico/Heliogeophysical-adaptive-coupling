"""
deep_hac_forecast.py
HAC 3.0 — Treinamento Deep Learning (LSTM + GRU) + salvamento de modelos
Versão compatível com hac_realtime_predict.py
"""

import os
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# LOGGING E CONFIG TENSORFLOW
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Deep - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HAC_Deep")

tf.config.optimizer.set_jit(True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# CARREGAR DADOS
# ============================================================

def load_and_validate_real_data():
    """Carrega o CSV mais recente das pastas conhecidas."""
    folders = ["data_real", "data", "data/raw", "data/processed"]

    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Pasta {folder} não encontrada, pulando.")
            continue

        files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".csv")],
            reverse=True,
        )
        if not files:
            continue

        latest = os.path.join(folder, files[0])
        logger.info(f"📥 Carregando dados de: {latest}")

        df = pd.read_csv(latest)

        if df.empty:
            logger.warning(f"Arquivo {latest} está vazio, tentando outro.")
            continue

        logger.info(f"✅ {len(df)} registros, {len(df.columns)} colunas.")
        logger.info(f"📋 Colunas: {list(df.columns)}")

        # Se tiver timestamp, converte
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    raise FileNotFoundError("❌ Nenhum CSV com dados encontrado.")


# ============================================================
# SEQUÊNCIAS PARA TREINO
# ============================================================

def create_sequences(df, features, target_col="speed",
                     horizon_hours=1, lookback=36, step=1):
    """
    Cria sequências (X) e alvos (y) para previsão.
    Previsão single-step em t + horizonte.
    """
    logger.info(f"🔄 Criando sequências: lookback={lookback}, horizonte={horizon_hours}h")

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"⚠️ Features faltando: {missing}")
    features = available_features

    # Escolher target: se não existir 'speed', usa primeira numérica
    if target_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("Nenhuma coluna numérica encontrada para usar como target.")
        target_col = num_cols[0]
        logger.warning(f"⚠️ Target 'speed' não encontrado. Usando '{target_col}'.")

    data = df[features].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(lookback, len(data) - horizon_hours, step):
        X.append(data[i - lookback:i])
        y.append(targets[i + horizon_hours - 1])

    if not X:
        raise ValueError("❌ Não foi possível criar sequências (dados insuficientes).")

    X = np.array(X)
    y = np.array(y)

    logger.info(f"📊 Sequências criadas: X{X.shape}, y{y.shape}")
    return X, y, features, target_col


# ============================================================
# MODELOS
# ============================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        LSTM(32, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        GRU(32, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def create_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ============================================================
# TREINO PARA UM HORIZONTE
# ============================================================

def train_model_for_horizon(df, features, horizon_h, model_type="lstm"):
    logger.info(f"\n🎯 Treinando {model_type.upper()} para horizonte {horizon_h}h")

    lookback = min(36, len(df) // 3)
    if lookback < 12:
        lookback = 12

    X, y, used_features, target_col = create_sequences(
        df, features, horizon_hours=horizon_h, lookback=lookback, step=1
    )

    if len(X) < 50:
        logger.warning(f"Dados insuficientes para treinamento: {len(X)} amostras.")
        return None

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    input_shape = (lookback, len(used_features))

    if model_type == "lstm":
        model = build_lstm(input_shape)
    elif model_type == "gru":
        model = build_gru(input_shape)
    else:
        raise ValueError(f"Modelo não suportado: {model_type}")

    logger.info(f"🏗️ Arquitetura {model_type}: {model.count_params()} parâmetros")

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=30,
        batch_size=16,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=create_callbacks(),
        verbose=0,
    )

    # Avaliação
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(
            np.mean(
                np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))
            ) * 100
        )

    logger.info(
        f"📊 {model_type.upper()} H{horizon_h}h -> "
        f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}% (target: {target_col})"
    )

    return {
        "model": model,
        "history": history.history,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "horizon": horizon_h,
            "target": target_col,
        },
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "lookback": lookback,
        "features": used_features,
    }


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def run_deep_hac():
    try:
        logger.info("🚀 Iniciando HAC Deep Learning 3.0 (treino + salvar modelos)")

        df = load_and_validate_real_data()

        possible_features = ["speed", "density", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
        available_features = [f for f in possible_features if f in df.columns]

        if len(available_features) < 2:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = num_cols[:3]
            if len(available_features) < 2:
                raise ValueError(f"❌ Features insuficientes. Colunas: {list(df.columns)}")

        logger.info(f"🎯 Features utilizadas: {available_features}")

        df_clean = df[available_features].dropna()
        if len(df_clean) < 50:
            raise ValueError(f"❌ Dados insuficientes após limpeza: {len(df_clean)} registros")

        horizons = [1, 3, 6]
        model_types = ["lstm", "gru"]

        results = {}

        for mtype in model_types:
            results[mtype] = {}
            for h in horizons:
                try:
                    res = train_model_for_horizon(df_clean, available_features, h, mtype)
                    results[mtype][h] = res
                except Exception as e:
                    logger.error(f"❌ Erro ao treinar {mtype} H{h}h: {e}")
                    results[mtype][h] = None

        logger.info("\n" + "=" * 60)
        logger.info("📈 RELATÓRIO FINAL HAC DEEP LEARNING")
        logger.info("=" * 60)
        logger.info(f"📊 Total de dados: {len(df_clean)} registros")
        logger.info(f"🎯 Features: {available_features}")

        for mtype in model_types:
            logger.info(f"\n🔮 {mtype.upper()}:")
            for h in horizons:
                res = results[mtype][h]
                if res:
                    met = res["metrics"]
                    logger.info(
                        f"  H{h:2d}h -> RMSE: {met['rmse']:.3f}, "
                        f"MAE: {met['mae']:.3f}, MAPE: {met['mape']:.2f}% "
                        f"(target: {met.get('target', 'speed')})"
                    )
                else:
                    logger.info(f"  H{h:2d}h -> ❌ Falha no treinamento")

        # ----------------------------------------------------
        # SALVAR MÉTRICAS
        # ----------------------------------------------------
        os.makedirs("results/deep_learning", exist_ok=True)
        metrics_summary = {}
        for mtype in model_types:
            metrics_summary[mtype] = {}
            for h in horizons:
                if results[mtype][h]:
                    metrics_summary[mtype][str(h)] = results[mtype][h]["metrics"]

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        metrics_path = f"results/deep_learning/metrics_{ts}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"💾 Métricas salvas em: {metrics_path}")

        # ----------------------------------------------------
        # SALVAR MODELOS EM models/deep_hac
        # ----------------------------------------------------
        os.makedirs("models/deep_hac", exist_ok=True)

        for mtype in model_types:
            for h in horizons:
                res = results[mtype][h]
                if res and res["model"] is not None:
                    model_fname = f"{mtype}_h{h}_{ts}.h5"
                    model_path = os.path.join("models/deep_hac", model_fname)
                    res["model"].save(model_path)
                    logger.info(f"💾 Modelo salvo: {model_path}")

        logger.info("✅ HAC Deep Learning concluído com sucesso!")
        return results

    except Exception as e:
        logger.error(f"❌ Falha no HAC Deep Learning: {e}")
        raise


if __name__ == "__main__":
    run_deep_hac()
