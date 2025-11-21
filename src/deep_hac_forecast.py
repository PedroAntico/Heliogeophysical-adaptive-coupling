"""
deep_hac_forecast.py
HAC 3.0 — Treino Deep Learning Avançado (LSTM + GRU + CNN)
Horizontes: 1h, 3h, 6h, 12h, 24h, 48h
Gera e salva modelos em models/deep_hac/
Salva métricas e visualizações em results/deep_learning/
"""

import os
import json
import logging
import warnings
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, Flatten, Input, 
                                   concatenate, Bidirectional)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, TensorBoard)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

# Configuração de logging avançada
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("hac_deep_training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HAC_Deep_Advanced")

# Supressão de warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurações globais - CONSTANTES (não alterar dinamicamente)
GLOBAL_HORIZONS_H = [1, 3, 6, 12, 24, 48]
GLOBAL_MODEL_TYPES = ["lstm", "gru", "bilstm", "cnn_lstm"]
LOOKBACK_RANGE = (24, 168)  # Min e max lookback
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 8

class Config:
    """Configurações do treinamento"""
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.15
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    L2_REGULARIZATION = 0.001

# ============================================================
# FUNÇÕES AUXILIARES AVANÇADAS
# ============================================================

def setup_gpu():
    """Configura otimizações para GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU configurada: {len(gpus)} dispositivos")
        except RuntimeError as e:
            logger.warning(f"⚠️ Erro na configuração GPU: {e}")
    else:
        logger.info("🔶 Treinando em CPU")

def load_and_validate_real_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega e valida dados com múltiplas estratégias de fallback
    """
    search_paths = ["data_real", "data/raw", "data/processed", "data", "dataset"]
    
    if data_path and os.path.exists(data_path):
        file_path = data_path
        logger.info(f"📥 Carregando dados do caminho especificado: {file_path}")
    else:
        file_path = None
        for folder in search_paths:
            if os.path.exists(folder):
                csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
                if csv_files:
                    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
                    file_path = os.path.join(folder, latest_file)
                    logger.info(f"📥 Carregando dados mais recentes: {file_path}")
                    break
        
        if not file_path:
            raise FileNotFoundError("❌ Nenhum arquivo CSV encontrado nas pastas de busca")

    try:
        # Tenta diferentes encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"✅ Arquivo carregado com encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Não foi possível decodificar o arquivo com nenhum encoding comum")

        # Análise inicial dos dados
        logger.info(f"📊 Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
        logger.info(f"📋 Colunas: {list(df.columns)}")
        logger.info(f"📅 Período: {df.index.min() if hasattr(df.index, 'min') else 'N/A'} - {df.index.max() if hasattr(df.index, 'max') else 'N/A'}")
        
        # Informações sobre dados faltantes
        missing_info = df.isnull().sum()
        if missing_info.any():
            logger.warning(f"⚠️ Dados faltantes: {dict(missing_info[missing_info > 0])}")
        
        return df

    except Exception as e:
        logger.error(f"❌ Erro ao carregar {file_path}: {e}")
        raise

def create_advanced_sequences(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "speed",
    horizon_hours: int = 1,
    lookback: int = 48,
    step: int = 1,
    multi_output: bool = False
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Cria sequências avançadas para treinamento com múltiplas opções
    """
    logger.info(f"🔄 Criando sequências: lookback={lookback}, horizon={horizon_hours}h")

    # Validação de features
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        logger.warning(f"⚠️ Features não encontradas: {missing_features}")
    
    if not available_features:
        raise ValueError("❌ Nenhuma feature válida disponível")

    # Seleção automática de target se necessário
    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("❌ Nenhuma coluna numérica encontrada")
        target_col = numeric_cols[0]
        logger.warning(f"⚠️ Target original não encontrado. Usando: {target_col}")

    # Preparação dos dados
    data = df[available_features].values
    targets = df[target_col].values

    X, y = [], []

    # Criação das sequências
    if multi_output:
        # Para múltiplos passos futuros
        for i in range(lookback, len(data) - horizon_hours, step):
            X.append(data[i - lookback:i])
            y.append(targets[i:i + horizon_hours])
    else:
        # Para passo único
        for i in range(lookback, len(data) - horizon_hours, step):
            X.append(data[i - lookback:i])
            y.append(targets[i + horizon_hours - 1])

    if not X:
        raise ValueError("❌ Não foi possível criar sequências. Dados insuficientes.")

    X = np.array(X)
    y = np.array(y)

    logger.info(f"📊 Sequências criadas: X{X.shape}, y{y.shape}")
    return X, y, target_col, available_features

def calculate_optimal_lookback(df: pd.DataFrame, max_lookback: int = 168) -> int:
    """
    Calcula lookback ótimo baseado na autocorrelação dos dados
    """
    if len(df) < 100:
        return min(48, len(df) // 2)
    
    # Usa a primeira coluna numérica para análise
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 48
    
    sample_col = numeric_cols[0]
    data = df[sample_col].dropna().values
    
    if len(data) < 50:
        return 24
    
    # Tenta usar statsmodels para autocorrelação, fallback se não disponível
    try:
        from statsmodels.tsa.stattools import acf
        autocorr = acf(data, nlags=min(100, len(data)//3), fft=True)
        
        # Encontra primeiro lag onde autocorrelação cai abaixo do threshold
        threshold = 0.1
        significant_lags = np.where(np.abs(autocorr) > threshold)[0]
        optimal_lookback = significant_lags[-1] if len(significant_lags) > 0 else 48
        
        # Limita ao range permitido
        optimal_lookback = max(LOOKBACK_RANGE[0], min(optimal_lookback, LOOKBACK_RANGE[1]))
        
        logger.info(f"🎯 Lookback ótimo calculado: {optimal_lookback}")
        return optimal_lookback
        
    except ImportError:
        logger.warning("📊 statsmodels não disponível, usando lookback padrão de 48")
        return 48
    except Exception as e:
        logger.warning(f"📊 Erro no cálculo de lookback: {e}, usando padrão 48")
        return 48

# ============================================================
# ARQUITETURAS AVANÇADAS DE MODELOS
# ============================================================

def build_hybrid_model(input_shape: Tuple[int, int], model_type: str = "cnn_lstm") -> Model:
    """
    Constrói modelos híbridos avançados
    """
    model_input = Input(shape=input_shape)
    
    if model_type == "cnn_lstm":
        # CNN + LSTM
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(model_input)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = LSTM(64, return_sequences=True, dropout=Config.DROPOUT_RATE)(x)
        x = LSTM(32, dropout=Config.DROPOUT_RATE)(x)
        
    elif model_type == "bilstm":
        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=Config.DROPOUT_RATE))(model_input)
        x = Bidirectional(LSTM(32, dropout=Config.DROPOUT_RATE))(x)
        
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")

    x = Dense(32, activation='relu', kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION))(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=model_input, outputs=outputs)
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    
    return model

def build_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """LSTM avançado com regularização"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, 
             dropout=Config.DROPOUT_RATE, recurrent_dropout=0.2,
             kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION)),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=Config.DROPOUT_RATE,
             kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION)),
        LSTM(32, dropout=Config.DROPOUT_RATE),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION)),
        Dropout(Config.DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def build_gru_model(input_shape: Tuple[int, int]) -> Model:
    """GRU avançado com regularização"""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape, 
            dropout=Config.DROPOUT_RATE, recurrent_dropout=0.2,
            kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION)),
        BatchNormalization(),
        GRU(64, return_sequences=True, dropout=Config.DROPOUT_RATE),
        GRU(32, dropout=Config.DROPOUT_RATE),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l2=Config.L2_REGULARIZATION)),
        Dropout(Config.DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def create_advanced_callbacks(model_name: str, horizon: int) -> List[tf.keras.callbacks.Callback]:
    """Cria callbacks avançados para treinamento"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"models/deep_hac/temp_best_{model_name}_h{horizon}.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # TensorBoard apenas se necessário para debug
    if logger.level <= logging.DEBUG:
        callbacks.append(
            TensorBoard(
                log_dir=f"logs/{model_name}_h{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                histogram_freq=1
            )
        )
    
    return callbacks

# ============================================================
# TREINAMENTO AVANÇADO
# ============================================================

def prepare_data_advanced(
    X: np.ndarray, 
    y: np.ndarray, 
    validation_split: float = 0.2,
    test_split: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Any, Any]:
    """
    Prepara dados com split temporal e normalização robusta
    """
    # Split temporal
    total_samples = len(X)
    test_idx = int(total_samples * (1 - test_split))
    val_idx = int(test_idx * (1 - validation_split))
    
    # Train/Validation/Test split
    X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
    y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]
    
    # Normalização robusta (menos sensível a outliers)
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    # Reshape para normalização
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit e transform
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    logger.info(f"📊 Split dos dados - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")
    
    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), scaler_X, scaler_y

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas abrangentes de avaliação"""
    metrics = {}
    
    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    
    # MAPE com proteção contra divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        metrics['mape'] = float(mape)
    
    # Additional metrics
    metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
    metrics['std_error'] = float(np.std(y_true - y_pred))
    
    return metrics

def plot_training_history(history: tf.keras.callbacks.History, model_name: str, horizon: int, save_path: str):
    """Plota e salva gráficos do histórico de treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', alpha=0.7)
    ax1.plot(history.history['val_loss'], label='Val Loss', alpha=0.7)
    ax1.set_title(f'Model Loss - {model_name.upper()} H{horizon}h')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    if 'mae' in history.history:
        ax2.plot(history.history['mae'], label='Train MAE', alpha=0.7)
        ax2.plot(history.history['val_mae'], label='Val MAE', alpha=0.7)
        ax2.set_title(f'Model MAE - {model_name.upper()} H{horizon}h')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, horizon: int, save_path: str):
    """Plota e salva gráficos de previsões vs valores reais"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Valores Reais')
    ax1.set_ylabel('Previsões')
    ax1.set_title(f'Previsões vs Reais - {model_name.upper()} H{horizon}h')
    ax1.grid(True, alpha=0.3)
    
    # Time series comparison (first 100 points)
    sample_size = min(100, len(y_true))
    ax2.plot(y_true[:sample_size], label='Real', alpha=0.7, marker='o', markersize=3)
    ax2.plot(y_pred[:sample_size], label='Predito', alpha=0.7, marker='s', markersize=3)
    ax2.set_title(f'Comparação Temporal (amostra) - {model_name.upper()} H{horizon}h')
    ax2.set_ylabel('Valor')
    ax2.set_xlabel('Amostra')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model_for_horizon(
    df: pd.DataFrame,
    features: List[str],
    horizon_h: int,
    model_type: str = "lstm",
    lookback: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Treina modelo avançado para horizonte específico
    """
    logger.info(f"\n🎯 Iniciando treino {model_type.upper()} para horizonte {horizon_h}h")
    
    # Determina lookback ótimo
    if lookback is None:
        lookback = calculate_optimal_lookback(df)
    
    lookback = max(LOOKBACK_RANGE[0], min(lookback, LOOKBACK_RANGE[1]))
    
    try:
        # Cria sequências
        X, y, target_col, used_features = create_advanced_sequences(
            df, features, horizon_hours=horizon_h, lookback=lookback
        )
        
        if len(X) < 100:
            logger.warning(f"⚠️ Dados insuficientes: {len(X)} amostras")
            return None
        
        # Prepara dados
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_data_advanced(X, y)
        
        # Constrói modelo
        input_shape = (lookback, len(used_features))
        
        if model_type == "lstm":
            model = build_lstm_model(input_shape)
        elif model_type == "gru":
            model = build_gru_model(input_shape)
        elif model_type in ["bilstm", "cnn_lstm"]:
            model = build_hybrid_model(input_shape, model_type)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        logger.info(f"🏗️ {model_type.upper()} construído: {model.count_params():,} parâmetros")
        
        # Callbacks
        callbacks = create_advanced_callbacks(model_type, horizon_h)
        
        # Treinamento
        history = model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.MAX_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Mantém ordem temporal
        )
        
        # Avaliação no conjunto de teste
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()  # CORREÇÃO APPLIED
        
        # Métricas
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        metrics.update({
            'horizon': horizon_h,
            'target': target_col,
            'features_used': used_features,
            'lookback': lookback,
            'model_params': model.count_params(),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
        logger.info(
            f"📊 {model_type.upper()} H{horizon_h}h -> "
            f"RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, "
            f"R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%"
        )
        
        return {
            'model': model,
            'history': history.history,
            'metrics': metrics,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'test_data': (X_test, y_test, y_pred, y_true)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no treino {model_type} H{horizon_h}h: {e}")
        return None

# ============================================================
# EXECUÇÃO PRINCIPAL AVANÇADA
# ============================================================

def run_advanced_deep_hac(data_path: Optional[str] = None, horizons: Optional[List[int]] = None, models: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
    """
    Executa pipeline completo de treinamento avançado
    """
    logger.info("🚀 INICIANDO HAC DEEP LEARNING 4.0")
    start_time = datetime.now()
    
    # Configuração inicial
    setup_gpu()
    
    # Usar configurações fornecidas ou padrões GLOBAIS
    HORIZONS_TO_USE = horizons if horizons else GLOBAL_HORIZONS_H
    MODELS_TO_USE = [m.lower() for m in models] if models else GLOBAL_MODEL_TYPES
    
    logger.info(f"🎯 Horizontes: {HORIZONS_TO_USE}")
    logger.info(f"🧠 Modelos: {MODELS_TO_USE}")
    
    # Criar diretórios
    os.makedirs("models/deep_hac", exist_ok=True)
    os.makedirs("results/deep_learning/plots", exist_ok=True)
    os.makedirs("results/deep_learning/models", exist_ok=True)
    
    # Carregar dados
    df = load_and_validate_real_data(data_path)
    
    # Engenharia de features básica
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Features preferenciais com fallback
    preferred_features = [
        "speed", "density", "temperature", 
        "bx_gse", "by_gse", "bz_gse", "bt",
        "vx_gse", "vy_gse", "vz_gse", "pressure"
    ]
    
    available_features = [f for f in preferred_features if f in df.columns]
    
    # Fallback para colunas numéricas
    if len(available_features) < 2:
        available_features = numeric_cols[:5]  # Primeiras 5 colunas numéricas
        logger.warning(f"⚠️ Usando fallback features: {available_features}")
    
    if len(available_features) < 2:
        raise ValueError(f"❌ Features insuficientes. Disponíveis: {numeric_cols}")
    
    # Limpeza final
    df_clean = df[available_features].dropna().copy()
    
    if len(df_clean) < 200:
        raise ValueError(f"❌ Dados insuficientes após limpeza: {len(df_clean)} registros")
    
    logger.info(f"🎯 Features finais ({len(available_features)}): {available_features}")
    logger.info(f"📊 Dados para treino: {len(df_clean)} registros")
    
    # ID da execução
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Estruturas para resultados
    all_results = {}
    metrics_summary = {}
    best_models = {}
    
    # Loop de treinamento
    for model_type in MODELS_TO_USE:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧠 TREINANDO {model_type.upper()}")
        logger.info(f"{'='*50}")
        
        all_results[model_type] = {}
        metrics_summary[model_type] = {}
        
        for horizon in HORIZONS_TO_USE:
            try:
                result = train_model_for_horizon(
                    df_clean, available_features, horizon, model_type
                )
                
                all_results[model_type][horizon] = result
                
                if result is None:
                    logger.warning(f"⏭️  Pulando {model_type} H{horizon}h")
                    continue
                
                # Salvar modelo
                model_filename = f"{model_type}_h{horizon}_{run_id}.keras"
                model_path = os.path.join("models/deep_hac", model_filename)
                result['model'].save(model_path)
                
                # Salvar scalers - usando pickle como fallback universal
                scaler_x_path = f"models/deep_hac/scaler_X_{model_type}_h{horizon}_{run_id}.pkl"
                scaler_y_path = f"models/deep_hac/scaler_y_{model_type}_h{horizon}_{run_id}.pkl"
                
                try:
                    # Tenta usar joblib primeiro (mais eficiente)
                    import joblib
                    joblib.dump(result['scaler_X'], scaler_x_path)
                    joblib.dump(result['scaler_y'], scaler_y_path)
                    logger.info("💾 Scalers salvos com joblib")
                except ImportError:
                    # Fallback para pickle
                    with open(scaler_x_path, 'wb') as f:
                        pickle.dump(result['scaler_X'], f)
                    with open(scaler_y_path, 'wb') as f:
                        pickle.dump(result['scaler_y'], f)
                    logger.info("💾 Scalers salvos com pickle (joblib não disponível)")
                
                # Gerar visualizações
                plot_training_history(
                    result['history'], model_type, horizon,
                    f"results/deep_learning/plots/training_{model_type}_h{horizon}_{run_id}.png"
                )
                
                if 'test_data' in result:
                    _, _, y_pred, y_true = result['test_data']
                    plot_predictions(
                        y_true, y_pred, model_type, horizon,
                        f"results/deep_learning/plots/predictions_{model_type}_h{horizon}_{run_id}.png"
                    )
                
                # Atualizar métricas
                metrics = result['metrics'].copy()
                metrics.update({
                    'model_file': model_filename,
                    'scaler_x_file': scaler_x_path,
                    'scaler_y_file': scaler_y_path,
                    'training_time': str(datetime.now() - start_time)
                })
                
                metrics_summary[model_type][str(horizon)] = metrics
                
                logger.info(f"💾 Modelo salvo: {model_path}")
                
            except Exception as e:
                logger.error(f"❌ Erro em {model_type} H{horizon}h: {e}")
                continue
    
    # Salvar resultados
    metrics_path = f"results/deep_learning/metrics_advanced_{run_id}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Relatório final
    execution_time = datetime.now() - start_time
    generate_final_report(metrics_summary, execution_time, run_id)
    
    logger.info("✅ TREINAMENTO HAC DEEP LEARNING CONCLUÍDO!")
    return all_results, metrics_summary

def generate_final_report(metrics_summary: Dict, execution_time: timedelta, run_id: str):
    """Gera relatório final completo"""
    report_path = f"results/deep_learning/report_{run_id}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HAC DEEP LEARNING - RELATÓRIO FINAL\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo de execução: {execution_time}\n\n")
        
        for model_type in metrics_summary.keys():
            f.write(f"\n{model_type.upper()}:\n")
            f.write("-" * 40 + "\n")
            
            horizons_data = metrics_summary.get(model_type, {})
            if not horizons_data:
                f.write("  Nenhum modelo treinado com sucesso\n")
                continue
            
            for horizon_str, metrics in horizons_data.items():
                horizon = int(horizon_str)
                f.write(
                    f"  H{horizon:2d}h -> RMSE: {metrics['rmse']:7.3f} | "
                    f"MAE: {metrics['mae']:6.3f} | R²: {metrics['r2']:6.3f} | "
                    f"MAPE: {metrics['mape']:6.2f}%\n"
                )
    
    logger.info(f"📋 Relatório salvo: {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HAC Deep Learning Training')
    parser.add_argument('--data_path', type=str, help='Caminho para o arquivo de dados')
    parser.add_argument('--horizons', nargs='+', type=int, help='Horizontes de previsão')
    parser.add_argument('--models', nargs='+', type=str, help='Tipos de modelo para treinar')
    
    args = parser.parse_args()
    
    # Executar treinamento com argumentos
    run_advanced_deep_hac(
        data_path=args.data_path,
        horizons=args.horizons,
        models=args.models
    )
    
