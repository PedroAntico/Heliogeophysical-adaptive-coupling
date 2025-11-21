"""
train_hac_multibase.py
HAC 5.1 — Treino Deep Learning Multi-Base, Multi-Horizonte e Multi-Alvo
"""

import os
import json
import logging
import warnings
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Train_MultiBase - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hac_train_multibase.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("HAC_Train_MultiBase")

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Horizontes e modelos padrão
DEFAULT_HORIZONS = [1, 3, 6, 12, 24, 48]
DEFAULT_MODEL_TYPES = ["lstm", "gru"]

# Targets canônicos
TARGET_MAP = {
    "speed": ["speed", "v_sw", "flow_speed", "vx", "vp", "proton_speed"],
    "bz_gse": ["bz_gse", "bz_gsm", "bz", "bzgse"],
    "density": ["density", "n_p", "np", "proton_density", "n"]
}

class TrainConfig:
    BATCH_SIZE = 32
    MAX_EPOCHS = 80
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    LEARNING_RATE = 0.001
    DROPOUT = 0.3
    MIN_SAMPLES = 300
    DEFAULT_LOOKBACK = 48
    MAX_LOOKBACK = 168
    MEMORY_SAFETY_FACTOR = 0.7  # Usar apenas 70% da memória estimada

# ============================================================
# CONFIGURAÇÕES AVANÇADAS DE PERFORMANCE
# ============================================================

def setup_advanced_training():
    """Configurações avançadas para performance e estabilidade"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    
    if gpus:
        try:
            # ✅ Ativar Mixed Precision para GPU (2x mais rápido, metade da memória)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("✅ Mixed Precision ativado (float16)")
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU detectada e configurada: {len(gpus)} dispositivo(s)")
            
        except Exception as e:
            logger.warning(f"⚠️ Configurações avançadas falharam: {e}")
            # Fallback para configuração básica
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        logger.info("🔶 Nenhuma GPU detectada, treinando em CPU")

# ============================================================
# VERIFICAÇÃO DE MEMÓRIA
# ============================================================

def estimate_memory_usage(n_samples: int, lookback: int, n_features: int, n_targets: int) -> float:
    """Estima uso de memória em GB para evitar OOM"""
    # Bytes por elemento (float32 = 4 bytes)
    bytes_per_element = 4
    
    # Memória para dados (X e y)
    data_memory = (n_samples * lookback * n_features + n_samples * n_targets) * bytes_per_element
    
    # Memória para treino (gradientes, otimizadores, etc.) - estimativa conservadora
    training_memory = data_memory * 3
    
    total_gb = (data_memory + training_memory) / (1024 ** 3)
    
    logger.info(f"🧠 Estimativa de memória: {total_gb:.2f} GB "
                f"(samples: {n_samples}, lookback: {lookback}, features: {n_features})")
    
    return total_gb

def check_memory_safety(n_samples: int, lookback: int, n_features: int, n_targets: int) -> bool:
    """Verifica se o treinamento é seguro em termos de memória"""
    estimated_gb = estimate_memory_usage(n_samples, lookback, n_features, n_targets)
    
    # Limite conservador: 8GB para CPU, 16GB para GPU
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        memory_limit = 16.0 if gpus else 8.0
    except:
        memory_limit = 8.0
    
    safe_memory = memory_limit * TrainConfig.MEMORY_SAFETY_FACTOR
    
    if estimated_gb > safe_memory:
        logger.warning(f"⚠️ Uso de memória estimado ({estimated_gb:.2f} GB) excede limite seguro ({safe_memory:.2f} GB)")
        return False
    
    logger.info(f"✅ Uso de memória dentro do limite seguro: {estimated_gb:.2f} GB < {safe_memory:.2f} GB")
    return True

# ============================================================
# ANÁLISE DE CORRELAÇÃO
# ============================================================

def analyze_feature_correlations(df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
    """Analisa correlações entre features e targets para detectar problemas"""
    analysis = {}
    
    try:
        # Calcular matriz de correlação
        corr_matrix = df.corr()
        
        # Correlações com targets
        target_correlations = {}
        for target in target_cols:
            if target in corr_matrix.columns:
                target_corr = corr_matrix[target].sort_values(ascending=False)
                target_correlations[target] = {
                    'top_positive': target_corr.head(6).to_dict(),
                    'top_negative': target_corr.tail(5).to_dict()
                }
        
        analysis['target_correlations'] = target_correlations
        
        # Detectar possíveis problemas
        issues = []
        
        # 1. Correlação muito baixa com targets
        for target in target_cols:
            if target in corr_matrix.columns:
                max_corr = corr_matrix[target].abs().max()
                if max_corr < 0.1:
                    issues.append(f"ALERTA: Target '{target}' tem correlação máxima muito baixa: {max_corr:.3f}")
        
        # 2. Features constantes ou quase constantes
        constant_features = df.std()[(df.std() < 1e-10) & (df.std() >= 0)].index.tolist()
        if constant_features:
            issues.append(f"ALERTA: Features constantes detectadas: {constant_features}")
        
        # 3. Correlações perfeitas (possível vazamento)
        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                if abs(corr_matrix.loc[col1, col2]) > 0.99:
                    high_corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
        
        if high_corr_pairs:
            issues.append(f"ALERTA: Correlações quase perfeitas (>0.99): {high_corr_pairs[:3]}")  # Mostra apenas as primeiras
        
        analysis['issues'] = issues
        analysis['correlation_matrix_shape'] = corr_matrix.shape
        
        # Log dos resultados
        logger.info("📊 Análise de correlação concluída:")
        for target in target_cols:
            if target in target_correlations:
                top_pos = list(target_correlations[target]['top_positive'].items())[:3]
                logger.info(f"   {target}: top correlations = {top_pos}")
        
        if issues:
            logger.warning("⚠️ Problemas detectados na análise de correlação:")
            for issue in issues[:3]:  # Mostra apenas os 3 primeiros problemas
                logger.warning(f"   - {issue}")
        
    except Exception as e:
        logger.error(f"❌ Erro na análise de correlação: {e}")
        analysis['error'] = str(e)
    
    return analysis

# ============================================================
# CARREGAMENTO MULTI-BASE (MANTIDO IGUAL)
# ============================================================

def load_csv_robust(path: str) -> Optional[pd.DataFrame]:
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar {path} com {enc}: {e}")
            return None
    return None

def find_all_csvs() -> List[str]:
    search_dirs = ["data_real", "data/raw", "data/processed", "data", "dataset"]
    csv_paths = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(".csv"):
                csv_paths.append(os.path.join(d, fname))
    return sorted(csv_paths)

def unify_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    cand = ["timestamp", "time", "datetime", "date_time", "date"]
    tcol = None
    for c in cand:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        return df.reset_index(drop=True)
    try:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol])
        df = df.sort_values(tcol)
        df = df.set_index(tcol)
        return df
    except Exception as e:
        logger.warning(f"⚠️ Falha ao tratar timestamp em {tcol}: {e}")
        return df.reset_index(drop=True)

def load_and_merge_multibase() -> pd.DataFrame:
    csv_paths = find_all_csvs()
    if not csv_paths:
        raise FileNotFoundError("❌ Nenhum CSV encontrado em data_real/, data/, data/raw/, data/processed/ ou dataset/")

    logger.info(f"📂 Encontrados {len(csv_paths)} arquivos CSV para mesclar")
    dfs = []

    for path in csv_paths:
        df = load_csv_robust(path)
        if df is None or df.empty:
            logger.warning(f"⚠️ Ignorando {path} (vazio ou erro)")
            continue
        df = unify_timestamp(df)
        dfs.append(df)
        logger.info(f"   + {path}: {df.shape[0]} linhas, {df.shape[1]} colunas")

    if not dfs:
        raise ValueError("❌ Nenhum dataset válido foi carregado")

    merged = pd.concat(dfs, axis=0, join="outer", ignore_index=False)
    merged = merged.sort_index() if isinstance(merged.index, pd.DatetimeIndex) else merged.reset_index(drop=True)

    logger.info(f"📊 Dataset unificado: {merged.shape[0]} linhas × {merged.shape[1]} colunas")
    return merged

# ============================================================
# DETECÇÃO DE TARGETS E FEATURES (MANTIDO IGUAL)
# ============================================================

def detect_target_columns(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    cols = [c.lower() for c in df.columns]

    for target_name, candidates in TARGET_MAP.items():
        for cand in candidates:
            if cand in cols:
                idx = cols.index(cand)
                real_name = df.columns[idx]
                detected[target_name] = real_name
                break

    if "speed" not in detected:
        raise ValueError(f"❌ Não encontrei nenhuma coluna de velocidade (candidatos: {TARGET_MAP['speed']})")

    if "bz_gse" not in detected:
        logger.warning("⚠️ Coluna de Bz não encontrada, alvo Bz será ignorado")
    if "density" not in detected:
        logger.warning("⚠️ Coluna de densidade não encontrada, alvo density será ignorado")

    logger.info(f"🎯 Targets detectados: {detected}")
    return detected

def select_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = list(dict.fromkeys(target_cols + numeric_cols))
    logger.info(f"🧬 Total de features numéricas: {len(feature_cols)}")
    return feature_cols

# ============================================================
# CRIAÇÃO DE SEQUÊNCIAS MULTI-ALVO (MANTIDO IGUAL)
# ============================================================

def create_sequences_multitarget(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    lookback: int,
    horizon_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df = df[feature_cols].dropna()
    data = df.values

    X, y = [], []
    n_total = len(df)

    for i in range(lookback, n_total - horizon_steps):
        x_window = data[i - lookback:i, :]
        target_idx = i + horizon_steps
        target_vals = df.iloc[target_idx][target_cols].values.astype(float)

        X.append(x_window)
        y.append(target_vals)

    if len(X) == 0:
        raise ValueError(f"❌ Não foi possível criar sequências: dados insuficientes (lookback={lookback}, horizon={horizon_steps})")

    X = np.array(X)
    y = np.array(y)

    logger.info(f"📊 Sequências criadas: X{X.shape}, y{y.shape} (targets: {target_cols})")
    return X, y

# ============================================================
# ARQUITETURAS DE MODELOS MULTI-SAÍDA (MANTIDO IGUAL)
# ============================================================

def build_multitarget_lstm(input_shape: Tuple[int, int], output_dim: int) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, dropout=TrainConfig.DROPOUT),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=TrainConfig.DROPOUT),
        LSTM(32, dropout=TrainConfig.DROPOUT),
        Dense(64, activation="relu"),
        Dropout(TrainConfig.DROPOUT),
        Dense(32, activation="relu"),
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=TrainConfig.LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_multitarget_gru(input_shape: Tuple[int, int], output_dim: int) -> Sequential:
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape, dropout=TrainConfig.DROPOUT),
        BatchNormalization(),
        GRU(64, return_sequences=True, dropout=TrainConfig.DROPOUT),
        GRU(32, dropout=TrainConfig.DROPOUT),
        Dense(64, activation="relu"),
        Dropout(TrainConfig.DROPOUT),
        Dense(32, activation="relu"),
        Dense(output_dim)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=TrainConfig.LEARNING_RATE),
        loss="mse", 
        metrics=["mae"]
    )
    return model

# ✅ CORREÇÃO CRÍTICA: ModelCheckpoint com run_id único
def create_callbacks(model_type: str, horizon: int, run_id: str) -> List[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        ),
        # ✅ CORREÇÃO: Arquivo temporário único por execução
        ModelCheckpoint(
            filepath=f"models/deep_hac/temp_best_{model_type}_h{horizon}_{run_id}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0  # Reduzido para evitar poluição visual
        )
    ]

# ============================================================
# PREPARAÇÃO DE DADOS E NORMALIZAÇÃO (MANTIDO IGUAL)
# ============================================================

def prepare_multitarget_data(
    X: np.ndarray, 
    y: np.ndarray,
    val_split: float = TrainConfig.VAL_SPLIT,
    test_split: float = TrainConfig.TEST_SPLIT
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Any, Any]:
    total_samples = len(X)
    test_idx = int(total_samples * (1 - test_split))
    val_idx = int(test_idx * (1 - val_split))
    
    X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
    y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    logger.info(f"📊 Split dos dados - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")
    
    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), scaler_X, scaler_y

# ============================================================
# AVALIAÇÃO MULTI-ALVO (MANTIDO IGUAL)
# ============================================================

def calculate_multitarget_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict[str, Any]:
    metrics = {}
    
    for i, target_name in enumerate(target_names):
        y_true_target = y_true[:, i]
        y_pred_target = y_pred[:, i]
        
        metrics[target_name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true_target, y_pred_target))),
            "mae": float(mean_absolute_error(y_true_target, y_pred_target)),
            "r2": float(r2_score(y_true_target, y_pred_target))
        }
        
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true_target - y_pred_target) / np.where(y_true_target != 0, y_true_target, 1))) * 100
            metrics[target_name]["mape"] = float(mape)
    
    metrics["global"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }
    
    return metrics

# ============================================================
# TREINAMENTO POR HORIZONTE - COM VERIFICAÇÃO DE MEMÓRIA
# ============================================================

def train_model_for_horizon(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    horizon_h: int,
    model_type: str = "lstm",
    lookback: int = TrainConfig.DEFAULT_LOOKBACK,
    run_id: str = None
) -> Optional[Dict[str, Any]]:
    """
    Treina um modelo multi-alvo para horizonte específico
    """
    logger.info(f"\n🎯 Treinando {model_type.upper()} multi-alvo para horizonte {horizon_h}h")
    
    try:
        # Cria sequências multi-alvo
        X, y = create_sequences_multitarget(
            df, feature_cols, target_cols, lookback, horizon_h
        )
        
        if len(X) < TrainConfig.MIN_SAMPLES:
            logger.warning(f"⚠️ Dados insuficientes para treinamento: {len(X)} < {TrainConfig.MIN_SAMPLES}")
            return None
        
        # ✅ VERIFICAÇÃO DE MEMÓRIA ANTES DO TREINO
        memory_safe = check_memory_safety(
            n_samples=len(X),
            lookback=lookback, 
            n_features=len(feature_cols),
            n_targets=len(target_cols)
        )
        
        if not memory_safe:
            logger.warning(f"⚠️ Pulando treino por questões de memória: {model_type} H{horizon}h")
            return None
        
        # Prepara dados
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = prepare_multitarget_data(X, y)
        
        # Constrói modelo
        input_shape = (lookback, len(feature_cols))
        output_dim = len(target_cols)
        
        if model_type == "lstm":
            model = build_multitarget_lstm(input_shape, output_dim)
        elif model_type == "gru":
            model = build_multitarget_gru(input_shape, output_dim)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        logger.info(f"🏗️ {model_type.upper()} multi-alvo construído: {model.count_params():,} parâmetros")
        
        # ✅ CORREÇÃO: Passar run_id para callbacks
        callbacks = create_callbacks(model_type, horizon_h, run_id)
        
        # Treinamento
        history = model.fit(
            X_train, y_train,
            batch_size=TrainConfig.BATCH_SIZE,
            epochs=TrainConfig.MAX_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        # Avaliação no conjunto de teste
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)
        
        # Métricas
        metrics = calculate_multitarget_metrics(y_true, y_pred, target_cols)
        
        # Log das métricas
        logger.info(f"📊 {model_type.upper()} H{horizon_h}h - Métricas globais:")
        logger.info(f"   RMSE: {metrics['global']['rmse']:.3f}, MAE: {metrics['global']['mae']:.3f}, R²: {metrics['global']['r2']:.3f}")
        
        for target in target_cols:
            target_metrics = metrics[target]
            logger.info(f"   {target:8} -> RMSE: {target_metrics['rmse']:6.3f}, MAE: {target_metrics['mae']:6.3f}, MAPE: {target_metrics['mape']:5.1f}%")
        
        return {
            "model": model,
            "history": history.history,
            "metrics": metrics,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "test_data": (X_test, y_test, y_pred, y_true),
            "targets": target_cols,
            "lookback": lookback,
            "features": feature_cols
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no treino {model_type} H{horizon_h}h: {e}")
        return None

# ============================================================
# EXECUÇÃO PRINCIPAL - COM ANÁLISE DE CORRELAÇÃO
# ============================================================

def run_multibase_training(
    horizons: List[int] = None,
    model_types: List[str] = None
) -> Tuple[Dict, Dict]:
    """
    Executa pipeline completo de treinamento multi-base e multi-alvo
    """
    logger.info("🚀 INICIANDO HAC 5.1 - TREINAMENTO MULTI-BASE E MULTI-ALVO")
    start_time = datetime.now()
    
    # ✅ CONFIGURAÇÕES AVANÇADAS
    setup_advanced_training()
    
    horizons = horizons or DEFAULT_HORIZONS
    model_types = model_types or DEFAULT_MODEL_TYPES
    
    # Criar diretórios
    os.makedirs("models/deep_hac", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)
    
    # 1. Carregar e unificar dados
    df = load_and_merge_multibase()
    
    # 2. Detectar targets e features
    target_mapping = detect_target_columns(df)
    target_cols = list(target_mapping.values())
    feature_cols = select_feature_columns(df, target_cols)
    
    # ✅ ANÁLISE DE CORRELAÇÃO ANTES DO TREINO
    correlation_analysis = analyze_feature_correlations(df[feature_cols], target_cols)
    
    # Limpeza final
    df_clean = df[feature_cols].dropna()
    if len(df_clean) < TrainConfig.MIN_SAMPLES + TrainConfig.MAX_LOOKBACK + max(horizons):
        raise ValueError(f"❌ Dados insuficientes após limpeza: {len(df_clean)} registros")
    
    logger.info(f"🎯 Targets finais: {target_cols}")
    logger.info(f"🧬 Features finais: {len(feature_cols)} colunas")
    logger.info(f"📊 Dados para treino: {len(df_clean)} registros")
    
    # ID da execução
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Estruturas para resultados
    all_results = {}
    metrics_summary = {}
    
    # Loop de treinamento
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 TREINANDO {model_type.upper()}")
        logger.info(f"{'='*60}")
        
        all_results[model_type] = {}
        metrics_summary[model_type] = {}
        
        for horizon in horizons:
            try:
                # ✅ CORREÇÃO: Passar run_id para o treino
                result = train_model_for_horizon(
                    df_clean, feature_cols, target_cols, horizon, model_type, TrainConfig.DEFAULT_LOOKBACK, run_id
                )
                
                all_results[model_type][horizon] = result
                
                if result is None:
                    logger.warning(f"⏭️  Pulando {model_type} H{horizon}h")
                    continue
                
                # Salvar modelo
                model_filename = f"{model_type}_h{horizon}_{run_id}.keras"
                model_path = os.path.join("models/deep_hac", model_filename)
                result["model"].save(model_path)
                
                # Salvar scalers
                scaler_x_filename = f"scaler_X_{model_type}_h{horizon}_{run_id}.pkl"
                scaler_y_filename = f"scaler_Y_{model_type}_h{horizon}_{run_id}.pkl"
                
                scaler_x_path = os.path.join("models/deep_hac", scaler_x_filename)
                scaler_y_path = os.path.join("models/deep_hac", scaler_y_filename)
                
                with open(scaler_x_path, "wb") as f:
                    pickle.dump(result["scaler_X"], f)
                with open(scaler_y_path, "wb") as f:
                    pickle.dump(result["scaler_y"], f)
                
                # Atualizar métricas
                metrics = result["metrics"].copy()
                metrics.update({
                    "model_file": model_filename,
                    "scaler_X_file": scaler_x_filename,
                    "scaler_Y_file": scaler_y_filename,
                    "targets": target_cols,  # ✅ ORDEM EXATA DOS TARGETS SALVA
                    "features": feature_cols,
                    "lookback": result["lookback"],
                    "training_time": str(datetime.now() - start_time)
                })
                
                # ✅ SALVAR ANÁLISE DE CORRELAÇÃO TAMBÉM
                metrics["correlation_analysis"] = correlation_analysis
                
                metrics_summary[model_type][str(horizon)] = metrics
                
                logger.info(f"💾 Modelo salvo: {model_path}")
                logger.info(f"💾 Scalers salvos: {scaler_x_filename}, {scaler_y_filename}")
                
            except Exception as e:
                logger.error(f"❌ Erro em {model_type} H{horizon}h: {e}")
                continue
    
    # Salvar resultados
    metrics_path = f"results/training/metrics_multitarget_{run_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Relatório final
    execution_time = datetime.now() - start_time
    generate_training_report(metrics_summary, execution_time, run_id, target_cols)
    
    logger.info("✅ TREINAMENTO HAC MULTI-BASE CONCLUÍDO!")
    return all_results, metrics_summary

def generate_training_report(metrics_summary: Dict, execution_time: timedelta, run_id: str, target_cols: List[str]):
    """Gera relatório final de treinamento"""
    report_path = f"results/training/report_multitarget_{run_id}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("HAC 5.1 - RELATÓRIO DE TREINAMENTO MULTI-ALVO\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Data de execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tempo de execução: {execution_time}\n")
        f.write(f"Targets: {', '.join(target_cols)}\n\n")
        
        for model_type in metrics_summary.keys():
            f.write(f"\n{model_type.upper()}:\n")
            f.write("-" * 50 + "\n")
            
            horizons_data = metrics_summary.get(model_type, {})
            if not horizons_data:
                f.write("  Nenhum modelo treinado com sucesso\n")
                continue
            
            for horizon_str, metrics in horizons_data.items():
                horizon = int(horizon_str)
                global_metrics = metrics.get("global", {})
                
                f.write(f"  H{horizon:2d}h -> RMSE: {global_metrics.get('rmse', 0):7.3f} | ")
                f.write(f"MAE: {global_metrics.get('mae', 0):6.3f} | ")
                f.write(f"R²: {global_metrics.get('r2', 0):6.3f}\n")
                
                for target in target_cols:
                    if target in metrics:
                        target_metrics = metrics[target]
                        f.write(f"        {target:8} -> RMSE: {target_metrics.get('rmse', 0):6.3f} | ")
                        f.write(f"MAE: {target_metrics.get('mae', 0):6.3f} | ")
                        f.write(f"MAPE: {target_metrics.get('mape', 0):5.1f}%\n")
    
    logger.info(f"📋 Relatório salvo: {report_path}")

# ============================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HAC 5.1 - Treinamento Multi-Base e Multi-Alvo")
    parser.add_argument("--horizons", nargs="+", type=int, help="Horizontes de previsão (ex: 1 3 6 12 24 48)")
    parser.add_argument("--models", nargs="+", type=str, help="Tipos de modelo (ex: lstm gru)")
    
    args = parser.parse_args()
    
    horizons = args.horizons if args.horizons else DEFAULT_HORIZONS
    models = [m.lower() for m in args.models] if args.models else DEFAULT_MODEL_TYPES
    
    logger.info(f"🎯 Horizontes: {horizons}")
    logger.info(f"🧠 Modelos: {models}")
    
    try:
        results, metrics = run_multibase_training(horizons, models)
        logger.info("✅ Processo de treinamento concluído com sucesso!")
    except Exception as e:
        logger.error(f"❌ Falha no treinamento: {e}")
        exit(1)
