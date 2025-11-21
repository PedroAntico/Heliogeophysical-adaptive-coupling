"""
hac_realtime_predict.py
HAC 5.2 — Previsor Real-Time Multi-Alvo (speed, Bz, density)
Compatível com: train_hac_multibase.py (HAC 5.1)
"""

import os
import re
import json
import pickle
import logging
import warnings
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HAC_Realtime_Multi - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hac_realtime_multitarget.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("HAC_Realtime_Multi")

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configurações globais
VALID_MODEL_TYPES = ["lstm", "gru", "bilstm", "cnn_lstm"]
VALID_HORIZONS = [1, 3, 6, 12, 24, 48]

ENSEMBLE_WEIGHTS = {
    "lstm": 0.4,
    "gru": 0.4,
    "bilstm": 0.1,
    "cnn_lstm": 0.1
}

class RealtimeConfig:
    MIN_DATA_POINTS = 24
    MAX_SEQUENCE_GAP = 100  # máximo de pontos faltantes permitidos
    MEMORY_CLEANUP_INTERVAL = 5  # limpar memória a cada 5 modelos

# ============================================================
# UTILS AVANÇADOS
# ============================================================

def setup_memory_optimization():
    """Configura otimizações de memória para TensorFlow"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ GPU configurada com memory growth")
        except RuntimeError as e:
            logger.warning(f"⚠️ Erro na configuração GPU: {e}")

def json_safe(obj: Any) -> Any:
    """Converte objetos complexos em tipos serializáveis para JSON com fallbacks robustos."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None

def safe_timestamp(x) -> str:
    """Conversão segura de timestamp com múltiplos fallbacks"""
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.isoformat()
    try:
        return pd.to_datetime(x).isoformat()
    except Exception:
        return datetime.utcnow().isoformat()

def clean_memory():
    """Limpeza agressiva de memória para evitar vazamentos"""
    gc.collect()
    tf.keras.backend.clear_session()
    logger.debug("🧹 Memória limpa")

# ============================================================
# 1. CARREGAR METADATA DE TREINO (metrics_multitarget_*.json) - CORRIGIDO
# ============================================================

def find_latest_metrics_file() -> str:
    """Encontra o último arquivo de métricas multitarget em results/training."""
    metrics_dir = "results/training"
    if not os.path.exists(metrics_dir):
        raise FileNotFoundError("❌ Diretório results/training não encontrado")

    files = [
        os.path.join(metrics_dir, f)
        for f in os.listdir(metrics_dir)
        if f.startswith("metrics_multitarget_") and f.endswith(".json")
    ]
    
    if not files:
        # Fallback: procurar em models/deep_hac por run_id
        logger.warning("📁 Nenhum metrics_*.json encontrado, buscando run_id nos modelos...")
        return find_run_id_from_models()
    
    # ✅ CORREÇÃO: Ordenar por data de criação real
    latest = max(files, key=os.path.getctime)
    logger.info(f"📄 Usando arquivo de métricas: {os.path.basename(latest)}")
    return latest

def find_run_id_from_models() -> str:
    """Fallback: extrai run_id dos arquivos de modelo quando metrics não existe"""
    model_dir = "models/deep_hac"
    if not os.path.exists(model_dir):
        raise FileNotFoundError("❌ Diretório models/deep_hac também não encontrado")
    
    run_ids = set()
    for fname in os.listdir(model_dir):
        if fname.endswith(".keras"):
            # Padrão: model_h1_20250101_120000.keras
            match = re.search(r"_(\d{8}_\d{6})\.keras$", fname)
            if match:
                run_ids.add(match.group(1))
    
    if not run_ids:
        raise FileNotFoundError("❌ Nenhum run_id encontrado nos arquivos de modelo")
    
    latest_run_id = sorted(run_ids)[-1]  # Pega o mais recente
    logger.info(f"🔍 Run_id detectado dos modelos: {latest_run_id}")
    
    # Cria um metrics_summary básico para continuar
    return create_fallback_metrics(latest_run_id)

def create_fallback_metrics(run_id: str) -> str:
    """Cria um arquivo de métricas fallback quando o original não existe"""
    fallback_path = f"results/training/metrics_multitarget_{run_id}.json"
    os.makedirs("results/training", exist_ok=True)
    
    fallback_data = {
        "lstm": {},
        "gru": {}
    }
    
    with open(fallback_path, "w", encoding="utf-8") as f:
        json.dump(fallback_data, f, indent=2)
    
    logger.warning(f"📝 Criado arquivo de métricas fallback: {fallback_path}")
    return fallback_path

def load_training_metadata() -> Tuple[Dict[str, Any], str]:
    """
    Carrega o último metrics_multitarget_*.json e retorna:
    - metrics_summary (dict)
    - run_id (str)
    """
    metrics_path = find_latest_metrics_file()
    
    # Extrai run_id do filename de forma robusta
    filename = os.path.basename(metrics_path)
    match = re.search(r"metrics_multitarget_(\d{8}_\d{6})\.json", filename)
    if not match:
        # Fallback: tenta qualquer formato numérico
        match = re.search(r"metrics_multitarget_(\d+)\.json", filename)
        if not match:
            raise ValueError(f"❌ Não consegui extrair run_id de {filename}")
    
    run_id = match.group(1)

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar {metrics_path}: {e}")
        # Fallback: metrics vazio
        metrics_summary = {"lstm": {}, "gru": {}}

    logger.info(f"🔎 Run_id detectado: {run_id}")
    return metrics_summary, run_id

# ============================================================
# 2. CARREGAR MODELOS E SCALERS - CORRIGIDO E MELHORADO
# ============================================================

def load_all_models_for_run(run_id: str, metrics_summary: Dict[str, Any]):
    """
    Carrega modelos e scalers apenas para o run_id fornecido.
    Agora com fallbacks robustos para metadata faltante.
    """
    model_dir = "models/deep_hac"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Diretório de modelos não encontrado: {model_dir}")

    models: Dict[str, Dict[int, Any]] = {}
    scalers: Dict[str, Dict[int, Dict[str, RobustScaler]]] = {}
    metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    loaded_count = 0

    # 2.1 Carregar modelos .keras que contenham esse run_id
    for fname in os.listdir(model_dir):
        if not fname.endswith(".keras"):
            continue
        if run_id not in fname:
            continue

        path = os.path.join(model_dir, fname)
        fname_lower = fname.lower()

        # ✅ PADRÃO CORRIGIDO: Suporta múltiplos formatos
        patterns = [
            r"(lstm|gru|bilstm|cnn_lstm)_h(\d+)_" + re.escape(run_id),
            r"(lstm|gru|bilstm|cnn_lstm)_h?(\d+).*" + re.escape(run_id),
        ]
        
        model_type, horizon = None, None
        for pattern in patterns:
            match = re.search(pattern, fname_lower)
            if match:
                model_type = match.group(1)
                horizon = int(match.group(2))
                break
        
        if not model_type or not horizon:
            logger.warning(f"⚠️ Nome de modelo não segue padrão esperado: {fname}")
            continue

        if model_type not in models:
            models[model_type] = {}
            metadata[model_type] = {}

        try:
            model = load_model(path, compile=False)
            model.compile(optimizer="adam", loss="mse")
            models[model_type][horizon] = model

            # ✅ METADATA COM FALLBACKS ROBUSTOS
            horizon_str = str(horizon)
            metrics = metrics_summary.get(model_type, {}).get(horizon_str, {})
            
            # Fallback para targets
            targets = metrics.get("targets", [])
            if not targets:
                # Tenta inferir do nome do arquivo ou usa padrão
                if "speed" in fname_lower or "bz" in fname_lower or "density" in fname_lower:
                    targets = ["speed", "bz_gse", "density"]
                else:
                    targets = ["speed", "bz_gse", "density"]  # padrão
            
            # Fallback para features  
            features = metrics.get("features", [])
            if not features:
                features = targets + ["temperature", "bt", "bx_gse", "by_gse"]
            
            # Fallback para lookback
            lookback = metrics.get("lookback", 48)
            if model.input_shape and len(model.input_shape) > 1:
                lookback = model.input_shape[1]  # prioridade: shape do modelo

            metadata[model_type][horizon] = {
                "filename": fname,
                "targets": targets,
                "features": features,
                "lookback": lookback,
                "params": model.count_params(),
                "input_shape": str(model.input_shape),
                "loaded_at": datetime.utcnow().isoformat()
            }

            loaded_count += 1
            logger.info(
                f"📦 Modelo carregado: {model_type.upper()} H{horizon} "
                f"(targets={len(targets)}, lookback={lookback}, params={model.count_params():,})"
            )

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo {fname}: {e}")
            continue

    # 2.2 Carregar scalers do mesmo run_id - COM FALLBACKS
    scaler_files_loaded = 0
    for fname in os.listdir(model_dir):
        if not fname.endswith(".pkl"):
            continue
        if run_id not in fname:
            continue

        path = os.path.join(model_dir, fname)
        fname_lower = fname.lower()

        # ✅ PADRÕES FLEXÍVEIS PARA SCALERS
        patterns = [
            r"scaler_(x|y)_(lstm|gru|bilstm|cnn_lstm)_h(\d+)_" + re.escape(run_id),
            r"scaler_(x|y).*" + re.escape(run_id),
        ]
        
        xy_flag, model_type, horizon = None, None, None
        for pattern in patterns:
            match = re.search(pattern, fname_lower)
            if match:
                xy_flag = match.group(1).upper()
                model_type = match.group(2) if len(match.groups()) > 1 else "lstm"  # fallback
                horizon = int(match.group(3)) if len(match.groups()) > 2 else 1     # fallback
                break
        
        if not xy_flag:
            continue

        if model_type not in scalers:
            scalers[model_type] = {}
        if horizon not in scalers[model_type]:
            scalers[model_type][horizon] = {}

        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            scalers[model_type][horizon][xy_flag] = scaler
            scaler_files_loaded += 1
            logger.debug(f"🔧 Scaler {xy_flag} carregado para {model_type.upper()} H{horizon}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar scaler {fname}: {e}")
            continue

    logger.info(f"📊 Total carregado: {loaded_count} modelos, {scaler_files_loaded} scalers (run_id={run_id})")
    
    if loaded_count == 0:
        raise RuntimeError(f"❌ Nenhum modelo válido carregado para run_id {run_id}")
    
    return models, scalers, metadata

# ============================================================
# 3. CARREGAR DADOS RECENTES - CORRIGIDO
# ============================================================

def load_latest_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega dados mais recentes com estratégias robustas e validação.
    """
    search_paths = ["data_real", "data", "data/raw", "data/processed", "dataset"]

    if data_path and os.path.exists(data_path):
        file_path = data_path
        logger.info(f"📥 Carregando dados do caminho especificado: {file_path}")
    else:
        file_path = None
        for folder in search_paths:
            if not os.path.exists(folder):
                continue
            try:
                csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
                if not csvs:
                    continue
                
                # ✅ CORREÇÃO: Usar data de modificação mais recente
                latest_file = max(
                    csvs,
                    key=lambda x: os.path.getmtime(os.path.join(folder, x))
                )
                file_path = os.path.join(folder, latest_file)
                logger.info(f"📥 Carregando dados mais recentes: {file_path}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Erro ao acessar {folder}: {e}")
                continue

    if not file_path:
        raise FileNotFoundError("❌ Nenhum CSV encontrado nas pastas de busca")

    # ✅ CARREGAMENTO ROBUSTO COM MÚLTIPLOS ENCODINGS
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "windows-1252"]
    df = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"✅ Arquivo carregado com encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"⚠️ Erro com encoding {enc}: {e}")
            continue

    if df is None or df.empty:
        raise ValueError(f"❌ Não foi possível carregar dados de {file_path}")

    # Normalizar colunas para lowercase (compatível com treino)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # ✅ DETECÇÃO ROBUSTA DE TIMESTAMP
    timestamp_candidates = ["timestamp", "time", "datetime", "date_time", "date"]
    tcol = None
    for c in timestamp_candidates:
        if c in df.columns:
            tcol = c
            break
    
    if tcol:
        try:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
            df = df.dropna(subset=[tcol])
            df = df.sort_values(tcol).reset_index(drop=True)
            logger.info(f"🕒 Coluna temporal usada: {tcol} ({len(df)} registros após limpeza)")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao tratar coluna temporal {tcol}: {e}")

    # ✅ VALIDAÇÃO DE DADOS MÍNIMOS
    if len(df) < RealtimeConfig.MIN_DATA_POINTS:
        raise ValueError(f"❌ Dados insuficientes: {len(df)} < {RealtimeConfig.MIN_DATA_POINTS}")

    logger.info(f"📊 Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    # Log de colunas numéricas disponíveis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"🔢 Colunas numéricas disponíveis: {numeric_cols}")
    
    return df

# ============================================================
# 4. PREPARAR SEQUÊNCIA - CORRIGIDO COM FALLBACKS
# ============================================================

def prepare_sequence_for_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    scaler_X: Optional[RobustScaler]
) -> np.ndarray:
    """
    Prepara uma sequência (1, lookback, n_features) para previsão.
    Com fallbacks robustos para dados faltantes.
    """
    # ✅ VERIFICAÇÃO ROBUSTA DE FEATURES
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = set(feature_cols) - set(available_features)
    
    if missing_features:
        logger.warning(f"⚠️ Features faltando: {missing_features}")
        
        if len(available_features) == 0:
            raise ValueError(f"❌ Nenhuma feature disponível de {feature_cols}")
        
        # Usa apenas as features disponíveis
        logger.info(f"🔄 Usando {len(available_features)} features disponíveis de {len(feature_cols)}")
        feature_cols = available_features

    # ✅ PREPARAÇÃO DE DADOS COM FALLBACKS
    if len(df) < lookback:
        available = len(df)
        logger.warning(f"⚠️ Dados insuficientes ({available} < {lookback}), replicando linhas")
        
        if available == 0:
            raise ValueError("❌ Nenhum dado disponível para criar sequência")
        
        # Replicação inteligente mantendo estatísticas
        reps = (lookback // available) + 1
        df_rep = pd.concat([df] * reps, ignore_index=True)
        data_slice = df_rep[feature_cols].iloc[-lookback:].values
    else:
        data_slice = df[feature_cols].iloc[-lookback:].values

    # ✅ NORMALIZAÇÃO COM FALLBACKS EM CASCATA
    if scaler_X is not None:
        try:
            orig_shape = data_slice.shape
            data_2d = data_slice.reshape(-1, orig_shape[-1])
            data_scaled = scaler_X.transform(data_2d)
            data_slice = data_scaled.reshape(orig_shape)
            logger.debug("🔧 Scaler X aplicado com sucesso")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao aplicar scaler_X: {e}, usando normalização simples")
            # Fallback 1: normalização por feature
            try:
                data_slice = (data_slice - data_slice.mean(axis=0)) / (data_slice.std(axis=0) + 1e-8)
            except Exception:
                # Fallback 2: normalização global
                data_slice = data_slice / (np.max(np.abs(data_slice)) + 1e-8)
    else:
        # Normalização básica como fallback
        data_slice = (data_slice - data_slice.mean(axis=0)) / (data_slice.std(axis=0) + 1e-8)

    return data_slice.reshape(1, lookback, len(feature_cols))

# ============================================================
# 5. CLASSIFICAÇÃO DE RISCO MELHORADA
# ============================================================

def classify_advanced_risk(speed: float, bz: float, density: float = None, bt: float = None) -> Dict[str, Any]:
    """
    Classificação de risco avançada com thresholds ajustáveis e múltiplos fatores.
    """
    score = 0
    triggers = []
    warnings = []

    # ✅ VALORES PADRÃO SEGUROS
    if speed is None:
        speed = 400
        warnings.append("Velocidade assumida como 400 km/s (valor padrão)")
    
    if bz is None:
        bz = -2
        warnings.append("Bz assumido como -2 nT (valor padrão)")

    # Velocidade do vento solar
    if speed > 500:
        score += 1
        triggers.append(f"VELOCIDADE ALTA: {speed:.0f} km/s")
    if speed > 600:
        score += 1
    if speed > 700:
        score += 2
        triggers.append(f"VELOCIDADE MUITO ALTA: {speed:.0f} km/s")
    if speed > 800:
        score += 2
        triggers.append(f"VELOCIDADE EXTREMA: {speed:.0f} km/s")

    # Componente Bz
    if bz < -5:
        score += 1
        triggers.append(f"BZ NEGATIVO: {bz:.1f} nT")
    if bz < -10:
        score += 2
        triggers.append(f"BZ FORTEMENTE NEGATIVO: {bz:.1f} nT")
    if bz < -15:
        score += 2
        triggers.append(f"BZ CRÍTICO: {bz:.1f} nT")

    # Densidade
    if density is not None:
        if density > 20:
            score += 1
            triggers.append(f"DENSIDADE ALTA: {density:.1f} p/cm³")
        if density > 50:
            score += 1
            triggers.append(f"DENSIDADE MUITO ALTA: {density:.1f} p/cm³")
    else:
        warnings.append("Densidade não disponível para avaliação")

    # Campo magnético total
    if bt is not None:
        if bt > 10:
            score += 1
        if bt > 20:
            score += 1
            triggers.append(f"CAMPO MAGNÉTICO FORTE: {bt:.1f} nT")
    else:
        warnings.append("Campo magnético total não disponível para avaliação")

    # ✅ CLASSIFICAÇÃO FINAL
    if score <= 2:
        level = "BAIXO"
        color = "green"
        emoji = "🟢"
    elif score <= 4:
        level = "MODERADO"
        color = "yellow"
        emoji = "🟡"
    elif score <= 7:
        level = "ALTO"
        color = "orange"
        emoji = "🟠"
    else:
        level = "CRÍTICO"
        color = "red"
        emoji = "🔴"

    return {
        "score": score,
        "level": level,
        "color": color,
        "emoji": emoji,
        "triggers": triggers,
        "warnings": warnings,
        "max_possible_score": 15
    }

# ============================================================
# 6. ENSEMBLE MULTI-ALVO - CORRIGIDO
# ============================================================

def calculate_ensemble_multitarget(
    predictions: Dict[str, Dict[int, Dict[str, float]]],
    targets_per_horizon: Dict[int, List[str]]
) -> Dict[int, Dict[str, Any]]:
    """
    Calcula ensemble multi-alvo com proteção contra divisão por zero e dados faltantes.
    """
    ensemble: Dict[int, Dict[str, Any]] = {}

    for horizon, target_list in targets_per_horizon.items():
        horizon_info = {
            "targets": target_list,
            "values": {},
            "stats": {},
            "models_contributing": {}
        }

        for tgt in target_list:
            vals = []
            weights = []
            models_used = []

            # Coleta previsões de todos os modelos para este target
            for mtype, horizons_data in predictions.items():
                if horizon in horizons_data and tgt in horizons_data[horizon]:
                    value = horizons_data[horizon][tgt]
                    vals.append(value)
                    weights.append(ENSEMBLE_WEIGHTS.get(mtype, 0.1))
                    models_used.append(mtype)

            if not vals:
                horizon_info["models_contributing"][tgt] = []
                continue

            # ✅ CÁLCULO ROBUSTO COM PROTEÇÃO CONTRA DIVISÃO POR ZERO
            vals_arr = np.array(vals, dtype=float)
            w = np.array(weights, dtype=float)
            
            # Normaliza pesos se soma for zero
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()

            value = float(np.average(vals_arr, weights=w))
            std = float(vals_arr.std())
            
            # ✅ PROTEÇÃO CONTRA DIVISÃO POR ZERO
            denominator = max(abs(np.mean(vals_arr)), 1e-6)
            confidence = max(0.0, min(1.0, 1.0 - (std / denominator)))

            horizon_info["values"][tgt] = value
            horizon_info["stats"][tgt] = {
                "min": float(vals_arr.min()),
                "max": float(vals_arr.max()),
                "std": std,
                "confidence": confidence,
                "n_models": len(vals_arr)
            }
            horizon_info["models_contributing"][tgt] = models_used

        if horizon_info["values"]:
            ensemble[horizon] = horizon_info

    return ensemble

# ============================================================
# 7. EXECUÇÃO PRINCIPAL - CORRIGIDA E MELHORADA
# ============================================================

def run_hac_realtime_multitarget(data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Pipeline completo com tratamento de erro robusto e otimizações de memória.
    """
    logger.info("🚀 INICIANDO HAC 5.2 – PREVISOR REAL-TIME MULTI-ALVO")
    start_time = datetime.utcnow()
    
    # ✅ CONFIGURAÇÃO INICIAL
    setup_memory_optimization()

    try:
        # 1) Carregar metadata de treinamento
        metrics_summary, run_id = load_training_metadata()

        # 2) Carregar modelos e scalers desse run_id
        models, scalers, model_metadata = load_all_models_for_run(run_id, metrics_summary)
        if not any(models.values()):
            raise RuntimeError("❌ Nenhum modelo carregado para o run_id atual")

        # 3) Carregar dados recentes
        df = load_latest_data(data_path)

        # 4) Determinar condições atuais
        current_conditions = {}
        
        # ✅ DETECÇÃO ROBUSTA DE VARIÁVEIS
        speed_candidates = ["speed", "flow_speed", "v_sw", "proton_speed", "vx", "vp"]
        bz_candidates = ["bz_gse", "bz_gsm", "bz", "bzgse"] 
        density_candidates = ["density", "n_p", "np", "proton_density", "n"]
        bt_candidates = ["bt", "btotal", "b_total", "bmag"]
        
        def find_first_available(candidates, default=None):
            for col in candidates:
                if col in df.columns and not df[col].isnull().all():
                    return float(df[col].iloc[-1])
            return default
        
        current_conditions["speed"] = find_first_available(speed_candidates, 400.0)
        current_conditions["bz_gse"] = find_first_available(bz_candidates, -2.0)
        current_conditions["density"] = find_first_available(density_candidates)
        current_conditions["bt"] = find_first_available(bt_candidates)

        # Timestamp atual dos dados
        for cand in ["timestamp", "time", "datetime", "date_time", "date"]:
            if cand in df.columns:
                current_conditions["timestamp"] = safe_timestamp(df[cand].iloc[-1])
                break
        if "timestamp" not in current_conditions:
            current_conditions["timestamp"] = datetime.utcnow().isoformat()

        logger.info(f"🌡️ Condições atuais: {current_conditions}")

        # 5) Previsões por modelo/horizonte
        predictions_by_model: Dict[str, Dict[int, Dict[str, float]]] = {}
        targets_per_horizon: Dict[int, List[str]] = {}
        
        model_count = 0

        for mtype, horizons in models.items():
            predictions_by_model[mtype] = {}

            for horizon, model in horizons.items():
                # ✅ LIMPEZA PERIÓDICA DE MEMÓRIA
                if model_count > 0 and model_count % RealtimeConfig.MEMORY_CLEANUP_INTERVAL == 0:
                    clean_memory()
                
                model_count += 1
                
                meta = model_metadata.get(mtype, {}).get(horizon, {})
                targets = meta.get("targets", [])
                features = meta.get("features", [])
                lookback = meta.get("lookback", 48)

                if not targets or not features:
                    logger.warning(f"⚠️ Targets ou features não definidos para {mtype} H{horizon}, pulando.")
                    continue

                # Registrar targets para esse horizonte
                if horizon not in targets_per_horizon:
                    targets_per_horizon[horizon] = targets

                # Pegando scalers
                scaler_X = scalers.get(mtype, {}).get(horizon, {}).get("X")
                scaler_Y = scalers.get(mtype, {}).get(horizon, {}).get("Y")
                
                if scaler_Y is None:
                    logger.warning(f"⚠️ Sem scaler_Y para {mtype} H{horizon}, pulando.")
                    continue

                try:
                    seq = prepare_sequence_for_model(df, features, lookback, scaler_X)
                    y_scaled = model.predict(seq, verbose=0)[0]  # shape: (n_targets,)
                    y_real = scaler_Y.inverse_transform(y_scaled.reshape(1, -1))[0]

                    # Montar dicionário de saída: target_name -> valor
                    pred_dict = {}
                    for idx, tname in enumerate(targets):
                        pred_dict[tname] = float(y_real[idx])

                    predictions_by_model[mtype][horizon] = pred_dict
                    logger.info(f"📈 {mtype.upper()} H{horizon}h -> {pred_dict}")

                except Exception as e:
                    logger.error(f"❌ Erro na previsão {mtype} H{horizon}: {e}")
                    continue

        # 6) Ensemble por horizonte / alvo
        ensemble_predictions = calculate_ensemble_multitarget(predictions_by_model, targets_per_horizon)

        # 7) Avaliar risco baseado nas condições ATUAIS
        risk_assessment = classify_advanced_risk(
            current_conditions.get("speed"),
            current_conditions.get("bz_gse"),
            current_conditions.get("density"),
            current_conditions.get("bt")
        )

        # 8) Montar saída final
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        output = {
            "metadata": {
                "version": "HAC 5.2",
                "run_id": run_id,
                "generated_at": datetime.utcnow().isoformat(),
                "execution_time_seconds": execution_time,
                "models_loaded": {m: list(models[m].keys()) for m in models},
                "n_models": sum(len(h) for h in models.values()),
                "n_predictions": len(predictions_by_model)
            },
            "current_conditions": current_conditions,
            "risk_assessment": risk_assessment,
            "predictions": {
                "by_model": predictions_by_model,
                "ensemble": ensemble_predictions
            }
        }

        # 9) Salvar em JSON
        os.makedirs("results/realtime", exist_ok=True)
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/realtime/hac_rt_multitarget_{ts_str}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(output), f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Resultado realtime salvo em: {out_path}")
        logger.info(f"⚠️ {risk_assessment['emoji']} Nível de risco: {risk_assessment['level']} (score={risk_assessment['score']})")
        logger.info(f"⏱️ Tempo total de execução: {execution_time:.2f}s")

        return output

    except Exception as e:
        logger.error(f"❌ Falha crítica na execução: {e}")
        clean_memory()  # Limpa memória mesmo em caso de erro
        raise

# ============================================================
# FUNÇÃO DE SERVIÇO PARA INTEGRAÇÃO
# ============================================================

def get_latest_prediction() -> Optional[Dict[str, Any]]:
    """
    Retorna a previsão mais recente do diretório de resultados.
    Útil para integração com outros sistemas.
    """
    results_dir = "results/realtime"
    if not os.path.exists(results_dir):
        return None
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        return None
    
    try:
        # ✅ ORDENAÇÃO CORRETA POR DATA DE CRIAÇÃO
        latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
        latest_path = os.path.join(results_dir, latest_file)
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar previsão mais recente: {e}")
        return None

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HAC 5.2 – Previsor Real-Time Multi-Alvo")
    parser.add_argument("--data_path", type=str, help="Caminho opcional para um CSV específico de dados")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso para debug")
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Modo verboso ativado")

    try:
        result = run_hac_realtime_multitarget(args.data_path)
        logger.info("✅ Previsão em tempo real concluída com sucesso!")
    except Exception as e:
        logger.error(f"❌ Falha na previsão em tempo real: {e}")
        exit(1)
