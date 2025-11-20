#!/usr/bin/env python3
"""
Setup completo do projeto Heliogeophysical com dados reais
"""
import os
from pathlib import Path

def create_structure():
    """Cria estrutura completa de diretórios"""
    dirs = [
        "src/config",
        "src/fetchers", 
        "src/utils",
        "src/processing",
        "src/detection",
        "src/model",
        "data/raw",
        "data/processed",
        "data/live",
        "data/historical",
        "logs",
        "models",
        "tests"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Diretório criado: {directory}")

def create_config_files():
    """Cria arquivos de configuração"""
    
    # config.yaml
    config_content = """# Configuração do Projeto Heliogeophysical
project:
  name: "Heliogeophysical Adaptive Coupling"
  version: "2.0.0"
  description: "Sistema de detecção de eventos heliofísicos em tempo real"

data_sources:
  noaa:
    base_url: "https://services.swpc.noaa.gov"
    endpoints:
      plasma: "/products/solar-wind/plasma-5-minute.json"
      mag: "/products/solar-wind/mag-5-minute.json"
      dscovr: "/products/solar-wind/dscovr_1m.json"
      alerts: "/products/alerts.json"
    timeout: 30
    retry_attempts: 3

processing:
  resample_frequency: "5T"
  interpolation_method: "linear"
  rolling_window: "1H"
  features:
    - rolling_mean
    - rolling_std
    - temporal_features
    - derived_parameters

detection:
  thresholds:
    strong_negative_bz: -10.0
    high_speed_stream: 600.0
    density_spike: 20.0
    temperature_anomaly: 100000.0
  min_event_duration: "5 minutes"

logging:
  level: "INFO"
  file: "logs/heliogeophysical.log"
  format: "%(asctime)s | %(levelname)-8s | %(message)s"
"""
    
    with open("src/config/config.yaml", "w") as f:
        f.write(config_content)
    print("✓ config.yaml criado")

def create_utils():
    """Cria módulos utilitários"""
    
    # utils/__init__.py
    Path("src/utils/__init__.py").touch()
    
    # utils/retries.py
    retries_content = '''import time
import random
import requests
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """Decorator para retry com backoff exponencial"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_exception = e
                    
                    if retries > max_retries:
                        break
                    
                    delay = min(max_delay, base_delay * (2 ** (retries - 1)))
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    
                    logger.warning(f"Tentativa {retries}/{max_retries} falhou. Retry em {total_delay:.1f}s: {e}")
                    time.sleep(total_delay)
            
            raise last_exception
        
        return wrapper
    return decorator

@retry_with_exponential_backoff(max_retries=3, base_delay=2.0)
def safe_http_request(url, timeout=30, **kwargs):
    """Faz requisição HTTP segura com retry"""
    response = requests.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response
'''
    
    with open("src/utils/retries.py", "w") as f:
        f.write(retries_content)
    print("✓ utils/retries.py criado")
    
    # utils/logger.py
    logger_content = '''import logging
import sys
from pathlib import Path

def setup_logging(log_file="logs/heliogeophysical.log", level="INFO"):
    """Configura o sistema de logging"""
    
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Garante que o diretório de logs existe
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configura o logger root
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Retorna logger específico para o projeto
    return logging.getLogger("heliogeophysical")
'''
    
    with open("src/utils/logger.py", "w") as f:
        f.write(logger_content)
    print("✓ utils/logger.py criado")

def create_fetchers():
    """Cria módulos de coleta de dados"""
    
    Path("src/fetchers/__init__.py").touch()
    
    # fetchers/noaa_fetcher.py
    noaa_content = '''import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ..utils.retries import safe_http_request

logger = logging.getLogger(__name__)

class NOAAFetcher:
    """Fetcher para dados em tempo real do NOAA SWPC"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://services.swpc.noaa.gov')
        self.endpoints = config.get('endpoints', {})
    
    def fetch_plasma_data(self, days: int = 3) -> Optional[pd.DataFrame]:
        """Busca dados de plasma solar (densidade, velocidade, temperatura)"""
        return self._fetch_noaa_data('plasma', days)
    
    def fetch_magnetic_data(self, days: int = 3) -> Optional[pd.DataFrame]:
        """Busca dados do campo magnético"""
        return self._fetch_noaa_data('mag', days)
    
    def fetch_dscovr_data(self, days: int = 3) -> Optional[pd.DataFrame]:
        """Busca dados do DSCOVR"""
        return self._fetch_noaa_data('dscovr', days)
    
    def _fetch_noaa_data(self, data_type: str, days: int) -> Optional[pd.DataFrame]:
        """Método interno para buscar dados NOAA"""
        endpoint = self.endpoints.get(data_type)
        if not endpoint:
            logger.error(f"Endpoint não configurado: {data_type}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"Buscando dados NOAA: {data_type}")
        
        try:
            response = safe_http_request(url)
            data = response.json()
            
            if not data or len(data) <= 1:
                logger.warning(f"Dados NOAA {data_type} vazios")
                return None
            
            # Converte JSON para DataFrame
            columns = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=columns)
            
            # Processa timestamp
            if 'time_tag' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time_tag'], utc=True)
                df = df.drop('time_tag', axis=1)
            
            # Filtra por data
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]
            
            # Converte colunas numéricas
            numeric_cols = [col for col in df.columns if col != 'timestamp']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove linhas completamente vazias
            df = df.dropna(how='all')
            
            logger.info(f"Dados NOAA {data_type} processados: {len(df)} linhas")
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados NOAA {data_type}: {e}")
            return None

def create_noaa_fetcher(config: Dict[str, Any]) -> NOAAFetcher:
    """Factory function para criar NOAAFetcher"""
    return NOAAFetcher(config)
'''
    
    with open("src/fetchers/noaa_fetcher.py", "w") as f:
        f.write(noaa_content)
    print("✓ fetchers/noaa_fetcher.py criado")

def create_processing():
    """Cria módulos de processamento"""
    
    Path("src/processing/__init__.py").touch()
    
    # processing/preprocessor.py
    processing_content = '''import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocessa dados heliofísicos para análise"""
    
    def __init__(self, resample_freq: str = "5T"):
        self.resample_freq = resample_freq
    
    def preprocess_data(self, datasets: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Preprocessa múltiplos datasets e os combina"""
        if not datasets:
            logger.warning("Nenhum dataset fornecido para preprocessing")
            return None
        
        # Combina datasets
        merged_data = self._merge_datasets(datasets)
        if merged_data.empty:
            return None
        
        # Resample e interpola
        processed_data = self._resample_data(merged_data)
        
        # Engenharia de features
        processed_data = self._feature_engineering(processed_data)
        
        logger.info(f"Dados preprocessados: {len(processed_data)} linhas")
        return processed_data
    
    def _merge_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combina múltiplos datasets por timestamp"""
        valid_datasets = [df for df in datasets if df is not None and not df.empty]
        
        if not valid_datasets:
            return pd.DataFrame()
        
        # Usa o dataset com mais dados como base
        base_df = max(valid_datasets, key=len)
        
        for df in valid_datasets:
            if df is base_df:
                continue
            
            # Encontra colunas únicas para merge
            unique_cols = [col for col in df.columns if col not in base_df.columns and col != 'timestamp']
            if not unique_cols:
                continue
            
            # Merge aproximado por timestamp
            merge_df = df[['timestamp'] + unique_cols].sort_values('timestamp')
            base_df = pd.merge_asof(
                base_df.sort_values('timestamp'),
                merge_df,
                on='timestamp',
                tolerance=pd.Timedelta('10min'),
                direction='nearest'
            )
        
        return base_df.sort_values('timestamp').reset_index(drop=True)
    
    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Redimensiona e interpola dados para frequência consistente"""
        if df.empty:
            return df
        
        df = df.set_index('timestamp')
        
        # Resample para frequência desejada
        resampled = df.resample(self.resample_freq).mean()
        
        # Interpola valores faltantes
        interpolated = resampled.interpolate(method='linear', limit=6)
        
        # Preenche valores restantes com forward fill
        filled = interpolated.ffill(limit=3)
        
        return filled.reset_index()
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features derivadas"""
        if df.empty:
            return df
        
        # Features temporais
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Features rolling para variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['hour', 'day_of_year']:
                # Média móvel
                df[f'{col}_rolling_mean_1h'] = df[col].rolling(window=12, min_periods=1).mean()
                # Desvio padrão móvel
                df[f'{col}_rolling_std_1h'] = df[col].rolling(window=12, min_periods=1).std()
        
        return df
'''
    
    with open("src/processing/preprocessor.py", "w") as f:
        f.write(processing_content)
    print("✓ processing/preprocessor.py criado")

def create_detection():
    """Cria módulos de detecção de eventos"""
    
    Path("src/detection/__init__.py").touch()
    
    # detection/event_detector.py
    detection_content = '''import logging
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class EventDetector:
    """Detector de eventos heliofísicos baseado em thresholds"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def detect_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta eventos heliofísicos baseado em thresholds"""
        events = []
        
        if df is None or df.empty:
            return events
        
        # 1. Detecção de Bz negativo forte (condição favorável para tempestades)
        if 'bz_gse' in df.columns:
            bz_events = self._detect_bz_events(df)
            events.extend(bz_events)
        
        # 2. Detecção de alta velocidade do vento solar
        if 'speed' in df.columns:
            speed_events = self._detect_speed_events(df)
            events.extend(speed_events)
        
        # 3. Detecção de picos de densidade
        if 'density' in df.columns:
            density_events = self._detect_density_events(df)
            events.extend(density_events)
        
        # 4. Detecção de anomalias de temperatura
        if 'temperature' in df.columns:
            temp_events = self._detect_temperature_events(df)
            events.extend(temp_events)
        
        logger.info(f"Detectados {len(events)} eventos")
        return events
    
    def _detect_bz_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta eventos de Bz negativo forte"""
        events = []
        threshold = self.thresholds.get('strong_negative_bz', -10.0)
        
        strong_bz = df[df['bz_gse'] <= threshold]
        
        for _, row in strong_bz.iterrows():
            events.append({
                'timestamp': row['timestamp'].isoformat(),
                'type': 'strong_negative_bz',
                'value': float(row['bz_gse']),
                'severity': self._classify_bz_severity(row['bz_gse']),
                'description': f'Bz negativo forte: {row["bz_gse"]:.1f} nT'
            })
        
        return events
    
    def _detect_speed_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta eventos de alta velocidade do vento solar"""
        events = []
        threshold = self.thresholds.get('high_speed_stream', 600.0)
        
        high_speed = df[df['speed'] >= threshold]
        
        for _, row in high_speed.iterrows():
            events.append({
                'timestamp': row['timestamp'].isoformat(),
                'type': 'high_speed_stream',
                'value': float(row['speed']),
                'severity': self._classify_speed_severity(row['speed']),
                'description': f'Alta velocidade do vento solar: {row["speed"]:.1f} km/s'
            })
        
        return events
    
    def _detect_density_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta picos de densidade do plasma"""
        events = []
        threshold = self.thresholds.get('density_spike', 20.0)
        
        density_spikes = df[df['density'] >= threshold]
        
        for _, row in density_spikes.iterrows():
            events.append({
                'timestamp': row['timestamp'].isoformat(),
                'type': 'density_spike',
                'value': float(row['density']),
                'severity': 'medium',
                'description': f'Pico de densidade: {row["density"]:.1f} p/cm³'
            })
        
        return events
    
    def _detect_temperature_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta anomalias de temperatura"""
        events = []
        threshold = self.thresholds.get('temperature_anomaly', 100000.0)
        
        temp_anomalies = df[df['temperature'] >= threshold]
        
        for _, row in temp_anomalies.iterrows():
            events.append({
                'timestamp': row['timestamp'].isoformat(),
                'type': 'temperature_anomaly',
                'value': float(row['temperature']),
                'severity': 'high',
                'description': f'Anomalia de temperatura: {row["temperature"]:.1f} K'
            })
        
        return events
    
    def _classify_bz_severity(self, bz_value: float) -> str:
        """Classifica severidade baseado no valor de Bz"""
        if bz_value <= -15:
            return 'critical'
        elif bz_value <= -10:
            return 'high'
        else:
            return 'medium'
    
    def _classify_speed_severity(self, speed_value: float) -> str:
        """Classifica severidade baseado na velocidade"""
        if speed_value >= 800:
            return 'critical'
        elif speed_value >= 600:
            return 'high'
        else:
            return 'medium'

def create_event_detector(thresholds: Dict[str, float]) -> EventDetector:
    """Factory function para criar EventDetector"""
    return EventDetector(thresholds)
'''
    
    with open("src/detection/event_detector.py", "w") as f:
        f.write(detection_content)
    print("✓ detection/event_detector.py criado")

def create_main_pipeline():
    """Cria o pipeline principal"""
    
    main_content = '''#!/usr/bin/env python3
"""
Pipeline Principal - Heliogeophysical Adaptive Coupling
Sistema de detecção de eventos heliofísicos em tempo real
"""
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path

# Importações internas
from src.utils.logger import setup_logging
from src.fetchers.noaa_fetcher import create_noaa_fetcher
from src.processing.preprocessor import DataPreprocessor
from src.detection.event_detector import create_event_detector
from src.utils.retries import safe_http_request

class HeliogeophysicalPipeline:
    """Pipeline principal para processamento heliofísico"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = setup_logging(
            self.config['logging']['file'],
            self.config['logging']['level']
        )
        
        # Inicializa componentes
        self.noaa_fetcher = create_noaa_fetcher(self.config['data_sources']['noaa'])
        self.preprocessor = DataPreprocessor(
            self.config['processing']['resample_frequency']
        )
        self.event_detector = create_event_detector(
            self.config['detection']['thresholds']
        )
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return {}
    
    def run_pipeline(self) -> dict:
        """Executa o pipeline completo"""
        self.logger.info("Iniciando pipeline de acoplamento heliogeofísico")
        
        try:
            # Fase 1: Coleta de dados
            self.logger.info("Coletando dados REALTIME")
            datasets = self._fetch_data()
            
            # Fase 2: Processamento
            self.logger.info("Processando séries temporais")
            processed_data = self._process_data(datasets)
            
            if processed_data is None or processed_data.empty:
                self.logger.warning("Nenhum dado processado disponível")
                return {"events": [], "status": "no_data"}
            
            # Fase 3: Detecção de eventos
            self.logger.info("Detectando eventos de interesse")
            events = self._detect_events(processed_data)
            
            # Fase 4: Salvar resultados
            self._save_results(processed_data, events)
            
            self.logger.info(f"Pipeline concluído — {len(events)} eventos detectados")
            
            return {
                "events": events,
                "processed_records": len(processed_data),
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline: {e}")
            return {
                "events": [],
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _fetch_data(self) -> list:
        """Coleta dados de todas as fontes"""
        datasets = []
        
        # Coleta dados NOAA
        try:
            plasma_data = self.noaa_fetcher.fetch_plasma_data(days=2)
            if plasma_data is not None:
                datasets.append(plasma_data)
                self.logger.info(f"Dados de plasma coletados: {len(plasma_data)} registros")
            
            mag_data = self.noaa_fetcher.fetch_magnetic_data(days=2)
            if mag_data is not None:
                datasets.append(mag_data)
                self.logger.info(f"Dados magnéticos coletados: {len(mag_data)} registros")
                
        except Exception as e:
            self.logger.error(f"Erro na coleta de dados NOAA: {e}")
        
        return datasets
    
    def _process_data(self, datasets: list):
        """Processa e combina os datasets"""
        return self.preprocessor.preprocess_data(datasets)
    
    def _detect_events(self, processed_data) -> list:
        """Detecta eventos nos dados processados"""
        return self.event_detector.detect_events(processed_data)
    
    def _save_results(self, processed_data, events: list):
        """Salva resultados em arquivos"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Salva dados processados
        processed_path = f"data/processed/helio_data_{timestamp}.csv"
        processed_data.to_csv(processed_path, index=False)
        self.logger.info(f"Dados processados salvos: {processed_path}")
        
        # Salva eventos detectados
        if events:
            events_path = f"data/processed/events_{timestamp}.json"
            with open(events_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
            self.logger.info(f"Eventos salvos: {events_path}")

def main():
    """Função principal"""
    pipeline = HeliogeophysicalPipeline()
    result = pipeline.run_pipeline()
    
    # Resumo executivo
    print(f"\\n{'='*50}")
    print("RELATÓRIO EXECUTIVO - HELIOGEOPHYSICAL PIPELINE")
    print(f"{'='*50}")
    print(f"Status: {result['status']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Registros processados: {result.get('processed_records', 0)}")
    print(f"Eventos detectados: {len(result['events'])}")
    
    if result['events']:
        print(f"\\nEventos detectados:")
        for event in result['events']:
            print(f"  • {event['type']}: {event['description']} (Severidade: {event['severity']})")
    
    print(f"{'='*50}")
    print("Projeto heliogeofísico operacional!")

if __name__ == "__main__":
    main()
'''
    
    with open("src/main.py", "w") as f:
        f.write(main_content)
    print("✓ src/main.py criado")

def create_requirements():
    """Cria requirements.txt atualizado"""
    requirements = '''# Core dependencies
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
urllib3>=2.0.0

# Data science
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
schedule>=1.2.0

# Testing
pytest>=7.0.0
pytest-mock>=3.10.0
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ requirements.txt atualizado")

def create_readme():
    """Cria README.md"""
    readme = '''# Heliogeophysical Adaptive Coupling

Sistema avançado para detecção de eventos heliofísicos em tempo real usando dados do NOAA SWPC e outras fontes.

## 🚀 Características

- **Coleta em tempo real** de dados solares e magnetosféricos
- **Processamento avançado** de séries temporais
- **Detecção inteligente** de eventos baseada em thresholds
- **Pipeline robusto** com retry automático e tratamento de erros
- **Monitoramento contínuo** do ambiente espacial

## 📊 Eventos Detectados

- **Bz negativo forte**: Condições favoráveis para tempestades geomagnéticas
- **Alta velocidade do vento solar**: Streams de alta velocidade
- **Picos de densidade**: Aumentos súbitos na densidade do plasma
- **Anomalias de temperatura**: Variações extremas de temperatura

## 🛠 Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar pipeline
python src/main.py

exit()

exit()
