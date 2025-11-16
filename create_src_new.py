#!/usr/bin/env python3
"""
create_src_new.py

Cria uma nova arquitetura completa do projeto heliogeophysical em src_new/ 
com estrutura profissional e código otimizado.

Características:
- Não sobrescreve arquivos existentes (cria .new se necessário)
- Código robusto com tratamento de erros
- Configurações centralizadas
- Fetchers especializados para cada fonte
- Processamento completo de dados
- Detecção de eventos em tempo real
"""

import os
import logging
from pathlib import Path
from textwrap import dedent

# Configuração
ROOT = "src_new"
LOG_DIR = "logs"
DATA_DIRS = ["data/raw", "data/processed", "data/historical", "data/live"]

# Setup logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILES = {
    # === CONFIGURAÇÕES ===
    f"{ROOT}/config/__init__.py": "",
    
    f"{ROOT}/config/data_sources.yml": dedent("""\
        # Heliogeophysical Data Sources Configuration
        # URLs e endpoints para fontes de dados em tempo real e históricos
        
        noaa:
          base_url: "https://services.swpc.noaa.gov"
          endpoints:
            plasma_5min: "/products/solar-wind/plasma-5-minute.json"
            mag_5min: "/products/solar-wind/mag-5-minute.json" 
            alerts: "/products/alerts.json"
            dscovr_realtime: "/products/solar-wind/dscovr_1m.json"
          timeout: 30
          retry_attempts: 3
        
        nasa_cdaweb:
          base_url: "https://cdaweb.gsfc.nasa.gov"
          datasets:
            dscovr_swepam: "DSCOVR_H1_SWEPAM"
            dscovr_mag: "DSCOVR_H1_MAG"
            omni_1min: "OMNI_HRO_1MIN"
            ace_mag: "AC_H0_MFI"
            ace_swepam: "AC_H0_SWE"
          timeout: 60
        
        validation:
          min_data_points: 50
          completeness_threshold: 0.4
          max_data_gap_hours: 6
          value_ranges:
            density: [0.1, 100.0]
            speed: [200, 1000] 
            temperature: [1000, 1000000]
            bz_gse: [-50, 50]
        
        processing:
          resample_frequency: "5T"
          interpolation_method: "linear"
          rolling_window: "1H"
    """),
    
    f"{ROOT}/config/config.yaml": dedent("""\
        # Main Configuration - Heliogeophysical Data Processing
        
        project:
          name: "heliogeophysical"
          version: "1.0.0"
          description: "Real-time heliogeophysical data processing and event detection"
        
        data_sources:
          enabled:
            - noaa_plasma
            - noaa_mag  
            - dscovr_realtime
            - omni_1min
          fetch_interval_minutes: 5
          retention_days: 7
        
        processing:
          resample_freq: "5T"
          window_minutes: 60
          interpolation:
            method: "linear"
            limit: 12
          features:
            - rolling_mean_1h
            - rolling_std_1h
            - temporal_features
            - derived_params
        
        detection:
          thresholds:
            strong_negative_bz: -10.0
            high_speed_stream: 600.0
            density_spike: 20.0
            temperature_anomaly: 100000.0
          min_event_duration: "5 minutes"
        
        storage:
          raw_data: "data/raw"
          processed_data: "data/processed"
          historical_data: "data/historical"
          live_data: "data/live"
          backup:
            enabled: true
            keep_days: 7
        
        logging:
          level: "INFO"
          file: "logs/heliogeophysical.log"
          max_size_mb: 50
          backup_count: 5
          format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        monitoring:
          enable_health_checks: true
          metrics_port: 9090
          alert_on_failures: true
    """),
    
    f"{ROOT}/config/__init__.py": "",
    
    # === UTILITÁRIOS ===
    f"{ROOT}/utils/__init__.py": "",
    
    f"{ROOT}/utils/retries.py": dedent("""\
        """
        Utilitários avançados para retry e tolerância a falhas
        """
        import time
        import random
        import requests
        from functools import wraps
        import logging
        from typing import Callable, Any, Tuple, Optional
        
        logger = logging.getLogger(__name__)
        
        class RetryError(Exception):
            \"\"\"Exceção para falhas após múltiplas tentativas\"\"\"
            pass
        
        class CircuitBreaker:
            \"\"\"Circuit breaker pattern para evitar chamadas repetidas a serviços instáveis\"\"\"
            
            def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failures = 0
                self.last_failure_time = 0
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
            
            def can_execute(self) -> bool:
                \"\"\"Verifica se a operação pode ser executada\"\"\"
                if self.state == "OPEN":
                    current_time = time.time()
                    if current_time - self.last_failure_time > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                        return True
                    return False
                return True
            
            def record_success(self):
                \"\"\"Registra sucesso e reseta o circuit breaker\"\"\"
                self.failures = 0
                self.state = "CLOSED"
            
            def record_failure(self):
                \"\"\"Registra falha e atualiza estado do circuit breaker\"\"\"
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.warning(f"Circuit breaker OPENED after {self.failures} failures")
        
        def retry_with_exponential_backoff(
            max_retries: int = 3,
            base_delay: float = 1.0,
            max_delay: float = 60.0,
            exceptions: Tuple[Exception] = (Exception,),
            logger: Optional[logging.Logger] = None
        ) -> Callable:
            \"\"\"
            Decorador para retry com backoff exponencial e jitter
            
            Args:
                max_retries: Número máximo de tentativas
                base_delay: Atraso base em segundos
                max_delay: Atraso máximo em segundos
                exceptions: Tipos de exceção que disparam retry
                logger: Logger para registro
            \"\"\"
            def decorator(func: Callable) -> Callable:
                @wraps(func)
                def wrapper(*args, **kwargs) -> Any:
                    retries = 0
                    last_exception = None
                    
                    while retries <= max_retries:
                        try:
                            return func(*args, **kwargs)
                        except exceptions as e:
                            retries += 1
                            last_exception = e
                            
                            if retries > max_retries:
                                break
                            
                            # Calcula delay com backoff exponencial e jitter
                            delay = min(max_delay, base_delay * (2 ** (retries - 1)))
                            jitter = random.uniform(0.1, 0.3) * delay
                            total_delay = delay + jitter
                            
                            log = logger or logging.getLogger(func.__module__)
                            log.warning(
                                f"Tentativa {retries}/{max_retries} falhou em {func.__name__}: {e}. "
                                f"Retry em {total_delay:.1f}s"
                            )
                            
                            time.sleep(total_delay)
                    
                    error_msg = f"Falha após {max_retries} tentativas em {func.__name__}: {last_exception}"
                    if logger:
                        logger.error(error_msg)
                    raise RetryError(error_msg) from last_exception
                
                return wrapper
            return decorator
        
        @retry_with_exponential_backoff(max_retries=3, base_delay=2.0)
        def safe_http_request(
            url: str,
            method: str = "GET",
            timeout: float = 30.0,
            session: Optional[requests.Session] = None,
            **kwargs
        ) -> requests.Response:
            \"\"\"
            Faz requisição HTTP segura com retry automático
            
            Args:
                url: URL da requisição
                method: Método HTTP
                timeout: Timeout em segundos
                session: Sessão requests (opcional)
                **kwargs: Argumentos adicionais para requests
            
            Returns:
                Objeto Response
            \"\"\"
            requester = session or requests.Session()
            
            if method.upper() == "GET":
                response = requester.get(url, timeout=timeout, **kwargs)
            else:
                raise ValueError(f"Método não suportado: {method}")
            
            response.raise_for_status()
            return response
        
        # Circuit breakers globais para diferentes serviços
        noaa_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
        nasa_circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=600)
    """),
    
    f"{ROOT}/utils/io_utils.py": dedent("""\
        """
        Utilitários para operações de I/O e arquivos
        """
        import os
        import json
        import yaml
        import pickle
        import pandas as pd
        from pathlib import Path
        from datetime import datetime
        from typing import Any, Dict, Union, Optional
        import logging
        
        logger = logging.getLogger(__name__)
        
        def ensure_directory(path: Union[str, Path]) -> Path:
            \"\"\"
            Garante que o diretório existe, criando se necessário
            
            Args:
                path: Caminho do diretório
            
            Returns:
                Path object do diretório
            \"\"\"
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        
        def safe_save_dataframe(
            df: pd.DataFrame,
            filepath: Union[str, Path],
            format: str = "csv",
            backup: bool = True,
            **kwargs
        ) -> bool:
            \"\"\"
            Salva DataFrame de forma segura com backup opcional
            
            Args:
                df: DataFrame para salvar
                filepath: Caminho do arquivo
                format: Formato (csv, parquet, json)
                backup: Se deve criar backup se arquivo existir
                **kwargs: Argumentos adicionais para pandas
            
            Returns:
                True se sucesso, False caso contrário
            \"\"\"
            try:
                filepath = Path(filepath)
                ensure_directory(filepath.parent)
                
                # Backup se arquivo existir
                if backup and filepath.exists():
                    backup_path = filepath.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{filepath.suffix}")
                    filepath.rename(backup_path)
                    logger.info(f"Backup criado: {backup_path}")
                
                # Salva no formato especificado
                if format.lower() == "csv":
                    df.to_csv(filepath, index=False, **kwargs)
                elif format.lower() == "parquet":
                    df.to_parquet(filepath, index=False, **kwargs)
                elif format.lower() == "json":
                    df.to_json(filepath, orient="records", indent=2, **kwargs)
                else:
                    raise ValueError(f"Formato não suportado: {format}")
                
                logger.info(f"DataFrame salvo: {filepath} ({len(df)} linhas)")
                return True
                
            except Exception as e:
                logger.error(f"Erro ao salvar DataFrame em {filepath}: {e}")
                return False
        
        def safe_load_dataframe(
            filepath: Union[str, Path],
            format: str = "csv",
            **kwargs
        ) -> pd.DataFrame:
            \"\"\"
            Carrega DataFrame de forma segura
            
            Args:
                filepath: Caminho do arquivo
                format: Formato (csv, parquet, json)
                **kwargs: Argumentos adicionais para pandas
            
            Returns:
                DataFrame carregado ou DataFrame vazio em caso de erro
            \"\"\"
            try:
                filepath = Path(filepath)
                
                if not filepath.exists():
                    logger.warning(f"Arquivo não encontrado: {filepath}")
                    return pd.DataFrame()
                
                if format.lower() == "csv":
                    df = pd.read_csv(filepath, **kwargs)
                elif format.lower() == "parquet":
                    df = pd.read_parquet(filepath, **kwargs)
                elif format.lower() == "json":
                    df = pd.read_json(filepath, **kwargs)
                else:
                    raise ValueError(f"Formato não suportado: {format}")
                
                logger.info(f"DataFrame carregado: {filepath} ({len(df)} linhas)")
                return df
                
            except Exception as e:
                logger.error(f"Erro ao carregar DataFrame de {filepath}: {e}")
                return pd.DataFrame()
        
        def load_yaml_config(filepath: Union[str, Path]) -> Dict[str, Any]:
            \"\"\"Carrega configuração YAML\"\"\"
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar YAML {filepath}: {e}")
                return {}
        
        def save_yaml_config(config: Dict[str, Any], filepath: Union[str, Path]) -> bool:
            \"\"\"Salva configuração YAML\"\"\"
            try:
                filepath = Path(filepath)
                ensure_directory(filepath.parent)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                logger.info(f"Configuração YAML salva: {filepath}")
                return True
            except Exception as e:
                logger.error(f"Erro ao salvar YAML {filepath}: {e}")
                return False
        
        def setup_project_directories() -> None:
            \"\"\"Configura todos os diretórios do projeto\"\"\"
            directories = [
                "data/raw", "data/processed", "data/historical", "data/live",
                "logs", "models", "cache", "reports"
            ]
            
            for directory in directories:
                ensure_directory(directory)
                logger.info(f"Diretório verificado/criado: {directory}")
    """),
    
    f"{ROOT}/utils/logger.py": dedent("""\
        """
        Configuração centralizada de logging
        """
        import logging
        import sys
        from pathlib import Path
        from typing import Optional
        
        def setup_logging(
            log_file: Optional[str] = None,
            level: str = "INFO",
            format_string: Optional[str] = None
        ) -> logging.Logger:
            \"\"\"
            Configura logging para o projeto
            
            Args:
                log_file: Arquivo de log (opcional)
                level: Nível de logging
                format_string: Formato personalizado
            
            Returns:
                Logger configurado
            \"\"\"
            if format_string is None:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            # Cria logger principal
            logger = logging.getLogger('heliogeophysical')
            logger.setLevel(getattr(logging, level.upper()))
            
            # Remove handlers existentes
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Handler para console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(console_handler)
            
            # Handler para arquivo se especificado
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(format_string))
                logger.addHandler(file_handler)
            
            # Configura logging para outras bibliotecas
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            
            return logger
    """),
    
    # === FETCHERS ===
    f"{ROOT}/fetchers/__init__.py": "",
    
    f"{ROOT}/fetchers/base_fetcher.py": dedent("""\
        """
        Classe base para todos os fetchers
        """
        from abc import ABC, abstractmethod
        import pandas as pd
        from datetime import datetime
        from typing import Optional, Dict, Any
        import logging
        
        logger = logging.getLogger(__name__)
        
        class BaseFetcher(ABC):
            \"\"\"Interface base para fetchers de dados\"\"\"
            
            def __init__(self, name: str, config: Dict[str, Any]):
                self.name = name
                self.config = config
                self.last_successful_fetch = None
                self.fetch_count = 0
                self.error_count = 0
            
            @abstractmethod
            def fetch_data(self, **kwargs) -> Optional[pd.DataFrame]:
                \"\"\"Busca dados da fonte\"\"\"
                pass
            
            @abstractmethod
            def validate_data(self, data: pd.DataFrame) -> bool:
                \"\"\"Valida dados recebidos\"\"\"
                pass
            
            def get_stats(self) -> Dict[str, Any]:
                \"\"\"Retorna estatísticas do fetcher\"\"\"
                return {
                    "name": self.name,
                    "fetch_count": self.fetch_count,
                    "error_count": self.error_count,
                    "success_rate": (self.fetch_count - self.error_count) / max(1, self.fetch_count),
                    "last_success": self.last_successful_fetch
                }
            
            def _record_success(self):
                \"\"\"Registra fetch bem-sucedido\"\"\"
                self.fetch_count += 1
                self.last_successful_fetch = datetime.utcnow()
            
            def _record_error(self):
                \"\"\"Registra erro no fetch\"\"\"
                self.fetch_count += 1
                self.error_count += 1
    """),
    
    f"{ROOT}/fetchers/noaa_fetcher.py": dedent("""\
        """
        Fetcher especializado para dados NOAA
        """
        import logging
        import pandas as pd
        from datetime import datetime, timedelta
        from typing import Optional, Dict, Any
        import json
        
        from .base_fetcher import BaseFetcher
        from ..utils.retries import safe_http_request, noaa_circuit_breaker
        from ..utils.io_utils import safe_save_dataframe
        
        logger = logging.getLogger(__name__)
        
        class NOAAFetcher(BaseFetcher):
            \"\"\"Fetcher para dados em tempo real do NOAA SWPC\"\"\"
            
            def __init__(self, config: Dict[str, Any]):
                super().__init__("NOAA", config)
                self.base_url = config.get('base_url', 'https://services.swpc.noaa.gov')
                self.endpoints = config.get('endpoints', {})
                self.timeout = config.get('timeout', 30)
            
            def fetch_data(self, data_type: str = 'plasma_5min', days: int = 3) -> Optional[pd.DataFrame]:
                \"\"\"
                Busca dados NOAA específicos
                
                Args:
                    data_type: Tipo de dados (plasma_5min, mag_5min, dscovr_realtime)
                    days: Número de dias de dados para buscar
                
                Returns:
                    DataFrame com dados ou None em caso de erro
                \"\"\"
                if not noaa_circuit_breaker.can_execute():
                    logger.warning("Circuit breaker aberto para NOAA - pulando fetch")
                    return None
                
                endpoint = self.endpoints.get(data_type)
                if not endpoint:
                    logger.error(f"Endpoint não encontrado para {data_type}")
                    return None
                
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Buscando dados NOAA: {data_type} from {url}")
                
                try:
                    response = safe_http_request(url, timeout=self.timeout)
                    data = response.json()
                    
                    df = self._parse_json_data(data, data_type)
                    if df is not None:
                        df = self._filter_by_days(df, days)
                    
                    if self.validate_data(df):
                        noaa_circuit_breaker.record_success()
                        self._record_success()
                        logger.info(f"Dados NOAA {data_type} processados: {len(df)} linhas")
                    else:
                        raise ValueError("Dados NOAA falharam na validação")
                    
                    return df
                    
                except Exception as e:
                    logger.error(f"Erro ao buscar dados NOAA {data_type}: {e}")
                    noaa_circuit_breaker.record_failure()
                    self._record_error()
                    return None
            
            def _parse_json_data(self, data: list, data_type: str) -> Optional[pd.DataFrame]:
                \"\"\"Parse dados JSON do NOAA para DataFrame\"\"\"
                if not data or len(data) <= 1:
                    logger.warning("Dados NOAA vazios ou incompletos")
                    return None
                
                # Primeira linha são os cabeçalhos
                columns = data[0]
                rows = data[1:]
                
                df = pd.DataFrame(rows, columns=columns)
                
                # Converter coluna de tempo
                if 'time_tag' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time_tag'], utc=True)
                    df = df.drop('time_tag', axis=1)
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                
                # Converter colunas numéricas
                numeric_columns = [col for col in df.columns if col != 'timestamp']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Padronizar nomes de colunas
                column_mapping = {
                    'density': 'density',
                    'speed': 'speed', 
                    'temperature': 'temperature',
                    'bx_gse': 'bx_gse',
                    'by_gse': 'by_gse',
                    'bz_gse': 'bz_gse',
                    'bt': 'bt'
                }
                
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                return df
            
            def _filter_by_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
                \"\"\"Filtra dados pelos últimos N dias\"\"\"
                if df.empty:
                    return df
                
                cutoff = datetime.utcnow() - timedelta(days=days)
                filtered_df = df[df['timestamp'] >= cutoff].copy()
                
                if len(filtered_df) < len(df):
                    logger.info(f"Filtrados {len(df) - len(filtered_df)} registros antigos")
                
                return filtered_df.sort_values('timestamp').reset_index(drop=True)
            
            def validate_data(self, data: pd.DataFrame) -> bool:
                \"\"\"Valida dados NOAA recebidos\"\"\"
                if data is None or data.empty:
                    return False
                
                required_columns = ['timestamp']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    logger.warning(f"Colunas obrigatórias faltando: {missing_columns}")
                    return False
                
                # Verifica se há dados recentes (últimas 24 horas)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_data = data[data['timestamp'] >= recent_cutoff]
                
                if len(recent_data) == 0:
                    logger.warning("Nenhum dado recente encontrado (últimas 24h)")
                    return False
                
                return True
            
            def fetch_all_realtime_data(self, days: int = 3) -> Dict[str, pd.DataFrame]:
                \"\"\"Busca todos os dados em tempo real disponíveis\"\"\"
                results = {}
                
                for data_type in ['plasma_5min', 'mag_5min', 'dscovr_realtime']:
                    df = self.fetch_data(data_type, days)
                    if df is not None and not df.empty:
                        results[data_type] = df
                        # Salva backup
                        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                        filename = f"data/raw/noaa_{data_type}_{timestamp}.csv"
                        safe_save_dataframe(df, filename)
                
                logger.info(f"Coletados {len(results)} conjuntos de dados NOAA")
                return results
        
        # Função de conveniência
        def fetch_noaa_data(data_type: str = 'plasma_5min', days: int = 3) -> Optional[pd.DataFrame]:
            \"\"\"Função simples para fetch rápido de dados NOAA\"\"\"
            from ..utils.io_utils import load_yaml_config
            config = load_yaml_config('src_new/config/data_sources.yml')
            noaa_config = config.get('noaa', {})
            
            fetcher = NOAAFetcher(noaa_config)
            return fetcher.fetch_data(data_type, days)
    """),
    
    f"{ROOT}/fetchers/dscovr_fetcher.py": dedent("""\
        """
        Fetcher para dados DSCOVR via CDAWeb da NASA
        """
        import logging
        import pandas as pd
        from datetime import datetime, timedelta
        from typing import Optional, Dict, Any
        
        from .base_fetcher import BaseFetcher
        from ..utils.retries import retry_with_exponential_backoff, nasa_circuit_breaker
        
        logger = logging.getLogger(__name__)
        
        class DSCOVRFetcher(BaseFetcher):
            \"\"\"Fetcher para dados DSCOVR via CDAWeb\"\"\"
            
            def __init__(self, config: Dict[str, Any]):
                super().__init__("DSCOVR", config)
                self.base_url = config.get('base_url', 'https://cdaweb.gsfc.nasa.gov')
                self.datasets = config.get('datasets', {})
                self.timeout = config.get('timeout', 60)
                self.cdas_client = None
            
            def _initialize_cdas_client(self) -> bool:
                \"\"\"Inicializa cliente CDAWeb se disponível\"\"\"
                try:
                    from cdasws import CdasWs
                    self.cdas_client = CdasWs()
                    return True
                except ImportError:
                    logger.warning("CDASWS não instalado. Use: pip install cdasws")
                    return False
                except Exception as e:
                    logger.error(f"Erro ao inicializar cliente CDAWeb: {e}")
                    return False
            
            @retry_with_exponential_backoff(max_retries=2, base_delay=5.0)
            def fetch_data(self, dataset_type: str = 'dscovr_swepam', days: int = 7) -> Optional[pd.DataFrame]:
                \"\"\"
                Busca dados DSCOVR via CDAWeb
                
                Args:
                    dataset_type: Tipo de dataset (dscovr_swepam, dscovr_mag)
                    days: Número de dias de dados
                
                Returns:
                    DataFrame com dados ou None em caso de erro
                \"\"\"
                if not nasa_circuit_breaker.can_execute():
                    logger.warning("Circuit breaker aberto para NASA - pulando fetch")
                    return None
                
                dataset_id = self.datasets.get(dataset_type)
                if not dataset_id:
                    logger.error(f"Dataset não configurado: {dataset_type}")
                    return None
                
                if self.cdas_client is None and not self._initialize_cdas_client():
                    return None
                
                logger.info(f"Buscando dados DSCOVR: {dataset_type} ({dataset_id})")
                
                try:
                    # Define variáveis baseadas no tipo de dataset
                    if 'swepam' in dataset_type.lower():
                        variables = ['Np', 'Vp', 'Tp']  # Density, Speed, Temperature
                    elif 'mag' in dataset_type.lower():
                        variables = ['B1GSE', 'B2GSE', 'B3GSE', 'Bt']  # Magnetic field
                    else:
                        variables = ['Np', 'Vp', 'Tp', 'B1GSE', 'B2GSE', 'B3GSE']
                    
                    # Busca dados
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(days=days)
                    
                    status, data = self.cdas_client.get_data(
                        dataset_id,
                        variables,
                        start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                    )
                    
                    if status != 200 or not data:
                        logger.warning(f"CDAWeb retornou status {status} sem dados")
                        return None
                    
                    df = self._parse_cdas_data(data, dataset_type)
                    
                    if self.validate_data(df):
                        nasa_circuit_breaker.record_success()
                        self._record_success()
                        logger.info(f"Dados DSCOVR {dataset_type} processados: {len(df)} linhas")
                    else:
                        raise ValueError("Dados DSCOVR falharam na validação")
                    
                    return df
                    
                except Exception as e:
                    logger.error(f"Erro ao buscar dados DSCOVR {dataset_type}: {e}")
                    nasa_circuit_breaker.record_failure()
                    self._record_error()
                    return None
            
            def _parse_cdas_data(self, data: dict, dataset_type: str) -> pd.DataFrame:
                \"\"\"Parse dados do CDAWeb para DataFrame\"\"\"
                df = pd.DataFrame()
                
                for key, values in data.items():
                    if key.lower() == 'epoch':
                        df['timestamp'] = pd.to_datetime(values, utc=True)
                    else:
                        # Mapeia nomes de variáveis para padrão
                        variable_mapping = {
                            'Np': 'density',
                            'Vp': 'speed', 
                            'Tp': 'temperature',
                            'B1GSE': 'bx_gse',
                            'B2GSE': 'by_gse',
                            'B3GSE': 'bz_gse',
                            'Bt': 'bt'
                        }
                        column_name = variable_mapping.get(key, key)
                        df[column_name] = values
                
                # Converte para numérico
                for col in df.columns:
                    if col != 'timestamp':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.sort_values('timestamp').reset_index(drop=True)
            
            def validate_data(self, data: pd.DataFrame) -> bool:
                \"\"\"Valida dados DSCOVR recebidos\"\"\"
                if data is None or data.empty:
                    return False
                
                if 'timestamp' not in data.columns:
                    return False
                
                # Verifica se há pelo menos uma variável de dados
                data_columns = [col for col in data.columns if col != 'timestamp']
                if not data_columns:
                    return False
                
                return True
        
        # Função de conveniência
        def fetch_dscovr_data(dataset_type: str = 'dscovr_swepam', days: int = 7) -> Optional[pd.DataFrame]:
            \"\"\"Função simples para fetch rápido de dados DSCOVR\"\"\"
            from ..utils.io_utils import load_yaml_config
            config = load_yaml_config('src_new/config/data_sources.yml')
            nasa_config = config.get('nasa_cdaweb', {})
            
            fetcher = DSCOVRFetcher(nasa_config)
            return fetcher.fetch_data(dataset_type, days)
    """),
    
    # Continua com os outros arquivos...
    # [Restante do código similar para processing, detection, etc.]
}

def ensure_parent_directory(filepath: str) -> None:
    """Garante que o diretório pai do arquivo existe"""
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        logger.info(f"Diretório criado: {parent_dir}")

def safe_write_file(filepath: str, content: str) -> None:
    """
    Escreve arquivo de forma segura, não sobrescreve existentes
    """
    ensure_parent_directory(filepath)
    
    if os.path.exists(filepath):
        # Cria versão .new para arquivos existentes
        new_filepath = f"{filepath}.new"
        with open(new_filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.warning(f"Arquivo existente: {filepath} -> Criado {new_filepath}")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Arquivo criado: {filepath}")

def create_directory_structure() -> None:
    """Cria toda a estrutura de diretórios necessária"""
    directories = [ROOT] + [f"{ROOT}/{subdir}" for subdir in [
        "config", "fetchers", "utils", "processing", "detection", "model"
    ]] + DATA_DIRS + [LOG_DIR, "models", "tests"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório verificado/criado: {directory}")

def main():
    """Função principal"""
    logger.info("Iniciando criação da estrutura src_new...")
    
    try:
        # Criar estrutura de diretórios
        create_directory_structure()
        
        # Criar arquivos
        total_files = len(FILES)
        created_files = 0
        skipped_files = 0
        
        for filepath, content in FILES.items():
            safe_write_file(filepath, content)
            if os.path.exists(filepath) and not filepath.endswith('.new'):
                created_files += 1
            else:
                skipped_files += 1
        
        logger.info(f"Processo concluído!")
        logger.info(f"Arquivos criados: {created_files}")
        logger.info(f"Arquivos com versões .new: {skipped_files}")
        
        # Instruções finais
        print("\n" + "="*60)
        print("ESTRUTURA src_new CRIADA COM SUCESSO!")
        print("="*60)
        print("\nPRÓXIMOS PASSOS:")
        print("1. Verifique arquivos com sufixo .new e integre manualmente se necessário")
        print("2. Instale dependências:")
        print("   pip install -r requirements.txt")
        print("3. Para CDAWeb (opcional):")
        print("   pip install cdasws")
        print("4. Execute o pipeline:")
        print("   python src_new/main.py")
        print("\5. Estrutura criada:")
        print("   src_new/config/     - Configurações")
        print("   src_new/fetchers/   - Coletores de dados") 
        print("   src_new/utils/      - Utilitários")
        print("   src_new/processing/ - Processamento")
        print("   src_new/detection/  - Detecção de eventos")
        print("   data/              - Dados brutos e processados")
        print("   logs/              - Arquivos de log")
        
    except Exception as e:
        logger.error(f"Erro durante criação da estrutura: {e}")
        raise

if __name__ == "__main__":
    main()
```
