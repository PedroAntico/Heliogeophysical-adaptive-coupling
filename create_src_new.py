cat > create_src_new.py << 'EOF'
#!/usr/bin/env python3
"""
create_src_new.py - Cria estrutura completa do projeto heliogeophysical
"""

import os
import logging
from pathlib import Path
from textwrap import dedent

# Configuração
ROOT = "src_new"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILES = {
    # requirements
    "requirements.txt": dedent("""\
        requests>=2.31.0
        pandas>=2.0.0
        numpy>=1.24.0
        scikit-learn>=1.3.0
        pyyaml>=6.0
        beautifulsoup4>=4.12.0
        urllib3>=2.0.0
        tqdm>=4.65.0
        python-dotenv>=1.0.0
        schedule>=1.2.0
        pytest>=7.0.0
    """),
    
    # config
    f"{ROOT}/__init__.py": "",
    f"{ROOT}/config/__init__.py": "",
    
    f"{ROOT}/config/data_sources.yml": dedent("""\
        noaa:
          base_url: "https://services.swpc.noaa.gov"
          endpoints:
            plasma_5min: "/products/solar-wind/plasma-5-minute.json"
            mag_5min: "/products/solar-wind/mag-5-minute.json"
            alerts: "/products/alerts.json"
            dscovr_realtime: "/products/solar-wind/dscovr_1m.json"
          timeout: 30

        nasa_cdaweb:
          base_url: "https://cdaweb.gsfc.nasa.gov"
          datasets:
            dscovr_swepam: "DSCOVR_H1_SWEPAM"
            dscovr_mag: "DSCOVR_H1_MAG"
            omni_1min: "OMNI_HRO_1MIN"
          timeout: 60

        validation:
          min_data_points: 50
          completeness_threshold: 0.4
    """),
    
    f"{ROOT}/config/config.yaml": dedent("""\
        project:
          name: "heliogeophysical"
          version: "1.0.0"

        data_sources:
          enabled:
            - noaa_plasma
            - noaa_mag
            - dscovr_realtime
          fetch_interval_minutes: 5

        processing:
          resample_freq: "5T"
          window_minutes: 60

        storage:
          raw_data: "data/raw"
          processed_data: "data/processed"

        logging:
          level: "INFO"
          file: "logs/heliogeophysical.log"
    """),
    
    # utils
    f"{ROOT}/utils/__init__.py": "",
    
    f"{ROOT}/utils/retries.py": dedent("""\
        import time
        import random
        import requests
        from functools import wraps
        import logging
        from typing import Callable, Any, Tuple

        logger = logging.getLogger(__name__)

        def retry_with_backoff(
            max_retries: int = 3,
            base_delay: float = 1.0,
            max_delay: float = 60.0,
            exceptions: Tuple[Exception] = (Exception,)
        ) -> Callable:
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
                            
                            delay = min(max_delay, base_delay * (2 ** (retries - 1)))
                            jitter = random.uniform(0.1, 0.3) * delay
                            total_delay = delay + jitter
                            
                            logger.warning(f"Tentativa {retries}/{max_retries} falhou. Retry em {total_delay:.1f}s: {e}")
                            time.sleep(total_delay)
                    
                    raise last_exception
                
                return wrapper
            return decorator

        @retry_with_backoff(max_retries=3, base_delay=2.0)
        def safe_request(url: str, timeout: float = 30.0, **kwargs) -> requests.Response:
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
    """),
    
    f"{ROOT}/utils/io_utils.py": dedent("""\
        import os
        import json
        import yaml
        import pandas as pd
        from pathlib import Path
        from typing import Any, Dict, Union
        import logging

        logger = logging.getLogger(__name__)

        def ensure_directory(path: Union[str, Path]) -> Path:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            return path

        def safe_save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], format: str = "csv") -> bool:
            try:
                filepath = Path(filepath)
                ensure_directory(filepath.parent)
                
                if format.lower() == "csv":
                    df.to_csv(filepath, index=False)
                elif format.lower() == "parquet":
                    df.to_parquet(filepath, index=False)
                else:
                    raise ValueError(f"Formato não suportado: {format}")
                
                logger.info(f"DataFrame salvo: {filepath} ({len(df)} linhas)")
                return True
                
            except Exception as e:
                logger.error(f"Erro ao salvar DataFrame: {e}")
                return False

        def load_yaml_config(filepath: Union[str, Path]) -> Dict[str, Any]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar YAML: {e}")
                return {}
    """),
    
    # fetchers
    f"{ROOT}/fetchers/__init__.py": "",
    
    f"{ROOT}/fetchers/noaa_fetcher.py": dedent("""\
        import logging
        import pandas as pd
        from datetime import datetime, timedelta
        from typing import Optional, Dict, Any

        from ..utils.retries import safe_request
        from ..utils.io_utils import safe_save_dataframe, load_yaml_config

        logger = logging.getLogger(__name__)

        class NOAAFetcher:
            def __init__(self):
                config = load_yaml_config('src_new/config/data_sources.yml')
                self.noaa_config = config.get('noaa', {})
                self.base_url = self.noaa_config.get('base_url', 'https://services.swpc.noaa.gov')
                self.endpoints = self.noaa_config.get('endpoints', {})

            def fetch_data(self, data_type: str = 'plasma_5min', days: int = 3) -> Optional[pd.DataFrame]:
                endpoint = self.endpoints.get(data_type)
                if not endpoint:
                    logger.error(f"Endpoint não encontrado: {data_type}")
                    return None

                url = f"{self.base_url}{endpoint}"
                logger.info(f"Buscando dados NOAA: {data_type}")

                try:
                    response = safe_request(url)
                    data = response.json()
                    
                    if not data or len(data) <= 1:
                        logger.warning("Dados NOAA vazios")
                        return None

                    columns = data[0]
                    rows = data[1:]
                    df = pd.DataFrame(rows, columns=columns)

                    if 'time_tag' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['time_tag'], utc=True)
                        df = df.drop('time_tag', axis=1)

                    cutoff = datetime.utcnow() - timedelta(days=days)
                    df = df[df['timestamp'] >= cutoff]

                    numeric_cols = [col for col in df.columns if col != 'timestamp']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"Dados NOAA {data_type} processados: {len(df)} linhas")
                    return df

                except Exception as e:
                    logger.error(f"Erro ao buscar dados NOAA: {e}")
                    return None

        def fetch_noaa_data(data_type: str = 'plasma_5min', days: int = 3) -> Optional[pd.DataFrame]:
            fetcher = NOAAFetcher()
            return fetcher.fetch_data(data_type, days)
    """),
    
    f"{ROOT}/fetchers/dscovr_fetcher.py": dedent("""\
        import logging
        import pandas as pd
        from datetime import datetime, timedelta
        from typing import Optional

        from ..utils.io_utils import load_yaml_config

        logger = logging.getLogger(__name__)

        class DSCOVRFetcher:
            def __init__(self):
                config = load_yaml_config('src_new/config/data_sources.yml')
                self.nasa_config = config.get('nasa_cdaweb', {})
                self.cdas_client = None

            def _init_client(self):
                try:
                    from cdasws import CdasWs
                    self.cdas_client = CdasWs()
                    return True
                except ImportError:
                    logger.warning("cdasws não instalado. Use: pip install cdasws")
                    return False

            def fetch_data(self, dataset_type: str = 'dscovr_swepam', days: int = 7) -> Optional[pd.DataFrame]:
                if not self.cdas_client and not self._init_client():
                    return None

                dataset_id = self.nasa_config.get('datasets', {}).get(dataset_type)
                if not dataset_id:
                    logger.error(f"Dataset não configurado: {dataset_type}")
                    return None

                logger.info(f"Buscando dados DSCOVR: {dataset_type}")

                try:
                    if 'swepam' in dataset_type:
                        variables = ['Np', 'Vp', 'Tp']
                    elif 'mag' in dataset_type:
                        variables = ['B1GSE', 'B2GSE', 'B3GSE']
                    else:
                        variables = ['Np', 'Vp', 'Tp']

                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(days=days)

                    status, data = self.cdas_client.get_data(
                        dataset_id,
                        variables,
                        start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                    )

                    if status != 200 or not data:
                        logger.warning("CDAWeb retornou sem dados")
                        return None

                    df = pd.DataFrame()
                    for key, values in data.items():
                        if key.lower() == 'epoch':
                            df['timestamp'] = pd.to_datetime(values, utc=True)
                        else:
                            mapping = {'Np': 'density', 'Vp': 'speed', 'Tp': 'temperature',
                                     'B1GSE': 'bx_gse', 'B2GSE': 'by_gse', 'B3GSE': 'bz_gse'}
                            col_name = mapping.get(key, key)
                            df[col_name] = values

                    for col in df.columns:
                        if col != 'timestamp':
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"Dados DSCOVR {dataset_type} processados: {len(df)} linhas")
                    return df

                except Exception as e:
                    logger.error(f"Erro ao buscar dados DSCOVR: {e}")
                    return None

        def fetch_dscovr_data(dataset_type: str = 'dscovr_swepam', days: int = 7) -> Optional[pd.DataFrame]:
            fetcher = DSCOVRFetcher()
            return fetcher.fetch_data(dataset_type, days)
    """),
    
    # processing
    f"{ROOT}/processing/__init__.py": "",
    
    f"{ROOT}/processing/merge.py": dedent("""\
        import pandas as pd
        import logging

        logger = logging.getLogger(__name__)

        def merge_datasets(datasets: list, tolerance: str = '10min') -> pd.DataFrame:
            if not datasets:
                return pd.DataFrame()

            main_df = datasets[0]
            for df in datasets[1:]:
                if df is None or df.empty:
                    continue
                    
                cols_to_merge = [col for col in df.columns if col not in main_df.columns and col != 'timestamp']
                if not cols_to_merge:
                    continue

                merge_df = df[['timestamp'] + cols_to_merge].sort_values('timestamp')
                main_df = pd.merge_asof(
                    main_df.sort_values('timestamp'),
                    merge_df,
                    on='timestamp',
                    tolerance=pd.Timedelta(tolerance),
                    direction='nearest'
                )
                logger.info(f"Mescladas {len(cols_to_merge)} colunas")

            return main_df.sort_values('timestamp').reset_index(drop=True)
    """),
    
    f"{ROOT}/processing/preprocess.py": dedent("""\
        import pandas as pd
        import logging

        logger = logging.getLogger(__name__)

        def resample_data(df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
            if df.empty:
                return df

            df = df.set_index('timestamp')
            df = df.resample(freq).mean()
            df = df.interpolate(limit=6)
            df = df.ffill(limit=6)
            df = df.reset_index()
            
            logger.info(f"Dados redimensionados para {freq}: {len(df)} linhas")
            return df
    """),
    
    # detection
    f"{ROOT}/detection/__init__.py": "",
    
    f"{ROOT}/detection/simple_detector.py": dedent("""\
        import logging
        from typing import List, Dict, Any
        import pandas as pd

        logger = logging.getLogger(__name__)

        def detect_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
            events = []
            if df is None or df.empty:
                return events

            # Detecção de Bz negativo forte
            if 'bz_gse' in df.columns:
                strong_bz = df[df['bz_gse'] <= -10]
                for _, row in strong_bz.iterrows():
                    events.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'type': 'strong_negative_bz',
                        'value': float(row['bz_gse']),
                        'severity': 'high'
                    })

            # Detecção de alta velocidade
            if 'speed' in df.columns:
                high_speed = df[df['speed'] >= 600]
                for _, row in high_speed.iterrows():
                    events.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'type': 'high_speed_stream',
                        'value': float(row['speed']),
                        'severity': 'medium'
                    })

            # Detecção de pico de densidade
            if 'density' in df.columns:
                density_spike = df[df['density'] >= 20]
                for _, row in density_spike.iterrows():
                    events.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'type': 'density_spike',
                        'value': float(row['density']),
                        'severity': 'low'
                    })

            logger.info(f"Detectados {len(events)} eventos")
            return events
    """),
    
    # main
    f"{ROOT}/main.py": dedent("""\
        #!/usr/bin/env python3
        """
        Pipeline principal do projeto Heliogeophysical
        """
        import logging
        import os
        from datetime import datetime
        from pathlib import Path

        from utils.io_utils import ensure_directory, safe_save_dataframe, load_yaml_config
        from utils.logger import setup_logging
        from fetchers.noaa_fetcher import fetch_noaa_data
        from fetchers.dscovr_fetcher import fetch_dscovr_data
        from processing.merge import merge_datasets
        from processing.preprocess import resample_data
        from detection.simple_detector import detect_events

        def main():
            # Configuração inicial
            config = load_yaml_config('src_new/config/config.yaml')
            
            # Setup logging
            log_config = config.get('logging', {})
            setup_logging(
                log_file=log_config.get('file', 'logs/heliogeophysical.log'),
                level=log_config.get('level', 'INFO')
            )
            
            logger = logging.getLogger(__name__)
            logger.info("Iniciando pipeline Heliogeophysical")

            try:
                # Criar diretórios necessários
                storage_config = config.get('storage', {})
                ensure_directory(storage_config.get('raw_data', 'data/raw'))
                ensure_directory(storage_config.get('processed_data', 'data/processed'))
                ensure_directory('logs')

                # Coletar dados
                logger.info("Coletando dados...")
                noaa_plasma = fetch_noaa_data('plasma_5min', days=3)
                noaa_mag = fetch_noaa_data('mag_5min', days=3)
                dscovr_swepam = fetch_dscovr_data('dscovr_swepam', days=3)

                # Filtrar datasets válidos
                datasets = [df for df in [noaa_plasma, noaa_mag, dscovr_swepam] if df is not None and not df.empty]
                
                if not datasets:
                    logger.error("Nenhum dado válido coletado")
                    return

                # Processar dados
                logger.info("Processando dados...")
                merged_data = merge_datasets(datasets)
                
                processing_config = config.get('processing', {})
                processed_data = resample_data(merged_data, freq=processing_config.get('resample_freq', '5T'))

                # Detectar eventos
                logger.info("Detectando eventos...")
                events = detect_events(processed_data)

                # Salvar resultados
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                safe_save_dataframe(processed_data, f"data/processed/helio_data_{timestamp}.csv")
                
                # Salvar eventos
                if events:
                    import json
                    with open(f"data/processed/events_{timestamp}.json", 'w') as f:
                        json.dump(events, f, indent=2)

                logger.info(f"Pipeline concluído! Processados {len(processed_data)} registros, detectados {len(events)} eventos")

            except Exception as e:
                logger.error(f"Erro no pipeline: {e}", exc_info=True)
                raise

        if __name__ == "__main__":
            main()
    """),
    
    f"{ROOT}/utils/logger.py": dedent("""\
        import logging
        import sys
        from pathlib import Path

        def setup_logging(log_file: str = None, level: str = "INFO"):
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            logger = logging.getLogger('heliogeophysical')
            logger.setLevel(getattr(logging, level.upper()))
            
            # Remove handlers existentes
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(console_handler)
            
            # File handler
  
