import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.retries import retry_with_exponential_backoff

logger = logging.getLogger(__name__)

class NASACDAWebFetcher:
    """
    Fetcher avançado para dados da NASA CDAWeb
    Acessa dados de múltiplos satélites e experimentos
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://cdaweb.gsfc.nasa.gov')
        self.datasets = config.get('datasets', {})
        self.cdas_client = None
    
    def _initialize_cdas_client(self) -> bool:
        """Inicializa cliente CDAWeb"""
        try:
            from cdasws import CdasWs
            self.cdas_client = CdasWs()
            return True
        except ImportError:
            logger.warning("cdasws não instalado. Use: pip install cdasws")
            return False
        except Exception as e:
            logger.error(f"Erro ao inicializar CDAWeb: {e}")
            return False
    
    @retry_with_exponential_backoff(max_retries=2, base_delay=10.0)
    def fetch_dataset(self, dataset_name: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Busca dados de um dataset específico do CDAWeb"""
        if not self.cdas_client and not self._initialize_cdas_client():
            return None
        
        dataset_id = self.datasets.get(dataset_name)
        if not dataset_id:
            logger.error(f"Dataset não configurado: {dataset_name}")
            return None
        
        logger.info(f"Buscando dados CDAWeb: {dataset_name} ({dataset_id})")
        
        try:
            # Define variáveis baseadas no dataset
            variable_mapping = {
                'dscovr_swepam': ['Np', 'Vp', 'Tp'],
                'dscovr_mag': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'omni_1min': ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'V', 'N', 'T', 'KP'],
                'ace_mag': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'ace_swepam': ['Np', 'Vp', 'Tp'],
                'wind_mfi': ['B1GSE', 'B2GSE', 'B3GSE', 'Bt'],
                'wind_swe': ['Np', 'Vp', 'Tp']
            }
            
            variables = variable_mapping.get(dataset_name, ['Np', 'Vp', 'Tp'])
            
            # Período de busca
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Busca dados
            status, data = self.cdas_client.get_data(
                dataset_id,
                variables,
                start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            if status != 200 or not data:
                logger.warning(f"CDAWeb retornou status {status} sem dados")
                return None
            
            # Processa dados
            df = self._process_cdas_data(data, dataset_name)
            logger.info(f"Dados {dataset_name} processados: {len(df)} linhas")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao buscar {dataset_name}: {e}")
            return None
    
    def _process_cdas_data(self, data: dict, dataset_name: str) -> pd.DataFrame:
        """Processa dados brutos do CDAWeb"""
        df = pd.DataFrame()
        
        # Mapeamento de variáveis para nomes padronizados
        standard_mapping = {
            'Np': 'density', 'Vp': 'speed', 'Tp': 'temperature',
            'B1GSE': 'bx_gse', 'B2GSE': 'by_gse', 'B3GSE': 'bz_gse', 'Bt': 'bt',
            'BX_GSE': 'bx_gse', 'BY_GSE': 'by_gse', 'BZ_GSE': 'bz_gse',
            'V': 'speed', 'N': 'density', 'T': 'temperature', 'KP': 'kp_index'
        }
        
        for key, values in data.items():
            if key.lower() == 'epoch':
                df['timestamp'] = pd.to_datetime(values, utc=True)
            else:
                standard_name = standard_mapping.get(key, key)
                df[standard_name] = values
        
        # Converte para numérico
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicatas de timestamp
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        return df.reset_index(drop=True)
    
    def fetch_multiple_datasets(self, dataset_names: list, days: int = 7) -> Dict[str, pd.DataFrame]:
        """Busca múltiplos datasets e retorna dicionário com DataFrames"""
        results = {}
        
        for dataset_name in dataset_names:
            df = self.fetch_dataset(dataset_name, days)
            if df is not None and not df.empty:
                results[dataset_name] = df
        
        logger.info(f"CDAWeb: {len(results)} datasets coletados com sucesso")
        return results

def create_nasa_fetcher(config: Dict[str, Any]) -> NASACDAWebFetcher:
    """Factory function para NASA CDAWeb Fetcher"""
    return NASACDAWebFetcher(config)
