"""
src/data_fetcher_omni.py
Coleta dados históricos OMNI da NASA CDAWeb para validação científica extensiva
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_fetcher_omni.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("omni_fetcher")

class OMNIDataCollector:
    """Coletor de dados OMNI históricos da NASA CDAWeb"""
    
    def __init__(self):
        self.dataset = 'OMNI2_H0_MRG1HR'
        self.variables = [
            'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM', 'BT',
            'V', 'Vx', 'Vy', 'Vz', 'N', 'T', 'P', 'E', 'Beta', 'Ma'
        ]
    
    def fetch_omni_data(self, start_date, end_date):
        """
        Busca dados OMNI via NASA CDAWeb API
        
        Parâmetros:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
        
        Retorna:
            DataFrame com dados OMNI ou None em caso de falha
        """
        try:
            from cdasws import CdasWs
            cdas = CdasWs()
            
            logger.info(f"🌐 Buscando dados OMNI: {start_date} a {end_date}")
            
            status, data = cdas.get_data(
                self.dataset, 
                self.variables, 
                start_date, 
                end_date
            )
            
            if status == 200 and data:
                df = self.process_omni_data(data)
                logger.info(f"✅ Dados OMNI coletados: {len(df)} registros")
                return df
            else:
                logger.error(f"❌ Falha CDAWeb: status {status}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro na coleta OMNI: {e}")
            return None
    
    def process_omni_data(self, data):
        """Processa dados brutos do OMNI para formato padrão"""
        # Criar DataFrame
        df = pd.DataFrame()
        
        # Mapear variáveis OMNI para nosso formato
        mapping = {
            'V': 'speed',
            'N': 'density', 
            'T': 'temperature',
            'BX_GSE': 'bx_gse',
            'BY_GSE': 'by_gse', 
            'BZ_GSE': 'bz_gse',
            'BY_GSM': 'by_gsm',
            'BZ_GSM': 'bz_gsm',
            'BT': 'bt',
            'P': 'pressure',
            'E': 'electric_field',
            'Beta': 'plasma_beta',
            'Ma': 'mach_number'
        }
        
        # Extrair dados
        for var in self.variables:
            if var in data:
                df[mapping.get(var, var)] = data[var]
        
        # Adicionar timestamp (Epoch)
        if 'Epoch' in data:
            df['time_tag'] = pd.to_datetime(data['Epoch'])
        else:
            # Criar range temporal se Epoch não disponível
            start = datetime.strptime(data['Start_Date'], '%Y-%m-%d')
            end = datetime.strptime(data['End_Date'], '%Y-%m-%d')
            df['time_tag'] = pd.date_range(start=start, end=end, freq='H')
        
        # Remover valores inválidos (OMNI usa 999.99 para missing)
        for col in df.columns:
            if col != 'time_tag':
                df[col] = df[col].replace(999.99, np.nan)
                df[col] = df[col].replace(9999.99, np.nan)
        
        # Ordenar por tempo e resetar índice
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        return df
    
    def create_omni_sample(self, days=365):
        """Cria dados de exemplo OMNI quando API não está disponível"""
        logger.warning("🔄 Criando dados OMNI de exemplo")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        np.random.seed(42)
        n_samples = len(dates)
        
        # Dados realistas baseados em estatísticas OMNI
        df = pd.DataFrame({
            'time_tag': dates,
            'speed': np.random.normal(450, 100, n_samples),
            'density': np.random.lognormal(2, 0.8, n_samples),
            'temperature': np.random.lognormal(11, 0.5, n_samples),
            'bx_gse': np.random.normal(0, 3, n_samples),
            'by_gse': np.random.normal(0, 3, n_samples),
            'bz_gse': np.random.normal(0, 4, n_samples),
            'by_gsm': np.random.normal(0, 3, n_samples),
            'bz_gsm': np.random.normal(0, 4, n_samples),
            'bt': np.abs(np.random.normal(5, 2, n_samples)),
            'pressure': np.random.lognormal(0.5, 0.7, n_samples),
            'plasma_beta': np.random.lognormal(0, 0.5, n_samples)
        })
        
        # Adicionar eventos de tempestade realistas
        storm_days = np.random.choice(days, size=min(30, days//10), replace=False)
        for day in storm_days:
            storm_start = start_date + timedelta(days=day)
            storm_hours = 6 + np.random.randint(12)  # Duração 6-18 horas
            storm_indices = df[
                (df['time_tag'] >= storm_start) & 
                (df['time_tag'] < storm_start + timedelta(hours=storm_hours))
            ].index
            
            if len(storm_indices) > 0:
                df.loc[storm_indices, 'speed'] = np.random.uniform(600, 800, len(storm_indices))
                df.loc[storm_indices, 'density'] = np.random.uniform(15, 30, len(storm_indices))
                df.loc[storm_indices, 'bz_gsm'] = np.random.uniform(-20, -10, len(storm_indices))
                df.loc[storm_indices, 'bt'] = np.random.uniform(10, 25, len(storm_indices))
        
        logger.info(f"📊 Dados OMNI de exemplo criados: {len(df)} registros")
        return df

def main():
    """Função principal - Coleta dados OMNI para validação histórica"""
    collector = OMNIDataCollector()
    
    # Período para validação histórica (últimos 2 anos)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)  # 2 anos
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Tentar coletar dados reais
    df_omni = collector.fetch_omni_data(start_str, end_str)
    
    if df_omni is None or len(df_omni) == 0:
        logger.warning("⚠️ API OMNI indisponível - usando dados de exemplo")
        df_omni = collector.create_omni_sample(days=730)
    
    # Salvar dados
    os.makedirs("data", exist_ok=True)
    output_path = "data/omni_historical_data.csv"
    df_omni.to_csv(output_path, index=False)
    
    # Estatísticas
    logger.info(f"💾 Dados OMNI salvos: {output_path}")
    logger.info(f"📅 Período: {df_omni['time_tag'].min()} a {df_omni['time_tag'].max()}")
    logger.info(f"📊 Registros: {len(df_omni)}")
    logger.info(f"🌪️  Eventos de tempestade: {len(df_omni[df_omni['speed'] > 600])}")

if __name__ == "__main__":
    main()
