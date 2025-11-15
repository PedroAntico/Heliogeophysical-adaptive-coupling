"""
src/data_fetcher_observational_fixed.py
Coletor CORRIGIDO com fontes reais verificadas
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_fetcher_observational.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('data_fetcher_observational')

class ObservationalDataCollectorFixed:
    """Coletor CORRIGIDO com fontes reais verificadas"""
    
    def __init__(self):
        self.data_source = None
        self.min_data_points = 144
        
    def fetch_noaa_realtime_verified(self):
        """
        Coleta dados NOAA em tempo real - FONTES VERIFICADAS
        """
        try:
            logger.info("🌐 Conectando às fontes NOAA verificadas...")
            
            # URLs VERIFICADAS da NOAA
            verified_urls = {
                'plasma': 'https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json',
                'mag': 'https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json',
                'ace_5min': 'https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json',
                'ace_mag_5min': 'https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json'
            }
            
            dfs = []
            successful_sources = []
            
            # Tentar múltiplas fontes NOAA
            for source_name, url in verified_urls.items():
                try:
                    logger.info(f"📡 Tentando {source_name}: {url}")
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 1:  # Tem dados além do header
                            df_temp = pd.DataFrame(data[1:], columns=data[0])
                            if len(df_temp) > 10:  # Dados suficientes
                                dfs.append((source_name, df_temp))
                                successful_sources.append(source_name)
                                logger.info(f"✅ {source_name}: {len(df_temp)} registros")
                            else:
                                logger.warning(f"⚠️ {source_name}: dados insuficientes ({len(df_temp)})")
                        else:
                            logger.warning(f"⚠️ {source_name}: estrutura de dados inválida")
                    else:
                        logger.warning(f"⚠️ {source_name}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Falha em {source_name}: {str(e)}")
            
            # Processar dados bem-sucedidos
            if len(dfs) >= 1:
                # Usar a fonte com mais dados
                best_source = max(dfs, key=lambda x: len(x[1]))
                source_name, best_df = best_source
                
                df_processed = self._process_noaa_data(best_df, source_name)
                
                if len(df_processed) >= self.min_data_points:
                    self.data_source = f'NOAA_{source_name.upper()}'
                    logger.info(f"✅ Dados NOAA coletados: {len(df_processed)} registros de {source_name}")
                    return df_processed
            
            logger.error("❌ Todas as fontes NOAA falharam")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro geral na coleta NOAA: {str(e)}")
            return None
    
    def _process_noaa_data(self, df, source_name):
        """Processa dados NOAA para formato padrão"""
        try:
            # Mapeamento de colunas baseado no formato NOAA
            if 'plasma' in source_name:
                if len(df.columns) >= 4:
                    df = df.rename(columns={
                        df.columns[0]: 'time_tag',
                        df.columns[1]: 'density',
                        df.columns[2]: 'speed', 
                        df.columns[3]: 'temperature'
                    })
            elif 'mag' in source_name:
                if len(df.columns) >= 5:
                    df = df.rename(columns={
                        df.columns[0]: 'time_tag',
                        df.columns[1]: 'bx_gse',
                        df.columns[2]: 'by_gse',
                        df.columns[3]: 'bz_gse',
                        df.columns[4]: 'bt'
                    })
            
            # Converter timestamp
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            
            # Ordenar por tempo
            df = df.sort_values('time_tag').reset_index(drop=True)
            
            # Limpeza básica
            df = df.dropna(subset=['time_tag'])
            
            # Filtrar dados recentes (últimos 30 dias)
            cutoff = datetime.utcnow() - timedelta(days=30)
            df = df[df['time_tag'] > cutoff]
            
            # Converter tipos numéricos
            numeric_cols = ['speed', 'density', 'temperature', 'bx_gse', 'by_gse', 'bz_gse', 'bt']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remover outliers baseados em física
            if 'speed' in df.columns:
                df = df[(df['speed'] > 200) & (df['speed'] < 1000)]
            if 'density' in df.columns:
                df = df[(df['density'] > 0.1) & (df['density'] < 100)]
            
            logger.info(f"📊 Dados processados: {len(df)} registros limpos")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento NOAA: {str(e)}")
            return pd.DataFrame()
    
    def fetch_omni_with_fallback(self):
        """
        Tenta OMNI com fallback robusto
        """
        try:
            logger.info("🔄 Tentando dados OMNI com fallback...")
            
            # Tentar importar cdasws
            try:
                from cdasws import CdasWs
                cdas = CdasWs()
                
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=30)
                
                status, data = cdas.get_data(
                    'OMNI_HRO2_1MIN',
                    ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'BT', 'V', 'N', 'T'],
                    start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                )
                
                if status == 200 and data:
                    df = self._process_omni_data(data)
                    if len(df) >= self.min_data_points:
                        self.data_source = 'NASA_OMNI'
                        return df
                        
            except ImportError:
                logger.warning("❌ cdasws não disponível")
            except Exception as e:
                logger.warning(f"⚠️ OMNI falhou: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro no OMNI: {str(e)}")
            return None
    
    def _process_omni_data(self, data):
        """Processa dados OMNI"""
        df = pd.DataFrame()
        
        # Mapeamento de variáveis OMNI
        mapping = {
            'V': 'speed', 'N': 'density', 'T': 'temperature',
            'BX_GSE': 'bx_gse', 'BY_GSE': 'by_gse', 'BZ_GSE': 'bz_gse', 'BT': 'bt'
        }
        
        for omni_var, our_var in mapping.items():
            if omni_var in data and data[omni_var] is not None:
                df[our_var] = data[omni_var]
        
        if 'Epoch' in data:
            df['time_tag'] = pd.to_datetime(data['Epoch'])
        
        df = df.dropna(subset=['time_tag']).sort_values('time_tag')
        return df
    
    def collect_data_robust(self):
        """
        Coleta robusta com múltiplas fontes
        """
        logger.info("🚀 INICIANDO COLETA ROBUSTA DE DADOS OBSERVACIONAIS")
        
        # 1. Tentar NOAA primeiro (mais confiável)
        df = self.fetch_noaa_realtime_verified()
        
        # 2. Tentar OMNI como alternativa
        if df is None or len(df) < self.min_data_points:
            logger.info("🔄 NOAA insuficiente, tentando OMNI...")
            df = self.fetch_omni_with_fallback()
        
        # 3. Verificar se temos dados suficientes
        if df is None or len(df) < self.min_data_points:
            logger.error("❌ COLETA FALHOU - Dados insuficientes de todas as fontes")
            logger.info("💡 Alternativas:")
            logger.info("   - Verificar conexão com internet")
            logger.info("   - Serviços NOAA podem estar temporariamente indisponíveis")
            logger.info("   - Tentar novamente mais tarde")
            return None, None, None
        
        # Análise dos dados coletados
        stats = self.analyze_data(df)
        
        logger.info("✅ COLETA CONCLUÍDA")
        logger.info(f"📊 Fonte: {self.data_source}")
        logger.info(f"📦 Registros: {len(df)}")
        logger.info(f"⏱️  Período: {stats['time_span_hours']:.1f} horas")
        
        return df, self.data_source, stats
    
    def analyze_data(self, df):
        """Analisa os dados coletados"""
        stats = {
            'data_points': len(df),
            'time_span_hours': (df['time_tag'].max() - df['time_tag'].min()).total_seconds() / 3600,
            'variables': [col for col in df.columns if col != 'time_tag'],
            'completeness': {}
        }
        
        for col in df.columns:
            if col != 'time_tag':
                stats['completeness'][col] = f"{df[col].notna().mean():.1%}"
        
        return stats

def main():
    """Função principal"""
    collector = ObservationalDataCollectorFixed()
    
    try:
        # Coleta robusta
        df, source, stats = collector.collect_data_robust()
        
        if df is None:
            logger.error("❌ Não foi possível coletar dados observacionais")
            print("\n❌ FALHA NA COLETA DE DADOS")
            print("💡 Possíveis causas:")
            print("   - Serviços NOAA indisponíveis")
            print("   - Problema de conexão com internet")
            print("   - Firewall bloqueando acesso")
            sys.exit(1)
        
        # Salvar dados
        os.makedirs('data_observational', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        output_path = f"data_observational/solar_observational_{timestamp}.csv"
        
        df.to_csv(output_path, index=False)
        
        # Salvar metadados
        metadata = {
            'collection_timestamp': datetime.utcnow().isoformat(),
            'data_source': source,
            'statistics': stats,
            'records': len(df),
            'time_range': {
                'start': df['time_tag'].min().isoformat(),
                'end': df['time_tag'].max().isoformat()
            }
        }
        
        import json
        with open(f'data_observational/observational_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ COLETA OBSERVACIONAL - CONCLUÍDA")
        print("="*70)
        print(f"📡 Fonte: {source}")
        print(f"📊 Registros: {len(df)}")
        print(f"⏱️  Período: {stats['time_span_hours']:.1f} horas")
        print(f"📈 Variáveis: {', '.join(stats['variables'])}")
        print(f"💾 Arquivo: {output_path}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO: {e}")
        print(f"\n❌ ERRO: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
