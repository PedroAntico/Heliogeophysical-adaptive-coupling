"""
src/data_fetcher_observational.py
Coletor STRICT de dados observacionais - SEM FALLBACK SINTÉTICO
Falha explicitamente se não conseguir dados reais
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

class ObservationalDataCollector:
    """Coletor STRICT de dados observacionais - ZERO SINTÉTICO"""
    
    def __init__(self):
        self.data_source = None
        self.quality_metrics = {}
        self.min_data_points = 144  # Mínimo 24h de dados (144 pontos de 10min)
        
    def fetch_omni_observational(self, days=30):
        """
        Coleta dados OMNI observacionais via NASA CDAWeb
        RETORNA APENAS DADOS OBSERVACIONAIS ou FALHA
        """
        try:
            from cdasws import CdasWs
            
            logger.info(f"🌐 Conectando à NASA CDAWeb para {days}d de dados OBSERVACIONAIS OMNI...")
            
            cdas = CdasWs()
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Variáveis OMNI essenciais - apenas observacionais
            variables = [
                'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BT',
                'V', 'N', 'T', 'TEMP', 'FLOW_SPEED', 'PROTON_DENSITY'
            ]
            
            logger.info(f"📡 Buscando dados de {start_time} a {end_time}")
            
            status, data = cdas.get_data(
                'OMNI_HRO2_1MIN',
                variables,
                start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            if status == 200 and data:
                df = self._process_observational_data(data)
                
                # VERIFICAÇÃO CRÍTICA: dados suficientes e reais
                if len(df) >= self.min_data_points:
                    self.data_source = 'NASA_OMNI_OBSERVATIONAL'
                    logger.info(f"✅ Dados OBSERVACIONAIS coletados: {len(df)} registros")
                    return df
                else:
                    logger.error(f"❌ Dados insuficientes: {len(df)} < {self.min_data_points}")
                    return None
            
            logger.error("❌ CDAWeb retornou status diferente de 200 ou dados vazios")
            return None
            
        except Exception as e:
            logger.error(f"❌ Falha CDAWeb: {str(e)}")
            return None
    
    def _process_observational_data(self, data):
        """Processa dados OBSERVACIONAIS brutos - SEM GERAÇÃO SINTÉTICA"""
        df = pd.DataFrame()
        
        # Mapeamento de variáveis OBSERVACIONAIS
        var_mapping = {
            'FLOW_SPEED': 'speed',
            'PROTON_DENSITY': 'density', 
            'TEMP': 'temperature',
            'BX_GSE': 'bx_gse',
            'BY_GSE': 'by_gse',
            'BZ_GSE': 'bz_gse', 
            'BT': 'bt'
        }
        
        # Extrair APENAS dados observacionais disponíveis
        for cda_var, our_var in var_mapping.items():
            if cda_var in data and data[cda_var] is not None:
                values = data[cda_var]
                # Verificar se são dados reais (não sintéticos)
                if len(values) > 0 and not all(np.isnan(values)):
                    df[our_var] = values
        
        # Timestamp OBSERVACIONAL
        if 'Epoch' in data:
            df['time_tag'] = pd.to_datetime(data['Epoch'])
        else:
            logger.error("❌ Sem timestamps observacionais - dados inválidos")
            return pd.DataFrame()
        
        # Limpeza CRÍTICA de dados
        df = df.dropna(subset=['time_tag']).sort_values('time_tag')
        
        # Remover outliers baseado em física observacional
        if 'speed' in df.columns:
            df = df[(df['speed'] > 200) & (df['speed'] < 1000)]
        if 'density' in df.columns:
            df = df[(df['density'] > 0.1) & (df['density'] < 100)]
        if 'temperature' in df.columns:
            df = df[(df['temperature'] > 10000) & (df['temperature'] < 500000)]
        
        logger.info(f"📊 Dados observacionais processados: {len(df)} registros limpos")
        return df.reset_index(drop=True)
    
    def fetch_noaa_observational(self, days=30):
        """
        Fonte alternativa: dados NOAA observacionais
        """
        try:
            logger.info("🔄 Tentando fonte alternativa: NOAA dados observacionais...")
            
            # URLs de dados OBSERVACIONAIS da NOAA
            urls = {
                'plasma': 'https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json',
                'mag': 'https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json'
            }
            
            dfs = []
            for data_type, url in urls.items():
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        json_data = response.json()
                        if len(json_data) > 1:  # Tem dados além do header
                            df_temp = pd.DataFrame(json_data[1:], columns=json_data[0])
                            if len(df_temp) > 0:
                                dfs.append(df_temp)
                                logger.info(f"📡 NOAA {data_type}: {len(df_temp)} registros")
                except Exception as e:
                    logger.warning(f"⚠️  Falha NOAA {data_type}: {e}")
            
            if len(dfs) >= 2:
                df = self._merge_noaa_observational(dfs[0], dfs[1])
                if len(df) >= self.min_data_points:
                    self.data_source = 'NOAA_OBSERVATIONAL'
                    logger.info(f"✅ Dados NOAA observacionais: {len(df)} registros")
                    return df
            
            logger.error("❌ NOAA não retornou dados observacionais suficientes")
            return None
            
        except Exception as e:
            logger.error(f"❌ Fallback NOAA falhou: {e}")
            return None
    
    def _merge_noaa_observational(self, plasma_df, mag_df):
        """Merge dados OBSERVACIONAIS plasma e magnéticos NOAA"""
        # Verificar dados não vazios
        if len(plasma_df) == 0 or len(mag_df) == 0:
            return pd.DataFrame()
        
        # Renomear colunas para dados OBSERVACIONAIS
        plasma_df = plasma_df.rename(columns={
            plasma_df.columns[0]: 'time_tag',
            plasma_df.columns[1]: 'density',
            plasma_df.columns[2]: 'speed', 
            plasma_df.columns[3]: 'temperature'
        })
        
        mag_df = mag_df.rename(columns={
            mag_df.columns[0]: 'time_tag',
            mag_df.columns[1]: 'bx_gse',
            mag_df.columns[2]: 'by_gse',
            mag_df.columns[3]: 'bz_gse', 
            mag_df.columns[4]: 'bt'
        })
        
        # Converter timestamps OBSERVACIONAIS
        try:
            plasma_df['time_tag'] = pd.to_datetime(plasma_df['time_tag'])
            mag_df['time_tag'] = pd.to_datetime(mag_df['time_tag'])
        except Exception as e:
            logger.error(f"❌ Erro na conversão de timestamps: {e}")
            return pd.DataFrame()
        
        # Merge de dados OBSERVACIONAIS
        df = pd.merge_asof(
            plasma_df.sort_values('time_tag'),
            mag_df.sort_values('time_tag'), 
            on='time_tag',
            tolerance=pd.Timedelta('10min'),
            direction='nearest'
        )
        
        # Filtrar período observacional recente
        cutoff = datetime.utcnow() - timedelta(days=30)
        df = df[df['time_tag'] > cutoff]
        
        return df
    
    def validate_observational_data(self, df):
        """
        Validação CRÍTICA: verifica se dados são realmente observacionais
        """
        if df.empty:
            return {'is_observational': False, 'issues': ['Dataset vazio']}
        
        issues = []
        
        # 1. Verificar tamanho mínimo
        if len(df) < self.min_data_points:
            issues.append(f"Dados insuficientes: {len(df)} < {self.min_data_points}")
        
        # 2. Verificar variabilidade real (não sintética)
        if 'speed' in df.columns:
            speed_std = df['speed'].std()
            if speed_std < 10:  # Dados sintéticos tendem a ter baixa variabilidade
                issues.append(f"Baixa variabilidade observacional em speed: {speed_std:.2f}")
        
        # 3. Verificar presença de valores extremos (comuns em dados reais)
        if 'speed' in df.columns:
            extreme_events = len(df[df['speed'] > 600])
            if extreme_events == 0:
                issues.append("Ausência de eventos de alta velocidade (suspeito)")
        
        # 4. Verificar padrões temporais realistas
        if 'time_tag' in df.columns:
            time_diffs = df['time_tag'].diff().dropna()
            if len(time_diffs) > 0:
                avg_interval = time_diffs.mean().total_seconds()
                if abs(avg_interval - 300) > 60:  # Esperado ~5min
                    issues.append(f"Intervalo temporal anômalo: {avg_interval:.1f}s")
        
        is_observational = len(issues) == 0
        
        validation_result = {
            'is_observational': is_observational,
            'issues': issues,
            'data_points': len(df),
            'time_span_hours': (df['time_tag'].max() - df['time_tag'].min()).total_seconds() / 3600,
            'variables_present': [col for col in df.columns if col != 'time_tag']
        }
        
        return validation_result
    
    def collect_observational_data(self):
        """
        Coleta STRICT de dados observacionais
        FALHA se não conseguir dados reais
        """
        logger.info("🚀 INICIANDO COLETA STRICT DE DADOS OBSERVACIONAIS")
        logger.info("📡 POLÍTICA: SEM FALLBACK SINTÉTICO - APENAS DADOS REAIS")
        
        # Tentar NASA OMNI primeiro
        df = self.fetch_omni_observational(days=30)
        
        # Tentar NOAA como alternativa
        if df is None or len(df) < self.min_data_points:
            logger.info("🔄 NASA OMNI falhou, tentando NOAA...")
            df = self.fetch_noaa_observational(days=30)
        
        # VALIDAÇÃO CRÍTICA
        if df is None:
            logger.error("❌ TODAS AS FONTES OBSERVACIONAIS FALHARAM")
            raise RuntimeError("COLETA OBSERVACIONAL FALHOU: Nenhuma fonte retornou dados reais")
        
        validation = self.validate_observational_data(df)
        
        if not validation['is_observational']:
            logger.error("❌ VALIDAÇÃO FALHOU - Dados não são observacionais")
            logger.error(f"📋 Issues: {validation['issues']}")
            raise RuntimeError(f"DADOS NÃO OBSERVACIONAIS: {validation['issues']}")
        
        logger.info("✅ COLETA OBSERVACIONAL VALIDADA COM SUCESSO")
        logger.info(f"📊 Fonte: {self.data_source}")
        logger.info(f"📦 Registros: {len(df)}")
        logger.info(f"⏱️  Período: {validation['time_span_hours']:.1f} horas")
        
        return df, self.data_source, validation

def main():
    """Função principal - FALHA se não conseguir dados observacionais"""
    collector = ObservationalDataCollector()
    
    try:
        # Coleta STRICT - falha se não for observacional
        df, source, validation = collector.collect_observational_data()
        
        # Salvar dados OBSERVACIONAIS
        os.makedirs('data_observational', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        output_path = f"data_observational/solar_observational_{timestamp}.csv"
        
        df.to_csv(output_path, index=False)
        
        # Salvar metadados de validação
        metadata = {
            'collection_timestamp': datetime.utcnow().isoformat(),
            'data_source': source,
            'validation_result': validation,
            'records': len(df),
            'time_range': {
                'start': df['time_tag'].min().isoformat(),
                'end': df['time_tag'].max().isoformat()
            },
            'data_quality': {
                'completeness': df.notna().mean().to_dict(),
                'basic_stats': {col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                } for col in df.columns if col != 'time_tag'}
            }
        }
        
        import json
        with open(f'data_observational/observational_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Dados OBSERVACIONAIS salvos em: {output_path}")
        
        print("\n" + "="*70)
        print("✅ COLETA OBSERVACIONAL 100% REAL - CONCLUÍDA")
        print("="*70)
        print(f"📡 Fonte: {source}")
        print(f"📊 Registros: {len(df)}")
        print(f"⏱️  Período: {validation['time_span_hours']:.1f} horas")
        print(f"💾 Arquivo: {output_path}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ FALHA CRÍTICA: {e}")
        print(f"\n❌ COLETA OBSERVACIONAL FALHOU: {e}")
        print("💡 Verifique:")
        print("   - Conexão com internet")
        print("   - Serviços NASA CDAWeb/NOAA")
        print("   - Biblioteca cdasws instalada")
        sys.exit(1)

if __name__ == '__main__':
    main()
