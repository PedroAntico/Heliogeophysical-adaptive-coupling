"""
src/data_fetcher_REAL.py
Coletor DEFINITIVO com variáveis REAIS que existem nos datasets
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

# Configurações com variáveis REAIS
from config.data_sources_REAL import (
    NASA_REAL_SOURCES, NOAA_REAL_SOURCES,
    VARIABLE_MAPPING_REAL, VALIDATION_CONFIG
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_fetcher_real.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('data_fetcher_real')

class RealDataCollector:
    """Coletor com variáveis REAIS que existem nos datasets"""
    
    def __init__(self):
        self.data_source = None
        self.collection_stats = {}
        
    def fetch_nasa_real(self, dataset='DSCOVR_SWEPAM_L1', days=30):
        """
        Coleta NASA com variáveis REAIS que existem
        """
        try:
            if dataset not in NASA_REAL_SOURCES:
                logger.error(f"❌ Dataset NASA não encontrado: {dataset}")
                return None
            
            config = NASA_REAL_SOURCES[dataset]
            logger.info(f"🌐 Conectando à NASA: {dataset}")
            logger.info(f"📡 Variáveis REAIS: {config.variables}")
            
            from cdasws import CdasWs
            cdas = CdasWs()
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # ✅ USAR VARIÁVEIS REAIS QUE EXISTEM
            status, data = cdas.get_data(
                dataset,
                config.variables,  # ✅ VARIÁVEIS QUE EXISTEM
                start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            if status == 200 and data:
                df = self._process_nasa_data_real(data, config)
                if len(df) >= VALIDATION_CONFIG['min_data_points']:
                    self.data_source = f'NASA_{dataset}'
                    logger.info(f"✅ NASA {dataset}: {len(df)} registros REAIS")
                    return df
                else:
                    logger.error(f"❌ Dados NASA insuficientes: {len(df)} registros")
            else:
                logger.error(f"❌ NASA retornou status {status}")
                if status != 200:
                    logger.error("💡 Provável erro: variáveis não existem no dataset")
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro NASA: {str(e)}")
            return None
    
    def _process_nasa_data_real(self, data, config):
        """Processa dados NASA com variáveis REAIS"""
        df = pd.DataFrame()
        
        # ✅ MAPEAMENTO CORRETO PARA VARIÁVEIS REAIS
        if config.name.startswith('DSCOVR_SWEPAM'):
            mapping = VARIABLE_MAPPING_REAL['DSCOVR_SWEPAM']
        elif config.name.startswith('DSCOVR_MAG'):
            mapping = VARIABLE_MAPPING_REAL['DSCOVR_MAG']
        else:  # OMNI
            mapping = VARIABLE_MAPPING_REAL['OMNI_HRO2']
        
        # Extrair variáveis REAIS disponíveis
        for real_var, our_var in mapping.items():
            if real_var in data and data[real_var] is not None:
                values = data[real_var]
                if len(values) > 0 and not all(pd.isna(values)):
                    df[our_var] = values
                    logger.debug(f"✅ Variável real extraída: {real_var} -> {our_var}")
        
        # Timestamp
        if 'Epoch' in data:
            df['time_tag'] = pd.to_datetime(data['Epoch'])
        else:
            logger.error("❌ Sem timestamps NASA")
            return pd.DataFrame()
        
        # Calcular BT se necessário (para datasets que não fornecem)
        if all(col in df.columns for col in ['bx_gse', 'by_gse', 'bz_gse']) and 'bt' not in df.columns:
            df['bt'] = np.sqrt(df['bx_gse']**2 + df['by_gse']**2 + df['bz_gse']**2)
            logger.info("✅ BT calculado a partir dos componentes")
        
        # Limpeza
        df = df.dropna(subset=['time_tag']).sort_values('time_tag')
        
        # Aplicar limites físicos
        if 'speed' in df.columns:
            df = df[(df['speed'] > 200) & (df['speed'] < 1000)]
        if 'density' in df.columns:
            df = df[(df['density'] > 0.1) & (df['density'] < 100)]
        
        logger.info(f"📊 NASA {config.name}: {len(df)} registros REAIS processados")
        return df.reset_index(drop=True)
    
    def fetch_noaa_real(self, data_type='PLASMA_5MIN'):
        """
        Coleta NOAA com parsing CORRETO
        """
        try:
            if data_type not in NOAA_REAL_SOURCES:
                logger.error(f"❌ Tipo NOAA não encontrado: {data_type}")
                return None
            
            config = NOAA_REAL_SOURCES[data_type]
            logger.info(f"📡 Buscando NOAA: {config.url}")
            
            response = requests.get(config.url, timeout=30)
            
            if response.status_code == 200:
                json_data = response.json()
                
                # ✅ CORREÇÃO: Primeira linha é header, dados da segunda em diante
                if len(json_data) > 1:
                    columns = json_data[0]  # Header
                    data_rows = json_data[1:]  # Dados
                    
                    df = pd.DataFrame(data_rows, columns=columns)
                    df_processed = self._process_noaa_data_real(df, config)
                    
                    if len(df_processed) >= VALIDATION_CONFIG['min_data_points']:
                        self.data_source = f'NOAA_{data_type}'
                        logger.info(f"✅ NOAA {data_type}: {len(df_processed)} registros REAIS")
                        return df_processed
                else:
                    logger.error("❌ NOAA: JSON sem dados suficientes")
            else:
                logger.error(f"❌ NOAA HTTP {response.status_code}")
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro NOAA: {str(e)}")
            return None
    
    def _process_noaa_data_real(self, df, config):
        """Processa dados NOAA com estrutura CORRETA"""
        try:
            # Converter timestamp
            if 'time_tag' in df.columns:
                df['time_tag'] = pd.to_datetime(df['time_tag'])
            else:
                logger.error("❌ NOAA: Sem coluna time_tag")
                return pd.DataFrame()
            
            # ✅ Converter colunas numéricas
            numeric_columns = ['density', 'speed', 'temperature', 'bx_gse', 'by_gse', 'bz_gse', 'bt']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ordenar por tempo
            df = df.sort_values('time_tag').reset_index(drop=True)
            
            # Remover duplicatas
            df = df.drop_duplicates(subset=['time_tag'], keep='first')
            
            # Filtrar dados recentes
            cutoff = datetime.utcnow() - timedelta(days=30)
            df = df[df['time_tag'] > cutoff]
            
            # Aplicar limites físicos
            if 'speed' in df.columns:
                df = df[(df['speed'] > 200) & (df['speed'] < 1000)]
            if 'density' in df.columns:
                df = df[(df['density'] > 0.1) & (df['density'] < 100)]
            
            logger.info(f"📊 NOAA: {len(df)} registros REAIS processados")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro processamento NOAA: {str(e)}")
            return pd.DataFrame()
    
    def collect_data_real(self):
        """
        Coleta com fontes REAIS e estratégia inteligente
        """
        logger.info("🚀 INICIANDO COLETA COM VARIÁVEIS REAIS")
        
        collected_data = []
        sources_used = []
        
        # ✅ ESTRATÉGIA: Tentar fontes com variáveis REAIS
        strategies = [
            # 1. DSCOVR (dados mais atuais)
            {'type': 'nasa', 'dataset': 'DSCOVR_SWEPAM_L1'},
            {'type': 'nasa', 'dataset': 'DSCOVR_MAG_L1'},
            # 2. OMNI (dados históricos)
            {'type': 'nasa', 'dataset': 'OMNI_HRO2_1MIN'},
            # 3. NOAA (tempo real)
            {'type': 'noaa', 'data_type': 'PLASMA_5MIN'},
            {'type': 'noaa', 'data_type': 'MAG_5MIN'}
        ]
        
        for strategy in strategies:
            logger.info(f"🔄 Tentando {strategy['type']}: {strategy.get('dataset', strategy.get('data_type'))}")
            
            if strategy['type'] == 'nasa':
                df = self.fetch_nasa_real(strategy['dataset'])
            else:
                df = self.fetch_noaa_real(strategy['data_type'])
            
            if df is not None and len(df) >= VALIDATION_CONFIG['min_data_points']:
                collected_data.append(df)
                source_name = f"{strategy['type'].upper()}_{strategy.get('dataset', strategy.get('data_type'))}"
                sources_used.append(source_name)
                logger.info(f"✅ Fonte REAL adicionada: {source_name} ({len(df)} registros)")
                
                # Se já temos dados suficientes, podemos parar
                if len(collected_data) >= 2:  # Pelo menos 2 fontes
                    break
        
        # Combinar dados
        if collected_data:
            if len(collected_data) == 1:
                final_df = collected_data[0]
            else:
                final_df = self._merge_datasets_real(collected_data)
            
            # Validar dataset final
            if self._validate_dataset_real(final_df):
                stats = self._generate_stats_real(final_df, sources_used)
                return final_df, stats
        
        logger.error("❌ TODAS AS FONTES REAIS FALHARAM")
        logger.info("💡 VERIFIQUE:")
        logger.info("   - As variáveis no data_sources_REAL.py estão corretas")
        logger.info("   - Os datasets ainda estão disponíveis na NASA/NOAA")
        logger.info("   - A conexão com a internet está funcionando")
        return None, {}
    
    def _merge_datasets_real(self, datasets):
        """Combina múltiplos datasets REAIS"""
        # Começar com o dataset mais completo
        main_df = max(datasets, key=lambda x: len(x))
        
        # Adicionar variáveis de outros datasets
        for dataset in datasets:
            if dataset is not main_df:
                new_cols = [col for col in dataset.columns if col not in main_df.columns and col != 'time_tag']
                if new_cols:
                    merge_df = dataset[['time_tag'] + new_cols]
                    main_df = pd.merge_asof(
                        main_df.sort_values('time_tag'),
                        merge_df.sort_values('time_tag'),
                        on='time_tag',
                        tolerance=pd.Timedelta('10min'),
                        direction='nearest'
                    )
                    logger.info(f"✅ Merge realizado: adicionadas {len(new_cols)} variáveis")
        
        return main_df
    
    def _validate_dataset_real(self, df):
        """Validação do dataset REAL"""
        if df.empty:
            return False
        
        if len(df) < VALIDATION_CONFIG['min_data_points']:
            logger.warning(f"⚠️  Dataset pequeno: {len(df)} registros")
            return False
        
        # Verificar variáveis essenciais
        essential_vars = ['time_tag', 'speed']
        missing_vars = [var for var in essential_vars if var not in df.columns]
        if missing_vars:
            logger.error(f"❌ Variáveis essenciais faltando: {missing_vars}")
            return False
        
        # Verificar se os dados são realistas
        if 'speed' in df.columns:
            avg_speed = df['speed'].mean()
            if avg_speed < 200 or avg_speed > 800:
                logger.warning(f"⚠️  Velocidade média suspeita: {avg_speed:.1f} km/s")
        
        logger.info(f"✅ Dataset REAL validado: {len(df)} registros")
        return True
    
    def _generate_stats_real(self, df, sources):
        """Gera estatísticas do dataset REAL"""
        stats = {
            'collection_time': datetime.utcnow().isoformat(),
            'data_sources': sources,
            'total_records': len(df),
            'time_span_hours': (df['time_tag'].max() - df['time_tag'].min()).total_seconds() / 3600,
            'variables_present': [col for col in df.columns if col != 'time_tag'],
            'data_quality': {},
            'provenance': '100% DADOS REAIS - ZERO SINTÉTICO'
        }
        
        # Estatísticas por variável
        for col in df.columns:
            if col != 'time_tag' and df[col].notna().any():
                stats['data_quality'][col] = {
                    'completeness': f"{df[col].notna().mean():.1%}",
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return stats

def main():
    """Função principal - Coleta 100% com variáveis REAIS"""
    collector = RealDataCollector()
    
    try:
        # Coleta com variáveis REAIS
        df, stats = collector.collect_data_real()
        
        if df is None:
            logger.error("❌ COLETA FALHOU - Nenhuma fonte REAL retornou dados")
            print("\n❌ FALHA NA COLETA COM VARIÁVEIS REAIS")
            print("💡 Isso significa:")
            print("   - As variáveis especificadas podem não existir mais")
            print("   - Os datasets podem ter mudado")
            print("   - Serviços NASA/NOAA podem estar indisponíveis")
            print("📞 Verifique os logs para detalhes")
            sys.exit(1)
        
        # Salvar dados REAIS
        os.makedirs('data_real', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        
        csv_path = f"data_real/solar_real_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Salvar metadados
        json_path = f"data_real/real_metadata_{timestamp}.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Relatório de sucesso
        print("\n" + "="*70)
        print("✅ COLETA 100% REAL - CONCLUÍDA")
        print("="*70)
        print(f"📡 Fontes REAIS: {', '.join(stats['data_sources'])}")
        print(f"📊 Registros REAIS: {stats['total_records']}")
        print(f"⏱️  Período: {stats['time_span_hours']:.1f} horas")
        print(f"📈 Variáveis REAIS: {', '.join(stats['variables_present'])}")
        print(f"💾 Dados: {csv_path}")
        print(f"📋 Metadados: {json_path}")
        print("🎯 STATUS: 100% DADOS REAIS - ZERO SINTÉTICO")
        print("="*70)
        
        # Mostrar qualidade dos dados REAIS
        print("\n📊 QUALIDADE DOS DADOS REAIS:")
        for var, quality in stats['data_quality'].items():
            print(f"   {var:>12}: {quality['completeness']} completos | Média: {quality['mean']:6.1f}")
        
        # Verificar realismo dos dados
        if 'speed' in stats['data_quality']:
            avg_speed = stats['data_quality']['speed']['mean']
            if 300 <= avg_speed <= 600:
                print(f"✅ Velocidade média realista: {avg_speed:.1f} km/s")
            else:
                print(f"⚠️  Velocidade média atípica: {avg_speed:.1f} km/s")
                
    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO: {e}")
        print(f"\n❌ ERRO: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
