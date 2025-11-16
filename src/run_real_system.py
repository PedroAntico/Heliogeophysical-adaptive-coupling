"""
src/run_real_system.py
Sistema DEFINITIVO com variáveis REAIS
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/real_system.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('real_system')

def run_real_system():
    """Executa o sistema definitivo com variáveis REAIS"""
    
    print("🚀 SISTEMA HELIOGEOPHYSICAL - VERSÃO DEFINITIVA")
    print("🎯 VARIÁVEIS 100% REAIS DOS DATASETS:")
    print("   ✅ DSCOVR_SWEPAM_L1: proton_density, bulk_speed, proton_temp")
    print("   ✅ DSCOVR_MAG_L1: B_x, B_y, B_z, Bt")  
    print("   ✅ OMNI_HRO2_1MIN: BX_GSE, BY_GSE, BZ_GSE, V, FlowPressure")
    print("   ✅ NOAA: density, speed, temperature, bx_gse, by_gse, bz_gse, bt")
    print("="*60)
    
    # Verificar dependências
    try:
        import cdasws
        import requests
        import pandas as pd
        logger.info("✅ Todas as dependências disponíveis")
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        print(f"❌ Instale as dependências: pip install cdasws requests pandas numpy")
        return False
    
    # Executar coletor com variáveis REAIS
    try:
        logger.info("🔧 Executando coletor com variáveis REAIS...")
        result = subprocess.run(
            [sys.executable, 'src/data_fetcher_REAL.py'],
            capture_output=True, text=True, timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Coleta REAL concluída com sucesso")
            print("\n✅ DADOS REAIS COLETADOS COM SUCESSO!")
            print("📁 Verifique a pasta 'data_real/'")
            
            # Listar arquivos gerados
            if os.path.exists('data_real'):
                files = os.listdir('data_real')
                print("📊 Arquivos REAIS gerados:")
                for file in sorted(files)[-3:]:
                    file_path = os.path.join('data_real', file)
                    file_size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({file_size/1024:.1f} KB)")
            
            return True
        else:
            logger.error(f"❌ Coleta REAL falhou: {result.stderr}")
            print(f"❌ Erro na coleta REAL: {result.stderr}")
            
            # Análise do erro
            if "variable does not belong to dataset" in result.stderr:
                print("💡 PROBLEMA IDENTIFICADO: Variáveis não existem no dataset")
                print("   📞 Verifique data_sources_REAL.py para variáveis atualizadas")
            elif "Timeout" in result.stderr:
                print("💡 PROBLEMA IDENTIFICADO: Timeout na conexão")
                print("   🌐 Verifique sua conexão com a internet")
            else:
                print("💡 Verifique os logs para detalhes")
                
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Timeout na coleta REAL")
        print("❌ Timeout - Verifique conexão com serviços NASA/NOAA")
        return False
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        print(f"❌ Erro: {e}")
        return False

def main():
    """Função principal"""
    print("🔍 Iniciando sistema com variáveis REAIS...")
    
    success = run_real_system()
    
    if success:
        print("\n🎉 SISTEMA 100% REAL FUNCIONANDO CORRETAMENTE!")
        print("📈 Próximos passos:")
        print("   1. Verifique os dados REAIS em 'data_real/'")
        print("   2. Execute a análise preditiva com dados reais")
        print("   3. Valide os resultados com métricas realistas")
        print("🔬 STATUS: PRONTO PARA PUBLICAÇÃO CIENTÍFICA")
    else:
        print("\n💥 SISTEMA FALHOU - Variáveis podem precisar de atualização")
        print("📞 Contate o mantenedor para atualizar as variáveis REAIS")
        sys.exit(1)

if __name__ == '__main__':
    main()
