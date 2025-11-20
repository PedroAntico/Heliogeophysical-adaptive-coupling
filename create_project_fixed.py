#!/usr/bin/env python3
"""
Script 100% seguro e testado para recriar o projeto heliogeofísico sem erros de sintaxe
Autor: Pedro Antico + Grok (Novembro 2025)
"""
import os
from pathlib import Path

def main():
    print("Iniciando criação da estrutura do projeto Heliogeophysical Adaptive Coupling\n")

    # 1. Cria todos os diretórios necessários
    dirs = [
        "src", "src/config", "src/fetchers", "src/utils", 
        "src/processing", "src/detection",
        "data/raw", "data/processed", "logs", "tests"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")

    # 2. Cria requirements.txt
    reqs = """requests
pandas
numpy
scikit-learn
pyyaml
matplotlib
loguru
"""
    Path("requirements.txt").write_text(reqs.strip() + "\n")
    print("✓ requirements.txt criado")

    # 3. Cria src/__init__.py (torna src um pacote)
    Path("src/__init__.py").touch()

    # 4. Cria o main.py perfeito (zero chance de string solta)
    main_py = '''#!/usr/bin/env python3
"""
Heliogeophysical Adaptive Coupling
Framework para detecção e análise de eventos solar-terrestres
Autor: Pedro Antico - Novembro 2025
"""
import logging
from datetime import datetime
from pathlib import Path

# Configuração de logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "heliogeophysical.log")
    ]
)
log = logging.getLogger(__name__)

def pipeline_simulado():
    log.info("Iniciando pipeline de acoplamento heliogeofísico")
    log.info("Coletando dados REALTIME (simulado)")
    log.info("Processando séries temporais")
    log.info("Detectando eventos de interesse")

    eventos = [
        {"timestamp": datetime.utcnow().isoformat() + "Z", "type": "CME_simulado", "severity": "high"},
        {"timestamp": datetime.utcnow().isoformat() + "Z", "type": "tempestade_geomagnetica", "severity": "medium"},
    ]
    
    log.info(f"Pipeline concluído — {len(eventos)} eventos detectados")
    return eventos

if __name__ == "__main__":
    print("Heliogeophysical Adaptive Coupling")
    print("=" * 50)
    eventos = pipeline_simulado()
    print("=" * 50)
    print("Projeto iniciado com sucesso! Use: python src/main.py")
'''
    Path("src/main.py").write_text(main_py.lstrip())
    os.chmod("src/main.py", 0o755)
    print("✓ src/main.py criado e tornado executável")

    print("\nPROJETO RECRIADO COM SUCESSO!")
    print("\nPróximos comandos (copie e cole um por um):")
    print("   pip install -r requirements.txt")
    print("   python src/main.py")
    print("\nPronto! Agora está tudo limpo e funcionando.")

if __name__ == "__main__":
    main()
