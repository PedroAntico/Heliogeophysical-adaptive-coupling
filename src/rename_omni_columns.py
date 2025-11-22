#!/usr/bin/env python3
"""
rename_omni_columns.py – Nomeia corretamente as variáveis OMNI do arquivo convertido
Formato oficial:
1 Bt
2 Bx
3 Bz_GSE
4 Bz_GSM
5 Temperature
6 Density
7 Speed
8 Pressure
"""

import pandas as pd

INPUT = "data_real/omni_converted.csv"
OUTPUT = "data_real/omni_labeled.csv"

COLUMN_MAP = {
    "var_1": "Bt",
    "var_2": "Bx",
    "var_3": "Bz_GSE",
    "var_4": "Bz_GSM",
    "var_5": "Temperature",
    "var_6": "Density",
    "var_7": "Speed",
    "var_8": "Pressure",
}

def main():
    print("📂 Carregando arquivo convertido...")
    df = pd.read_csv(INPUT)

    print("🔧 Aplicando renomeação oficial OMNI...")
    df = df.rename(columns=COLUMN_MAP)

    print("💾 Salvando arquivo final:", OUTPUT)
    df.to_csv(OUTPUT, index=False)

    print("\n🎉 Pronto!")
    print("Arquivo com nomes completos salvo:", OUTPUT)
    print("Colunas finais:", list(df.columns))

if __name__ == "__main__":
    main()
