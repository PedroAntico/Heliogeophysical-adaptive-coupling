# data_fetcher.py
import os
import pandas as pd
import requests
from datetime import datetime  # <-- Adicione esta linha

def fetch_real_time_solar_data():
    # Garante que o diretório 'data' existe antes de salvar
    os.makedirs("data", exist_ok=True)

    # --- Aqui entra o código que coleta os dados reais ---
    df = pd.DataFrame({
        "timestamp": ["2025-11-11T00:00Z"],
        "speed_km_s": [429],
        "density_pcm3": [3.44],
        "bz_nT": [0.69]
    })

    # Salva os dados
    df.to_csv("data/solar_data_latest.csv", index=False)
    print("✅ Dados solares salvos em data/solar_data_latest.csv")
    print(f"✅ Dados salvos: {len(df)} registros ({datetime.utcnow()} UTC)")  # <-- Linha que falhou

if __name__ == "__main__":
    fetch_real_time_solar_data()
    
def fetch_real_time_solar_data():
    """
    Coleta dados de vento solar em tempo real do SWPC (NOAA)
    e salva em data/solar_data_latest.csv
    """
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    r = requests.get(url, timeout=30)
    data = r.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df = df.astype({
        "density": float,
        "speed": float,
        "temperature": float
    })

    df.to_csv("data/solar_data_latest.csv", index=False)
    print(f"✅ Dados salvos: {len(df)} registros ({datetime.utcnow()} UTC)")

if __name__ == "__main__":
    fetch_real_time_solar_data()
