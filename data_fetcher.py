# data_fetcher.py
import requests
import pandas as pd
from datetime import datetime

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
