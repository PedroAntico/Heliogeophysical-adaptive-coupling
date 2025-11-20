import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="HAC 3.0 - Brasil Espacial", layout="wide")
st.title("🌞 Heliogeophysical Adaptive Coupling 3.0")
st.subheader("Monitoramento em tempo real do vento solar · NOAA/DSCOVR · 20 de novembro de 2025")

@st.cache_data(ttl=300)  # atualiza a cada 5 minutos
def get_data():
    plasma = requests.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json").json()
    mag = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json").json()
    df_p = pd.DataFrame(plasma[1:], columns=plasma[0])
    df_m = pd.DataFrame(mag[1:], columns=mag[0])
    df_p['time_tag'] = pd.to_datetime(df_p['time_tag'])
    df_m['time_tag'] = pd.to_datetime(df_m['time_tag'])
    return df_p, df_m

df_p, df_m = get_data()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Velocidade", f"{df_p.iloc[-1]['speed']} km/s")
with col2:
    st.metric("Densidade", f"{df_p.iloc[-1]['density']} p/cm³")
with col3:
    st.metric("Temperatura", f"{int(float(df_p.iloc[-1]['temperature'] or 0)):,} K")
with col4:
    bz = df_m.iloc[-1].get('bz_gse', 'N/D')
    delta = "🟢 Calmo" if float(bz or 0) > -5 else "🟡 Atenção" if float(bz or 0) > -10 else "🔴 TEMPESTADE!"
    st.metric("Bz (nT)", bz, delta=delta)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_p['time_tag'][-50:], y=df_p['speed'][-50:], name="Velocidade"))
fig.add_trace(go.Scatter(x=df_m['time_tag'][-50:], y=df_m['bz_gse'][-50:], name="Bz", yaxis="y2"))
fig.update_layout(title="Vento Solar nas últimas 4 horas", yaxis2=dict(title="Bz (nT)", overlaying="y", side="right"))
st.plotly_chart(fig, use_container_width=True)

st.success("HAC 3.0 rodando ao vivo · Criado por Pedro Guilherme Antico · 20/11/2025")
