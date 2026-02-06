import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(page_title="Nutriwash Dashboard", layout="wide", page_icon="🌱")

# Título e Introdução
st.title("🌱 Nutriwash: Gestão Inteligente de Resíduos")
st.markdown("""
    **Operação: Ribeirão Preto/SP** Monitoramento em tempo real dos reatores anaeróbicos instalados em varejistas de frutas e vegetais.
    *Tecnologia de vedação total para zero emissões pré-descarte.*
""")

# --- SIMULAÇÃO DE DADOS ---
@st.cache_data
def load_data():
    # Simulando 30 dias de operação
    dates = pd.date_range(end=datetime.now(), periods=30)
    data = pd.DataFrame({
        'Data': dates,
        'Coleta (kg)': np.random.normal(100, 5, 30), # Média de 100kg conforme solicitado
        'Emissões Evitadas (kg CO2e)': np.random.normal(45, 3, 30),
        'Eficiência do Reator (%)': np.random.uniform(95, 99.8, 30)
    })
    return data

df = load_data()

# --- SIDEBAR (Filtros e Info) ---
st.sidebar.header("Configurações do Painel")
unidade = st.sidebar.selectbox("Selecione a Unidade", ["Todos os Varejistas", "Ceasa Ribeirão", "Rede Local A", "Hortifruti B"])
st.sidebar.info(f"Status do Reator: **Operacional** 🟢\n\nVedação: **100% Hermético**")

# --- MÉTRICAS PRINCIPAIS ---
col1, col2, col3 = st.columns(3)
total_coletado = df['Coleta (kg)'].sum()
media_diaria = df['Coleta (kg)'].mean()

col1.metric("Total Coletado (Mês)", f"{total_coletado:,.0f} kg", "+12%")
col2.metric("Média Diária", f"{media_diaria:.1f} kg", "Meta batida")
col3.metric("Emissões de Metano Retidas", "99.9%", "Tecnologia Nutriwash")

st.divider()

# --- GRÁFICOS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Volume de Coleta Diária (kg)")
    fig_coleta = px.line(df, x='Data', y='Coleta (kg)', markers=True, 
                         color_discrete_sequence=['#2E7D32'])
    fig_coleta.add_hline(y=100, line_dash="dot", annotation_text="Meta Diária (100kg)")
    st.plotly_chart(fig_coleta, use_container_width=True)

with c2:
    st.subheader("Impacto Ambiental: CO2 Evitado")
    fig_env = px.bar(df, x='Data', y='Emissões Evitadas (kg CO2e)', 
                     color_discrete_sequence=['#81C784'])
    st.plotly_chart(fig_env, use_container_width=True)

# --- DETALHES TÉCNICOS DOS REATORES ---
st.subheader("Status dos Reatores nos Pontos de Coleta")
st.write("Sensores de pressão e vedação nos varejistas parceiros:")

# Criando uma tabela fictícia de status por local
locais_df = pd.DataFrame({
    'Ponto de Venda': ['Varejista Centro', 'Horti-Sertãozinho', 'Mercado RP Sul', 'Quitanda da Avenida'],
    'Capacidade Atual': ['85%', '40%', '92%', '15%'],
    'Vedação': ['Ativa', 'Ativa', 'Ativa', 'Ativa'],
    'Última Coleta': ['Há 2h', 'Há 5h', 'Há 30min', 'Ontem']
})
st.table(locais_df)

st.success("✅ Sistema operando dentro das normas ambientais de Ribeirão Preto.")
