import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

np.random.seed(50)  # Garante reprodutibilidade

# Configurações iniciais
st.set_page_config(page_title="Simulador de Emissões CO₂eq", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Título do aplicativo
st.title("Simulador de Emissões de tCO₂eq")
st.markdown("""
Esta ferramenta calcula as emissões de gases de efeito estufa para dois contextos de gestão de resíduos,
aterro sanitário vs. vermicompostagem (Contexto: Proposta da Tese) e aterro sanitário vs. compostagem (Contexto: UNFCCC).
""")

# =============================================
# Funções de formatação no padrão brasileiro
# =============================================

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    numero = round(float(numero), 2)
    if numero.is_integer():
        return f"{int(numero):,}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def aplicar_formatacao_br(df):
    df_formatado = df.copy()
    for col in df_formatado.columns:
        if df_formatado[col].dtype in ['float64', 'int64']:
            df_formatado[col] = df_formatado[col].apply(formatar_br)
    return df_formatado

# Para gráficos
def br_format_inteiro(x, pos):
    return f'{x:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def br_format_decimal(x, pos):
    return f'{x:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}"
    if x.is_integer():
        return br_format_inteiro(x, pos)
    return br_format_decimal(x, pos)

br_formatter_inteiro = FuncFormatter(br_format_inteiro)
br_formatter_decimal = FuncFormatter(br_format_decimal)

# =============================================================
# Sidebar para entrada de parâmetros
# =============================================================
with st.sidebar:
    st.header("Parâmetros de Entrada")
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", min_value=10, max_value=1000, value=100, step=10)
    st.subheader("Parâmetros Operacionais")
    umidade_valor = st.slider("Umidade do resíduo", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    st.write(f"Umidade selecionada: {formatar_br(umidade_valor)}%")
    massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10)
    h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1)
    st.subheader("Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)
    if st.button("Executar Simulação"):
        st.session_state.run_simulation = True
    else:
        st.session_state.run_simulation = False

# =============================================================
# Parâmetros fixos
# =============================================================
T = 25
DOC = 0.15
DOCf_val = 0.0147 * T + 0.28
MCF = 1
F = 0.5
OX = 0.1
Ri = 0.0
k_ano = 0.06
TOC_YANG = 0.436
TN_YANG = 14.2 / 1000
CH4_C_FRAC_YANG = 0.13 / 100
N2O_N_FRAC_YANG = 0.92 / 100
DIAS_COMPOSTAGEM = 50

# =============================================================
# Aqui entram todas as funções originais de cálculo do app
# =============================================================

# =============================================================
# Execução da simulação e exibição de resultados
# =============================================================
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # --- aqui ficam os cálculos originais do app.py ---
        # exemplo: df, df_anual_revisado, df_comp_anual_revisado criados

        # Exibição de métricas
        st.header("Resultados da Simulação")
        col1, col2 = st.columns(2)
        with col1:
            total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
            st.metric("Total de emissões evitadas (Tese)", f"{formatar_br(total_evitado_tese)} tCO₂eq")
        with col2:
            total_evitado_unfccc = df_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
            st.metric("Total de emissões evitadas (UNFCCC)", f"{formatar_br(total_evitado_unfccc)} tCO₂eq")

        # Exibir DataFrames formatados
        st.subheader("Resultados detalhados")
        st.dataframe(aplicar_formatacao_br(df))

        st.subheader("Resumo anual - Proposta da Tese")
        st.dataframe(aplicar_formatacao_br(df_anual_revisado))

        st.subheader("Resumo anual - UNFCCC")
        st.dataframe(aplicar_formatacao_br(df_comp_anual_revisado))

        # Gráficos com formatação brasileira nos eixos e labels
        fig, ax = plt.subplots()
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], label='Aterro')
        ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], label='Vermi')
        ax.set_ylabel("tCO₂eq acumulado")
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        for line in ax.get_lines():
            y_data = line.get_ydata()
            x_data = line.get_xdata()
            for x, y in zip(x_data[::len(x_data)//10], y_data[::len(y_data)//10]):  # marca 10 pontos
                ax.annotate(formatar_br(y), (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        bars = ax2.bar(df_anual_revisado['Year'], df_anual_revisado['Emission reductions (t CO₂eq)'])
        ax2.set_ylabel("Redução anual (tCO₂eq)")
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(formatar_br(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        st.pyplot(fig2)
