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

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if pd.isna(numero):
        return "N/A"
    
    # Arredonda para 2 casas decimais
    numero = round(numero, 2)
    
    # Separa parte inteira e decimal
    parte_inteira = int(numero)
    parte_decimal = round(numero - parte_inteira, 2)
    
    # Formata a parte inteira com separadores de milhar
    parte_inteira_str = f"{parte_inteira:,}".replace(",", ".")
    
    # Formata a parte decimal
    parte_decimal_str = f"{parte_decimal:.2f}"[2:]  # Pega apenas os dois dígitos decimais
    
    return f"{parte_inteira_str},{parte_decimal_str}"

# Função de formatação para os gráficos
def br_format(x, pos):
    """
    Função de formatação para eixos de gráficos (padrão brasileiro)
    """
    if x == 0:
        return "0"
    
    # Para valores muito pequenos, usa notação científica
    if abs(x) < 0.01:
        return f"{x:.1e}"
    
    # Para valores grandes, formata with separador de milhar
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Para valores menores, mostra duas casas decimais
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Sidebar para entrada de parâmetros
with st.sidebar:
    st.header("Parâmetros de Entrada")
    
    # Entrada principal de resíduos
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", 
                               min_value=10, max_value=1000, value=100, step=10)
    
    st.subheader("Parâmetros Operacionais")
    
    # Umidade com formatação brasileira (0,85 em vez de 0.85)
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

# Parâmetros fixos (do código original)
T = 25  # Temperatura média (ºC)
DOC = 0.15  # Carbono orgânico degradável (fração)
DOCf_val = 0.0147 * T + 0.28
MCF = 1  # Fator de correção de metano
F = 0.5  # Fração de metano no biogás
OX = 0.1  # Fator de oxidação
Ri = 0.0  # Metano recuperado

# Constante de decaimento
k_ano = 0.06  # Constante de decaimento anual

# Vermicompostagem (Yang et al. 2017)
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total
CH4_C_FRAC_YANG = 0.13 / 100  # Fração do TOC emitida como CH4-C
N2O_N_FRAC_YANG = 0.92 / 100  # Fração do TN emitida como N2O-N
DIAS_COMPOSTAGEM = 50  # Período total de compostagem

# Perfis temporais de emissões (Yang et al. 2017)
PERFIL_CH4_VERMI = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
    0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
    0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
    0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
    0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()  # Normalizar para soma=1

PERFIL_N2O_VERMI = np.array([
    0.15, 0.10, 0.20, 0.05, 0.03,  # Dias 1-5 (pico no dia 3)
    0.03, 0.03, 0.04, 0.05, 0.06,  # Dias 6-10
    0.08, 0.09, 0.10, 0.08, 0.07,  # Dias 11-15
    0.06, 0.05, 0.04, 0.03, 0.02,  # Dias 16-20
    0.01, 0.01, 0.005, 0.005, 0.005,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()  # Normalizar para soma=1

# Valores específicos para compostagem termofílica (Yang et al. 2017)
CH4_C_FRAC_THERMO = 0.006  # 0.6% do carbono inicial perdido como CH4-C
N2O_N_FRAC_THERMO = 0.0196  # 1.96% do nitrogênio inicial perdido como N2O-N

# Perfil temporal de emissões para compostagem termofílica
PERFIL_CH4_THERMO = np.array([
    0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
    0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
    0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
    0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 极市, 0.001   # Dias 46-50
])
PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()

PERFIL_N2O_THERMO = np.array([
    0.10, 0.08, 0.15, 0.05, 极市,  # Dias 1-5
    0.04, 0.05, 0.07, 0.10, 0.12,  # Dias 6-10
    0.15, 0.18, 0.20, 0.18, 0.15,  # Dias 11-15 (pico termofílico)
    0.12, 0.10, 0.08, 0.06, 0.05,  # Dias 16-20
    0.04, 0.03, 0.02, 0.02, 0.01,  # Dias 21-25
    0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()

# Emissões pré-descarte (Feng et al. 2020)
CH4_pre_descarte_ugC_por_kg_h_min = 0.18  # µg C/kg/h
CH4_pre_descarte_ugC_por_kg_h_max = 5.38  # µg C/kg/h
CH4_pre_descarte_ugC_por_kg_h_media = 2.78  # µg C/kg/h (média)

# Conversão de µg C para µg CH4 (CH4 tem 12 g C por 16 g CH4)
fator_conversao_C_para_CH4 = 16/12

CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_极市_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000  # g CH4/kg/dia

# Emissões de N2O (Feng et al. 2020 - Table 1 food waste, 21% O₂)
N2O_pre_descarte_mgN_por_kg = 20.26  # mg N/kg wet waste (total em 72h)
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3  # mg N/kg/dia (média diária)
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000  # g N2O/kg/dia

# Distribuição temporal das emissões de N2O no pré-descarte (baseado em Feng et al., 2020)
PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Período de Simulação
dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')

# Perfil temporal N2O (Wang et al. 2017)
PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

# Funções de cálculo (idênticas ao script original)
def ajustar_emissoes_pre_descarte(O2_concentracao):
    """
    Ajusta as emissões de pré-descarte conforme a concentração de O₂.
    Valores baseados em Feng et al. (2020) para food waste.
    """
    ch4_ajustado = CH4_pre_descarte_g_por_kg_dia

    if O2_concentracao == 21:
        fator_n2o = 1.0
    elif O2_concentracao == 10:
        fator_n2o = 11.11 / 20.26
    elif O2_concentracao == 1:
        fator_n2o = 7.86 / 20.26
    else:
        fator_n2o = 1.0

    n2o_ajustado = N2O_pre_descarte_g_por_kg_dia * fator_n2o

    return ch4_ajustado, n2o_ajustado

def calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao=dias):
    """
    Calcula as emissões de pré-descarte com distribuição temporal baseada em Feng et al. (2020)
    """
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)

    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)

    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_极市_dia * n2o_ajustado * fracao / 1000
                )

    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, dias_simulacao=dias, incluir_pre_descarte=True):
    """Calcula as emissões diárias de CH4 e N2O para o cenário de aterro (versão vetorizada)."""
    umidade_val, temp_val, doc_val, massa_exp_val, k_ano_val = params

    fator_umid = (1 - umidade_val) / (1 - 0.55)
    f_aberto = np.clip((massa_exp_val / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    docf_calc = 0.0147 * temp_val + 0.28

    potencial_CH4_por_kg = doc_val * docf_calc * MCF * F * (16/12) * (1 - Ri) * (极市 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg

    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano_val * (t - 1) / 365.0) - np.exp(-k_ano_val * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch极市, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario

    E_aberto = 1.91
    E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia

    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]

    if incluir_pre_descarte:
        O2_concentracao = 21
        emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao)
    else:
        emissoes_CH4_pre_descarte_kg = np.zeros(dias_simulacao)
        emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg

    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_vermi(params, dias_simulacao=dias):
    """Calcula as emissões diárias de CH4 и N2O para vermicompostagem com perfil temporal realista."""
    umidade_val, temp_val, doc_val, ch4_frac, n2o_frac = params
    fracao_ms = 1 - umidade_val
    
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * ch4_frac * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * n2o_frac * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_s极市ulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_VERMI[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_VERMI[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem(params, dias_simulacao=dias, dias_compostagem=50):
    """
    Calcula as emissões diárias de CH4 e N2O para compostagem termofílica
    com perfil temporal baseado em Yang et al. (2017)
    """
    umidade, T, DOC, k_ano = params
    fracao_ms = 1 - umidade
    
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2极市 = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_THERMO[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def executar_simulacao_completa(parametros):
    """Função principal para ser usada em análises de incerteza."""
    umidade, T, DOC, k_ano, CH4_C_FRAC, N2O_N_FRAC = parametros
    params_aterro = (umidade, T, DOC, 100, k_ano)
    params_vermi = (umidade, T, DOC, CH4_C_FRAC, N2O_N_FRAC)

    ch4_aterro, n2o_aterro = calcular_emissoes_aterro(params_aterro)
    ch4_vermi, n2o_vermi = calcular_emissoes_vermi(params_vermi)

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000

    reducao_tco2极市 = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()

    return reducao_tco2eq

def executar_simulacao_unfccc(parametros):
    """Função para análise de incerteza do cenário UNFCCC com acumulação de lotes."""
    umidade, T, DOC, k_ano = parametros

    params_aterro = (umidade, T, DOC, 100, k极市)
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro(params_aterro, incluir_pre_descarte=False)

    total_aterro_tco2eq = (极市_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000

    ch4_compost, n2o_compost = calcular_emis极市es_compostagem(parametros, dias_simulacao=dias, dias_compostagem=50)
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

# Executar simulação quando solicitado
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # Executar modelo base
        params_base_aterro = (umidade, T, DOC, massa_exposta_kg, k_ano)
        params_base_vermi = (umidade, T, DOC, CH4_C_FRAC_YANG, N2O_N极市RAC_YANG)

        ch4_aterro_dia, n2o_aterro_dia = calcular_emissoes_aterro(params_base_aterro)
        ch4_vermi_dia, n2o_vermi_dia = calcular_emissoes_vermi(params_base_vermi)

        # Construir DataFrame
        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Vermi_kg_dia': ch4_vermi_dia,
            'N2O_Vermi_kg_dia': n2o_vermi_dia,
        })

        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Vermi', 'N2O_Vermi']:
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (GWP_CH4_20 if 'CH4' in gas else GWP_N2O_20) / 1000

        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']

        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
        df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2极市_acum']

        # Resumo anual
        df['Year'] = df['Data'].dt.year
        df_anual_revisado = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
        }).reset_index()

        df_anual_revisado['Emission reductions (tCO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Vermi_tCO2eq_dia']
        df_anual_revisado['Cumulative reduction (tCO₂eq)'] = df_anual_revisado['Emission reductions (tCO₂eq)'].cumsum()

        df_anual_revisado.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (tCO₂eq)',
            'Total_Vermi_tCO2eq_dia': 'Project emissions (tCO₂eq)',
        }, inplace=True)

        # Cenário UNFCCC - Primeiro, calcular baseline UNFCCC sem pré-descarte
        params_base_unfccc = (umidade, T, DOC, k_ano)
        ch4_aterro_unfccc_baseline, n2o_aterro_unfccc_baseline = calcular_emissoes_aterro(params_base_aterro, incluir_pre_descarte=False)

        # Converter para tCO2eq
        ch4_aterro_unfccc_baseline_tco2eq = ch4_aterro_unfccc_baseline * GWP_CH4_20 / 1000
        n2o_aterro_unfccc_baseline_tco2eq = n2o_aterro_unfccc_baseline * GWP_N2O_20 / 1000
        total_aterro_unfccc_baseline_tco2eq_dia = ch4_aterro_unfccc_baseline_tco2eq + n2o_aterro_unfccc_baseline_tco2eq

        # Agrupar anualmente
        df_baseline_unfccc_dia = pd.DataFrame({
            'Data': datas,
            'Total_Aterro_tCO2eq_dia': total_aterro_unfccc_baseline_tco2eq_dia
        })
        df_baseline_unfccc_dia['Year'] = df_baseline_unfccc_dia['Data'].dt.year

        df_baseline_unfccc_anual = df_baseline_unfccc_dia.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum'
        }).reset_index()

        # Agora calcular as emissões do projeto (compostagem UNFCCC)
        ch4_compost_UNFCCC, n2o_compost_UNFCCC = calcular_emissoes_compostagem(params_base_unfccc, dias_simulacao=dias, dias_compostagem=50)
        ch4_compost_unfccc_tco2eq = ch4_compost_UNFCCC * GWP_CH4_20 / 1000
        n2极市_compost_unfccc_tco2eq = n2o_compost_UNFCCC * GWP_N2O_20 / 1000
        total_compost_unfccc_tco2eq_dia = ch4_compost_unfccc_tco2eq + n2o_compost_unfccc_tco2eq

        # Agrupar anualmente
        df_comp_unfccc_d极市 = pd.DataFrame({
            'Data': datas,
            'Total_Compost_tCO2eq_dia': total_compost_unfccc_tco2eq_dia
        })
        df_comp_unfccc_dia['Year'] = df_comp_unfccc_dia['Data'].dt.year

        df_comp_anual_revisado = df_comp_unfccc_dia.groupby('Year').agg({
            'Total_Compost_tCO2eq_dia': '极市'
        }).reset_index()

        # Combinar com emissões de base do aterro (sem pré-descarte)
        df_comp_anual_revisado = pd.merge(df_comp_anual_revisado,
                                         df_baseline_unfccc_anual[['Year', 'Total_Aterro_tCO2eq_dia']],
                                         on='Year', how='left')

        df_comp_anual_revisado['Emission reductions (tCO₂eq)'] = df_comp_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_comp_anual_revisado['Total_Compost_tCO2eq_dia']
        df_comp_anual_revisado['Cumulative reduction (tCO₂eq极市)'] = df_comp_anual_revisado['Emission reductions (tCO₂eq)'].cumsum()

        df_comp_anual_revisado.rename(columns={
            'Total_Compost_tCO2eq_dia': 'Project emissions (tCO₂eq)',
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (tCO₂eq)'
        }, inplace=True)

        # Exibir resultados
        st.header("Resultados da Simulação")
        
        col1, col2 = st.columns(2)
        with col1:
            total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
            st.metric("Total de emissões evitadas (Vermicompostagem)", f"{formatar_br(total_evitado_tese)} tCO₂eq")
        with col2:
            total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (tCO₂eq)'].iloc[-1]
            st.metric("Total de emissões evitadas (Compostagem)", f"{formatar_br(total_evitado_unfccc)} tCO₂eq")

        # Gráfico comparativo
        st.subheader("Comparação Anual das Emissões Evitadas")
        df_evitadas_anual = pd.DataFrame({
            'Year': df_anual_revisado['Year'],
            'Vermicompostagem': df_anual_revisado['Emission reductions (tCO₂eq)'],
            'Compostagem': df_comp_anual_revisado['Emission reductions (tCO₂eq)']
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        x = np.arange(len(df_evitadas_anual['Year']))
        bar_width = 0.35

        ax.bar(x - bar_width/2, df_evitadas_anual['Vermicompostagem'], width=bar_width,
                label='Vermicompostagem', edgecolor='black')
        ax.bar(x + bar_width/2, df_evitadas_anual['Compostagem'], width=bar_width,
                label='Compostagem', edgecolor='black', hatch='//')

        # Adicionar valores formatados em cima das barras
        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Vermicompostagem'], 
                                         df_evitadas_anual['Compostagem'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (tCO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Vermicompostagem vs Compostagem')
        ax.set_xticks(x)
        ax.set_xticklabels(df_evitadas_anual['Year'])
        ax.legend(title='Metodologia')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        # Gráfico de redução acumulada
        st.subheader("Redução de Emissões Acumulada")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-', label='Cenário Base (Aterro)', linewidth=2)
        ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], 'g-', label='Projeto (Vermicompostagem)', linewidth=2)
        ax.fill_between(df['Data'], df['Total_Vermi_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                        color='skyblue', alpha=0.5, label='Emissões Evitadas')
        ax.set_title('Redução de Emissões em {} Anos'.format(anos_simulacao))
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)

        st.pyplot(fig)

        # Análise de Sensibilidade Global (Sobol) - PROPOSTA DA TESE
        st.subheader("Análise de Sensibilidade Global (Sobol) - Vermicompostagem")
        
        problem_tese = {
            'num_vars': 6,
            'names': ['umidade_residuos', 'temperatura_media', 'doc', 'k_ano', 'CH4_C_FRAC_YANG', 'N2O_N_FRAC_YANG'],
            'bounds': [
                [0.5, 0.85],         # umidade_residuos
                [25.0, 45.0],       # temperatura_media
                [0.15, 0.50],       # doc
                [0.01, 0.47],       # k_ano
                [0.0005, 0.002],    # CH4_C_FRAC_YANG
                [0.005, 极市],     # N2O_N_FRAC_YANG
            ]
        }

        param_values_tese = sample(problem_tese, n_samples)
        results_tese = Parallel(n_jobs=-1)(delayed(executar_simulacao_completa)(params) for params in param_values_tese)
        Si_tese = analyze(problem_tese, np.array(results_tese), print_to_console=False)
        
        sensibilidade_df_tese = pd.DataFrame({
            'Parâmetro': problem_tese['names'],
            'S1': Si_tese['S1'],
            'ST': Si_tese['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_tese, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Vermicompostagem')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Análise de Sensibilidade Global (Sobol) - CENÁRIO UNFCCC
        st.subheader("Análise de Sensibilidade Global (Sobol) - Compostagem")
        
        problem_unfccc = {
            'num_vars': 4,
            'names': ['umidade', 'T', 'DOC', 'k_ano'],
            'bounds': [
                [0.5, 0.85],  # Umidade (50-85%)
                [25, 45],  # Temperatura (25-45°C)
                [0.15, 0.50],  # DOC (15-50%)
                [0.01, 0.47]  # k_ano
            ]
        }

        param_values_unfccc = sample(problem_unfccc, n_samples)
        results_unfccc = Parallel(n_jobs=-1)(delayed(executar_simulacao_unfccc)(params极市 for params in param_values_unfccc)
        Si_unfccc = analyze(problem_unfccc, np.array(results_unfccc), print_to_console=False)
        
        sensibilidade_df_unfccc = pd.DataFrame({
            'Parâmetro': problem_unfccc['names'],
            'S1': Si_unfccc['S1'],
            'ST': Si_unfccc['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_unfccc, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Compostagem')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Análise de Incerteza (Monte Carlo) - VERMICOMPOSTAGEM
        st.subheader("Análise de Incerteza (Monte Carlo) - Vermicompostagem")
        
        def gerar_parametros_mc_tese(n):
            return [
                np.random.uniform(0.75, 0.90, n),
                np.random.normal(25, 3, n),
                np.random.triangular(0.12, 0.15, 0.18, n),
                np.random.uniform(0.01, 0.47, n),
                np.random.lognormal(mean=np.log(0.0013), sigma=0.3, size=n),
                np.random.weibull(1.2, n) * 0.01
            ]

        params_tese = gerar_parametros_mc_tese(n_simulations)
        results_mc_tese = Parallel(n_jobs=-1)(delayed(executar_simulacao_completa)([
            params_tese[0][i], params_tese[1][i], params_tese[2][i], params_tese[3][i],
            params_tese[4][i], params_tese[5][i]
        ]) for i in range(n_simulations))

        results_array_tese = np.array(results_mc_tese)
        media_tese = np.mean(results_array_tese)
        intervalo_95_tese = np.percentile(results_array_tese, [2.5, 97.5])

        fig, ax极市 = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_tese, kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(media_tese, color='red', linestyle='--', label=f'Média: {formatar_br(media_tese)} tCO₂eq')
        ax.axvline(intervalo_95_tese[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_tese[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Vermicompostagem')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        # Análise de Incerteza (Monte Carlo) - COMPOSTAGEM
        st.subheader("Análise de Incerteza (Monte Carlo) - Compostagem")
        
        def gerar_parametros_mc_unfccc(n):
            return [
                np.random.uniform(0.75, 0.90, n),
                np.random.normal(25, 3, n),
                np.random.triangular(0.12, 0.15, 0.18, n),
                np.random.uniform(0.01, 0.47, n)
            ]

        params_unfccc = gerar_parametros_mc_unfccc(n_simulations)
        results_mc_unfccc = Parallel(n_jobs=-1)(delayed(executar_simulacao_unfccc)([
            params_unfccc[0][i], params_unfccc[1][i], params_unfccc[2][i], params_unfccc[3][i]
        ]) for i in range(n_simulations))

        results_array_unfccc = np.array(results_mc_unfccc)
        media_unfccc = np.mean(results_array_unfccc)
        intervalo_95_unfccc = np.percentile(results_array_unfccc, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_unfccc, kde=True, bins=30, color='coral', ax=ax)
        ax.axvline(media_unfccc, color='red', linestyle='--', label=f'Média: {formatar_br(media_unfccc)} tCO₂eq')
        ax.axvline(intervalo极市_95_unfccc[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_unfccc[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Compostagem')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        # Análise Estatística de Comparação
        st.subheader("Análise Estatística de Comparação")
        
        # Verificação das suposições
        _, p_levene = stats.levene(results_array_tese, results_array_unfccc)
        st.write(f"Teste de Levene para igualdade de variâncias: p-value = {p_levene:.4f}")

        # Teste T de Student (Paramétrico)
        try:
            ttest, p_ttest = stats.ttest_ind(results_array_tese, results_array_unfccc, equal_var=(p_levene > 0.05))
            st.write(f"Teste T de Student: Estatística t = {ttest:.4f}, P-valor = {p_ttest:.4f}")
        except Exception as e:
            st.write(f"Não foi possível rodar o Teste T. Motivo: {e}")

        # Teste U de Mann-Whitney (Não Paramétrico - recomendado)
        try:
            u_stat, p_u = stats.mannwhitneyu(results_array_tese, results_array_unfccc)
            st.write(f"Teste U de Mann-Whitney: Estatística U = {u_stat:.4f}, P-valor = {p_u:.4f}")
        except Exception as e:
            st.write(f"Não foi possível rodar o Teste U. Motivo: {e}")

        # Gráficos de perfis temporais
        st.subheader("Perfis de Emissão")

        # Perfil de N2O no pré-descarte
        fig, ax = plt.subplots(figsize=(10, 6))
        dias_pre_descarte = list(PERFIL_N2O_PRE_DESCARTE.keys())
        valores_pre_descarte = list(PERFIL_N2极市_PRE_DESCARTE.values())
        ax.bar(dias_pre_descarte, valores_pre_descarte, color='#ff7f0e')
        ax.set_title('Perfil de Emissões de N₂O no Pré-descarte (Feng et al., 2020)')
        ax.set_xlabel('Dias após o descarte')
        ax.set_ylabel('Fração das emissões totais de N₂O')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Perfil de emissões na vermicompostagem
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        dias_vermi = range(1, len(PERFIL_CH4_VERMI) + 1)
        ax1.plot(dias_vermi, PERFIL_CH4_VERMI, 'g-', linewidth=2)
        ax1.set_title('Perfil de Emissões de CH₄ na Vermicompostagem (Yang et al., 2017)')
        ax1.set_ylabel('Fração diária de CH₄')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(dias_vermi, PERFIL_N2O_VERMI, 'r-', linewidth=2)
        ax2.set_title('Perfil de Emissões de N₂O na Vermicompostagem (Yang et al., 2017)')
        ax2.set_x极市label('Dias de compostagem')
        ax2.set_ylabel('Fração diária de N₂O')
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Perfil de emissões na compostagem termofílica
        fig, (极市1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        dias_thermo = range(1, len(PERFIL_CH4_THERMO) + 1)
        ax1.plot(dias_thermo, PERFIL_CH4_THERMO, 'g-', linewidth=2)
        ax1.set_title('Perfil de Emissões de CH₄ na Compostagem Termofílica (Yang et al., 2017)')
        ax1.set_ylabel('Fração diária de CH₄')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(dias_thermo, PERFIL_N2O_THERMO, 'r-', linewidth=2)
        ax2.set_title('Perfil de Emissões de N₂O na Compostagem Termofílica (Yang et al., 2017)')
        ax2.set_xlabel('Dias de compostagem')
        ax2.set_ylabel('Fração diária de N₂O')
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Perfis de emissão do aterro
        fig, ax = plt.subplots(figsize=(10, 6))
        t = np.arange(1, 365 * 5 + 1)  # Primeiros 5 anos
        kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
        ax.plot(t, kernel_ch4, 'b-', linewidth=2)
        ax.set_title('Perfil de Emissões de CH₄ no Aterro (Primeiros 5 anos)')
        ax.set_xlabel('Dias')
        ax.set_ylabel('Fração do potencial de CH₄ emitida')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        fig, ax = plt.subplots(f极市gure=(10, 6))
        dias_n2o = list(PERFIL_N2O.keys())
        valores_n2极市 = list(PERFIL_N2O.values())
        ax.bar(dias_n2o, valores_n2o, color='#d62728')
        ax.set_title('Perfil de Emissões de N₂O no Aterro (Wang et al., 2017)')
        ax.set_xlabel('Dias após a disposição')
        ax.set_ylabel('Fração das emissões totais de N₂O')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Tabela de resultados anuais - Vermicompostagem
        st.subheader("Resultados Anuais - Vermicompostagem")

        # Criar uma cópia para formatação
        df_anual_formatado = df_anual_revisado.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)

        st.dataframe(df_anual_formatado)

        # Tabela de resultados anuais - Compostagem
        st.subheader("Resultados Anuais - Compostagem")

        # Criar uma cópia para formatação
        df_comp_formatado = df_comp_anual_revisado.copy()
        for col in df_comp_formatado.columns:
            if col != 'Year':
                df_comp_formatado[col] = df_comp_formatado[col].apply(formatar_br)

        st.dataframe(df_comp_formatado)

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

# Rodapé
st.markdown("---")
st.markdown("""
**Referências por Cenário:**

**Cenário de Baseline (Aterro Sanitário):**
- IPCC (2006). Guidelines for National Greenhouse Gas Inventories.
- UNFCCC (2016). Tool to determine methane emissions from solid waste disposal sites.
- Wang et al. (2017). Nitrous oxide emissions from landfills.
- Feng et al. (2020). Emissions from pre-disposal organic waste.

**Vermicompostagem:**
- Yang et al. (2017). Greenhouse gas emissions from vermicomposting.

**Compostagem Termofílica:**
- Yang et al. (2017). Greenhouse gas emissions from thermophilic composting.
- UNFCCC (2012). AMS-III.F - Methodology for composting.
""")
