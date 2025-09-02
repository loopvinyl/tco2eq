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

# =============================================
# 0. CONFIGURAÇÕES INICIAIS
# =============================================
st.set_page_config(page_title="Simulador de Emissões tCO₂eq", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================
# 1. TÍTULO E INTRODUÇÃO DO APP
# =============================================
st.title("Simulador de Emissões de tCO₂eq")
st.markdown("""
Esta ferramenta calcula e compara as emissões de gases de efeito estufa (GEE) para três contextos de gestão de resíduos,
com base em dados científicos.
---
""")

# =============================================
# 2. FUNÇÕES DE FORMATAÇÃO E CÁLCULO
# =============================================

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    parte_inteira = int(numero)
    parte_decimal = round(numero - parte_inteira, 2)
    parte_inteira_str = f"{parte_inteira:,}".replace(",", ".")
    parte_decimal_str = f"{parte_decimal:.2f}"[2:]
    return f"{parte_inteira_str},{parte_decimal_str}"

# Função de formatação para os gráficos
def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Funções de cálculo (importadas e adaptadas de tco2eq3set.txt)
def ajustar_emissoes_pre_descarte(O2_concentracao):
    CH4_pre_descarte_g_por_kg_dia = 2.78 * (16/12) * 24 / 1_000_000
    N2O_pre_descarte_mgN_por_kg = 20.26
    N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
    N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000
    
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

def calcular_emissoes_pre_descarte(O2_concentracao, residuos_kg_dia, dias_simulacao):
    PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)
    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)
    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (residuos_kg_dia * n2o_ajustado * fracao / 1000)
    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, residuos_kg_dia, h_exposta, dias_simulacao, incluir_pre_descarte=True):
    umidade_val, temp_val, doc_val, massa_exp_val, k_ano_val = params
    MCF = 1; F = 0.5; Ri = 0.0; OX = 0.1
    PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}
    fator_umid = (1 - umidade_val) / (1 - 0.55)
    f_aberto = np.clip((massa_exp_val / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    docf_calc = 0.0147 * temp_val + 0.28
    potencial_CH4_por_kg = doc_val * docf_calc * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg
    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano_val * (t - 1) / 365.0) - np.exp(-k_ano_val * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario
    E_aberto = 1.91; E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia
    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]
    
    total_ch4_aterro_kg, total_n2o_aterro_kg = emissoes_CH4, emissoes_N2O
    
    if incluir_pre_descarte:
        O2_concentracao = 21
        emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, residuos_kg_dia, dias_simulacao)
        total_ch4_aterro_kg += emissoes_CH4_pre_descarte_kg
        total_n2o_aterro_kg += emissoes_N2O_pre_descarte_kg
    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_vermi(params, residuos_kg_dia, dias_simulacao):
    umidade_val, temp_val, doc_val, ch4_frac, n2o_frac = params
    TOC_YANG = 0.436; TN_YANG = 14.2 / 1000
    fracao_ms = 1 - umidade_val
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * ch4_frac * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * n2o_frac * (44/28) * fracao_ms)
    
    DIAS_COMPOSTAGEM = 50
    PERFIL_CH4_VERMI = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001])
    PERFIL_N2O_VERMI = np.array([0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()
    PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()
    
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_VERMI[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_VERMI[dia_compostagem]
    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem(params, residuos_kg_dia, dias_simulacao):
    umidade_val, temp_val, doc_val, ch4_frac, n2o_frac = params
    TOC_YANG = 0.436; TN_YANG = 14.2 / 1000
    fracao_ms = 1 - umidade_val
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * ch4_frac * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * n2o_frac * (44/28) * fracao_ms)
    
    DIAS_COMPOSTAGEM = 50
    PERFIL_CH4_THERMO = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    PERFIL_N2O_THERMO = np.array([0.10, 0.08, 0.15, 0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()
    PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()
    
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_THERMO[dia_compostagem]
    return emissoes_CH4, emissoes_N2O

def run_single_simulation(residuos_kg_dia, umidade, massa_exposta_kg, h_exposta, anos_simulacao, k_ano, ch4_frac_vermi, n2o_frac_vermi, ch4_frac_thermo, n2o_frac_thermo, gwp_ch4, gwp_n2o):
    dias_simulacao = anos_simulacao * 365
    
    # Parâmetros IPCC 2006 (aterro)
    temp_aterro = 25  # T em ºC
    doc_aterro = 0.15 # Carbono orgânico degradável (fração)
    params_aterro = (umidade, temp_aterro, doc_aterro, massa_exposta_kg, k_ano)
    
    # Parâmetros vermicompostagem (Tese)
    params_vermi = (umidade, temp_aterro, doc_aterro, ch4_frac_vermi, n2o_frac_vermi)
    
    # Parâmetros compostagem termofílica (UNFCCC)
    params_thermo = (umidade, temp_aterro, doc_aterro, ch4_frac_thermo, n2o_frac_thermo)
    
    # Calcular emissões diárias para cada cenário
    em_ch4_aterro, em_n2o_aterro = calcular_emissoes_aterro(params_aterro, residuos_kg_dia, h_exposta, dias_simulacao)
    em_ch4_vermi, em_n2o_vermi = calcular_emissoes_vermi(params_vermi, residuos_kg_dia, dias_simulacao)
    em_ch4_thermo, em_n2o_thermo = calcular_emissoes_compostagem(params_thermo, residuos_kg_dia, dias_simulacao)
    
    # Converter para tCO₂eq
    em_aterro_co2eq_t = (em_ch4_aterro * gwp_ch4 + em_n2o_aterro * gwp_n2o) / 1000
    em_vermi_co2eq_t = (em_ch4_vermi * gwp_ch4 + em_n2o_vermi * gwp_n2o) / 1000
    em_thermo_co2eq_t = (em_ch4_thermo * gwp_ch4 + em_n2o_thermo * gwp_n2o) / 1000
    
    return em_aterro_co2eq_t, em_vermi_co2eq_t, em_thermo_co2eq_t

def run_simulations_parallel(n_simulations, residuos_kg_dia, umidade, massa_exposta_kg, h_exposta, anos_simulacao, k_ano_val, ch4_frac_vermi, n2o_frac_vermi, ch4_frac_thermo, n2o_frac_thermo, gwp_ch4, gwp_n2o):
    results = Parallel(n_jobs=-1)(delayed(run_single_simulation)(
        residuos_kg_dia, umidade, massa_exposta_kg, h_exposta, anos_simulacao, k_ano_val, ch4_frac_vermi, n2o_frac_vermi, ch4_frac_thermo, n2o_frac_thermo, gwp_ch4, gwp_n2o
    ) for _ in range(n_simulations))
    
    em_aterro = np.array([r[0] for r in results])
    em_vermi = np.array([r[1] for r in results])
    em_thermo = np.array([r[2] for r in results])
    
    return em_aterro, em_vermi, em_thermo

# =============================================
# 3. BARRA LATERAL (ENTRADA DE PARÂMETROS)
# =============================================
with st.sidebar:
    st.header("Parâmetros de Entrada")
    
    residuos_kg_dia = st.slider("Quantidade de resíduos (kg/dia)", min_value=10, max_value=1000, value=100, step=10)
    
    st.subheader("Parâmetros Operacionais")
    umidade_valor = st.slider("Umidade do resíduo (%)", 50, 95, 85, 1)
    umidade = umidade_valor / 100.0
    
    massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10)
    h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1)
    
    st.subheader("Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    
    st.markdown("---")
    st.subheader("Parâmetros Fixos")
    st.markdown(f"- Constante de decaimento (k): **0.06**")
    st.markdown(f"- Fração de CH₄ (Vermicompostagem): **0.13%**")
    st.markdown(f"- Fração de CH₄ (Compostagem): **0.6%**")
    st.markdown(f"- Fração de N₂O (Vermicompostagem): **0.92%**")
    st.markdown(f"- Fração de N₂O (Compostagem): **1.96%**")
    
    if st.button("Executar Simulação"):
        st.session_state.run_simulation = True
    else:
        st.session_state.run_simulation = False

# =============================================
# 4. VARIÁVEIS FIXAS
# =============================================
k_ano_val = 0.06
GWP_CH4 = 79.7
GWP_N2O = 273
CH4_C_FRAC_VERMI = 0.13 / 100
N2O_N_FRAC_VERMI = 0.92 / 100
CH4_C_FRAC_THERMO = 0.006
N2O_N_FRAC_THERMO = 0.0196

# =============================================
# 5. EXECUÇÃO DA SIMULAÇÃO
# =============================================
if st.session_state.get("run_simulation", False):
    with st.spinner("Executando simulações... Isso pode levar alguns minutos."):
        emissoes_aterro_t, emissoes_vermi_t, emissoes_thermo_t = run_simulations_parallel(
            n_simulations, residuos_kg_dia, umidade, massa_exposta_kg, h_exposta, anos_simulacao, k_ano_val,
            CH4_C_FRAC_VERMI, N2O_N_FRAC_VERMI, CH4_C_FRAC_THERMO, N2O_N_FRAC_THERMO,
            GWP_CH4, GWP_N2O
        )
    
    # Reduções e Estatísticas
    reducao_vermi_t = emissoes_aterro_t - emissoes_vermi_t
    reducao_thermo_t = emissoes_aterro_t - emissoes_thermo_t

    media_vermi = np.mean(reducao_vermi_t)
    media_thermo = np.mean(reducao_thermo_t)

    # Teste T Pareado
    t_stat_pareado, p_ttest_pareado = stats.ttest_rel(reducao_vermi_t, reducao_thermo_t)

    # Teste de Wilcoxon (não paramétrico)
    wilcoxon_stat, p_wilcoxon = stats.wilcoxon(reducao_vermi_t, reducao_thermo_t)

    st.success("Simulação concluída!")

    # =============================================
    # 6. EXIBIÇÃO DE RESULTADOS
    # =============================================
    
    # 6.1 Resumo dos resultados
    st.header("Resultados da Simulação")
    st.subheader("Resumo das Emissões Evitadas (tCO₂eq)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Redução Média (Proposta da Tese - Vermicompostagem)", value=f"{formatar_br(media_vermi)} tCO₂eq")
    with col2:
        st.metric(label="Redução Média (UNFCCC - Compostagem)", value=f"{formatar_br(media_thermo)} tCO₂eq")
    
    st.markdown("---")

    # 6.2 Análise estatística
    st.subheader("Análise Estatística da Diferença de Reduções")
    st.markdown(f"""
    - **Diferença Média (Tese vs UNFCCC):** {formatar_br(media_vermi - media_thermo)} tCO₂eq
    - **Teste T Pareado (p-value):** {p_ttest_pareado:.4f}
    - **Teste de Wilcoxon (p-value):** {p_wilcoxon:.4f}
    """)
    if p_ttest_pareado < 0.05 and p_wilcoxon < 0.05:
        st.success("A diferença nas reduções de emissões entre os dois cenários é estatisticamente significativa.")
    else:
        st.warning("A diferença não é estatisticamente significativa ao nível de 95% de confiança.")

    st.markdown("---")

    # 6.3 Gráficos
    st.subheader("Visualização dos Resultados")
    
    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = pd.DataFrame({
        'Vermicompostagem': reducao_vermi_t,
        'Compostagem': reducao_thermo_t
    })
    sns.boxplot(data=data, ax=ax)
    ax.set_title('Distribuição das Emissões Evitadas')
    ax.set_ylabel('tCO₂eq')
    st.pyplot(fig)
    
    # Histograma das diferenças
    fig, ax = plt.subplots(figsize=(10, 6))
    diferencas = reducao_vermi_t - reducao_thermo_t
    sns.histplot(diferencas, bins=30, kde=True, ax=ax)
    ax.set_xlabel('Diferença de Redução (Vermicompostagem - Compostagem) em tCO₂eq')
    ax.set_ylabel('Frequência')
    ax.set_title('Histograma da Diferença nas Reduções de Emissões')
    st.pyplot(fig)

    # Gráfico de série temporal
    st.subheader("Série Temporal de Emissões")
    
    # Média ao longo do tempo
    df_series = pd.DataFrame({
        'Aterro': np.mean(emissoes_aterro_t, axis=0),
        'Vermicompostagem': np.mean(emissoes_vermi_t, axis=0),
        'Compostagem': np.mean(emissoes_thermo_t, axis=0)
    })
    
    # Agrupamento para média anual
    dias = anos_simulacao * 365
    datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
    df_series['Date'] = datas
    df_series.set_index('Date', inplace=True)
    df_anual = df_series.resample('A').sum().reset_index()
    df_anual['Year'] = df_anual['Date'].dt.year
    df_anual.drop('Date', axis=1, inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_anual['Year'], df_anual['Aterro'], label='Aterro Sanitário (Baseline)', marker='o')
    ax.plot(df_anual['Year'], df_anual['Vermicompostagem'], label='Vermicompostagem (Tese)', marker='x')
    ax.plot(df_anual['Year'], df_anual['Compostagem'], label='Compostagem Termofílica (UNFCCC)', marker='^')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Emissões Anuais (tCO₂eq)')
    ax.set_title('Comparação Anual de Emissões por Cenário')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(br_format))
    st.pyplot(fig)

    # 6.4 Tabelas
    st.subheader("Dados Anuais (Valores Médios)")
    df_anual_formatado = df_anual.copy()
    for col in df_anual_formatado.columns:
        if col != 'Year':
            df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)
    st.dataframe(df_anual_formatado)

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

# Rodapé
st.markdown("---")
st.markdown("""
**Referências:**

**Cenário de Baseline (Aterro Sanitário):**
- IPCC (2006). Guidelines for National Greenhouse Gas Inventories.
- UNFCCC (2016). Tool to determine methane emissions from solid waste disposal sites.
- Wang et al. (2017). Nitrous oxide emissions from landfills.
- Feng et al. (2020). Gaseous emissions during waste sorting.
- Wang et al. (2023). A review of methane generation models for landfills.

**Cenário Proposta da Tese (Vermicompostagem):**
- Yang et al. (2017). Methane and nitrous oxide emissions from different composting systems.

**Cenário UNFCCC (Compostagem Termofílica):**
- Yang et al. (2017). Methane and nitrous oxide emissions from different composting systems.
""")
