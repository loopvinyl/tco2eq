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

# Emissões pré-descarte (Feng et al. 2020)
CH4_pre_descarte_ugC_por_kg_h_media = 2.78  # µg C/kg/h (média)
fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000  # g CH4/kg/dia

N2O_pre_descarte_mgN_por_kg = 20.26  # mg N/kg wet waste (total em 72h)
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3  # mg N/kg/dia (média diária)
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000  # g N2O/kg/dia

# Distribuição temporal das emissões de N2O no pré-descarte
PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Perfil temporal N2O (Wang et al. 2017)
PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

# Fatores de emissão baseados em Zhu-Barker et al. (2017)
EF_CH4_COMPOST_MEDIA = 0.000546816  # kg CH4 / kg WW / dia
EF_CH4_COMPOST_DP = 0.000500  # kg CH4 / kg WW / dia
EF_N2O_COMPOST_MEDIA = 0.000742912   # kg N2O / kg WW / dia
EF_N2O_COMPOST_DP = 0.000569  # kg N2O / kg WW / dia

# Período de Simulação
dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')

# Funções de cálculo (adaptadas do código original)
def ajustar_emissoes_pre_descarte(O2_concentracao):
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
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)

    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_kg_dia * n2o_ajustado * fracao / 1000
                )

    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val, massa_exp_val, k_ano_val = params

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

    E_aberto = 1.91
    E_fechado = 2.15
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia

    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]

    O2_concentracao = 21
    emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao)

    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg

    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_vermi(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val, ch4_frac, n2o_frac = params
    fracao_ms = 1 - umidade_val
    dias_comp = DIAS_COMPOSTAGEM
    
    ch4_total_por_lote = residuos_kg_dia * (TOC_YANG * ch4_frac * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_YANG * n2o_frac * (44/28) * fracao_ms)
    
    emissao_diaria_por_lote_ch4 = ch4_total_por_lote / dias_comp
    emissao_diaria_por_lote_n2o = n2o_total_por_lote / dias_comp
    
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(dias_comp):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += emissao_diaria_por_lote_ch4
                emissoes_N2O[dia_emissao] += emissao_diaria_por_lote_n2o
                
    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem(params, dias_simulacao=dias, dias_compostagem=62):
    umidade, T, DOC, k_ano = params

    ef_ch4 = np.random.normal(EF_CH4_COMPOST_MEDIA, EF_CH4_COMPOST_DP)
    ef_n2o = np.random.normal(EF_N2O_COMPOST_MEDIA, EF_N2O_COMPOST_DP)

    emissao_diaria_por_lote_ch4 = residuos_kg_dia * ef_ch4 / dias_compostagem
    emissao_diaria_por_lote_n2o = residuos_kg_dia * ef_n2o / dias_compostagem

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_comp in range(dias_compostagem):
            dia_emissao = dia_entrada + dia_comp
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += emissao_diaria_por_lote_ch4
                emissoes_N2O[dia_emissao] += emissao_diaria_por_lote_n2o

    return emissoes_CH4, emissoes_N2O

def executar_simulacao_completa(parametros):
    umidade, T, DOC, k_ano, CH4_C_FRAC, N2O_N_FRAC = parametros
    params_aterro = (umidade, T, DOC, 100, k_ano)
    params_vermi = (umidade, T, DOC, CH4_C_FRAC, N2O_N_FRAC)

    ch4_aterro, n2o_aterro = calcular_emissoes_aterro(params_aterro)
    ch4_vermi, n2o_vermi = calcular_emissoes_vermi(params_vermi)

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()
    return reducao_tco2eq

def executar_simulacao_unfccc(parametros):
    umidade, T, DOC, k_ano = parametros

    params_aterro = (umidade, T, DOC, 100, k_ano)
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro(params_aterro)
    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000

    ch4_compost, n2o_compost = calcular_emissoes_compostagem(parametros, dias_simulacao=dias, dias_compostagem=62)
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

# Executar simulação quando solicitado
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # Executar modelo base
        params_base_aterro = (umidade, T, DOC, massa_exposta_kg, k_ano)
        params_base_vermi = (umidade, T, DOC, CH4_C_FRAC_YANG, N2O_N_FRAC_YANG)

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
        df['Reducao_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']

        # Resumo anual
        df['Year'] = df['Data'].dt.year
        df_anual_revisado = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
        }).reset_index()

        df_anual_revisado['Emission reductions (t CO₂eq)'] = df_anual_revisado['Total_Aterro_tCO2eq_dia'] - df_anual_revisado['Total_Vermi_tCO2eq_dia']
        df_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()

        df_anual_revisado.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
            'Total_Vermi_tCO2eq_dia': 'Project emissions (t CO₂eq)',
        }, inplace=True)

        # Cenário UNFCCC
        params_base_unfccc = (umidade, T, DOC, k_ano)
        ch4_compost_UNFCCC, n2o_compost_UNFCCC = calcular_emissoes_compostagem(params_base_unfccc, dias_simulacao=dias, dias_compostagem=62)
        ch4_compost_unfccc_tco2eq = ch4_compost_UNFCCC * GWP_CH4_20 / 1000
        n2o_compost_unfccc_tco2eq = n2o_compost_UNFCCC * GWP_N2O_20 / 1000
        total_compost_unfccc_tco2eq_dia = ch4_compost_unfccc_tco2eq + n2o_compost_unfccc_tco2eq

        df_comp_unfccc_dia = pd.DataFrame({
            'Data': datas,
            'Total_Compost_tCO2eq_dia': total_compost_unfccc_tco2eq_dia
        })
        df_comp_unfccc_dia['Year'] = df_comp_unfccc_dia['Data'].dt.year

        df_comp_anual_revisado = df_comp_unfccc_dia.groupby('Year').agg({
            'Total_Compost_tCO2eq_dia': 'sum'
        }).reset_index()

        df_comp_anual_revisado = pd.merge(df_comp_anual_revisado,
                                         df_anual_revisado[['Year', 'Baseline emissions (t CO₂eq)']],
                                         on='Year', how='left')

        df_comp_anual_revisado['Emission reductions (t CO₂eq)'] = df_comp_anual_revisado['Baseline emissions (t CO₂eq)'] - df_comp_anual_revisado['Total_Compost_tCO2eq_dia']
        df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'] = df_comp_anual_revisado['Emission reductions (t CO₂eq)'].cumsum()
        df_comp_anual_revisado.rename(columns={'Total_Compost_tCO2eq_dia': 'Project emissions (t CO₂eq)'}, inplace=True)

        # Exibir resultados
        st.header("Resultados da Simulação")
        
        col1, col2 = st.columns(2)
        with col1:
            total_evitado_tese = df['Reducao_tCO2eq_acum'].iloc[-1]
            st.metric("Total de emissões evitadas (Tese)", f"{formatar_br(total_evitado_tese)} tCO₂eq")
        with col2:
            total_evitado_unfccc = df_comp_anual_revisado['Cumulative reduction (t CO₂eq)'].iloc[-1]
            st.metric("Total de emissões evitadas (UNFCCC)", f"{formatar_br(total_evitado_unfccc)} tCO₂eq")

        # Gráfico comparativo
        st.subheader("Comparação Anual das Emissões Evitadas")
        df_evitadas_anual = pd.DataFrame({
            'Year': df_anual_revisado['Year'],
            'Proposta da Tese': df_anual_revisado['Emission reductions (t CO₂eq)'],
            'UNFCCC (2012)': df_comp_anual_revisado['Emission reductions (t CO₂eq)']
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        x = np.arange(len(df_evitadas_anual['Year']))
        bar_width = 0.35

        ax.bar(x - bar_width/2, df_evitadas_anual['Proposta da Tese'], width=bar_width,
                label='Proposta da Tese', edgecolor='black')
        ax.bar(x + bar_width/2, df_evitadas_anual['UNFCCC (2012)'], width=bar_width,
                label='UNFCCC (2012)', edgecolor='black', hatch='//')

        # Adicionar valores formatados em cima das barras
        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Proposta da Tese'], 
                                         df_evitadas_anual['UNFCCC (2012)'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Proposta da Tese vs UNFCCC (2012)')
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
        st.subheader("Análise de Sensibilidade Global (Sobol) - Proposta da Tese")
        
        problem_tese = {
            'num_vars': 6,
            'names': ['umidade_residuos', 'temperatura_media', 'doc', 'k_ano', 'CH4_C_FRAC_YANG', 'N2O_N_FRAC_YANG'],
            'bounds': [
                [0.5, 0.85],         # umidade_residuos
                [25.0, 45.0],       # temperatura_media
                [0.15, 0.50],       # doc
                [0.01, 0.47],       # k_ano
                [0.0005, 0.002],    # CH4_C_FRAC_YANG
                [0.005, 0.015],     # N2O_N_FRAC_YANG
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
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Proposta da Tese')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Análise de Sensibilidade Global (Sobol) - CENÁRIO UNFCCC
        st.subheader("Análise de Sensibilidade Global (Sobol) - Cenário UNFCCC")
        
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
        results_unfccc = Parallel(n_jobs=-1)(delayed(executar_simulacao_unfccc)(params) for params in param_values_unfccc)
        Si_unfccc = analyze(problem_unfccc, np.array(results_unfccc), print_to_console=False)
        
        sensibilidade_df_unfccc = pd.DataFrame({
            'Parâmetro': problem_unfccc['names'],
            'S1': Si_unfccc['S1'],
            'ST': Si_unfccc['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_unfccc, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros (Índice Sobol Total) - Cenário UNFCCC')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Análise de Incerteza (Monte Carlo) - PROPOSTA DA TESE
        st.subheader("Análise de Incerteza (Monte Carlo) - Proposta da Tese")
        
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

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_tese, kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(media_tese, color='red', linestyle='--', label=f'Média: {formatar_br(media_tese)} tCO₂eq')
        ax.axvline(intervalo_95_tese[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_tese[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Proposta da Tese')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        # Análise de Incerteza (Monte Carlo) - CENÁRIO UNFCCC
        st.subheader("Análise de Incerteza (Monte Carlo) - Cenário UNFCCC")
        
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
        ax.axvline(intervalo_95_unfccc[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_unfccc[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas (Simulação Monte Carlo) - Cenário UNFCCC')
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

        # Tabela de resultados anuais - Proposta da Tese
        st.subheader("Resultados Anuais - Proposta da Tese")

        # Criar uma cópia para formatação
        df_anual_formatado = df_anual_revisado.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)

        st.dataframe(df_anual_formatado)

        # Tabela de resultados anuais - Metodologia UNFCCC
        st.subheader("Resultados Anuais - Metodologia UNFCCC")

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

**Proposta da Tese (Vermicompostagem):**
- Yang et al. (2017). Greenhouse gas emissions from vermicomposting.

**Cenário UNFCCC (Compostagem):**
- UNFCCC (2012). AMS-III.F - Methodology for compostage.
- Zhu-Barker et al. (2017). Greenhouse gas emissions from composting.
""")
