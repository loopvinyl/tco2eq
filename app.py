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

# Função para formatar números no padrão brasileiro
def formatar_br(numero):
    """
    Formata um número para o padrão brasileiro (separador de milhares como ponto e decimal como vírgula).
    """
    if isinstance(numero, (float, np.float64)):
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return numero

# Função para gerar a série temporal de emissões de CH4 da vermicompostagem
@st.cache_data(show_spinner=False)
def get_vermi_ch4_temporal_profile(dias_operacao):
    """
    Gera a série temporal de emissões de CH4 da vermicompostagem baseada em Yang et al. (2017).
    """
    dias_vermicompostagem = 50
    pico_dia = 1
    # Distribuição que simula o perfil de emissão, com pico no início
    x = np.linspace(-3, 10, dias_vermicompostagem)
    y = np.exp(-((x - (pico_dia-1)) / 1.5)**2) + np.exp(-((x - (pico_dia-1)) / 10)**2)
    y_normalizada = y / np.sum(y)
    
    # Repete o perfil para todo o período de operação
    perfil_temporal = np.tile(y_normalizada, int(np.ceil(dias_operacao / dias_vermicompostagem)))[:dias_operacao]
    
    return perfil_temporal

# Função para gerar a série temporal de emissões de N2O da vermicompostagem
@st.cache_data(show_spinner=False)
def get_vermi_n2o_temporal_profile(dias_operacao):
    """
    Gera a série temporal de emissões de N2O da vermicompostagem baseada em Yang et al. (2017).
    """
    dias_vermicompostagem = 50
    pico_dia = 3
    # Distribuição que simula o perfil de emissão, com pico no terceiro dia
    x = np.arange(dias_vermicompostagem)
    # Pico no dia 3
    y = np.exp(-0.2 * (x - pico_dia)**2) + 0.1 * np.exp(-0.05 * (x - pico_dia)**2)
    y_normalizada = y / np.sum(y)
    
    # Repete o perfil para todo o período de operação
    perfil_temporal = np.tile(y_normalizada, int(np.ceil(dias_operacao / dias_vermicompostagem)))[:dias_operacao]
    
    return perfil_temporal

# Funções de cálculo de emissões
@st.cache_data(show_spinner=False)
def calculate_landfill_emissions(parametros, dias_operacao, **kwargs):
    """
    Calcula as emissões de metano do aterro sanitário ao longo do tempo.
    """
    
    # Desempacotar os parâmetros
    residuos_ano = parametros.get('residuos_ano', 0)
    fração_organica = parametros.get('fração_organica', 0)
    k_constante = parametros.get('k_constante', 0)
    taxa_N2O = parametros.get('taxa_N2O', 0)
    eficiencia_ch4 = parametros.get('eficiencia_ch4', 0)
    L0_fator = parametros.get('L0_fator', 0)
    GWP_CH4_Tese = parametros.get('GWP_CH4_Tese', 0)
    GWP_N2O_Tese = parametros.get('GWP_N2O_Tese', 0)

    # Conversão de unidades
    emissao_anual_toneladas = residuos_ano * fração_organica
    emissao_diaria_kg = (emissao_anual_toneladas * 1000) / 365
    
    # Emissões de CH4
    emissao_diaria_ch4 = emissao_diaria_kg * L0_fator * k_constante
    decay_series = np.exp(-k_constante * np.arange(dias_operacao))
    emissao_ch4_dia_kg = fftconvolve(np.ones(dias_operacao) * emissao_diaria_ch4, decay_series, 'full')[:dias_operacao]

    # Emissões de N2O
    emissao_n2o_dia_kg = np.ones(dias_operacao) * taxa_N2O

    # Total em tCO2eq
    emissao_ch4_tco2eq = (emissao_ch4_dia_kg * GWP_CH4_Tese) / 1000
    emissao_n2o_tco2eq = (emissao_n2o_dia_kg * GWP_N2O_Tese) / 1000
    emissao_total_tco2eq = emissao_ch4_tco2eq + emissao_n2o_tco2eq
    
    return emissao_total_tco2eq

@st.cache_data(show_spinner=False)
def calculate_compost_emissions(parametros, dias_operacao, context, **kwargs):
    """
    Calcula as emissões de gases da compostagem (metodologia UNFCCC) ou vermicompostagem (Tese).
    """

    # Desempacotar os parâmetros
    residuos_ano = parametros.get('residuos_ano', 0)
    fator_CH4 = parametros.get('fator_CH4', 0)
    fator_N2O = parametros.get('fator_N2O', 0)
    GWP_CH4_UNFCCC = parametros.get('GWP_CH4_UNFCCC', 0)
    GWP_N2O_UNFCCC = parametros.get('GWP_N2O_UNFCCC', 0)
    GWP_CH4_Tese = parametros.get('GWP_CH4_Tese', 0)
    GWP_N2O_Tese = parametros.get('GWP_N2O_Tese', 0)

    # Conversão de unidades
    emissao_anual_toneladas = residuos_ano
    emissao_diaria_toneladas = emissao_anual_toneladas / 365
    emissao_diaria_kg = emissao_diaria_toneladas * 1000

    if context == 'UNFCCC':
        # Emissões UNFCCC (compostagem)
        emissao_ch4_dia_kg = emissao_diaria_kg * fator_CH4
        emissao_n2o_dia_kg = emissao_diaria_kg * fator_N2O
        
        emissao_ch4_tco2eq = (emissao_ch4_dia_kg * GWP_CH4_UNFCCC) / 1000
        emissao_n2o_tco2eq = (emissao_n2o_dia_kg * GWP_N2O_UNFCCC) / 1000
        
        emissao_total_tco2eq = emissao_ch4_tco2eq + emissao_n2o_tco2eq
        
        return np.ones(dias_operacao) * emissao_total_tco2eq

    elif context == 'Tese':
        
        # Emissões Tese (vermicompostagem)
        fator_CH4_vermi_dia = 0.00022672
        fator_N2O_vermi_dia = 0.00006159
        
        emissao_ch4_total_50dias_kg = 0.011336
        emissao_n2o_total_50dias_kg = 0.00307937
        
        perfil_ch4 = get_vermi_ch4_temporal_profile(dias_operacao)
        perfil_n2o = get_vermi_n2o_temporal_profile(dias_operacao)
        
        # Escalar as emissões diárias pelo perfil temporal e para a quantidade de resíduos
        emissao_ch4_dia_kg = perfil_ch4 * (emissao_ch4_total_50dias_kg * (residuos_ano/365)/100)
        emissao_n2o_dia_kg = perfil_n2o * (emissao_n2o_total_50dias_kg * (residuos_ano/365)/100)
        
        emissao_ch4_tco2eq = (emissao_ch4_dia_kg * GWP_CH4_Tese) / 1000
        emissao_n2o_tco2eq = (emissao_n2o_dia_kg * GWP_N2O_Tese) / 1000
        
        emissao_total_tco2eq = emissao_ch4_tco2eq + emissao_n2o_tco2eq

        return emissao_total_tco2eq

# Sidebar para input de dados
st.sidebar.header("Parâmetros do Projeto")
# Definir valores padrão
params_default = {
    'residuos_ano': 36500.0,
    'anos_projeto': 20,
    'fração_organica': 0.5,
    'k_constante': 0.05,
    'L0_fator': 0.15,
    'taxa_N2O': 0.001,
    'eficiencia_ch4': 0.0,  # Corrigido para float
    'fator_CH4': 0.004,
    'fator_N2O': 0.001,
    'GWP_CH4_Tese': 79.7,
    'GWP_N2O_Tese': 273,
    'GWP_CH4_UNFCCC': 25,
    'GWP_N2O_UNFCCC': 298
}

params = {}
params['residuos_ano'] = st.sidebar.number_input(
    'Massa de resíduos orgânicos anuais (t)', 
    min_value=1.0, 
    value=params_default['residuos_ano'], 
    step=100.0, 
    help="Quantidade anual de resíduos orgânicos processados (toneladas)."
)

params['anos_projeto'] = st.sidebar.number_input(
    'Anos do projeto', 
    min_value=1, 
    value=params_default['anos_projeto'], 
    step=1
)

st.sidebar.subheader("Parâmetros do Cenário de Baseline (Aterro Sanitário)")
params['fração_organica'] = st.sidebar.slider(
    'Fração de resíduos orgânicos decomponível', 
    min_value=0.0, 
    max_value=1.0, 
    value=params_default['fração_organica'], 
    step=0.01
)
params['k_constante'] = st.sidebar.slider(
    'Constante k (decomposição)', 
    min_value=0.0, 
    max_value=1.0, 
    value=params_default['k_constante'], 
    step=0.01
)
params['L0_fator'] = st.sidebar.slider(
    'Fator L0 (máximo CH4)', 
    min_value=0.0, 
    max_value=1.0, 
    value=params_default['L0_fator'], 
    step=0.01
)
params['eficiencia_ch4'] = st.sidebar.slider(
    'Eficiência de recuperação de CH4', 
    min_value=0.0, 
    max_value=1.0, 
    value=params_default['eficiencia_ch4'], 
    step=0.01
)
params['taxa_N2O'] = st.sidebar.number_input(
    'Taxa de emissão de N2O (kg/t resíduo)', 
    min_value=0.0, 
    value=params_default['taxa_N2O'], 
    step=0.001
)

st.sidebar.subheader("Parâmetros da Tecnologia Ambiental")
params['fator_CH4'] = st.sidebar.number_input(
    'Fator de Emissão de CH4 (UNFCCC)', 
    min_value=0.0, 
    value=params_default['fator_CH4'], 
    step=0.001,
    help="Aplica-se ao cálculo de compostagem pela metodologia UNFCCC."
)
params['fator_N2O'] = st.sidebar.number_input(
    'Fator de Emissão de N2O (UNFCCC)', 
    min_value=0.0, 
    value=params_default['fator_N2O'], 
    step=0.001,
    help="Aplica-se ao cálculo de compostagem pela metodologia UNFCCC."
)

st.sidebar.subheader("Valores de GWP (Potencial de Aquecimento Global)")
params['GWP_CH4_Tese'] = st.sidebar.number_input(
    'GWP do CH4 (Tese)', 
    min_value=1.0, 
    value=params_default['GWP_CH4_Tese'], 
    step=1.0
)
params['GWP_N2O_Tese'] = st.sidebar.number_input(
    'GWP do N2O (Tese)', 
    min_value=1.0, 
    value=params_default['GWP_N2O_Tese'], 
    step=1.0
)
params['GWP_CH4_UNFCCC'] = st.sidebar.number_input(
    'GWP do CH4 (UNFCCC)', 
    min_value=1.0, 
    value=params_default['GWP_CH4_UNFCCC'], 
    step=1.0
)
params['GWP_N2O_UNFCCC'] = st.sidebar.number_input(
    'GWP do N2O (UNFCCC)', 
    min_value=1.0, 
    value=params_default['GWP_N2O_UNFCCC'], 
    step=1.0
)

# Botão de simulação
if st.sidebar.button('Executar Simulação'):
    st.session_state.run_simulation = True

# Execução da simulação
if st.session_state.get('run_simulation', False):
    dias_operacao = params['anos_projeto'] * 365

    # Cálculo das emissões
    emissao_aterro = calculate_landfill_emissions(params, dias_operacao)
    emissao_vermicompostagem = calculate_compost_emissions(params, dias_operacao, context='Tese')
    emissao_compostagem = calculate_compost_emissions(params, dias_operacao, context='UNFCCC')

    # Anos e dias
    dias = np.arange(1, dias_operacao + 1)
    anos = dias / 365

    # Criar DataFrame com os resultados diários
    df = pd.DataFrame({
        'Dia': dias,
        'Ano': anos,
        'Total_Aterro_tCO2eq_dia': emissao_aterro,
        'Total_Vermi_tCO2eq_dia': emissao_vermicompostagem
    })
    
    # Criar o DataFrame com os resultados anuais
    df_anual = df.groupby(np.floor(df['Ano'])).sum()
    df_anual['Year'] = df_anual.index.astype(int)
    df_anual.rename(columns={'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
                             'Total_Vermi_tCO2eq_dia': 'Project emissions (t CO₂eq)'}, inplace=True)
    df_anual_revisado = df_anual[['Year', 'Baseline emissions (t CO₂eq)', 'Project emissions (t CO₂eq)']].copy()
    
    # Criar o DataFrame de comparação anual (UNFCCC)
    df_comp = pd.DataFrame({
        'Dia': dias,
        'Total_Aterro_tCO2eq_dia': emissao_aterro,
        'Total_Compost_tCO2eq_dia': emissao_compostagem
    })

    df_comp_anual = df_comp.groupby(np.floor(df_comp['Dia']/365)).sum()
    df_comp_anual['Year'] = df_comp_anual.index.astype(int) + 1
    df_comp_anual.rename(columns={'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
                                  'Total_Compost_tCO2eq_dia': 'Project emissions (t CO₂eq)'}, inplace=True)
    df_comp_anual_revisado = df_comp_anual[['Year', 'Baseline emissions (t CO₂eq)', 'Project emissions (t CO₂eq)']].copy()

    # Exibição dos resultados
    st.header("Resultados da Simulação")

    # Cálculos para exibição dos percentuais
    emissao_aterro_dia_1 = df['Total_Aterro_tCO2eq_dia'][0]
    emissao_vermi_dia_1 = df['Total_Vermi_tCO2eq_dia'][0]
    
    # Redução no primeiro dia (perfil temporal)
    reducao_vermi_dia_1 = (emissao_aterro_dia_1 - emissao_vermi_dia_1) / emissao_aterro_dia_1 * 100
    
    # Redução total em 20 anos
    emissao_aterro_total = df_anual_revisado['Baseline emissions (t CO₂eq)'].sum()
    emissao_vermi_total = df_anual_revisado['Project emissions (t CO₂eq)'].sum()
    reducao_vermi_total = (emissao_aterro_total - emissao_vermi_total) / emissao_aterro_total * 100
    
    # Exibição dos percentuais de redução
    st.subheader("Redução de Emissões - Vermicompostagem vs. Aterro Sanitário")
    
    st.markdown(f"""
    Com base nos resultados da simulação, a vermicompostagem se destaca como uma tecnologia ambiental que pode reduzir significativamente as emissões de gases de efeito estufa.
    
    * **Primeiro dia:**
        * Emissões diárias médias (exemplo didático): redução de **89,0%**.
        * Emissões com perfil temporal: redução de **{formatar_br(reducao_vermi_dia_1)}%**.
    * **Redução acumulada em 20 anos:**
        * As emissões totais são reduzidas em **{formatar_br(reducao_vermi_total)}%**.
    """)

    st.subheader("Emissões Evitadas (Tese)")
    
    # Exibir valor total de emissões evitadas (tCO2eq)
    total_evitado = emissao_aterro.sum() - emissao_vermicompostagem.sum()
    st.metric("Total de emissões evitadas (Tese)", f"{total_evitado:,.2f} tCO₂eq".replace(",", "X").replace(".", ",").replace("X", "."))

    # Tabela de resultados anuais - Tese
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
- Feng et al. (20...
""")
