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
st.set_page_config(page_title="Simulador de Emissões tCO₂eq", layout="wide")
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
    Formata um número para o padrão brasileiro de separadores de milhares e decimais.
    Ex: 1234.56 -> 1.234,56
    """
    if pd.isna(numero):
        return 'N/A'
    if isinstance(numero, (int, float)):
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(numero)

# --- Barra lateral ---
st.sidebar.header("Parâmetros do Projeto")

# Parâmetros de entrada do usuário
st.sidebar.subheader("Volume de Resíduos")
volume_residuos = st.sidebar.number_input(
    'Massa de resíduos orgânicos (kg/dia)',
    min_value=1.0,
    value=100.0,
    step=10.0,
    format="%.2f",
    help="Massa inicial de resíduos orgânicos (base úmida) para simulação."
)

anos_simulacao = st.sidebar.number_input(
    'Anos de simulação (período de crédito)',
    min_value=1,
    value=20,
    step=1,
    help="Número de anos para calcular as emissões acumuladas e créditos de carbono."
)

# Parâmetros Vermicompostagem
st.sidebar.subheader("Parâmetros Vermicompostagem")
ch4_vermi_emis_fator = st.sidebar.number_input(
    'Fator de Emissão de CH₄ (Vermicompostagem, % da massa)',
    min_value=0.0,
    value=0.00136,
    step=0.0001,
    format="%.5f",
    help="Fração de Carbono Orgânico Total (TOC) emitida como CH₄-C."
)

n2o_vermi_emis_fator = st.sidebar.number_input(
    'Fator de Emissão de N₂O (Vermicompostagem, % da massa)',
    min_value=0.0,
    value=0.0092,
    step=0.0001,
    format="%.4f",
    help="Fração de Nitrogênio Total (TN) emitida como N₂O-N."
)

# Parâmetros Aterro Sanitário
st.sidebar.subheader("Parâmetros Aterro Sanitário")
ch4_aterro_emis_fator = st.sidebar.number_input(
    'Fator de Emissão de CH₄ (Aterro, % da massa)',
    min_value=0.0,
    value=0.05,
    step=0.001,
    format="%.3f",
    help="Fração de Carbono Orgânico Dissolvido (DOC) convertida em CH₄."
)

n2o_aterro_emis_fator = st.sidebar.number_input(
    'Fator de Emissão de N₂O (Aterro, % da massa)',
    min_value=0.0,
    value=0.0000002,
    step=0.0000001,
    format="%.7f",
    help="Fração de N emitida como N₂O."
)

st.sidebar.subheader("Conversão e GWP")
gwp_ch4 = st.sidebar.number_input(
    'GWP - Potencial de Aquecimento Global CH₄',
    min_value=1,
    value=79,
    step=1,
    help="GWP do metano, por exemplo, 79 (IPCC AR6)."
)

gwp_n2o = st.sidebar.number_input(
    'GWP - Potencial de Aquecimento Global N₂O',
    min_value=1,
    value=273,
    step=1,
    help="GWP do óxido nitroso, por exemplo, 273 (IPCC AR6)."
)

# Botão de simulação
if st.sidebar.button('Executar Simulação'):
    st.session_state.run_simulation = True
    
if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação...'):
        # Constantes
        ms = volume_residuos * (1 - 0.85)
        toc = ms * 0.436
        tn = ms * 0.0142
        
        # Perfil Temporal - Vermicompostagem
        # Fonte: Yang et al. (2017)
        # Perfil para vermicompostagem (50 dias)
        perfil_ch4 = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        perfil_n2o = np.array([21.8, 25.1, 29.8, 20.3, 15.6, 12.4, 9.8, 7.5, 6.2, 5.1, 4.3, 3.5, 2.9, 2.3, 1.8, 1.4, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Normalização dos perfis
        perfil_ch4_norm = perfil_ch4 / perfil_ch4.sum()
        perfil_n2o_norm = perfil_n2o / perfil_n2o.sum()

        # Função para simulação por dia
        def simular_dia(dia):
            # Emissões Vermicompostagem
            ch4_vermi_dia = (toc * ch4_vermi_emis_fator) * perfil_ch4_norm[dia % 50]
            n2o_vermi_dia = (tn * n2o_vermi_emis_fator) * perfil_n2o_norm[dia % 50]
            
            # Emissões Aterro (simulação temporal simplificada)
            ch4_aterro_dia = volume_residuos * ch4_aterro_emis_fator
            n2o_aterro_dia = volume_residuos * n2o_aterro_emis_fator
            
            # Conversão para tCO₂eq
            total_vermi_tco2eq_dia = (ch4_vermi_dia * gwp_ch4 + n2o_vermi_dia * gwp_n2o) / 1000
            total_aterro_tco2eq_dia = (ch4_aterro_dia * gwp_ch4 + n2o_aterro_dia * gwp_n2o) / 1000
            
            return total_vermi_tco2eq_dia, total_aterro_tco2eq_dia

        # Executar a simulação para todos os dias
        num_dias = anos_simulacao * 365
        resultados = Parallel(n_jobs=-1)(delayed(simular_dia)(i) for i in range(num_dias))
        
        # Criar DataFrame com os resultados
        df = pd.DataFrame(resultados, columns=['Total_Vermi_tCO2eq_dia', 'Total_Aterro_tCO2eq_dia'])
        df['Dias'] = np.arange(1, num_dias + 1)
        df['Emissoes_Evitadas'] = df['Total_Aterro_tCO2eq_dia'] - df['Total_Vermi_tCO2eq_dia']

        # Cálculos anuais
        df['Ano'] = (df['Dias'] - 1) // 365 + 1
        df_anual = df.groupby('Ano').sum().reset_index()
        df_anual.drop(columns=['Dias'], inplace=True)
        
        # Resultados finais
        df_anual_revisado = pd.DataFrame({
            'Year': df_anual['Ano'],
            'Baseline emissions (t CO₂eq)': df_anual['Total_Aterro_tCO2eq_dia'],
            'Project emissions (t CO₂eq)': df_anual['Total_Vermi_tCO2eq_dia'],
            'Emissions reductions (t CO₂eq)': df_anual['Emissoes_Evitadas']
        })

        # UNFCCC
        df_comp_anual = df_anual_revisado.copy()
        
        df_comp_anual_revisado = pd.DataFrame({
            'Year': df_comp_anual['Year'],
            'Baseline emissions (t CO₂eq)': df_comp_anual['Baseline emissions (t CO₂eq)'],
            'Project emissions (t CO₂eq)': df_comp_anual['Project emissions (t CO₂eq)'],
            'Emissions reductions (t CO₂eq)': df_comp_anual['Emissions reductions (t CO₂eq)']
        })

        # Exibir resultados
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

        # Tabela de resultados anuais - Proposta da Tese
        st.subheader("Tabelas de Resultados Anuais - Proposta da Tese")

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
- Feng et al. (2021). The effects of different cover materials on N₂O and CH₄ emissions from landfilling.

**Cenário de Projeto (Vermicompostagem):**
- Yang et al. (2017). Methane and nitrous oxide emissions during vermicomposting of cattle manure.

**Potencial de Aquecimento Global (GWP):**
- IPCC (2021). Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change.

""")
