import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Orçamento de Emissões BR",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🌍 Calculadora de Orçamento de Emissões Brasileiras</h1>', unsafe_allow_html=True)

# Barra lateral para parâmetros
with st.sidebar:
    st.header("⚙️ Parâmetros de Configuração")
    
    st.subheader("Período de Análise")
    ano_inicio = st.number_input("Ano Inicial", min_value=1990, max_value=2030, value=2020, step=1)
    ano_fim = st.number_input("Ano Final", min_value=2021, max_value=2100, value=2050, step=1)
    
    st.subheader("Emissões Atuais (MtCO₂e/ano)")
    emissao_atual = st.number_input("Emissões atuais", min_value=100.0, max_value=5000.0, value=1500.0, step=50.0)
    
    st.subheader("Taxas de Crescimento/Redução (% ao ano)")
    taxa_energia = st.slider("Setor Energia", -10.0, 10.0, 1.5, 0.1)
    taxa_agropecuaria = st.slider("Agropecuária", -10.0, 10.0, 0.8, 0.1)
    taxa_mudanca_uso_solo = st.slider("Mudança Uso Solo", -20.0, 10.0, -3.0, 0.5)
    taxa_processos_industriais = st.slider("Processos Industriais", -5.0, 5.0, 0.5, 0.1)
    taxa_residuos = st.slider("Resíduos", -5.0, 5.0, 1.0, 0.1)
    
    st.subheader("Meta de Redução")
    meta_reducao = st.slider("Redução até 2050 (%)", 0, 100, 50, 5)
    
    st.subheader("Parâmetros Econômicos")
    crescimento_pib = st.slider("Crescimento anual do PIB (%)", 0.0, 10.0, 2.0, 0.1)
    intensidade_carbono = st.slider("Intensidade carbono-PIB (tCO₂/R$ mil)", 0.01, 2.0, 0.15, 0.01)

# Funções de cálculo
def calcular_emissoes_projetadas(ano_inicio, ano_fim, emissao_atual, taxas):
    """Calcula as emissões projetadas por setor"""
    anos = np.arange(ano_inicio, ano_fim + 1)
    n_anos = len(anos)
    
    # Estrutura dos setores
    setores = {
        'Energia': emissao_atual * 0.45,
        'Agropecuária': emissao_atual * 0.25,
        'Mudança Uso Solo': emissao_atual * 0.20,
        'Processos Industriais': emissao_atual * 0.07,
        'Resíduos': emissao_atual * 0.03
    }
    
    # Projeções por setor
    proj_setores = {}
    for setor, emissao_setor in setores.items():
        taxa = taxas[setor]
        proj = emissao_setor * (1 + taxa/100) ** np.arange(n_anos)
        proj_setores[setor] = proj
    
    # Total
    total = np.zeros(n_anos)
    for proj in proj_setores.values():
        total += proj
    
    return anos, proj_setores, total

def calcular_orcamento_carbono(total_emissoes, meta_reducao, ano_inicio, ano_fim):
    """Calcula o orçamento de carbono restante"""
    ano_base = ano_inicio
    ano_meta = 2050
    meta_2050 = total_emissoes[0] * (1 - meta_reducao/100)
    
    # Trajetória linear de redução
    anos_trajetoria = np.arange(ano_base, ano_fim + 1)
    trajetoria = total_emissoes[0] + (meta_2050 - total_emissoes[0]) * (anos_trajetoria - ano_base) / (ano_meta - ano_base)
    trajetoria[anos_trajetoria > ano_meta] = meta_2050
    
    # Orçamento acumulado
    orcamento_trajetoria = np.trapz(trajetoria, anos_trajetoria)
    orcamento_real = np.trapz(total_emissoes[:len(anos_trajetoria)], anos_trajetoria)
    
    return anos_trajetoria, trajetoria, orcamento_trajetoria, orcamento_real

# Cálculos principais
taxas = {
    'Energia': taxa_energia,
    'Agropecuária': taxa_agropecuaria,
    'Mudança Uso Solo': taxa_mudanca_uso_solo,
    'Processos Industriais': taxa_processos_industriais,
    'Resíduos': taxa_residuos
}

anos, proj_setores, total_emissoes = calcular_emissoes_projetadas(
    ano_inicio, ano_fim, emissao_atual, taxas
)

anos_trajetoria, trajetoria, orcamento_trajetoria, orcamento_real = calcular_orcamento_carbono(
    total_emissoes, meta_reducao, ano_inicio, ano_fim
)

# Layout principal
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Orçamento Restante (MtCO₂)",
        value=f"{orcamento_trajetoria - orcamento_real:,.0f}",
        delta=f"{((orcamento_trajetoria - orcamento_real)/orcamento_trajetoria*100):.1f}% do total"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Emissões em 2050 (MtCO₂)",
        value=f"{total_emissoes[anos == 2050][0]:,.0f}",
        delta=f"{((total_emissoes[anos == 2050][0] - trajetoria[anos_trajetoria == 2050][0])/trajetoria[anos_trajetoria == 2050][0]*100):+.1f}% vs meta"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Intensidade Carbono (tCO₂/R$ mil)",
        value=f"{intensidade_carbono:.3f}",
        delta=f"{intensidade_carbono * (1 - crescimento_pib/100) ** (2050-ano_inicio) - intensidade_carbono:.3f} projeção 2050"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Gráficos
tab1, tab2, tab3, tab4 = st.tabs(["📈 Projeções", "🌡️ Sensibilidade", "📊 Setores", "📋 Relatório"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de linhas - emissões totais
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=anos,
            y=total_emissoes,
            mode='lines',
            name='Projeção Atual',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=anos_trajetoria,
            y=trajetoria,
            mode='lines',
            name='Trajetória 1.5°C',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Projeção de Emissões vs Meta',
            xaxis_title='Ano',
            yaxis_title='Emissões (MtCO₂e)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de barras - contribuição setorial
        fig2 = go.Figure()
        
        # Último ano
        contrib_setores = [proj[-1] for proj in proj_setores.values()]
        
        fig2.add_trace(go.Bar(
            x=list(proj_setores.keys()),
            y=contrib_setores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        ))
        
        fig2.update_layout(
            title=f'Contribuição Setorial em {ano_fim}',
            xaxis_title='Setor',
            yaxis_title='Emissões (MtCO₂e)',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Análise de Sensibilidade (Método Sobol)")
    
    # Definir o problema para análise Sobol
    problem = {
        'num_vars': 5,
        'names': ['taxa_energia', 'taxa_agro', 'taxa_solo', 'taxa_ind', 'taxa_res'],
        'bounds': [
            [-5.0, 5.0],      # taxa_energia
            [-5.0, 5.0],      # taxa_agro
            [-15.0, 5.0],     # taxa_solo
            [-3.0, 3.0],      # taxa_ind
            [-3.0, 3.0]       # taxa_res
        ]
    }
    
    if st.button("Executar Análise de Sensibilidade"):
        with st.spinner("Executando análise... (pode levar alguns segundos)"):
            # Gerar amostras
            n_samples = 1000
            param_values = saltelli.sample(problem, n_samples)
            
            # Avaliar modelo para cada conjunto de parâmetros
            Y = np.zeros([param_values.shape[0]])
            
            for i, params in enumerate(param_values):
                # Extrair parâmetros
                taxas_sim = {
                    'Energia': params[0],
                    'Agropecuária': params[1],
                    'Mudança Uso Solo': params[2],
                    'Processos Industriais': params[3],
                    'Resíduos': params[4]
                }
                
                # Calcular emissões
                _, _, total_sim = calcular_emissoes_projetadas(
                    ano_inicio, ano_fim, emissao_atual, taxas_sim
                )
                
                # Usar emissão em 2050 como métrica
                Y[i] = total_sim[anos == 2050][0]
            
            # Realizar análise Sobol
            Si = sobol.analyze(problem, Y)
            
            # Gráfico de sensibilidade
            fig_sens = go.Figure()
            
            fig_sens.add_trace(go.Bar(
                x=problem['names'],
                y=Si['S1'],
                name='Efeito principal',
                marker_color='lightblue'
            ))
            
            fig_sens.add_trace(go.Bar(
                x=problem['names'],
                y=Si['ST'],
                name='Efeito total',
                marker_color='darkblue'
            ))
            
            fig_sens.update_layout(
                title='Índices de Sensibilidade Sobol',
                xaxis_title='Parâmetro',
                yaxis_title='Índice de Sensibilidade',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # Tabela de resultados
            st.subheader("Resultados da Análise")
            sens_df = pd.DataFrame({
                'Parâmetro': problem['names'],
                'Efeito Principal (S1)': Si['S1'],
                'Efeito Total (ST)': Si['ST'],
                'Variância Explicada (%)': Si['S1'] / Si['S1'].sum() * 100
            })
            
            st.dataframe(sens_df.style.format({
                'Efeito Principal (S1)': '{:.4f}',
                'Efeito Total (ST)': '{:.4f}',
                'Variância Explicada (%)': '{:.1f}'
            }))

with tab3:
    # Gráfico de área - evolução setorial
    fig_area = go.Figure()
    
    # Preparar dados
    anos_array = np.tile(anos, (5, 1)).T.flatten()
    setores_array = np.repeat(list(proj_setores.keys()), len(anos))
    valores_array = np.concatenate([proj_setores[setor] for setor in proj_setores.keys()])
    
    df_area = pd.DataFrame({
        'Ano': anos_array,
        'Setor': setores_array,
        'Emissões': valores_array
    })
    
    fig_area = px.area(df_area, x='Ano', y='Emissões', color='Setor',
                       title='Evolução das Emissões por Setor',
                       color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig_area.update_layout(height=500)
    st.plotly_chart(fig_area, use_container_width=True)

with tab4:
    st.subheader("Relatório de Análise")
    
    # Gerar relatório
    relatorio = f"""
    ## Relatório de Orçamento de Emissões - Brasil
    
    ### 1. Resumo Executivo
    - **Período analisado**: {ano_inicio} - {ano_fim}
    - **Emissões atuais**: {emissao_atual:,.0f} MtCO₂e
    - **Orçamento restante**: {orcamento_trajetoria - orcamento_real:,.0f} MtCO₂
    - **Gap em 2050**: {total_emissoes[anos == 2050][0] - trajetoria[anos_trajetoria == 2050][0]:,.0f} MtCO₂e
    
    ### 2. Projeções por Setor ({ano_fim})
    """
    
    for setor, proj in proj_setores.items():
        relatorio += f"- **{setor}**: {proj[-1]:,.0f} MtCO₂e ({proj[-1]/total_emissoes[-1]*100:.1f}% do total)\n"
    
    relatorio += f"""
    
    ### 3. Recomendações
    1. **Prioridade de ação**: Setor {max(proj_setores.items(), key=lambda x: x[1][-1])[0]} apresenta maior contribuição
    2. **Taxa necessária**: Redução adicional de {abs((total_emissoes[anos == 2050][0] - trajetoria[anos_trajetoria == 2050][0])/trajetoria[anos_trajetoria == 2050][0]*100):.1f}% para atingir meta
    3. **Orçamento anual médio**: {(orcamento_trajetoria - orcamento_real)/(ano_fim - ano_inicio):,.0f} MtCO₂e/ano
    
    ### 4. Limitações
    - Projeções baseadas em tendências lineares
    - Não considera políticas futuras
    - Baseado em dados históricos até {ano_inicio}
    
    **Gerado em**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    """
    
    st.markdown(relatorio)
    
    # Botão para download
    csv_data = pd.DataFrame({
        'Ano': anos,
        'Total_Emissoes': total_emissoes,
        'Meta_Trajetoria': trajetoria[:len(anos)],
        'Gap': total_emissoes - trajetoria[:len(anos)]
    })
    
    for setor, proj in proj_setores.items():
        csv_data[f'Emissoes_{setor}'] = proj
    
    st.download_button(
        label="📥 Baixar Dados Completos (CSV)",
        data=csv_data.to_csv(index=False).encode('utf-8'),
        file_name=f"orcamento_emissoes_br_{ano_inicio}_{ano_fim}.csv",
        mime="text/csv"
    )

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>⚠️ <strong>Aviso</strong>: Esta ferramenta é para fins educacionais e de planejamento. 
    Consulte especialistas para análises detalhadas.</p>
    <p>Desenvolvido para análise de orçamento de carbono brasileiro • Dados simulados para demonstração</p>
</div>
""", unsafe_allow_html=True)
