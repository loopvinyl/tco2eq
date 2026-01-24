import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from SALib.sample import saltelli
from SALib.analyze import sobol
from datetime import datetime
import io

# Configuração da página
st.set_page_config(
    page_title="Orçamento de Emissões BR",
    page_icon="🌳",
    layout="wide"
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
    
    # Distribuição setorial (baseada em dados brasileiros)
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
        value=f"{max(0, orcamento_trajetoria - orcamento_real):,.0f}",
        delta=f"{((orcamento_trajetoria - orcamento_real)/orcamento_trajetoria*100):.1f}% do orçamento"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    try:
        emissao_2050 = total_emissoes[anos == 2050][0]
    except:
        emissao_2050 = total_emissoes[-1]
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="Emissões em 2050 (MtCO₂)",
        value=f"{emissao_2050:,.0f}",
        delta=f"{((emissao_2050 - trajetoria[-1])/trajetoria[-1]*100):+.1f}% vs meta"
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

# Gráficos e análises
tab1, tab2, tab3 = st.tabs(["📈 Projeções", "🌡️ Análise de Sensibilidade", "📋 Relatório"])

with tab1:
    st.subheader("Projeção de Emissões")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1: Projeção vs Meta
    ax1.plot(anos, total_emissoes, 'r-', linewidth=3, label='Projeção Atual')
    ax1.plot(anos_trajetoria, trajetoria, 'g--', linewidth=3, label='Meta de Redução')
    ax1.fill_between(anos, total_emissoes, trajetoria[:len(anos)], 
                     where=(total_emissoes > trajetoria[:len(anos)]), 
                     color='red', alpha=0.3, label='Excesso de Emissões')
    ax1.fill_between(anos, total_emissoes, trajetoria[:len(anos)], 
                     where=(total_emissoes <= trajetoria[:len(anos)]), 
                     color='green', alpha=0.3, label='Dentro da Meta')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Emissões (MtCO₂e)')
    ax1.set_title('Projeção vs Meta de Emissões')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Contribuição Setorial
    ultimo_ano_idx = -1
    contribuicoes = [proj[ultimo_ano_idx] for proj in proj_setores.values()]
    setores = list(proj_setores.keys())
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    ax2.bar(setores, contribuicoes, color=cores)
    ax2.set_xlabel('Setor')
    ax2.set_ylabel('Emissões (MtCO₂e)')
    ax2.set_title(f'Contribuição Setorial em {ano_fim}')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Gráfico 3: Evolução setorial
    st.subheader("Evolução das Emissões por Setor")
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    
    for i, (setor, proj) in enumerate(proj_setores.items()):
        ax3.plot(anos, proj, label=setor, linewidth=2)
    
    ax3.set_xlabel('Ano')
    ax3.set_ylabel('Emissões (MtCO₂e)')
    ax3.set_title('Evolução das Emissões por Setor Econômico')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig2)

with tab2:
    st.subheader("Análise de Sensibilidade (Método Sobol)")
    
    # Definir o problema
    problem = {
        'num_vars': 5,
        'names': ['taxa_energia', 'taxa_agro', 'taxa_solo', 'taxa_ind', 'taxa_res'],
        'bounds': [
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-15.0, 5.0],
            [-3.0, 3.0],
            [-3.0, 3.0]
        ]
    }
    
    if st.button("Executar Análise de Sensibilidade", type="primary"):
        with st.spinner("Executando análise... (isso pode levar alguns segundos)"):
            # Gerar amostras
            n_samples = 512  # Número reduzido para performance
            param_values = saltelli.sample(problem, n_samples)
            
            # Avaliar o modelo
            Y = np.zeros(param_values.shape[0])
            
            for i, params in enumerate(param_values):
                taxas_sim = {
                    'Energia': params[0],
                    'Agropecuária': params[1],
                    'Mudança Uso Solo': params[2],
                    'Processos Industriais': params[3],
                    'Resíduos': params[4]
                }
                
                _, _, total_sim = calcular_emissoes_projetadas(
                    ano_inicio, ano_fim, emissao_atual, taxas_sim
                )
                Y[i] = total_sim[-1]  # Emissões no último ano
            
            # Realizar análise Sobol
            Si = sobol.analyze(problem, Y)
            
            # Gráfico de sensibilidade
            fig3, ax4 = plt.subplots(figsize=(10, 6))
            
            indices_s1 = Si['S1']
            indices_st = Si['ST']
            nomes = problem['names']
            x_pos = np.arange(len(nomes))
            
            ax4.bar(x_pos - 0.2, indices_s1, 0.4, label='Efeito Principal (S1)', alpha=0.8)
            ax4.bar(x_pos + 0.2, indices_st, 0.4, label='Efeito Total (ST)', alpha=0.8)
            
            ax4.set_xlabel('Parâmetro')
            ax4.set_ylabel('Índice de Sensibilidade')
            ax4.set_title('Índices de Sensibilidade Sobol')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(['Energia', 'Agro', 'Uso Solo', 'Industrial', 'Resíduos'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            st.pyplot(fig3)
            
            # Tabela de resultados
            st.subheader("Resultados da Análise")
            resultados = pd.DataFrame({
                'Parâmetro': nomes,
                'Efeito Principal (S1)': indices_s1,
                'Efeito Total (ST)': indices_st,
                'Contribuição Relativa (%)': indices_s1 / indices_s1.sum() * 100
            })
            
            st.dataframe(resultados.style.format({
                'Efeito Principal (S1)': '{:.4f}',
                'Efeito Total (ST)': '{:.4f}',
                'Contribuição Relativa (%)': '{:.1f}'
            }))
            
            # Interpretação
            st.info("""
            **Interpretação dos resultados:**
            - **Efeito Principal (S1)**: Mede a contribuição individual de cada parâmetro
            - **Efeito Total (ST)**: Mede a contribuição total (incluindo interações)
            - **Parâmetros com maior ST** são os mais importantes para a incerteza do modelo
            """)

with tab3:
    st.subheader("Relatório de Análise")
    
    # Calcular métricas chave
    gap_2050 = emissao_2050 - trajetoria[-1]
    anos_restantes = 2050 - ano_inicio
    reducao_necessaria_ano = (total_emissoes[0] - trajetoria[-1]) / anos_restantes
    
    # Relatório
    relatorio = f"""
    ## 📊 Relatório de Orçamento de Carbono - Brasil
    
    ### 1. RESUMO EXECUTIVO
    
    **Período Analisado**: {ano_inicio}-{ano_fim}
    **Emissões Iniciais**: {emissao_atual:,.0f} MtCO₂e/ano
    **Meta de Redução**: {meta_reducao}% até 2050
    
    ### 2. RESULTADOS PRINCIPAIS
    
    - **Orçamento Restante**: {max(0, orcamento_trajetoria - orcamento_real):,.0f} MtCO₂
    - **Emissões Projetadas 2050**: {emissao_2050:,.0f} MtCO₂e
    - **Gap em 2050**: {gap_2050:,.0f} MtCO₂e ({gap_2050/trajetoria[-1]*100:+.1f}% acima da meta)
    - **Redução Necessária/Ano**: {reducao_necessaria_ano:,.0f} MtCO₂e/ano
    
    ### 3. CONTRIBUIÇÃO SETORIAL ({ano_fim})
    
    """
    
    for setor, proj in proj_setores.items():
        contrib = proj[-1]
        percentual = contrib / total_emissoes[-1] * 100
        relatorio += f"- **{setor}**: {contrib:,.0f} MtCO₂e ({percentual:.1f}%)\n"
    
    relatorio += f"""
    
    ### 4. RECOMENDAÇÕES
    
    1. **Ação Prioritária**: Foco no setor de maior crescimento
    2. **Taxa de Redução**: Aumentar para {abs(reducao_necessaria_ano/emissao_atual*100):.1f}% ao ano
    3. **Monitoramento**: Revisar metas a cada 5 anos
    4. **Políticas**: Implementar precificação de carbono e incentivos à descarbonização
    
    ### 5. LIMITAÇÕES
    
    - Baseado em projeções lineares
    - Não considera mudanças tecnológicas disruptivas
    - Baseado em dados disponíveis até {datetime.now().year}
    
    ---
    *Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}*
    """
    
    st.markdown(relatorio)
    
    # Botão para download dos dados
    st.subheader("📥 Exportar Dados")
    
    # Criar DataFrame com resultados
    dados_exportacao = pd.DataFrame({
        'Ano': anos,
        'Emissões_Total': total_emissoes,
        'Meta_Trajetoria': trajetoria[:len(anos)],
        'Gap': total_emissoes - trajetoria[:len(anos)]
    })
    
    for setor, proj in proj_setores.items():
        dados_exportacao[f'Emissões_{setor}'] = proj
    
    # Converter para CSV
    csv = dados_exportacao.to_csv(index=False)
    
    st.download_button(
        label="Baixar Dados Completos (CSV)",
        data=csv,
        file_name=f"orcamento_emissoes_brasil_{ano_inicio}_{ano_fim}.csv",
        mime="text/csv"
    )

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📌 <strong>Nota</strong>: Esta ferramenta é para fins educacionais e de planejamento.</p>
    <p>Fonte: Baseado em metodologias do IPCC e dados do SEEG Brasil</p>
</div>
""", unsafe_allow_html=True)
