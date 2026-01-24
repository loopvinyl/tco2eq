import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border: none;
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
    
    st.subheader("Parâmetros de Sensibilidade")
    n_simulacoes = st.slider("Número de simulações", 10, 1000, 100, 10)
    incerteza_taxas = st.slider("Incerteza nas taxas (%)", 0, 50, 20, 5)

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
    
    # Emissões no ano base
    emissao_base = total_emissoes[0]
    
    # Meta para 2050
    meta_2050 = emissao_base * (1 - meta_reducao/100)
    
    # Trajetória linear de redução até 2050, constante após
    anos_trajetoria = np.arange(ano_base, ano_fim + 1)
    trajetoria = np.zeros_like(anos_trajetoria, dtype=float)
    
    for i, ano in enumerate(anos_trajetoria):
        if ano <= ano_meta:
            # Redução linear até 2050
            progresso = (ano - ano_base) / (ano_meta - ano_base)
            trajetoria[i] = emissao_base + (meta_2050 - emissao_base) * progresso
        else:
            # Mantém constante após 2050
            trajetoria[i] = meta_2050
    
    # Cálculo do orçamento (integral das emissões)
    # Usando regra do trapézio
    def calcular_integral(y, x):
        integral = 0
        for i in range(1, len(x)):
            integral += (y[i] + y[i-1]) * (x[i] - x[i-1]) / 2
        return integral
    
    # Garantir que temos o mesmo número de pontos
    n_pontos = min(len(anos_trajetoria), len(total_emissoes))
    anos_common = anos_trajetoria[:n_pontos]
    trajetoria_common = trajetoria[:n_pontos]
    total_common = total_emissoes[:n_pontos]
    
    orcamento_trajetoria = calcular_integral(trajetoria_common, anos_common)
    orcamento_real = calcular_integral(total_common, anos_common)
    
    return anos_trajetoria, trajetoria, orcamento_trajetoria, orcamento_real

def analise_sensibilidade_monte_carlo(n_simulacoes, taxas_base, incerteza, ano_inicio, ano_fim, emissao_atual, meta_reducao):
    """Análise de sensibilidade usando Monte Carlo"""
    resultados = []
    emissoes_2050 = []
    
    for _ in range(n_simulacoes):
        # Adicionar incerteza às taxas
        taxas_sim = {}
        for setor, taxa in taxas_base.items():
            # Adicionar variação aleatória baseada na incerteza
            variacao = np.random.uniform(-incerteza/100, incerteza/100) * taxa
            taxas_sim[setor] = taxa + variacao
        
        # Calcular emissões
        anos, proj_setores, total = calcular_emissoes_projetadas(
            ano_inicio, ano_fim, emissao_atual, taxas_sim
        )
        
        # Encontrar emissões em 2050
        idx_2050 = np.where(anos == 2050)[0]
        if len(idx_2050) > 0:
            emissao_2050 = total[idx_2050[0]]
        else:
            emissao_2050 = total[-1]
        
        resultados.append(taxas_sim)
        emissoes_2050.append(emissao_2050)
    
    return resultados, np.array(emissoes_2050)

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
    orcamento_restante = max(0, orcamento_trajetoria - orcamento_real)
    percentual_restante = (orcamento_restante / orcamento_trajetoria * 100) if orcamento_trajetoria > 0 else 0
    st.metric(
        label="Orçamento Restante (MtCO₂)",
        value=f"{orcamento_restante:,.0f}",
        delta=f"{percentual_restante:.1f}% do total"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Encontrar emissões em 2050
    idx_2050 = np.where(anos == 2050)[0]
    if len(idx_2050) > 0:
        emissao_2050 = total_emissoes[idx_2050[0]]
        idx_traj_2050 = np.where(anos_trajetoria == 2050)[0]
        trajetoria_2050 = trajetoria[idx_traj_2050[0]] if len(idx_traj_2050) > 0 else trajetoria[-1]
    else:
        emissao_2050 = total_emissoes[-1]
        trajetoria_2050 = trajetoria[-1]
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    delta_percent = ((emissao_2050 - trajetoria_2050) / trajetoria_2050 * 100) if trajetoria_2050 > 0 else 0
    st.metric(
        label="Emissões em 2050 (MtCO₂)",
        value=f"{emissao_2050:,.0f}",
        delta=f"{delta_percent:+.1f}% vs meta"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    crescimento_pib = 2.0  # Valor padrão
    intensidade_carbono = 0.15  # Valor padrão
    int_carbono_2050 = intensidade_carbono * (1 - crescimento_pib/100) ** (2050 - ano_inicio)
    delta_intensidade = int_carbono_2050 - intensidade_carbono
    st.metric(
        label="Redução Necessária/Ano",
        value=f"{(total_emissoes[0] - trajetoria_2050) / (2050 - ano_inicio):,.0f}",
        delta=f"MtCO₂e/ano"
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
    
    # Preencher área entre as curvas
    ax1.fill_between(anos, total_emissoes, trajetoria[:len(anos)], 
                     where=(total_emissoes > trajetoria[:len(anos)]), 
                     color='red', alpha=0.3, label='Excesso')
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
    
    # Adicionar valores nas barras
    for i, v in enumerate(contribuicoes):
        ax2.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
    
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
    st.subheader("Análise de Sensibilidade (Monte Carlo)")
    
    if st.button("Executar Análise de Sensibilidade", type="primary"):
        with st.spinner(f"Executando {n_simulacoes} simulações..."):
            resultados, emissoes_2050 = analise_sensibilidade_monte_carlo(
                n_simulacoes, taxas, incerteza_taxas, ano_inicio, ano_fim, emissao_atual, meta_reducao
            )
            
            # Estatísticas
            media_2050 = np.mean(emissoes_2050)
            mediana_2050 = np.percentile(emissoes_2050, 50)
            p10_2050 = np.percentile(emissoes_2050, 10)
            p90_2050 = np.percentile(emissoes_2050, 90)
            
            # Gráfico de distribuição
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histograma
            ax4.hist(emissoes_2050, bins=30, edgecolor='black', alpha=0.7, color='lightblue')
            ax4.axvline(media_2050, color='red', linestyle='--', linewidth=2, label=f'Média: {media_2050:,.0f}')
            ax4.axvline(trajetoria_2050, color='green', linestyle='-', linewidth=2, label=f'Meta: {trajetoria_2050:,.0f}')
            ax4.set_xlabel('Emissões em 2050 (MtCO₂e)')
            ax4.set_ylabel('Frequência')
            ax4.set_title('Distribuição das Emissões em 2050')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Boxplot
            ax5.boxplot(emissoes_2050, vert=True, patch_artist=True)
            ax5.set_ylabel('Emissões em 2050 (MtCO₂e)')
            ax5.set_title('Boxplot das Emissões em 2050')
            ax5.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média 2050", f"{media_2050:,.0f} MtCO₂e")
            with col2:
                st.metric("Mediana 2050", f"{mediana_2050:,.0f} MtCO₂e")
            with col3:
                st.metric("Percentil 10%", f"{p10_2050:,.0f} MtCO₂e")
            with col4:
                st.metric("Percentil 90%", f"{p90_2050:,.0f} MtCO₂e")
            
            # Análise de correlação
            st.subheader("Análise de Influência dos Parâmetros")
            
            # Converter resultados para DataFrame
            df_resultados = pd.DataFrame(resultados)
            df_resultados['Emissao_2050'] = emissoes_2050
            
            # Calcular correlações
            correlacoes = {}
            for setor in taxas.keys():
                correlacao = np.corrcoef(df_resultados[setor], emissoes_2050)[0, 1]
                correlacoes[setor] = correlacao
            
            # Gráfico de correlações
            fig4, ax6 = plt.subplots(figsize=(10, 5))
            
            setores_list = list(correlacoes.keys())
            valores_corr = list(correlacoes.values())
            
            bars = ax6.bar(setores_list, valores_corr, color=['red' if v > 0 else 'green' for v in valores_corr])
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.set_xlabel('Setor')
            ax6.set_ylabel('Correlação com Emissões 2050')
            ax6.set_title('Correlação entre Taxas e Emissões em 2050')
            ax6.tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for bar, v in zip(bars, valores_corr):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{v:.3f}', ha='center', va='bottom' if height > 0 else 'top')
            
            ax6.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig4)
            
            # Tabela de resultados
            st.subheader("Resumo das Simulações")
            df_resumo = pd.DataFrame({
                'Setor': setores_list,
                'Taxa Base (%)': [taxas[s] for s in setores_list],
                'Correlação': valores_corr,
                'Influência': ['Alta' if abs(v) > 0.3 else 'Média' if abs(v) > 0.1 else 'Baixa' for v in valores_corr]
            })
            st.dataframe(df_resumo)

with tab3:
    st.subheader("Relatório de Análise")
    
    # Calcular métricas chave
    emissao_inicial = total_emissoes[0]
    reducao_necessaria = (emissao_inicial - trajetoria_2050) / max(1, 2050 - ano_inicio)
    gap_2050 = emissao_2050 - trajetoria_2050
    
    # Relatório
    relatorio = f"""
    ## 📊 Relatório de Orçamento de Carbono - Brasil
    
    ### 1. RESUMO EXECUTIVO
    
    **Período Analisado**: {ano_inicio}-{ano_fim}
    **Emissões Iniciais**: {emissao_inicial:,.0f} MtCO₂e/ano
    **Meta de Redução**: {meta_reducao}% até 2050
    
    ### 2. RESULTADOS PRINCIPAIS
    
    - **Orçamento Restante**: {orcamento_restante:,.0f} MtCO₂
    - **Emissões Projetadas 2050**: {emissao_2050:,.0f} MtCO₂e
    - **Meta para 2050**: {trajetoria_2050:,.0f} MtCO₂e
    - **Gap em 2050**: {gap_2050:,.0f} MtCO₂e ({gap_2050/trajetoria_2050*100:+.1f}% acima da meta)
    - **Redução Necessária/Ano**: {reducao_necessaria:,.0f} MtCO₂e/ano
    
    ### 3. CONTRIBUIÇÃO SETORIAL ({ano_fim})
    """
    
    total_atual = total_emissoes[-1]
    for setor, proj in proj_setores.items():
        contrib = proj[-1]
        percentual = (contrib / total_atual * 100) if total_atual > 0 else 0
        relatorio += f"\n- **{setor}**: {contrib:,.0f} MtCO₂e ({percentual:.1f}%)"
    
    relatorio += f"""
    
    ### 4. RECOMENDAÇÕES
    
    1. **Ação Prioritária**: Foco no setor de maior contribuição
    2. **Taxa de Redução**: Reduzir {reducao_necessaria/emissao_inicial*100:.1f}% ao ano
    3. **Monitoramento**: Revisar metas anualmente
    4. **Políticas**: Implementar mecanismos de mercado de carbono
    
    ### 5. LIMITAÇÕES
    
    - Projeções baseadas em crescimento composto
    - Incertezas econômicas e tecnológicas não consideradas
    - Cenários climáticos simplificados
    
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
        'Meta_Trajetoria': trajetoria[:len(anos)]
    })
    
    # Calcular gap
    dados_exportacao['Gap'] = dados_exportacao['Emissões_Total'] - dados_exportacao['Meta_Trajetoria']
    
    # Adicionar dados setoriais
    for setor, proj in proj_setores.items():
        dados_exportacao[f'Emissões_{setor}'] = proj
    
    # Converter para CSV
    csv = dados_exportacao.to_csv(index=False)
    
    st.download_button(
        label="📥 Baixar Dados Completos (CSV)",
        data=csv,
        file_name=f"orcamento_emissoes_brasil_{ano_inicio}_{ano_fim}.csv",
        mime="text/csv",
        type="primary"
    )

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📌 <strong>Nota</strong>: Esta ferramenta é para fins educacionais e de planejamento.</p>
    <p>Fonte: Baseado em metodologias do IPCC e dados do SEEG Brasil • Desenvolvido com Python e Streamlit</p>
</div>
""", unsafe_allow_html=True)
