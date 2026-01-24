import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    
    # Distribuição setorial típica do Brasil
    distribuicao_setorial = {
        'Energia': 0.45,
        'Agropecuária': 0.25,
        'Mudança Uso Solo': 0.20,
        'Processos Industriais': 0.07,
        'Resíduos': 0.03
    }
    
    # Projeções por setor
    proj_setores = {}
    for setor, proporcao in distribuicao_setorial.items():
        emissao_setor = emissao_atual * proporcao
        taxa = taxas[setor]
        # Cálculo de projeção com crescimento anual composto
        proj = emissao_setor * np.power(1 + taxa/100, np.arange(n_anos))
        proj_setores[setor] = proj
    
    # Total de emissões por ano
    total = np.sum(list(proj_setores.values()), axis=0)
    
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
    # Usando regra do trapézio simples
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
        # Se 2050 não estiver no intervalo, usar o último ano
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
    int_carbono_2050 = intensidade_carbono * np.power(1 - crescimento_pib/100, 2050 - ano_inicio)
    delta_intensidade = int_carbono_2050 - intensidade_carbono
    st.metric(
        label="Intensidade Carbono (tCO₂/R$ mil)",
        value=f"{intensidade_carbono:.3f}",
        delta=f"{delta_intensidade:.3f} em 2050"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Gráficos e análises
tab1, tab2, tab3 = st.tabs(["📈 Projeções", "🌡️ Análise de Sensibilidade", "📋 Relatório"])

with tab1:
    st.subheader("Projeção de Emissões")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Projeção vs Meta
    ax1.plot(anos, total_emissoes, 'r-', linewidth=3, label='Projeção Atual')
    ax1.plot(anos_trajetoria, trajetoria, 'g--', linewidth=3, label='Meta de Redução')
    
    # Preencher área entre as curvas
    anos_comum = np.intersect1d(anos, anos_trajetoria)
    idx_anos = np.searchsorted(anos, anos_comum)
    idx_traj = np.searchsorted(anos_trajetoria, anos_comum)
    
    ax1.fill_between(anos_comum, total_emissoes[idx_anos], trajetoria[idx_traj], 
                     where=(total_emissoes[idx_anos] > trajetoria[idx_traj]), 
                     color='red', alpha=0.3, label='Excesso de Emissões')
    ax1.fill_between(anos_comum, total_emissoes[idx_anos], trajetoria[idx_traj], 
                     where=(total_emissoes[idx_anos] <= trajetoria[idx_traj]), 
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
            n_samples = 256  # Número reduzido para performance no Streamlit Cloud
            try:
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
                nomes_legiveis = ['Energia', 'Agropecuária', 'Uso do Solo', 'Industrial', 'Resíduos']
                x_pos = np.arange(len(nomes))
                
                ax4.bar(x_pos - 0.2, indices_s1, 0.4, label='Efeito Principal (S1)', alpha=0.8, color='lightblue')
                ax4.bar(x_pos + 0.2, indices_st, 0.4, label='Efeito Total (ST)', alpha=0.8, color='darkblue')
                
                ax4.set_xlabel('Parâmetro')
                ax4.set_ylabel('Índice de Sensibilidade')
                ax4.set_title('Índices de Sensibilidade Sobol')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(nomes_legiveis)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
                
                # Tabela de resultados
                st.subheader("Resultados da Análise")
                resultados = pd.DataFrame({
                    'Parâmetro': nomes_legiveis,
                    'Efeito Principal (S1)': indices_s1,
                    'Efeito Total (ST)': indices_st,
                    'Contribuição Relativa (%)': (indices_s1 / indices_s1.sum() * 100) if indices_s1.sum() > 0 else 0
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
                
            except Exception as e:
                st.error(f"Erro na análise de sensibilidade: {str(e)}")
                st.info("Tente reduzir o número de amostras ou verificar os parâmetros.")

with tab3:
    st.subheader("Relatório de Análise")
    
    # Calcular métricas chave
    emissao_inicial = total_emissoes[0]
    reducao_necessaria = (emissao_inicial - trajetoria[-1]) / max(1, 2050 - ano_inicio)
    
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
    - **Gap em 2050**: {emissao_2050 - trajetoria_2050:,.0f} MtCO₂e
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
    
    1. **Ação Prioritária**: Foco no setor de maior contribuição atual
    2. **Taxa de Redução**: Aumentar para {abs(reducao_necessaria/emissao_inicial*100):.1f}% ao ano
    3. **Monitoramento**: Revisar metas a cada 5 anos
    4. **Políticas**: Implementar mecanismos de precificação de carbono
    
    ### 5. LIMITAÇÕES
    
    - Baseado em projeções lineares e crescimento composto
    - Não considera mudanças tecnológicas disruptivas
    - Cenários econômicos simplificados
    
    ---
    *Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}*
    """
    
    st.markdown(relatorio)
    
    # Botão para download dos dados
    st.subheader("📥 Exportar Dados")
    
    # Criar DataFrame com resultados
    dados_exportacao = pd.DataFrame({
        'Ano': anos,
        'Emissões_Total': total_emissoes
    })
    
    # Adicionar trajetória meta (alinhar anos)
    trajetoria_df = pd.DataFrame({
        'Ano': anos_trajetoria,
        'Meta_Trajetoria': trajetoria
    })
    
    dados_exportacao = pd.merge(dados_exportacao, trajetoria_df, on='Ano', how='left')
    
    # Calcular gap
    dados_exportacao['Gap'] = dados_exportacao['Emissões_Total'] - dados_exportacao['Meta_Trajetoria']
    
    # Adicionar dados setoriais
    for setor, proj in proj_setores.items():
        dados_exportacao[f'Emissões_{setor}'] = proj[:len(dados_exportacao)]
    
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
    <p>Fonte: Baseado em metodologias do IPCC e dados do SEEG Brasil</p>
</div>
""", unsafe_allow_html=True)
