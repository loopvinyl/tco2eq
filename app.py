import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import math

# Configuração da página
st.set_page_config(
    page_title="Calculadora de Orçamento de Carbono",
    page_icon="🌳",
    layout="wide"
)

# Título principal
st.title("🌍 Calculadora de Orçamento de Emissões - Brasil")

# Barra lateral para parâmetros
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("Período de Análise")
    ano_inicio = st.number_input("Ano Inicial", 1990, 2030, 2020)
    ano_fim = st.number_input("Ano Final", 2021, 2100, 2050)
    
    st.subheader("Emissões Atuais")
    emissao_atual = st.number_input("Emissões atuais (MtCO₂e/ano)", 100.0, 5000.0, 1500.0, 50.0)
    
    st.subheader("Taxas de Crescimento (%/ano)")
    taxa_energia = st.slider("Energia", -10.0, 10.0, 1.5, 0.1)
    taxa_agropecuaria = st.slider("Agropecuária", -10.0, 10.0, 0.8, 0.1)
    taxa_mudanca_solo = st.slider("Mudança Uso Solo", -20.0, 10.0, -3.0, 0.5)
    taxa_industrial = st.slider("Processos Industriais", -5.0, 5.0, 0.5, 0.1)
    taxa_residuos = st.slider("Resíduos", -5.0, 5.0, 1.0, 0.1)
    
    st.subheader("Meta de Redução")
    meta_reducao = st.slider("Redução até 2050 (%)", 0, 100, 50, 5)
    
    st.subheader("Análise de Sensibilidade")
    realizar_sensibilidade = st.checkbox("Realizar análise de sensibilidade", value=True)
    if realizar_sensibilidade:
        n_simulacoes = st.slider("Número de simulações", 10, 500, 100)

# Funções de cálculo
def calcular_projecao(ano_inicio, ano_fim, emissao_atual, taxas):
    """Calcula projeção de emissões"""
    anos = list(range(ano_inicio, ano_fim + 1))
    n_anos = len(anos)
    
    # Distribuição setorial
    distribuicao = {
        'Energia': 0.45,
        'Agropecuária': 0.25,
        'Mudança Uso Solo': 0.20,
        'Processos Industriais': 0.07,
        'Resíduos': 0.03
    }
    
    # Calcular emissões por setor
    emissoes_setores = {}
    for setor, proporcao in distribuicao.items():
        emissao_inicial = emissao_atual * proporcao
        taxa = taxas[setor]
        emissoes = []
        for t in range(n_anos):
            emissao = emissao_inicial * ((1 + taxa/100) ** t)
            emissoes.append(emissao)
        emissoes_setores[setor] = emissoes
    
    # Calcular total
    emissoes_total = []
    for i in range(n_anos):
        total_ano = sum(emissoes_setores[setor][i] for setor in emissoes_setores)
        emissoes_total.append(total_ano)
    
    return anos, emissoes_setores, emissoes_total

def calcular_meta(emissao_atual, meta_reducao, anos):
    """Calcula trajetória de meta"""
    emissao_2050 = emissao_atual * (1 - meta_reducao/100)
    trajetoria = []
    
    for ano in anos:
        if ano <= 2050:
            # Redução linear até 2050
            progresso = (ano - anos[0]) / (2050 - anos[0])
            emissao_meta = emissao_atual + (emissao_2050 - emissao_atual) * progresso
        else:
            emissao_meta = emissao_2050
        trajetoria.append(emissao_meta)
    
    return trajetoria

def calcular_orcamento(emissoes, anos):
    """Calcula orçamento de carbono acumulado"""
    orcamento = 0
    for i in range(1, len(anos)):
        area = (emissoes[i] + emissoes[i-1]) * (anos[i] - anos[i-1]) / 2
        orcamento += area
    return orcamento

def analise_sensibilidade_monte_carlo(n_simulacoes, taxas_base, emissao_atual, ano_fim):
    """Análise de sensibilidade simplificada"""
    resultados = []
    nomes_setores = list(taxas_base.keys())
    
    for _ in range(n_simulacoes):
        # Gerar taxas aleatórias com ±50% de variação
        taxas_aleatorias = {}
        for setor, taxa in taxas_base.items():
            variacao = np.random.uniform(-0.5, 0.5)  # ±50%
            taxas_aleatorias[setor] = taxa * (1 + variacao)
        
        # Calcular emissão final
        anos = [2020, ano_fim]
        _, _, emissao_final = calcular_projecao(2020, ano_fim, emissao_atual, taxas_aleatorias)
        
        resultados.append({
            'taxas': taxas_aleatorias,
            'emissao_final': emissao_final[-1]
        })
    
    return resultados

# Dicionário de taxas
taxas = {
    'Energia': taxa_energia,
    'Agropecuária': taxa_agropecuaria,
    'Mudança Uso Solo': taxa_mudanca_solo,
    'Processos Industriais': taxa_industrial,
    'Resíduos': taxa_residuos
}

# Cálculos principais
anos, emissoes_setores, emissoes_total = calcular_projecao(
    ano_inicio, ano_fim, emissao_atual, taxas
)

trajetoria_meta = calcular_meta(emissao_atual, meta_reducao, anos)
orcamento_total = calcular_orcamento(emissoes_total, anos)
orcamento_meta = calcular_orcamento(trajetoria_meta, anos)
orcamento_restante = max(0, orcamento_meta - orcamento_total)

# Exibir métricas principais
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Orçamento Restante",
        f"{orcamento_restante:,.0f} MtCO₂",
        f"{orcamento_restante/orcamento_meta*100:.1f}%"
    )

with col2:
    emissao_final = emissoes_total[-1]
    meta_final = trajetoria_meta[-1]
    delta = ((emissao_final - meta_final) / meta_final * 100) if meta_final > 0 else 0
    st.metric(
        "Emissões 2050",
        f"{emissao_final:,.0f} MtCO₂",
        f"{delta:+.1f}%"
    )

with col3:
    reducao_necessaria = (emissao_atual - meta_final) / (2050 - ano_inicio)
    st.metric(
        "Redução Necessária/Ano",
        f"{reducao_necessaria:,.0f} MtCO₂",
        f"{(reducao_necessaria/emissao_atual*100):.1f}%/ano"
    )

# Tabs para diferentes visualizações
tab1, tab2, tab3 = st.tabs(["📊 Gráficos", "📈 Dados", "📋 Relatório"])

with tab1:
    st.subheader("Projeção de Emissões")
    
    # Criar DataFrame para gráfico
    df_grafico = pd.DataFrame({
        'Ano': anos,
        'Projeção': emissoes_total,
        'Meta': trajetoria_meta
    })
    
    # Gráfico de linha usando streamlit
    st.line_chart(df_grafico.set_index('Ano'))
    
    # Gráfico de barras por setor
    st.subheader("Contribuição por Setor")
    
    # Dados do último ano
    dados_setores = {}
    for setor, emissoes in emissoes_setores.items():
        dados_setores[setor] = emissoes[-1]
    
    df_setores = pd.DataFrame({
        'Setor': list(dados_setores.keys()),
        'Emissões': list(dados_setores.values())
    })
    
    st.bar_chart(df_setores.set_index('Setor'))

with tab2:
    st.subheader("Dados Detalhados")
    
    # Criar DataFrame com todos os dados
    dados = []
    for i, ano in enumerate(anos):
        linha = {
            'Ano': ano,
            'Total': emissoes_total[i],
            'Meta': trajetoria_meta[i],
            'Gap': emissoes_total[i] - trajetoria_meta[i]
        }
        for setor in emissoes_setores:
            linha[setor] = emissoes_setores[setor][i]
        dados.append(linha)
    
    df_detalhado = pd.DataFrame(dados)
    st.dataframe(df_detalhado)
    
    # Botão para download
    csv = df_detalhado.to_csv(index=False)
    st.download_button(
        "📥 Baixar Dados (CSV)",
        csv,
        f"orcamento_carbono_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

with tab3:
    st.subheader("Relatório de Análise")
    
    # Gerar relatório
    relatorio = f"""
    ## Relatório de Orçamento de Carbono
    
    ### Configurações da Simulação
    - **Período**: {ano_inicio} - {ano_fim}
    - **Emissões iniciais**: {emissao_atual:,.0f} MtCO₂e/ano
    - **Meta de redução**: {meta_reducao}% até 2050
    
    ### Resultados Principais
    1. **Orçamento de carbono restante**: {orcamento_restante:,.0f} MtCO₂
    2. **Emissões em {ano_fim}**: {emissao_final:,.0f} MtCO₂e
    3. **Meta para {ano_fim}**: {meta_final:,.0f} MtCO₂e
    4. **Gap em {ano_fim}**: {emissao_final - meta_final:,.0f} MtCO₂e
    
    ### Contribuição Setorial ({ano_fim})
    """
    
    for setor, emissao in dados_setores.items():
        percentual = (emissao / emissao_final * 100) if emissao_final > 0 else 0
        relatorio += f"\n- **{setor}**: {emissao:,.0f} MtCO₂e ({percentual:.1f}%)"
    
    relatorio += f"""
    
    ### Recomendações
    
    1. **Ações prioritárias**: Concentrar esforços nos setores com maior contribuição
    2. **Taxa de redução**: Necessário reduzir {reducao_necessaria/emissao_atual*100:.1f}% ao ano
    3. **Monitoramento**: Acompanhar indicadores anualmente
    4. **Políticas**: Implementar medidas específicas por setor
    
    ---
    *Relatório gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}*
    """
    
    st.markdown(relatorio)

# Análise de sensibilidade
if realizar_sensibilidade and 'n_simulacoes' in locals():
    st.divider()
    st.subheader("🔬 Análise de Sensibilidade")
    
    if st.button("Executar Simulações"):
        with st.spinner(f"Executando {n_simulacoes} simulações..."):
            resultados = analise_sensibilidade_monte_carlo(
                n_simulacoes, taxas, emissao_atual, ano_fim
            )
            
            # Extrair resultados
            emissoes_finais = [r['emissao_final'] for r in resultados]
            
            # Estatísticas
            media = np.mean(emissoes_finais)
            mediana = np.percentile(emissoes_finais, 50)
            p10 = np.percentile(emissoes_finais, 10)
            p90 = np.percentile(emissoes_finais, 90)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média", f"{media:,.0f}")
            with col2:
                st.metric("Mediana", f"{mediana:,.0f}")
            with col3:
                st.metric("P10", f"{p10:,.0f}")
            with col4:
                st.metric("P90", f"{p90:,.0f}")
            
            # Histograma simples
            st.subheader("Distribuição das Emissões Finais")
            
            # Criar histograma usando pandas
            hist_data = pd.DataFrame({'Emissões Finais': emissoes_finais})
            st.bar_chart(hist_data)

# Informações finais
st.divider()
st.info("""
**Sobre esta ferramenta**: 
Esta calculadora estima o orçamento de carbono disponível para o Brasil 
considerando diferentes cenários de emissões e metas de redução.

**Metodologia**:
- Projeções baseadas em crescimento composto por setor
- Meta de redução linear até 2050
- Cálculo de orçamento por integração numérica simples
""")
