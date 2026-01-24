# app.py - Dashboard de Análise de Emissões SEEG
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard SEEG - Emissões de GEE",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funções auxiliares
@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados do SEEG"""
    try:
        df = pd.read_csv('SEEG.csv', encoding='utf-8')
        
        # Transpor os dados para ter anos como linhas
        df_transposed = df.set_index('Categoria').T.reset_index()
        df_transposed = df_transposed.rename(columns={'index': 'Ano'})
        df_transposed['Ano'] = pd.to_numeric(df_transposed['Ano'])
        
        return df, df_transposed
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None, None

def formatar_numero(valor):
    """Formata números para exibição amigável"""
    if valor >= 1e9:
        return f"R$ {valor/1e9:.2f} Bilhões"
    elif valor >= 1e6:
        return f"R$ {valor/1e6:.2f} Milhões"
    elif valor >= 1e3:
        return f"R$ {valor/1e3:.2f} Mil"
    else:
        return f"R$ {valor:.2f}"

def calcular_crescimento(df, ano_inicio, ano_fim):
    """Calcula crescimento entre dois anos"""
    crescimentos = {}
    for categoria in df['Categoria'].unique():
        valor_inicio = df.loc[df['Categoria'] == categoria, str(ano_inicio)].values[0]
        valor_fim = df.loc[df['Categoria'] == categoria, str(ano_fim)].values[0]
        
        if valor_inicio != 0:
            crescimento = ((valor_fim - valor_inicio) / valor_inicio) * 100
        else:
            crescimento = 0
        
        crescimentos[categoria] = {
            'inicio': valor_inicio,
            'fim': valor_fim,
            'crescimento': crescimento,
            'absoluto': valor_fim - valor_inicio
        }
    
    return crescimentos

# Interface principal
def main():
    # Cabeçalho
    st.title("🌍 Dashboard de Análise - Emissões SEEG")
    st.markdown("""
    Sistema de Estimativas de Emissões de Gases de Efeito Estufa
    *Análise de dados de 1990 a 2024*
    """)
    
    # Carregar dados
    df, df_transposed = carregar_dados()
    
    if df is None:
        st.warning("Não foi possível carregar os dados. Verifique se o arquivo SEEG.csv está na pasta correta.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("Filtros de Análise")
        ano_min = int(df_transposed['Ano'].min())
        ano_max = int(df_transposed['Ano'].max())
        
        anos_selecionados = st.slider(
            "Período de Análise",
            min_value=ano_min,
            max_value=ano_max,
            value=(2010, 2024),
            step=1
        )
        
        todas_categorias = df['Categoria'].unique().tolist()
        categorias_selecionadas = st.multiselect(
            "Categorias para Análise",
            options=todas_categorias,
            default=todas_categorias
        )
        
        st.subheader("Métricas de Exibição")
        mostrar_valores = st.checkbox("Mostrar valores numéricos", value=True)
        usar_log = st.checkbox("Usar escala logarítmica", value=False)
        
        st.markdown("---")
        st.info(f"""
        **Informações do Dataset:**
        - {len(df)} categorias de emissão
        - Período: {ano_min} - {ano_max}
        - Total de anos: {len(df_transposed)}
        """)
    
    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "📈 Análise Temporal",
        "🔍 Comparativo",
        "📋 Dados Brutos"
    ])
    
    with tab1:
        # Visão Geral
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ano_atual = df_transposed['Ano'].max()
            st.metric(
                "Ano Mais Recente",
                ano_atual,
                delta=f"{ano_max - ano_min} anos de dados"
            )
        
        with col2:
            total_emissao = df[str(ano_atual)].sum()
            st.metric(
                "Emissões Totais (Último Ano)",
                formatar_numero(total_emissao),
                delta="Todos os setores"
            )
        
        with col3:
            maior_setor = df.loc[df[str(ano_atual)].idxmax(), 'Categoria']
            maior_valor = df[str(ano_atual)].max()
            st.metric(
                "Maior Emissor (Atual)",
                maior_setor,
                delta=formatar_numero(maior_valor)
            )
        
        # Gráfico de pizza do último ano
        st.subheader(f"Distribuição das Emissões por Setor ({ano_atual})")
        
        fig_pizza = px.pie(
            df,
            values=str(ano_atual),
            names='Categoria',
            color='Categoria',
            hover_data=[str(ano_atual)],
            labels={'Categoria': 'Setor', str(ano_atual): 'Emissões'}
        )
        
        fig_pizza.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Emissões: %{value:,.0f}<br>Percentual: %{percent}'
        )
        
        fig_pizza.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_pizza, use_container_width=True)
    
    with tab2:
        # Análise Temporal
        st.subheader("Evolução Temporal das Emissões")
        
        # Preparar dados para gráfico de linha
        df_analise = df[df['Categoria'].isin(categorias_selecionadas)]
        
        # Criar DataFrame para Plotly
        anos_lista = [str(ano) for ano in range(anos_selecionados[0], anos_selecionados[1] + 1)]
        
        plot_data = []
        for _, row in df_analise.iterrows():
            categoria = row['Categoria']
            for ano in anos_lista:
                plot_data.append({
                    'Categoria': categoria,
                    'Ano': int(ano),
                    'Emissões': row[ano]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Gráfico de linha
        fig_linha = px.line(
            df_plot,
            x='Ano',
            y='Emissões',
            color='Categoria',
            title=f"Evolução das Emissões ({anos_selecionados[0]} - {anos_selecionados[1]})",
            markers=True,
            line_shape='spline'
        )
        
        if usar_log:
            fig_linha.update_yaxes(type="log")
        
        fig_linha.update_layout(
            height=500,
            hovermode='x unified',
            xaxis=dict(tickmode='linear', dtick=2),
            yaxis_title="Emissões (unidades)",
            xaxis_title="Ano"
        )
        
        st.plotly_chart(fig_linha, use_container_width=True)
        
        # Gráfico de área
        st.subheader("Contribuição Acumulada por Setor")
        
        fig_area = px.area(
            df_plot,
            x='Ano',
            y='Emissões',
            color='Categoria',
            title="Evolução da Contribuição por Setor",
            groupnorm='percent'
        )
        
        fig_area.update_layout(
            height=400,
            yaxis_title="Percentual das Emissões Totais (%)",
            xaxis_title="Ano"
        )
        
        st.plotly_chart(fig_area, use_container_width=True)
    
    with tab3:
        # Análise Comparativa
        st.subheader("Análise Comparativa entre Períodos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ano_comparacao1 = st.selectbox(
                "Ano Inicial para Comparação",
                options=sorted(df_transposed['Ano'].unique()),
                index=0
            )
        
        with col2:
            ano_comparacao2 = st.selectbox(
                "Ano Final para Comparação",
                options=sorted(df_transposed['Ano'].unique()),
                index=len(df_transposed['Ano'].unique()) - 1
            )
        
        if ano_comparacao1 and ano_comparacao2:
            # Calcular crescimento
            crescimentos = calcular_crescimento(df, ano_comparacao1, ano_comparacao2)
            
            # Criar DataFrame para exibição
            comparacao_data = []
            for categoria, dados in crescimentos.items():
                comparacao_data.append({
                    'Categoria': categoria,
                    f'Emissões {ano_comparacao1}': dados['inicio'],
                    f'Emissões {ano_comparacao2}': dados['fim'],
                    'Variação Absoluta': dados['absoluto'],
                    'Variação Percentual (%)': dados['crescimento']
                })
            
            df_comparacao = pd.DataFrame(comparacao_data)
            
            # Exibir tabela
            st.dataframe(
                df_comparacao.style.format({
                    f'Emissões {ano_comparacao1}': '{:,.0f}',
                    f'Emissões {ano_comparacao2}': '{:,.0f}',
                    'Variação Absoluta': '{:,.0f}',
                    'Variação Percentual (%)': '{:.2f}%'
                }).background_gradient(
                    subset=['Variação Percentual (%)'],
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            # Gráfico de barras comparativo
            st.subheader(f"Comparação: {ano_comparacao1} vs {ano_comparacao2}")
            
            fig_comparacao = go.Figure()
            
            for i, categoria in enumerate(categorias_selecionadas):
                if categoria in crescimentos:
                    fig_comparacao.add_trace(go.Bar(
                        name=f'{categoria} - {ano_comparacao1}',
                        x=[categoria],
                        y=[crescimentos[categoria]['inicio']],
                        marker_color=px.colors.qualitative.Set1[i],
                        showlegend=True,
                        hovertemplate='<b>%{x}</b><br>Ano: ' + str(ano_comparacao1) + '<br>Emissões: %{y:,.0f}'
                    ))
                    
                    fig_comparacao.add_trace(go.Bar(
                        name=f'{categoria} - {ano_comparacao2}',
                        x=[categoria],
                        y=[crescimentos[categoria]['fim']],
                        marker_color=px.colors.qualitative.Set2[i],
                        showlegend=True,
                        hovertemplate='<b>%{x}</b><br>Ano: ' + str(ano_comparacao2) + '<br>Emissões: %{y:,.0f}'
                    ))
            
            fig_comparacao.update_layout(
                barmode='group',
                height=500,
                title=f"Comparação entre {ano_comparacao1} e {ano_comparacao2}",
                yaxis_title="Emissões",
                xaxis_title="Categoria"
            )
            
            st.plotly_chart(fig_comparacao, use_container_width=True)
    
    with tab4:
        # Dados Brutos
        st.subheader("Dados Completos do SEEG")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            formato_exibicao = st.radio(
                "Formato de Exibição",
                ["Original (Anos como colunas)", "Transposto (Anos como linhas)"],
                index=0
            )
        
        if formato_exibicao == "Original (Anos como colunas)":
            st.dataframe(
                df.style.format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x),
                use_container_width=True
            )
            
            # Botão para download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download dos Dados (CSV)",
                data=csv,
                file_name="SEEG_dados_completos.csv",
                mime="text/csv"
            )
        else:
            st.dataframe(
                df_transposed.style.format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x),
                use_container_width=True
            )
        
        # Estatísticas descritivas
        st.subheader("Estatísticas Descritivas")
        
        anos_para_analise = [str(ano) for ano in range(anos_selecionados[0], anos_selecionados[1] + 1)]
        df_estatisticas = df[['Categoria'] + anos_para_analise]
        
        estatisticas = []
        for categoria in df_estatisticas['Categoria']:
            dados_categoria = df_estatisticas[df_estatisticas['Categoria'] == categoria].iloc[0, 1:].values
            estatisticas.append({
                'Categoria': categoria,
                'Média': np.mean(dados_categoria),
                'Mediana': np.median(dados_categoria),
                'Desvio Padrão': np.std(dados_categoria),
                'Mínimo': np.min(dados_categoria),
                'Máximo': np.max(dados_categoria),
                'Crescimento Total': dados_categoria[-1] - dados_categoria[0],
                'Taxa Crescimento (%)': ((dados_categoria[-1] - dados_categoria[0]) / dados_categoria[0] * 100) if dados_categoria[0] != 0 else 0
            })
        
        df_stats = pd.DataFrame(estatisticas)
        st.dataframe(
            df_stats.style.format({
                'Média': '{:,.0f}',
                'Mediana': '{:,.0f}',
                'Desvio Padrão': '{:,.0f}',
                'Mínimo': '{:,.0f}',
                'Máximo': '{:,.0f}',
                'Crescimento Total': '{:,.0f}',
                'Taxa Crescimento (%)': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    # Rodapé
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("📊 **Dashboard SEEG**")
        st.caption("Versão 1.0 - Análise de Emissões")
    
    with col2:
        st.caption("🔄 **Dados Atualizados**")
        st.caption(f"Período: {ano_min} - {ano_max}")
    
    with col3:
        st.caption("🔍 **Análise Completa**")
        st.caption("5 categorias | 35 anos de dados")

if __name__ == "__main__":
    main()
