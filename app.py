# Substituir a função formatar_br existente por:
def formatar_br(numero):
    """
    Formata números no padrão brasileiro: 1.234,56
    """
    if pd.isna(numero):
        return "N/A"
    
    # Verificar se é inteiro ou decimal
    if numero == int(numero):
        # Formatação para inteiros
        return f"{int(numero):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        # Formatação para decimais
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Substituir a função br_format existente por:
def br_format_inteiro(x, pos):
    return f'{x:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def br_format_decimal(x, pos):
    return f'{x:.2f}'.replace('.', ',')

# Criar os formatadores
br_formatter_inteiro = FuncFormatter(br_format_inteiro)
br_formatter_decimal = FuncFormatter(br_format_decimal)

# Atualizar todos os gráficos para usar os novos formatadores
# Exemplo de modificação em um gráfico (aplicar similarmente a todos):
def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}"
    if abs(x) >= 1000:
        return br_format_inteiro(x, pos)
    return br_format_decimal(x, pos)

# Aplicar a formatação brasileira em todas as tabelas:
# Para cada dataframe exibido com st.dataframe, aplicar:
df_formatado = df_original.copy()
for col in df_formatado.columns:
    if df_formatado[col].dtype in ['float64', 'int64']:
        df_formatado[col] = df_formatado[col].apply(formatar_br)

# Atualizar as métricas para usar formatação brasileira:
st.metric("Label", f"{formatar_br(valor)} unidades")
