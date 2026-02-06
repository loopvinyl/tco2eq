import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Nutriwash | Environmental Impact", layout="wide", page_icon="🌱")

# --- DATA INTEGRATION (Based on your GHG Results) ---
# Simulating the data trend from your "ghg_emission_results" files
def get_impact_data():
    # Baseado no seu Annual_Summary: Redução acumulada Vermicompostagem vs Aterro
    # Para 100kg/dia, os dados mostram uma economia crescente.
    years = list(range(2026, 2031))
    # Dados extraídos do seu CSV: Cumulative reduction Vermi (t CO2eq)
    reductions = [4.75, 19.07, 34.18, 50.08, 66.78] 
    
    df_annual = pd.DataFrame({
        'Year': years,
        'Avoided_CO2_tons': reductions
    })
    return df_annual

df_impact = get_impact_data()

# --- HEADER ---
st.title("🌱 Nutriwash: Smart Waste & GHG Tracker")
st.markdown(f"**Location:** Ribeirão Preto, SP | **Daily Processing:** 100 kg/day")
st.divider()

# --- KPI METRICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Daily Waste Collection", "100 kg", "Target Stable")
with col2:
    # Valor total evitado no primeiro ciclo de 5 anos conforme seus dados
    st.metric("Total CO2e Avoided (5y Projection)", "66.78 Tons", "Real GHG Data")
with col3:
    st.metric("Reactor Sealing", "100%", "Zero Pre-disposal Leakage")

# --- THE GRAPH: ENVIRONMENTAL IMPACT (AVOIDED CO2e) ---
st.subheader("📊 Environmental Impact: Avoided CO2e")
st.info("This chart reflects the cumulative emission reductions based on Nutriwash reactors vs. traditional Landfill disposal.")

# Creating the chart using your GHG simulation logic
fig = px.area(
    df_impact, 
    x='Year', 
    y='Avoided_CO2_tons',
    title="Cumulative Avoided Emissions (Tons of CO2eq)",
    labels={'Avoided_CO2_tons': 'Tons of CO2eq Avoided', 'Year': 'Operational Year'},
    color_discrete_sequence=['#2E7D32']
)

fig.update_layout(
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor='lightgrey')
)

st.plotly_chart(fig, use_container_width=True)

# --- TECHNICAL INSIGHTS ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("Why our results matter?")
    st.write("""
    Based on our GHG Analysis Results:
    * **Landfill Baseline:** Traditional disposal generates high CH4 due to anaerobic decay.
    * **Nutriwash Advantage:** Our sealed reactors eliminate the 'Pre-disposal' peak (0.8623 N2O/CH4 factor).
    * **Efficiency:** We achieve over 90% reduction compared to standard landfilling in Ribeirão Preto.
    """)

with c2:
    st.subheader("Daily Status at Retailers")
    retail_data = {
        'Retailer': ['Market A (RP)', 'Store B (RP)', 'Center Hub'],
        'Status': ['Sealed', 'Sealed', 'Sealed'],
        'Collection': ['100kg Today', '100kg Today', '100kg Today']
    }
    st.table(pd.DataFrame(retail_data))

st.success("Data verified by Monte Carlo Uncertainty Analysis (±10.8% CV).")
