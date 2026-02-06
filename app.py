import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Nutriwash | Waste-to-Value", layout="wide", page_icon="🌱")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Introduction
st.title("🌱 Nutriwash: Smart Waste Management")
st.markdown("""
    **Location: Ribeirão Preto, Brazil** Real-time monitoring of anaerobic reactors installed at fruit and vegetable retailers.  
    *Total sealing technology ensuring zero pre-disposal methane emissions.*
""")

# --- DATA SIMULATION ---
@st.cache_data
def load_data():
    # Simulating 30 days of operations
    dates = pd.date_range(end=datetime.now(), periods=30)
    # We maintain the 100kg average as requested
    data = pd.DataFrame({
        'Date': dates,
        'Collected Waste (kg)': np.random.normal(100, 3, 30), 
        'Avoided Emissions (kg CO2e)': np.random.normal(42, 2, 30),
        'Reactor Efficiency (%)': np.random.uniform(98.5, 99.9, 30)
    })
    return data

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Dashboard Settings")
unit_filter = st.sidebar.selectbox("Select Unit", ["All Retailers", "Ceasa RP", "Local Network A", "Green Market B"])
st.sidebar.divider()
st.sidebar.write("**Reactor Status:** 🟢 Operational")
st.sidebar.write("**Sealing Integrity:** 100% Hermetic")
st.sidebar.info("Since the project started, 100% of the collected organic waste has been diverted from landfills.")

# --- KEY METRICS ---
col1, col2, col3 = st.columns(3)
current_total = df['Collected Waste (kg)'].sum()
avg_daily = df['Collected Waste (kg)'].mean()

with col1:
    st.metric("Total Collected (30d)", f"{current_total:,.0f} kg", "Target Met")
with col2:
    # Highlighting the 100kg requirement
    st.metric("Daily Average", f"{avg_daily:.1f} kg", "Target: 100kg/day")
with col3:
    st.metric("Methane Capture Rate", "99.9%", "Proprietary Tech")

st.divider()

# --- CHARTS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Daily Collection Volume (kg)")
    fig_coleta = px.line(df, x='Date', y='Collected Waste (kg)', markers=True, 
                         color_discrete_sequence=['#2E7D32'])
    # Goal line at 100kg
    fig_coleta.add_hline(y=100, line_dash="dot", 
                         annotation_text="Daily Goal (100kg)", 
                         annotation_position="bottom right")
    st.plotly_chart(fig_coleta, use_container_width=True)

with c2:
    st.subheader("Environmental Impact: Avoided CO2e")
    fig_env = px.area(df, x='Date', y='Avoided Emissions (kg CO2e)', 
                     color_discrete_sequence=['#81C784'])
    st.plotly_chart(fig_env, use_container_width=True)

# --- OPERATIONAL STATUS TABLE ---
st.subheader("Retailer Point Status")
st.write("Real-time sensor data from our sealed anaerobic reactors:")

status_df = pd.DataFrame({
    'Retailer Location': ['Downtown Market', 'Sertãozinho Hub', 'RP South Grocery', 'Avenue Green Shop'],
    'Current Capacity': ['82%', '35%', '91%', '12%'],
    'Seal Status': ['Secured', 'Secured', 'Secured', 'Secured'],
    'Last Pickup': ['2h ago', '5h ago', '15min ago', 'Yesterday']
})

# Displaying table without index
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.success("✅ System operating under full compliance with environmental standards in Ribeirão Preto.")
