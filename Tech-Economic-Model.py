import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import networkx as nx
import json

# -------------------------------------------
# Page Config
# -------------------------------------------
st.set_page_config(page_title="Techno-Economic Decision Support", layout="wide")

# -------------------------------------------
# Settings & Global Parameters
# -------------------------------------------
np.random.seed(42)
years = np.arange(2025, 2036)
future_years = np.arange(2036, 2041)

def default_multipliers():
    return {
        'Baseline':               {'yield':1.00, 'carbon':1.00, 'water':1.00, 'policy':0.0},
        'AI Adoption':            {'yield':1.05, 'carbon':0.95, 'water':0.90, 'policy':0.0},
        'Quantum Leap':           {'yield':1.10, 'carbon':0.90, 'water':0.85, 'policy':0.0},
        'Blockchain Integration': {'yield':1.02, 'carbon':0.93, 'water':0.95, 'policy':0.0},
        'Satellite Farming':      {'yield':1.08, 'carbon':0.92, 'water':0.88, 'policy':0.0}
    }

scenarios = list(default_multipliers().keys())

# -------------------------------------------
# Sidebar: Scenario Builder & Controls
# -------------------------------------------
st.sidebar.title("Scenario Builder & Controls")

multipliers = {}
for scen in scenarios:
    st.sidebar.subheader(scen)
    y = st.sidebar.slider(f"Yield Boost {scen}",   0.5, 1.5, default_multipliers()[scen]['yield'],  0.01)
    c = st.sidebar.slider(f"Carbon Factor {scen}",  0.5, 1.5, default_multipliers()[scen]['carbon'], 0.01)
    w = st.sidebar.slider(f"Water Factor {scen}",   0.5, 1.5, default_multipliers()[scen]['water'],  0.01)
    p = st.sidebar.slider(f"Policy Shock {scen}",  -0.5, 0.5, default_multipliers()[scen]['policy'], 0.05)
    multipliers[scen] = {'yield':y,'carbon':c,'water':w,'policy':p}

mc_runs = st.sidebar.slider("Monte Carlo Runs", 100, 2000, 500, step=100)
yield_threshold = st.sidebar.number_input("Yield Alert Threshold", 0.0, 200.0, 85.0)

view = st.sidebar.selectbox(
    "Select View",
    [
        "Choropleth Map",
        "Risk Dashboard",
        "Forecast Comparison",
        "3D PCA",
        "Agent-Based Model",
        "Supply Chain",
        "Economic ROI",
        "Download & Reports"
    ]
)

# -------------------------------------------
# Data Simulation Functions
# -------------------------------------------
def simulate_trend(years, base, slope, noise):
    return base + slope*(years - years[0]) + np.random.normal(0, noise, len(years))

def build_df(multipliers):
    records = []
    counties = ['Carlow','Cavan','Clare','Cork','Donegal','Dublin','Galway','Kerry',
                'Kildare','Kilkenny','Laois','Leitrim','Limerick','Longford','Louth',
                'Mayo','Meath','Monaghan','Offaly','Roscommon','Sligo','Tipperary',
                'Waterford','Westmeath','Wexford','Wicklow']
    for scen in scenarios:
        m = multipliers[scen]
        y_ser = simulate_trend(years, 100*m['yield'], 1.5*m['yield'], 3)
        c_ser = simulate_trend(years,  70*m['carbon'], -0.5*m['carbon'], 2)
        w_ser = simulate_trend(years,  50*m['water'],  -1.0*m['water'], 2)
        shock_idx = len(years) // 2
        y_ser[shock_idx:] *= (1 + m['policy'])
        for i, yr in enumerate(years):
            records.append({
                'Year': yr,
                'Scenario': scen,
                'AgriYield': y_ser[i],
                'CarbonFootprint': c_ser[i],
                'WaterUsage': w_ser[i],
                'County': counties[i % len(counties)]
            })
    return pd.DataFrame(records)

@st.cache_data
def get_data():
    return build_df(multipliers)

df = get_data()

def monte_carlo_ci(series, runs):
    sims = np.random.choice(series, (runs, len(series)))
    return np.percentile(sims, [5, 95], axis=0)

# -------------------------------------------
# Forecasting Functions
# -------------------------------------------
def compute_forecasts(df, indicator='AgriYield'):
    results = {}
    for scen in scenarios:
        sub = df[df.Scenario == scen][['Year', indicator]].set_index('Year')
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(sub)
        window = 5
        X, y = [], []
        for i in range(len(scaled)-window):
            X.append(scaled[i:i+window])
            y.append(scaled[i+window])
        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        # LSTM
        model_l = Sequential([LSTM(16, input_shape=(window,1)), Dense(1)])
        model_l.compile('adam','mse')
        model_l.fit(X_train, y_train, epochs=10, verbose=0)
        # GRU
        model_g = Sequential([GRU(16, input_shape=(window,1)), Dense(1)])
        model_g.compile('adam','mse')
        model_g.fit(X_train, y_train, epochs=10, verbose=0)
        # Forecast loop
        seq, pl, pg = scaled[-window:], [], []
        for _ in future_years:
            inp = seq.reshape(1, window, 1)
            ol = model_l.predict(inp, verbose=0)[0]
            og = model_g.predict(inp, verbose=0)[0]
            pl.append(ol); pg.append(og)
            seq = np.vstack([seq[1:], ol])
        pl = scaler.inverse_transform(np.array(pl))
        pg = scaler.inverse_transform(np.array(pg))
        # Linear baseline
        lr = LinearRegression().fit(sub.index.values.reshape(-1,1), sub[indicator])
        lin = lr.predict(future_years.reshape(-1,1))
        results[scen] = {'Linear': lin, 'LSTM': pl.flatten(), 'GRU': pg.flatten()}
    return results

# -------------------------------------------
# Views Implementation
# -------------------------------------------

if view == "Choropleth Map":
    st.header("Ireland County-Level Choropleth")
    geo_file = st.file_uploader("Upload Ireland GeoJSON", type=['geojson'])
    metric = st.selectbox("Select Metric", ['AgriYield','CarbonFootprint','WaterUsage'])
    year_sel = st.slider("Select Year", int(years.min()), int(years.max()), int(years.mean()))
    scen_sel = st.selectbox("Select Scenario", scenarios)

    if geo_file:
        geojson = json.load(geo_file)
        subset = df[(df.Year==year_sel)&(df.Scenario==scen_sel)]
        agg = subset.groupby('County')[metric].mean().reset_index()
        fig = px.choropleth(
            agg,
            geojson=geojson,
            locations='County',
            featureidkey='properties.name',
            color=metric,
            color_continuous_scale='Viridis',
            projection='mercator'
        )
        fig.update_geos(fitbounds='locations', visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a valid County GeoJSON file.")

elif view == "Risk Dashboard":
    st.header("Monte Carlo Risk Analysis")
    for scen in scenarios:
        sub = df[df.Scenario==scen]
        low, high = monte_carlo_ci(sub.AgriYield.values, mc_runs)
        fig = go.Figure([
            go.Scatter(x=years, y=high, mode='lines', name='95th %ile'),
            go.Scatter(x=years, y=low,  mode='lines', fill='tonexty', name='5th %ile'),
            go.Scatter(x=years, y=sub.AgriYield, mode='lines', name='Mean')
        ])
        st.plotly_chart(fig, use_container_width=True)

elif view == "Forecast Comparison":
    st.header("Forecast Comparison")
    indicator = st.selectbox("Select Indicator", ['AgriYield','CarbonFootprint','WaterUsage'])
    forecasts = compute_forecasts(df, indicator)
    for scen in scenarios:
        if st.checkbox(f"Show {scen}", True):
            hist = df[df.Scenario==scen]
            fc = forecasts[scen]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.Year, y=hist[indicator], mode='lines', name='Historical'))
            for k, series in fc.items():
                fig.add_trace(go.Scatter(x=future_years, y=series, mode='lines', name=k))
            st.plotly_chart(fig, use_container_width=True)

elif view == "3D PCA":
    st.header("3D PCA Scenario View")
    feats = ['AgriYield','CarbonFootprint','WaterUsage']
    pcs = PCA(n_components=3).fit_transform(df[feats])
    df_pca = df.copy()
    df_pca[['PC1','PC2','PC3']] = pcs
    fig = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3',
        color='Scenario', animation_frame='Year'
    )
    st.plotly_chart(fig, use_container_width=True)

elif view == "Agent-Based Model":
    st.header("Agent-Based Adoption Simulation")
    risk = st.slider("Agent Risk Aversion", 0.0, 1.0, 0.5)
    conn = st.slider("Network Connectivity", 1, 10, 3)
    G = nx.erdos_renyi_graph(100, conn/100)
    adoption = np.zeros((100,len(years)))
    seeds = np.random.choice(list(G.nodes()),5,replace=False)
    adoption[seeds,0] = 1
    for t in range(1,len(years)):
        for i in G.nodes():
            if adoption[i,t-1]==0:
                neigh = list(G.neighbors(i))
                if np.mean(adoption[neigh,t-1] if neigh else [0]) > risk:
                    adoption[i,t]=1
    rate = adoption.mean(axis=0)
    fig = px.line(x=years, y=rate, labels={'x':'Year','y':'Adoption Rate'})
    st.plotly_chart(fig, use_container_width=True)

elif view == "Supply Chain":
    st.header("Supply-Chain Flow & Emissions")
    nodes = ['Farm','Processor','Distributor','Retail']
    flow  = [100,80,60]
    sankey = go.Sankey(
        node=dict(label=nodes),
        link=dict(source=[0,1,2],target=[1,2,3],value=flow)
    )
    fig = go.Figure(sankey)
    st.plotly_chart(fig, use_container_width=True)

elif view == "Economic ROI":
    st.header("ROI & NPV Calculator")
    capex     = st.number_input("CAPEX (€)", 100000.0)
    opex      = st.number_input("Annual OPEX (€)", 20000.0)
    discount  = st.slider("Discount Rate (%)", 0.0, 20.0, 5.0)/100
    period    = st.slider("Analysis Period (years)",1,20,10)
    revenues  = [capex * 0.2]*period
    cashflows = [-capex] + [r-opex for r in revenues]
    npv = sum(cf/(1+discount)**i for i,cf in enumerate(cashflows))
    st.metric("NPV (€)", f"{npv:,.2f}")

else:  # Download & Reports
    st.header("Download & Report Generation")
    st.download_button("Download CSV", df.to_csv(index=False), "data.csv", "text/csv")
    st.download_button("Download JSON", df.to_json(orient='records'), "data.json", "application/json")
    if st.button("Generate PDF Summary"):
        st.info("PDF export stub – integrate pdfkit or WeasyPrint.")

# -------------------------------------------
# Threshold Alerts
# -------------------------------------------
avg_yield = df.groupby('Scenario').AgriYield.mean()
for scen, val in avg_yield.items():
    if val < yield_threshold:
        st.warning(f"Avg yield under '{scen}' below threshold: {val:.2f}")
