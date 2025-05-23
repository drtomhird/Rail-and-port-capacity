import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# --- Sidebar Inputs ---
st.sidebar.header("Model Inputs")

# General parameters
years = st.sidebar.slider("Model horizon (years)", min_value=1, max_value=30, value=10)
discount_rate = st.sidebar.number_input("Discount rate (annual %)", min_value=0.0, max_value=30.0, value=8.0) / 100

# Costs
haulage_cost = st.sidebar.number_input("Haulage cost ($/Mt·km)", min_value=0.0, value=0.05)
handling_cost = st.sidebar.number_input("Port handling cost ($/Mt)", min_value=0.0, value=2.0)

# Investment specification (lumpy build)
rail_increments = st.sidebar.multiselect(
    "Rail capacity increments (Mtpa)",
    options=[50, 100, 150, 200],
    default=[50]
)
port_increments = st.sidebar.multiselect(
    "Port capacity increments (Mtpa)",
    options=[125, 250, 375],
    default=[125]
)

# Minimum base capacity
min_capacity = st.sidebar.number_input("Minimum rail capacity per segment (Mtpa)", value=200)

# Mine locations and default DN values
st.sidebar.subheader("Mine distances (km) from DBCT")
default_DN = [55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
DN = []
for i, dist in enumerate(default_DN, start=1):
    DN.append(st.sidebar.number_input(f"Mine {i} distance", value=dist))

# Annual mine volumes (Mtpa)
st.sidebar.subheader("Annual outputs (Mtpa) per mine")
T = []
for i in range(len(DN)):
    T.append(st.sidebar.number_input(f"Mine {i+1} output", value=10.0))

# --- Main model logic ---
# Construct segment nodes
nodes = [0] + DN + [200]
segments = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

# Precompute flows by segment
flows = []
for k, (start, end) in enumerate(segments, start=1):
    # sum of T_i for mines at distance >= end
    flow = sum(T[i] for i, d in enumerate(DN) if d >= end)
    flows.append(flow)

# Capacity per segment starts at min_capacity
cap_current = [min_capacity] * len(segments)

# Investment schedule: assume at year 1 add all increments
# For simplicity, expansions applied at t=0
cap_schedule = pd.DataFrame(
    index=range(0, years+1),
    columns=[f"Seg {i}" for i in range(1, len(segments)+1)]
)
for year in cap_schedule.index:
    for idx in range(len(segments)):
        cap_schedule.at[year, f"Seg {idx+1}"] = cap_current[idx]

# Placeholder: apply rail increments evenly across segments
for inc in rail_increments:
    for idx in range(len(segments)):
        cap_schedule.at[1, f"Seg {idx+1}"] += inc

# Compute annual costs and NPV
timeline = range(0, years+1)
results = []
for y in timeline:
    # Expansion costs at t=0 -> assume cost per unit capacity
    if y == 0:
        rail_inv_cost = sum(rail_increments) * 1000  # $1000 per Mtpa placeholder
        port_inv_cost = sum(port_increments) * 2000  # $2000 per Mtpa placeholder
    else:
        rail_inv_cost = 0
        port_inv_cost = 0
    # Haulage & handling mid-year
    mid_discount = (1 + discount_rate) ** (y + 0.5)
    vol = sum(T)
    haul_cost = vol * haulage_cost * 200  # average distance placeholder
    handle_cost = vol * handling_cost
    npv_haul = haul_cost / mid_discount
    npv_handle = handle_cost / mid_discount
    npv_rail_inv = rail_inv_cost / ((1 + discount_rate)**y)
    npv_port_inv = port_inv_cost / ((1 + discount_rate)**y)
    results.append({
        "Year": y,
        "Rail Inv Cost": rail_inv_cost,
        "Port Inv Cost": port_inv_cost,
        "Haulage Cost": haul_cost,
        "Handling Cost": handle_cost,
        "NPV Rail Inv": npv_rail_inv,
        "NPV Port Inv": npv_port_inv,
        "NPV Haulage": npv_haul,
        "NPV Handling": npv_handle
    })

df_res = pd.DataFrame(results)

df_res["Cumulative NPV"] = df_res[["NPV Rail Inv", "NPV Port Inv", "NPV Haulage", "NPV Handling"]].sum(axis=1).cumsum()

# --- Outputs ---
st.header("Model Outputs")
st.dataframe(df_res)

# Rail capacity by segment by year
st.subheader("Rail Capacity Schedule (Mtpa)")
st.dataframe(cap_schedule)

# Download CSV
csv = df_res.to_csv(index=False)
st.download_button("Download results CSV", data=csv, file_name="model_outputs.csv", mime="text/csv")
