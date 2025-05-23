import streamlit as st
import pandas as pd
import numpy as np

# --- Sidebar Inputs ---
st.sidebar.header("Model Inputs")

# General parameters
years = st.sidebar.slider("Model horizon (years)", min_value=1, max_value=30, value=10)
discount_rate = st.sidebar.number_input("Discount rate (annual %)", min_value=0.0, max_value=30.0, value=10.0) / 100

# Costs
haulage_cost = st.sidebar.number_input("Haulage cost ($/Mt·km)", min_value=0.0, value=1.0)
handling_cost_dbct = st.sidebar.number_input("Port handling cost DBCT ($/Mt)", min_value=0.0, value=1.0)
handling_cost_appt = st.sidebar.number_input("Port handling cost APPT ($/Mt)", min_value=0.0, value=1.0)

# Investment specification (lumpy build)
st.sidebar.subheader("Rail expansion specification")
rail_increment = st.sidebar.number_input("Rail capacity increment (Mtpa)", min_value=0.0, value=50.0)
rail_expansion_cost_per_mtpa = st.sidebar.number_input("Rail capex ($ per Mtpa)", min_value=0.0, value=1000.0)

st.sidebar.subheader("Port expansion specification")
port_increment_dbct = st.sidebar.number_input("DBCT capacity increment (Mtpa)", min_value=0.0, value=125.0)
port_expansion_cost_per_mtpa_dbct = st.sidebar.number_input("DBCT capex ($ per Mtpa)", min_value=0.0, value=2000.0)
port_increment_appt = st.sidebar.number_input("APPT capacity increment (Mtpa)", min_value=0.0, value=125.0)
port_expansion_cost_per_mtpa_appt = st.sidebar.number_input("APPT capex ($ per Mtpa)", min_value=0.0, value=2000.0)

# Minimum base rail capacity
min_capacity = st.sidebar.number_input("Minimum rail capacity per segment (Mtpa)", min_value=0.0, value=200.0)

# Mine distances and assignments
st.sidebar.subheader("Mine distances & port assignment")
default_DN = [55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
DN, T, port_assign = [], [], []
for i, dist in enumerate(default_DN, start=1):
    DN.append(st.sidebar.number_input(f"Mine {i} distance (km)", value=dist))
    T.append(st.sidebar.number_input(f"Mine {i} output (Mtpa)", value=100.0))
    port_assign.append(st.sidebar.selectbox(f"Mine {i} port", options=["DBCT", "APPT"], index=0))

# --- Main model logic ---
# Nodes and segments
nodes = [0] + DN + [200]
segments = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

# Compute train flows per segment (return trips)
flows = []
for start, end in segments:
    flow = sum(T[i] for i, d in enumerate(DN) if d >= end)
    flows.append(flow)

# Base capacities
cap_current = [min_capacity] * len(segments)

# Capacity schedule DataFrame
years_idx = range(0, years+1)
cap_schedule = pd.DataFrame(index=years_idx,
                             columns=[f"Seg {i+1}" for i in range(len(segments))])
for y in years_idx:
    for i in range(len(segments)):
        cap_schedule.at[y, f"Seg {i+1}"] = cap_current[i]

# Apply expansions at t=0
# Rail
cap_schedule.iloc[1] = cap_schedule.iloc[1] + rail_increment
# Ports: track as special nodes? (Not shown in cap_schedule)

# Annual costs and NPV calculation
results = []
for y in years_idx:
    # Investment costs at t=0
    if y == 0:
        rail_inv_cost = rail_increment * rail_expansion_cost_per_mtpa
        port_inv_dbct = port_increment_dbct * port_expansion_cost_per_mtpa_dbct
        port_inv_appt = port_increment_appt * port_expansion_cost_per_mtpa_appt
    else:
        rail_inv_cost = port_inv_dbct = port_inv_appt = 0.0

    # Operational costs mid-year
    mid_disc = (1 + discount_rate) ** (y + 0.5)
    total_haul = sum(T) * haulage_cost * 200  # average distance placeholder
    # Handling by port
    haul_by_port = {"DBCT": 0.0, "APPT": 0.0}
    for i, vol in enumerate(T):
        dist = DN[i] if port_assign[i] == "DBCT" else (200 - DN[i])
        haul_by_port[port_assign[i]] += vol * haulage_cost * dist
    handle_dbct = sum(T[i] for i in range(len(T)) if port_assign[i] == "DBCT") * handling_cost_dbct
    handle_appt = sum(T[i] for i in range(len(T)) if port_assign[i] == "APPT") * handling_cost_appt

    # Discount to NPV
    npv_haul = total_haul / mid_disc
    npv_handle_dbct = handle_dbct / mid_disc
    npv_handle_appt = handle_appt / mid_disc
    npv_rail_inv = rail_inv_cost / ((1 + discount_rate) ** y)
    npv_port_dbct = port_inv_dbct / ((1 + discount_rate) ** y)
    npv_port_appt = port_inv_appt / ((1 + discount_rate) ** y)

    results.append({
        "Year": y,
        "Rail Inv Cost": rail_inv_cost,
        "Port Inv Cost DBCT": port_inv_dbct,
        "Port Inv Cost APPT": port_inv_appt,
        "NPV Rail Inv": npv_rail_inv,
        "NPV Port Inv DBCT": npv_port_dbct,
        "NPV Port Inv APPT": npv_port_appt,
        "NPV Haulage": npv_haul,
        "NPV Handle DBCT": npv_handle_dbct,
        "NPV Handle APPT": npv_handle_appt
    })

# Results DataFrame
df_res = pd.DataFrame(results)
df_res["Cumulative NPV"] = df_res[["NPV Rail Inv", "NPV Port Inv DBCT", "NPV Port Inv APPT", "NPV Haulage", "NPV Handle DBCT", "NPV Handle APPT"]].sum(axis=1).cumsum()

# --- Outputs ---
st.header("Model Outputs")
st.dataframe(df_res)
st.subheader("Rail Capacity Schedule (Mtpa)")
st.dataframe(cap_schedule)

# CSV Download
csv = df_res.to_csv(index=False)
st.download_button("Download results CSV", data=csv, file_name="model_outputs.csv", mime="text/csv")
