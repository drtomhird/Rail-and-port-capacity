# Rails and Ports Capacity Model

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Sidebar Inputs ---
st.sidebar.header("Model Inputs")

# Railway and port settings
distance = st.sidebar.number_input("Total rail distance (km)", value=200.0)
num_mines = st.sidebar.number_input("Number of mines", min_value=1, max_value=20, value=10)

data = {
    "Distance": [],
    "Output0": [],
    "Growth%": [],
    "FixedPort": []
}
for i in range(int(num_mines)):
    data["Distance"].append(
        st.sidebar.number_input(f"Mine {i+1} distance (km)", value=55+10*i, key=f"d{i}")
    )
    data["Output0"].append(
        st.sidebar.number_input(f"Mine {i+1} output Year0 (units)", value=100.0, key=f"o{i}")
    )
    data["Growth%"].append(
        st.sidebar.number_input(f"Mine {i+1} growth rate (%)", value=10.0, key=f"g{i}" )/100
    )
    data["FixedPort"].append(
        st.sidebar.selectbox(
            f"Mine {i+1} fixed port (Fixed model)", ["DBCT","APPT"], key=f"p{i}"
        )
    )
mines = pd.DataFrame(data)

# Cost and increment settings
port_inc = st.sidebar.number_input("Port capacity increment (units)", value=125.0)
port_cost = st.sidebar.number_input("Port increment cost ($ per unit)", value=10.0)
haul_cost = st.sidebar.number_input("Haulage cost ($ per unit·km)", value=0.1)
discount = st.sidebar.number_input("Discount rate (%)", value=10.0)/100

# Model horizon
years = st.sidebar.number_input("Model horizon (years)", min_value=1, max_value=50, value=20)

# Prepare results holder
years_idx = np.arange(0, years+1)
fixed_exp = []
flex_exp = []
flex_haul_diff = []
flex_vol_reroute = []

# Initial port capacities
db_cap_fixed = ap_cap_fixed = port_inc*4   # 4*125=500 default Year0

db_cap_flex = ap_cap_flex = port_inc*4

def annualize(cap, rate):
    return cap * rate

for y in years_idx:
    # compute outputs
grow = (1+mines["Growth%"])
if y==0:
    out = mines["Output0"].values.copy()
else:
    out *= grow
# total demand
 demand = out.sum()

# Fixed model port expansions
db_flow = out[mines.FixedPort=="DBCT"].sum()
ap_flow = out[mines.FixedPort=="APPT"].sum()
fixed_capex = 0
if db_flow>db_cap_fixed:
    fixed_capex += port_inc * port_cost
    db_cap_fixed += port_inc
if ap_flow>ap_cap_fixed:
    fixed_capex += port_inc * port_cost
    ap_cap_fixed += port_inc
fixed_exp.append(fixed_capex)

# Flexible model decision: consider build DBCT only, APPT only, both or none
best_val = float('inf')
best = None
for build_db in [0,1]:
    for build_ap in [0,1]:
        cap_db = db_cap_flex + build_db*port_inc
        cap_ap = ap_cap_flex + build_ap*port_inc
        # allocate flow: minimize system cost = capex annualized + haulage cost
        capex_ann = annualize((build_db+build_ap)*port_inc, discount)*port_cost
        # allocate each mine: try assign to cheaper available port, overflow to other
        haul = 0
        reroute_vol = 0
        for d,o,p in zip(mines.Distance, out, mines.FixedPort):
            cost_db = d*haul_cost
            cost_ap = (distance-d)*haul_cost
            # primary cheapest
            if cost_db<cost_ap:
                send_db = min(o, cap_db - 0)
                send_ap = o - send_db
            else:
                send_ap = min(o, cap_ap - 0)
                send_db = o - send_ap
            haul += send_db*d*haul_cost + send_ap*(distance-d)*haul_cost
            reroute_vol += send_ap if p=="DBCT" else send_db
        total = capex_ann + haul
        if total<best_val:
            best_val,total,reroute_vol = total,total,reroute_vol
            best=(build_db,build_ap)
flex_exp.append((best[0]+best[1])*port_inc*port_cost)
flex_haul_diff.append(haul - (db_flow*d*haul_cost + ap_flow*(distance-ap_flow/len(out))*haul_cost))
flex_vol_reroute.append(reroute_vol)
# apply best
db_cap_flex += best[0]*port_inc
ap_cap_flex += best[1]*port_inc

# Assemble into DataFrame
results = pd.DataFrame({
    'Year': years_idx,
    'FixedPortCapEx': fixed_exp,
    'FlexPortCapEx': flex_exp,
    'FlexHaulDiff': flex_haul_diff,
    'FlexRerouteVol': flex_vol_reroute
})

# calculate NPV from year1 onwards
results['FixedNPV'] = ((results.FixedPortCapEx/(1+discount)**results.Year).cumsum())
results['FlexNPV']  = ((results.FlexPortCapEx/(1+discount)**results.Year).cumsum())

# Outputs
st.header("Comparison of Fixed vs Flexible")
st.line_chart(results.set_index('Year')[['FixedNPV','FlexNPV']].iloc[1:])
st.bar_chart(results.set_index('Year')[['FixedPortCapEx','FlexPortCapEx']].iloc[1:])
st.bar_chart(results.set_index('Year')['FlexHaulDiff'].iloc[1:])
st.bar_chart(results.set_index('Year')['FlexRerouteVol'].iloc[1:])

st.download_button("Download CSV", data=results.to_csv(index=False), file_name="model_results.csv")

# Summary metrics
table = pd.DataFrame({
    'Metric': ['NPV Fixed minus Flexible','% of Flexible Port NPV'],
    'Value': [results['FixedNPV'].iloc[1:].iloc[-1] - results['FlexNPV'].iloc[1:].iloc[-1],
              (results['FixedNPV'].iloc[1:].iloc[-1] - results['FlexNPV'].iloc[1:].iloc[-1]) / results['FlexNPV'].iloc[1:].iloc[-1] * 100]
})
st.table(table)
