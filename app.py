import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Sidebar Inputs ---
st.sidebar.header("Model Inputs")

# Model horizon and discount rate
years = st.sidebar.number_input("Model horizon (years)", min_value=1, max_value=50, value=20)
discount_rate = st.sidebar.number_input("Discount rate (%)", min_value=0.0, value=10.0) / 100

# Total distance between terminals
distance_total = st.sidebar.number_input("Total rail distance (km)", value=200.0)

# --- Mine definitions ---
st.sidebar.subheader("Mine settings")
mine_data = []
num_mines = st.sidebar.number_input("Number of mines", min_value=1, max_value=15, value=10)
for i in range(int(num_mines)):
    d = st.sidebar.number_input(f"Mine {i+1} distance (km)", value=55 + 10*i, key=f"d{i}")
    o = st.sidebar.number_input(f"Mine {i+1} initial output (Mtpa)", value=100.0, key=f"o{i}")
    g = st.sidebar.number_input(f"Mine {i+1} growth rate (%)", value=10.0, key=f"g{i}") / 100
    # default port assignment based on nearest: DBCT if closer to 0km, else APPT
    default_idx = 0 if d <= distance_total/2 else 1
    p = st.sidebar.selectbox(f"Mine {i+1} fixed port", ["DBCT", "APPT"], index=default_idx, key=f"p{i}")
    mine_data.append({'dist': d, 'out0': o, 'growth': g, 'port_fixed': p})

# --- Capacity increments & capex ---
st.sidebar.subheader("Capacity increments & costs")
rail_inc = st.sidebar.number_input("Rail capacity increment (Mtpa)", value=50.0)
rail_cost = st.sidebar.number_input("Rail capex ($/Mtpa)", value=1000.0)
port_inc_db = st.sidebar.number_input("DBCT increment (Mtpa)", value=125.0)
port_cost_db = st.sidebar.number_input("DBCT capex ($/Mtpa)", value=2000.0)
port_inc_ap = st.sidebar.number_input("APPT increment (Mtpa)", value=125.0)
port_cost_ap = st.sidebar.number_input("APPT capex ($/Mtpa)", value=2000.0)

# --- Operating cost rates ---
st.sidebar.subheader("Operating cost rates")
haul_rate = st.sidebar.number_input("Haulage cost ($/Mt·km)", value=1.0)
handle_rate_db = st.sidebar.number_input("DBCT handling cost ($/Mt)", value=1.0)
handle_rate_ap = st.sidebar.number_input("APPT handling cost ($/Mt)", value=1.0)

# --- Build segments ---
nodes = [0] + [m['dist'] for m in mine_data] + [distance_total]
segments = list(zip(nodes[:-1], nodes[1:]))

# --- Year 0 flows for seeding base capacities ---
init_out = [m['out0'] for m in mine_data]
seg_flow0 = [sum(o for o, m in zip(init_out, mine_data) if m['dist'] >= b) for (_, b) in segments]
flow_db0 = sum(m['out0'] for m in mine_data if m['port_fixed'] == 'DBCT')
flow_ap0 = sum(m['out0'] for m in mine_data if m['port_fixed'] == 'APPT')

# --- Seed base capacities ---
min_rail = rail_inc * 4  # at least 4 increments per segment
base_seg_cap = [max(min_rail, math.ceil(f / rail_inc) * rail_inc) for f in seg_flow0]
base_db_cap = max(port_inc_db, math.ceil(flow_db0 / port_inc_db) * port_inc_db)
base_ap_cap = max(port_inc_ap, math.ceil(flow_ap0 / port_inc_ap) * port_inc_ap)

# --- Generate possible build actions ---
def gen_actions():
    actions = []
    # up to two total increments distributed across rail, DBCT, APPT
    for r in range(3):
        for d in range(3):
            for a in range(3):
                if r + d + a <= 2:
                    actions.append((r, d, a))
    return actions
actions = gen_actions()

# --- Simulation function with hypothetical routing per action ---
def simulate(flexible):
    # Initialize capacities
    seg_cap = base_seg_cap.copy()
    db_cap = base_db_cap
    ap_cap = base_ap_cap
    records = []

    # Copy outputs
    out = [m['out0'] for m in mine_data]

    for year in range(years + 1):
        # Grow outputs
        if year > 0:
            out = [o * (1 + m['growth']) for o, m in zip(out, mine_data)]

        # Compute current routing under existing capacities
        def route_under_caps(c_db, c_ap):
    # Allocate mines partially when port capacity insufficient
    # First, group mines by fixed choice cost ranking
    # Sort mines by distance proximity to DBCT and APPT
    # But simplest: iterate nearest-to-port order for non-expanding port
    # Build lists for initial assignment
    routed = []  # list of (volume, port)
    # Determine order for filling non-expanding ports
    # We'll fill DBCT first if c_db < c_ap else APPT
    # But here for partial: allocate to both accordingly
    for o, m in zip(out, mine_data):
        # cost-based preference
        cost_db = m['dist'] * haul_rate + handle_rate_db
        cost_ap = (distance_total - m['dist']) * haul_rate + handle_rate_ap
        # initial desired port
        if cost_db < cost_ap:
            primary, secondary = 'DBCT', 'APPT'
        elif cost_ap < cost_db:
            primary, secondary = 'APPT', 'DBCT'
        else:
            primary, secondary = m['port_fixed'], ('APPT' if m['port_fixed']=='DBCT' else 'DBCT')
        # allocate to primary up to its remaining cap
        rem_p = c_db - sum(q for q,p in routed if p==primary) if primary=='DBCT' else c_ap - sum(q for q,p in routed if p==primary)
        vol_primary = min(o, max(rem_p,0))
        routed.append((vol_primary, primary))
        # allocate leftover to secondary
        rem_s = c_ap - sum(q for q,p in routed if p=='APPT') if primary=='DBCT' else c_db - sum(q for q,p in routed if p=='DBCT')
        vol_secondary = o - vol_primary
        if vol_secondary>0:
            routed.append((vol_secondary, secondary))
    return routed

        # Decide on best build action by minimizing immediate NPV with re-routing under each hypothetical
        best_val = float('inf')
        best_act = (0, 0, 0)
        best_routed = None
        for r, d, a in actions:
            # Hypothetical capacities
            h_seg = [c + r * rail_inc for c in seg_cap]
            h_db = db_cap + d * port_inc_db
            h_ap = ap_cap + a * port_inc_ap
            # Re-route under hypothetical port caps
            routed = route_under_caps(h_db, h_ap)
            # CapEx cost
            capex = (r * rail_inc * rail_cost +
                     d * port_inc_db * port_cost_db +
                     a * port_inc_ap * port_cost_ap) / ((1 + discount_rate) ** year)
            # OpEx under hypothetical routing
            haul = sum(q * (m['dist'] if p == 'DBCT' else distance_total - m['dist']) * haul_rate
                       for (q, p), m in zip(routed, mine_data))
            handle = sum(q * (handle_rate_db if p == 'DBCT' else handle_rate_ap)
                         for (q, p) in routed) / ((1 + discount_rate) ** (year + 0.5))
            total = capex + haul + handle
            if total < best_val:
                best_val = total
                best_act = (r, d, a)
                best_routed = routed
        # Apply best action
        r, d, a = best_act
        seg_cap = [c + r * rail_inc for c in seg_cap]
        db_cap += d * port_inc_db
        ap_cap += a * port_inc_ap
        # Record results
        haul = sum(q * (m['dist'] if p == 'DBCT' else distance_total - m['dist']) * haul_rate
                   for (q, p), m in zip(best_routed, mine_data))
        handle = sum(q * (handle_rate_db if p == 'DBCT' else handle_rate_ap)
                     for (q, p) in best_routed) / ((1 + discount_rate) ** (year + 0.5))
        records.append({
            'Year': year,
            'RailInv': r * rail_inc * rail_cost,
            'PortInvDBCT': d * port_inc_db * port_cost_db,
            'PortInvAPPT': a * port_inc_ap * port_cost_ap,
            'OpHaul': haul,
            'OpHandle': handle,
            'NPVYear': best_val
        })
    df = pd.DataFrame(records)
    df['CumulativeNPV'] = df['NPVYear'].cumsum()
    return df

# --- Run both scenarios ---
# --- Run both scenarios ---
df_fixed = simulate(False)
df_flex = simulate(True)

# --- Summary table ---
df_sum = pd.DataFrame([{
    'NPV Fixed': df_fixed['NPVYear'].sum(),
    'NPV Flexible': df_flex['NPVYear'].sum(),
    'Saving': df_fixed['NPVYear'].sum() - df_flex['NPVYear'].sum()
}])

# --- Outputs ---
st.header("Results Summary")
st.dataframe(df_sum)

st.subheader("Cumulative NPV Over Time")
fig, ax = plt.subplots()
ax.plot(df_fixed['Year'], df_fixed['CumulativeNPV'], label='Fixed')
ax.plot(df_flex['Year'], df_flex['CumulativeNPV'], label='Flexible')
ax.legend()
st.pyplot(fig)

st.subheader("Year-0 Breakdown by Category")
b0_fixed = df_fixed.loc[0, ['RailInv','PortInvDBCT','PortInvAPPT','OpHaul','OpHandle']]
b0_flex = df_flex.loc[0, ['RailInv','PortInvDBCT','PortInvAPPT','OpHaul','OpHandle']]
df_break = pd.DataFrame({'Fixed': b0_fixed, 'Flexible': b0_flex})
st.dataframe(df_break)

# --- CSV download ---
df_out = df_fixed[['Year','NPVYear']].merge(
    df_flex[['Year','NPVYear']], on='Year', suffixes=('_fixed','_flex'))
df_out['Difference'] = df_out['NPVYear_fixed'] - df_out['NPVYear_flex']
st.download_button(
    'Download CSV', data=df_out.to_csv(index=False), file_name='model_compare.csv', key='download'
)
