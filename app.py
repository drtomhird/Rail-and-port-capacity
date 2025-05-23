import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Sidebar Inputs ---
st.sidebar.header("Model Inputs")

# General model settings
years = st.sidebar.number_input("Model horizon (years)", min_value=1, max_value=50, value=20)
discount_rate = st.sidebar.number_input("Discount rate (annual %)", min_value=0.0, value=10.0) / 100
distance_total = st.sidebar.number_input("Distance DBCT–APPT (km)", min_value=0.0, value=200.0)

# Mine table inputs
st.sidebar.subheader("Mine settings")
num_mines = st.sidebar.number_input("Number of mines", min_value=1, max_value=15, value=10)

mine_data = []
for i in range(1, int(num_mines)+1):
    col1, col2, col3, col4, col5 = st.sidebar.columns([1,1,1,1,1])
    d = col1.number_input(f"D{i} (km)", value=55 + 10*(i-1), key=f"d{i}")
    o = col2.number_input(f"O{i} (Mtpa)", value=100.0, key=f"o{i}")
    g = col3.number_input(f"g{i} (%)", value=10.0, key=f"g{i}" )/100
    assign = col4.selectbox(f"Port{i} disc", ["DBCT","APPT"], key=f"p{i}")
    mine_data.append({'dist': d, 'out0': o, 'growth': g, 'port_fixed': assign})

# Cost and capacity increments table
st.sidebar.subheader("Capacity increments & costs")
inc_cols = st.sidebar.columns([2,1,1,1])
inc_header = ["Facility","Inc Mtpa","$/Mtpa","Cost per inc"]
defaults = [
    ("DBCT",125,10), ("APPT",125,10), ("Rail seg",50,10)
]
increments = {}
for name, inc0, cost0 in defaults:
    inc = st.sidebar.number_input(f"{name} increment (Mtpa)", value=inc0, key=f"inc_{name}")
    unit = st.sidebar.number_input(f"{name} capex ($/Mtpa)", value=cost0, key=f"cap_{name}")
    increments[name] = {'inc': inc, 'unit_cost': unit}

# Throughput costs
st.sidebar.subheader("Throughput costs")
haulage_cost = st.sidebar.number_input("Haulage ($/Mt·km)", value=1.0)
handling_cost_dbct = st.sidebar.number_input("DBCT handling ($/Mt)", value=1.0)
handling_cost_appt = st.sidebar.number_input("APPT handling ($/Mt)", value=1.0)

# --- Helper function: run scenario ---
def run_scenario(flexible: bool):
    # Initialize capacities and results
    seg_nodes = [0] + [m['dist'] for m in mine_data] + [distance_total]
    segments = list(zip(seg_nodes[:-1], seg_nodes[1:]))
    num_segs = len(segments)

    cap_seg = [max(increments['Rail seg']['inc'], increments['Rail seg']['inc']*0)]*num_segs
    cap_db = increments['DBCT']['inc']
    cap_ap = increments['APPT']['inc']

    # prepare time series
    records = []
    cap_over_time = []
    out = [m['out0'] for m in mine_data]

    for year in range(0, years+1):
        # grow production
        if year > 0:
            out = [o*(1+m['growth']) for o, m in zip(out, mine_data)]
        # port assignment
        if flexible:
            ports = []
            for m,o in zip(mine_data, out):
                d = m['dist']
                cost_db = d*haulage_cost + handling_cost_dbct
                cost_ap = (distance_total-d)*haulage_cost + handling_cost_appt
                ports.append('DBCT' if cost_db <= cost_ap else 'APPT')
        else:
            ports = [m['port_fixed'] for m in mine_data]

        # compute flows
        seg_flow = []
        for a,b in segments:
            seg_flow.append(sum(o for o, m in zip(out, mine_data) if m['dist']>=b))
        flow_db = sum(o for o,p in zip(out, ports) if p=='DBCT')
        flow_ap = sum(o for o,p in zip(out, ports) if p=='APPT')

        # trigger expansions
        inv_rail = 0.0
        if any(f>c for f,c in zip(seg_flow, cap_seg)):
            cap_seg = [c+increments['Rail seg']['inc'] for c in cap_seg]
            inv_rail = increments['Rail seg']['inc'] * increments['Rail seg']['unit_cost']
        inv_db = 0.0
        if flow_db > cap_db:
            cap_db += increments['DBCT']['inc']
            inv_db = increments['DBCT']['inc'] * increments['DBCT']['unit_cost']
        inv_ap = 0.0
        if flow_ap > cap_ap:
            cap_ap += increments['APPT']['inc']
            inv_ap = increments['APPT']['inc'] * increments['APPT']['unit_cost']

        # operational costs mid-year
        haul = sum(o*m['dist']*haulage_cost for o,m in zip(out, mine_data)) + sum(o*(distance_total-m['dist'])*haulage_cost for o,m,p in zip(out, mine_data, ports) if p=='APPT')
        handle_db = sum(o for o,p in zip(out, ports) if p=='DBCT')*handling_cost_dbct
        handle_ap = sum(o for o,p in zip(out, ports) if p=='APPT')*handling_cost_appt

        # discount
        df = (1+discount_rate)**year
        dfm = (1+discount_rate)**(year+0.5)
        record = {
            'Year': year,
            'InvRail': inv_rail/df,
            'InvDBCT': inv_db/df,
            'InvAPPT': inv_ap/df,
            'OpHaul': haul/dfm,
            'OpDBCT': handle_db/dfm,
            'OpAPPT': handle_ap/dfm
        }
        records.append(record)
        cap_over_time.append({'Year': year, **{f'Seg{i+1}': c for i,c in enumerate(cap_seg)}, 'CapDBCT': cap_db, 'CapAPPT': cap_ap})

    df_costs = pd.DataFrame(records)
    df_cap = pd.DataFrame(cap_over_time)
    df_costs['Total'] = df_costs.sum(axis=1)
    df_costs['Cumulative'] = df_costs['Total'].cumsum()
    return df_costs, df_cap

# Run both scenarios
df_conn, cap_conn = run_scenario(True)
df_disc, cap_disc = run_scenario(False)

# Compare NPVs
npv_conn = df_conn['Total'].sum()
pv_disc = df_disc['Total'].sum()
savings = pv_disc - pv_conn = df_disc['Total'].sum() - df_conn['Total'].sum()

# Results summary
df_summary = pd.DataFrame([{ 
    'NPV Connected': npv_conn,
    'NPV Disconnected': npv_disc,
    'Absolute Saving': npv_disc - npv_conn,
    'Pct Saving': (npv_disc - npv_conn)/npv_disc,
    'Pct Saving vs APPT Inv': (npv_disc - npv_conn)/ (df_conn['InvAPPT'].sum())
}])

# --- Outputs ---
st.header("Results Summary")
st.dataframe(df_summary)

# Time-series plots
st.subheader("Cumulative NPV Over Time")
fig, ax = plt.subplots()
ax.plot(df_conn['Year'], df_conn['Cumulative'], label='Connected')
ax.plot(df_disc['Year'], df_disc['Cumulative'], label='Disconnected')
ax.plot(df_disc['Year'], df_disc['Cumulative'] - df_conn['Cumulative'], label='Difference')
ax.set_xlabel('Year'); ax.set_ylabel('Cumulative NPV'); ax.legend()
st.pyplot(fig)

st.subheader("Component NPV Differences Over Time")
fig2, ax2 = plt.subplots()
diff = df_disc[['InvRail','InvDBCT','InvAPPT','OpHaul','OpDBCT','OpAPPT']].cumsum() - df_conn[['InvRail','InvDBCT','InvAPPT','OpHaul','OpDBCT','OpAPPT']].cumsum()
adf = diff.rename(columns={c: c.replace('Inv',''), for c in diff.columns})
for col in diff.columns:
    ax2.plot(df_conn['Year'], diff[col].cumsum(), label=col)
ax2.set_xlabel('Year'); ax2.set_ylabel('NPV Difference'); ax2.legend()
st.pyplot(fig2)

st.subheader("Volume Routed to Non-Assigned Port (Connected)")
vols = []
for year in df_conn['Year']:
    # in connected, count volumes where port assignment != fixed
    # Placeholder: needs tracking in run_scenario
    vols.append(0)
fig3, ax3 = plt.subplots()
ax3.plot(df_conn['Year'], vols)
ax3.set_xlabel('Year'); ax3.set_ylabel('Volume Non-Assigned')
st.pyplot(fig3)

# CSV download
df_out = pd.merge(df_conn[['Year','Total']], df_disc[['Year','Total']], on='Year', suffixes=('_conn','_disc'))
df_out['Difference'] = df_out['Total_disc'] - df_out['Total_conn']
st.download_button('Download results CSV', data=df_out.to_csv(index=False), file_name='comparison.csv')
