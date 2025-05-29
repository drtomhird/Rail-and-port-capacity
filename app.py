import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- Data classes ---
class Mine:
    def __init__(self, name, distance_to_dbct, output0, growth_rate):
        self.name = name
        self.distance_to_dbct = distance_to_dbct
        self.distance_to_appt = None
        self.output0 = output0
        self.growth_rate = growth_rate

class Port:
    def __init__(self, name, initial_capacity, plump, expansion_cost):
        self.name = name
        self.capacity = initial_capacity
        self.plump = plump
        self.expansion_cost = expansion_cost

# --- Core functions ---
def compute_yearly_outputs(mines, year):
    return {m.name: m.output0 * ((1 + m.growth_rate) ** year) for m in mines}

def compute_haulage_costs(assignments, haulage_rate, mines):
    total = 0.0
    for port_name, volumes in assignments.items():
        for mine_name, vol in volumes.items():
            m = next(x for x in mines if x.name == mine_name)
            dist = m.distance_to_dbct if port_name == 'DBCT' else m.distance_to_appt
            total += dist * haulage_rate * vol
    return total

# Fixed and Flexible model definitions as before...
# (omitted here for brevity; assume they remain unchanged)
# --- NPV helper ---
def compute_npv(series, rate):
    return sum(series[t] / ((1 + rate) ** t) for t in series.index)

# --- Streamlit App ---
st.title("Rail-Port Capacity Simulation")

# Sidebar inputs
with st.sidebar:
    RAIL_LENGTH   = st.number_input("Railway length (km)", value=200)
    YEARS         = st.number_input("Simulation horizon (years)", value=20)
    DISCOUNT_RATE = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE  = st.number_input("Haulage cost per unit-km", value=0.1)
    PLUMP         = st.number_input("Capacity chunk (plump)", value=125)
    EXP_COST      = st.number_input("Expansion cost/plump", value=1250)

    # Two mines scenario
    base_df = pd.DataFrame({
        "Name": ["Mine 1", "Mine 2"],
        "Distance to DBCT": [80, 130],
        "Output0": [500, 500],
        "Growth rate": [0.05, 0.05]
    })
    try:
        mines_df = st.data_editor(base_df, use_container_width=True)
    except AttributeError:
        mines_df = st.experimental_data_editor(base_df, use_container_width=True)

    # Handle the run button with persistent state
    if "run" not in st.session_state:
        st.session_state.run = False
    if st.button("Run simulation"):
        st.session_state.run = True

# Only display once run is triggered
if st.session_state.run:
    # Prepare data
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _, r in mines_df.iterrows()]
    for m in mines:
        m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports_fixed = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]
    ports_flex  = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]

    fixed = fixed_model(mines, ports_fixed, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    flex  = flexible_model(mines, ports_flex,  YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    df_f  = pd.DataFrame(fixed)
    df_x  = pd.DataFrame(flex)

    # Results
    npv_diff       = compute_npv(df_f['total_cost'], DISCOUNT_RATE) - compute_npv(df_x['total_cost'], DISCOUNT_RATE)
    pv_appt_fixed  = compute_npv(df_f['appt_cost'], DISCOUNT_RATE)
    pv_appt_flex   = compute_npv(df_x['appt_cost'], DISCOUNT_RATE)
    pct            = npv_diff / pv_appt_fixed * 100
    npv_appt_diff  = pv_appt_fixed - pv_appt_flex

    st.subheader(f"Total NPV difference (Fixed - Flexible) at {YEARS} years = {npv_diff:.2f}")
    st.subheader(f"{npv_diff:.2f} as % of PV of APPT port expansion costs = {pct:.2f}%")
    st.subheader(f"NPV of APPT port expansion costs (Fixed - Flexible) = {npv_appt_diff:.2f}")

    # Chart 1
    st.subheader("Cumulative APPT Port-Expansion Cost Difference ($)")
    port_diff = df_f['appt_cost'].cumsum() - df_x['appt_cost'].cumsum()
    port_df   = pd.DataFrame({'Year': port_diff.index, 'Cost Diff': port_diff.values})
    ch1       = alt.Chart(port_df).mark_bar().encode(x='Year:O', y='Cost Diff:Q')
    st.altair_chart(ch1, use_container_width=True)

    # Chart 2
    st.subheader("Cumulative Haulage Cost Difference ($)")
    haul_diff = df_f['haulage_cost'].cumsum() - df_x['haulage_cost'].cumsum()
    haul_df   = pd.DataFrame({'Year': haul_diff.index, 'Cost Diff': haul_diff.values})
    ch2       = alt.Chart(haul_df).mark_line(point=True).encode(x='Year:Q', y='Cost Diff:Q')
    st.altair_chart(ch2, use_container_width=True)

    # Chart 3
    st.subheader("Present Value of Cost Differences Over Time ($)")
    perf = []
    for t in range(YEARS+1):
        pv_port = sum((df_f['appt_cost'] - df_x['appt_cost']).iloc[:t+1] / ((1 + DISCOUNT_RATE)**np.arange(t+1)))
        pv_haul = sum((df_f['haulage_cost'] - df_x['haulage_cost']).iloc[:t+1] / ((1 + DISCOUNT_RATE)**np.arange(t+1)))
        perf.append(pv_port + pv_haul)
    pv_df = pd.DataFrame({'Year': range(YEARS+1), 'PV Diff': perf})
    ch3    = alt.Chart(pv_df).mark_line(point=True).encode(x='Year:Q', y='PV Diff:Q')
    st.altair_chart(ch3, use_container_width=True)

    # Download
    combined = pd.concat([df_f.add_prefix('fixed_'), df_x.add_prefix('flex_')], axis=1)
    st.download_button("Download CSV", combined.to_csv(index=False), file_name='results.csv')
