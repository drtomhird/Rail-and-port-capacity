import streamlit as st
import numpy as np
import pandas as pd

# --- Data classes ---
class Mine:
    def __init__(self, name, distance_to_dbct, output0, growth_rate):
        self.name = name
        self.distance_to_dbct = distance_to_dbct
        self.distance_to_appt = None  # set after rail length
        self.output0 = output0
        self.growth_rate = growth_rate

class Port:
    def __init__(self, name, initial_capacity, plump, expansion_cost):
        self.name = name
        self.capacity = initial_capacity
        self.plump = plump
        self.expansion_cost = expansion_cost

# --- Core model functions ---
def compute_yearly_outputs(mines, year):
    """Returns dict of mine outputs for a given year"""
    return {m.name: m.output0 * ((1 + m.growth_rate) ** year) for m in mines}


def compute_haulage_costs(assignments, haulage_rate, mines):
    """Compute total haulage cost given assignments: {port: {mine: volume}}"""
    total = 0.0
    for port_name, mine_vols in assignments.items():
        for mine_name, vol in mine_vols.items():
            # find the mine object
            m = next(m for m in mines if m.name == mine_name)
            dist = m.distance_to_dbct if port_name == 'DBCT' else m.distance_to_appt
            total += dist * haulage_rate * vol
    return total


def fixed_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        # assign to closest port
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            port = 'DBCT' if m.distance_to_dbct <= m.distance_to_appt else 'APPT'
            assignments[port][m.name] = vol
            usage[port] += vol
        # capacity expansion and port cost
        port_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                port_cost += chunks * expansion_cost
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        total_cost = port_cost + haulage_cost
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = total_cost
    return results


def flexible_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        # initial preferred assignment
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            cost_dbct = m.distance_to_dbct * haulage_rate
            cost_appt = m.distance_to_appt * haulage_rate
            preferred = 'DBCT' if cost_dbct <= cost_appt else 'APPT'
            assignments[preferred][m.name] = outputs[m.name]
            usage[preferred] += outputs[m.name]
        # reroute to delay expansions
        port_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                # sort by per-unit haulage cost descending
                items = sorted(
                    assignments[p.name].items(),
                    key=lambda x: (next(m for m in mines if m.name == x[0]).distance_to_dbct if p.name=='DBCT' else next(m for m in mines if m.name == x[0]).distance_to_appt) * haulage_rate,
                    reverse=True
                )
                # reroute highest-cost first
                other = 'APPT' if p.name == 'DBCT' else 'DBCT'
                for mine_name, vol in items:
                    if usage[p.name] <= p.capacity:
                        break
                    # move
                    assignments[p.name].pop(mine_name)
                    assignments[other][mine_name] = vol
                    usage[p.name] -= vol
                    usage[other] += vol
                # if still over, expand
                if usage[p.name] > p.capacity:
                    extra = usage[p.name] - p.capacity
                    chunks = int(np.ceil(extra / plump))
                    p.capacity += chunks * plump
                    port_cost += chunks * expansion_cost
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        total_cost = port_cost + haulage_cost
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = total_cost
    return results


def compute_npv(cash_flows, discount_rate):
    return sum(cf / ((1 + discount_rate) ** year) for year, cf in cash_flows.items())

# --- Streamlit UI ---
st.title("Rail-Port Capacity Simulation")
with st.sidebar:
    RAIL_LENGTH = st.number_input("Length of railway (km)", value=200, step=1)
    YEARS = st.number_input("Simulation horizon (years)", value=20, step=1)
    DISCOUNT_RATE = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE = st.number_input("Haulage cost per unit per km", value=0.1)
    PLUMP = st.number_input("Discrete capacity chunk (plump)", value=125)
    EXPANSION_COST = st.number_input("Cost per plump", value=1250)

    base_df = pd.DataFrame({
        "Name": [f"Mine {i+1}" for i in range(10)],
        "Distance to DBCT": [55 + 10*i for i in range(10)],
        "Output0": [100]*10,
        "Growth rate": [0.10]*10
    })
    try:
        mines_df = st.data_editor(base_df, use_container_width=True)
    except AttributeError:
        mines_df = st.experimental_data_editor(base_df, use_container_width=True)

if st.sidebar.button("Run simulation"):
    # instantiate
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _, r in mines_df.iterrows()]
    for m in mines:
        m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports = [Port("DBCT", 500, PLUMP, EXPANSION_COST), Port("APPT", 500, PLUMP, EXPANSION_COST)]

    fixed = fixed_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)
    # reset port capacities for flexible run
    ports = [Port("DBCT", 500, PLUMP, EXPANSION_COST), Port("APPT", 500, PLUMP, EXPANSION_COST)]
    flex = flexible_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)

    # prepare DataFrame
    df_fixed = pd.DataFrame(fixed)
    df_flex = pd.DataFrame(flex)

    # plots
    st.line_chart(df_fixed['port_cost'].cumsum() - df_flex['port_cost'].cumsum(), height=300)
    st.line_chart(df_fixed['haulage_cost'].cumsum() - df_flex['haulage_cost'].cumsum(), height=300)

    # NPV comparison
    npv_diff = compute_npv(fixed['total_cost'], DISCOUNT_RATE) - compute_npv(flex['total_cost'], DISCOUNT_RATE)
    st.write(f"NPV difference (Fixed - Flexible): {npv_diff:.2f}")

    # download
    combined = pd.concat([df_fixed.add_prefix('fixed_'), df_flex.add_prefix('flex_')], axis=1)
    st.download_button("Download results CSV", combined.to_csv(index=False), file_name='results.csv')
