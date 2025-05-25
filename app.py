import streamlit as st
import numpy as np
import pandas as pd

# --- Data classes ---
class Mine:
    def __init__(self, name, distance_to_dbct, output0, growth_rate):
        self.name = name
        self.distance_to_dbct = distance_to_dbct
        self.distance_to_appt = None  # set after rail length input
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
    """Returns output volume per mine for a given year"""
    return {m.name: m.output0 * ((1 + m.growth_rate) ** year) for m in mines}


def compute_haulage_costs(assignments, haulage_rate, mines):
    """Compute total haulage cost: assignments is {port_name: {mine_name: volume}}"""
    total = 0.0
    for port_name, mine_vols in assignments.items():
        for mine_name, vol in mine_vols.items():
            m = next(m for m in mines if m.name == mine_name)
            dist = m.distance_to_dbct if port_name == 'DBCT' else m.distance_to_appt
            total += dist * haulage_rate * vol
    return total


def fixed_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    """Always expand each port individually when local demand > capacity"""
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        # assign each mine to its closest port
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            port = 'DBCT' if m.distance_to_dbct <= m.distance_to_appt else 'APPT'
            assignments[port][m.name] = vol
            usage[port] += vol
        # expand ports independently
        port_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                port_cost += chunks * expansion_cost
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        total = port_cost + haulage_cost
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = total
    return results


def flexible_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    """Allocate a system-wide number of plumps each year to minimize unmet demand across both ports"""
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            port = 'DBCT' if m.distance_to_dbct <= m.distance_to_appt else 'APPT'
            assignments[port][m.name] = vol
            usage[port] += vol
        # determine system-wide plumps needed
        capacities = {p.name: p.capacity for p in ports}
        total_excess = sum(max(0, usage[name] - capacities[name]) for name in capacities)
        total_plumps = int(np.ceil(total_excess / plump))
        # sequentially allocate plumps to ports
        allocated = {p.name: 0 for p in ports}
        for _ in range(total_plumps):
            # compute remaining unmet demand per port after allocated plumps
            rem = {name: usage[name] - (capacities[name] + allocated[name] * plump) for name in capacities}
            # if any port still has unmet >= plump, allocate there (choose highest unmet)
            elig = [name for name, d in rem.items() if d >= plump]
            if elig:
                choice = max(elig, key=lambda n: rem[n])
            else:
                # final plump goes to port with highest remaining demand
                choice = max(rem, key=rem.get)
            allocated[choice] += 1
        # apply expansions and cost
        port_cost = 0.0
        for p in ports:
            chunks = allocated[p.name]
            p.capacity += chunks * plump
            port_cost += chunks * expansion_cost
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        total = port_cost + haulage_cost
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = total
    return results


def compute_npv(cash_flows, discount_rate):
    return sum(cf / ((1 + discount_rate) ** yr) for yr, cf in cash_flows.items())

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
        "Distance to DBCT": [55 + 10 * i for i in range(10)],
        "Output0": [100] * 10,
        "Growth rate": [0.10] * 10
    })
    try:
        mines_df = st.data_editor(base_df, use_container_width=True)
    except AttributeError:
        mines_df = st.experimental_data_editor(base_df, use_container_width=True)

if st.sidebar.button("Run simulation"):
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _, r in mines_df.iterrows()]
    for m in mines:
        m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports = [Port("DBCT", 500, PLUMP, EXPANSION_COST), Port("APPT", 500, PLUMP, EXPANSION_COST)]

    fixed = fixed_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)
    # reset for flexible run
    ports = [Port("DBCT", 500, PLUMP, EXPANSION_COST), Port("APPT", 500, PLUMP, EXPANSION_COST)]
    flex = flexible_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)

    df_fixed = pd.DataFrame(fixed)
    df_flex = pd.DataFrame(flex)

    st.line_chart(df_fixed['port_cost'].cumsum() - df_flex['port_cost'].cumsum(), height=300)
    st.line_chart(df_fixed['haulage_cost'].cumsum() - df_flex['haulage_cost'].cumsum(), height=300)

    npv_diff = compute_npv(fixed['total_cost'], DISCOUNT_RATE) - compute_npv(flex['total_cost'], DISCOUNT_RATE)
    st.write(f"NPV difference (Fixed - Flexible): {npv_diff:.2f}")

    combined = pd.concat([df_fixed.add_prefix('fixed_'), df_flex.add_prefix('flex_')], axis=1)
    st.download_button("Download results CSV", combined.to_csv(index=False), file_name='results.csv')
