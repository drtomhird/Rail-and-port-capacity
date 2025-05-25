import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

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
    return {m.name: m.output0 * ((1 + m.growth_rate) ** year) for m in mines}

def compute_haulage_costs(assignments, haulage_rate, mines):
    total = 0.0
    for port_name, mine_vols in assignments.items():
        for mine_name, vol in mine_vols.items():
            m = next(m for m in mines if m.name == mine_name)
            dist = m.distance_to_dbct if port_name == 'DBCT' else m.distance_to_appt
            total += dist * haulage_rate * vol
    return total


def fixed_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name == 'DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        port_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                port_cost += chunks * expansion_cost
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = port_cost + haulage_cost
    return results


def flexible_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    results = {'port_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name == 'DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        capacities = {p.name: p.capacity for p in ports}
        total_excess = sum(max(0, usage[n] - capacities[n]) for n in capacities)
        total_plumps = int(np.ceil(total_excess / plump))
        allocated = {p.name: 0 for p in ports}
        for _ in range(total_plumps):
            rem = {n: usage[n] - (capacities[n] + allocated[n]*plump) for n in capacities}
            elig = [n for n, d in rem.items() if d >= plump]
            choice = max(elig, key=lambda n: rem[n]) if elig else max(rem, key=rem.get)
            allocated[choice] += 1
        port_cost = 0.0
        for p in ports:
            chunks = allocated[p.name]
            p.capacity += chunks * plump
            port_cost += chunks * expansion_cost
        usage = {p.name: sum(assignments[p.name].values()) for p in ports}
        for p in ports:
            if usage[p.name] > p.capacity:
                excess_ports = [o for o in ports if usage[o.name] < o.capacity]
                if not excess_ports:
                    continue
                items = []
                for mine_name, vol in assignments[p.name].items():
                    m = next(m for m in mines if m.name == mine_name)
                    costs = {o.name: ((m.distance_to_dbct if o.name=='DBCT' else m.distance_to_appt)*haulage_rate) for o in excess_ports}
                    target = min(costs, key=costs.get)
                    items.append((mine_name, vol, costs[target], target))
                items.sort(key=lambda x: x[2])
                for mine_name, vol, _, target in items:
                    if usage[p.name] <= p.capacity:
                        break
                    move_vol = min(vol, usage[p.name] - p.capacity)
                    assignments[p.name][mine_name] -= move_vol
                    if assignments[p.name][mine_name] <= 0:
                        del assignments[p.name][mine_name]
                    assignments[target][mine_name] = assignments[target].get(mine_name, 0) + move_vol
                    usage[p.name] -= move_vol
                    usage[target] += move_vol
        haulage_cost = compute_haulage_costs(assignments, haulage_rate, mines)
        results['port_cost'][year] = port_cost
        results['haulage_cost'][year] = haulage_cost
        results['total_cost'][year] = port_cost + haulage_cost
    return results


def compute_npv(cash_flows, discount_rate):
    return sum(cf / ((1 + discount_rate) ** yr) for yr, cf in cash_flows.items())

# --- Streamlit UI ---
st.title("Rail-Port Capacity Simulation")
with st.sidebar:
    RAIL_LENGTH    = st.number_input("Length of railway (km)", value=200)
    YEARS          = st.number_input("Simulation horizon (years)", value=20)
    DISCOUNT_RATE  = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE   = st.number_input("Haulage cost per unit per km", value=0.1)
    PLUMP          = st.number_input("Discrete capacity chunk (plump)", value=125)
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
    ports = [Port("DBCT", 500, PLUMP, EXPANSION_COST), Port("APPT", 500, PLUMP, EXPANSION_COST)]
    flex  = flexible_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)
    df_f = pd.DataFrame(fixed)
    df_x = pd.DataFrame(flex)

    # Dynamic NPV summary at top matching subheader size
    npv_diff = compute_npv(df_f['total_cost'], DISCOUNT_RATE) - compute_npv(df_x['total_cost'], DISCOUNT_RATE)
    st.subheader(f"NPV difference (Fixed - Flexible) at {YEARS} years = {npv_diff:.2f}")

    # Chart 1: Cumulative Port-Expansion Cost Difference
    st.subheader("Cumulative Port-Expansion Cost Difference ($)")
    port_diff = df_f['port_cost'].cumsum() - df_x['port_cost'].cumsum()
    port_df = pd.DataFrame({'Year': port_diff.index, 'Cost Difference': port_diff.values})
    chart1 = alt.Chart(port_df).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Cost Difference:Q', title='Cumulative Cost Difference ($)')
    ).properties(height=300)
    st.altair_chart(chart1, use_container_width=True)

    # Chart 2: Cumulative Haulage Cost Difference
    st.subheader("Cumulative Haulage Cost Difference ($)")
    haul_diff = df_f['haulage_cost'].cumsum() - df_x['haulage_cost'].cumsum()
    haul_df = pd.DataFrame({'Year': haul_diff.index, 'Cost Difference': haul_diff.values})
    chart2 = alt.Chart(haul_df).mark_line(point=True).encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('Cost Difference:Q', title='Cumulative Cost Difference ($)')
    ).properties(height=300)
    st.altair_chart(chart2, use_container_width=True)

    # Chart 3: Present Value of Cost Differences Over Time
    st.subheader("Present Value of Cost Differences Over Time ($)")
    port_yr = df_f['port_cost'] - df_x['port_cost']
    haul_yr = df_f['haulage_cost'] - df_x['haulage_cost']
    years_idx = list(port_yr.index)
    pv_values = []
    for t in years_idx:
        pv_port = sum(port_yr.iloc[:t+1] / ((1 + DISCOUNT_RATE) ** np.array(years_idx[:t+1])))
        pv_haul = sum(haul_yr.iloc[:t+1] / ((1 + DISCOUNT_RATE) ** np.array(years_idx[:t+1])))
        pv_values.append(pv_port + pv_haul)
    pv_df = pd.DataFrame({'Year': years_idx, 'PV Cost Difference': pv_values})
    chart3 = alt.Chart(pv_df).mark_line(point=True).encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('PV Cost Difference:Q', title='Present Value ($)')
    ).properties(height=300)
    st.altair_chart(chart3, use_container_width=True)

    # Download results
    combined = pd.concat([df_f.add_prefix('fixed_'), df_x.add_prefix('flex_')], axis=1)
    st.download_button("Download results CSV", combined.to_csv(index=False), file_name='results.csv')
