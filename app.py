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

# --- Fixed model ---
def fixed_model(mines, ports, years, plump, exp_cost, haulage_rate, discount_rate):
    results = {'dbct_cost': {}, 'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        # assign volumes to nearest port
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name=='DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        # expand ports independently
        dbct_cost = appt_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                if p.name == 'DBCT':
                    dbct_cost += chunks * exp_cost
                else:
                    appt_cost += chunks * exp_cost
        # haulage
        haul = compute_haulage_costs(assignments, haulage_rate, mines)
        total = dbct_cost + appt_cost + haul
        results['dbct_cost'][year] = dbct_cost
        results['appt_cost'][year] = appt_cost
        results['haulage_cost'][year] = haul
        results['total_cost'][year] = total
    return results

# --- Flexible model with cost trade-off ---
def flexible_model(mines, ports, years, plump, exp_cost, haulage_rate, discount_rate):
    results = {'dbct_cost': {}, 'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        # 1) assign volumes to nearest port
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name=='DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        # record prior capacities
        cap_prior = {p.name: p.capacity for p in ports}
        system_demand = sum(usage.values())
        system_capacity = sum(cap_prior.values())
        # 2) no expansion if no shortfall
        if system_demand <= system_capacity:
            dbct_cost = appt_cost = 0.0
            haul = compute_haulage_costs(assignments, haulage_rate, mines)
        else:
            # 3) minimal plumps needed
            N = int(np.ceil((system_demand - system_capacity) / plump))
            # allocate N by largest excess demand per port
            excess = {p.name: usage[p.name] - cap_prior[p.name] for p in ports}
            alloc = {p.name: 0 for p in ports}
            for _ in range(N):
                choice = max(excess, key=excess.get)
                alloc[choice] += 1
                excess[choice] = usage[choice] - (cap_prior[choice] + alloc[choice] * plump)
            # simulate flexible scenario: reroute
            caps_flex = {p.name: cap_prior[p.name] + alloc[p.name] * plump for p in ports}
            assign_flex = {k: dict(v) for k, v in assignments.items()}
            usage_flex = usage.copy()
            for p in ports:
                if usage_flex[p.name] > caps_flex[p.name]:
                    targets = [o for o in ports if usage_flex[o.name] < caps_flex[o.name]]
                    items = sorted(assign_flex[p.name].items(),
                                   key=lambda mv: ((next(m for m in mines if m.name==mv[0]).distance_to_dbct if p.name=='DBCT' else next(m for m in mines if m.name==mv[0]).distance_to_appt) * haulage_rate))
                    for mn, vol in items:
                        if usage_flex[p.name] <= caps_flex[p.name]: break
                        move = min(vol, usage_flex[p.name] - caps_flex[p.name])
                        assign_flex[p.name][mn] -= move; usage_flex[p.name] -= move
                        assign_flex[targets[0].name][mn] = assign_flex[targets[0].name].get(mn, 0) + move; usage_flex[targets[0].name] += move
            haul_flex = compute_haulage_costs(assign_flex, haulage_rate, mines)
            cost_flex = (alloc['DBCT'] + alloc['APPT']) * exp_cost + haul_flex
            # simulate fixed scenario: reroute
            fixed_chunks = {p.name: int(np.ceil(max(0, usage[p.name] - cap_prior[p.name]) / plump)) for p in ports}
            caps_fixed = {p.name: cap_prior[p.name] + fixed_chunks[p.name] * plump for p in ports}
            assign_fixed = {k: dict(v) for k, v in assignments.items()}
            usage_fixed = usage.copy()
            for p in ports:
                if usage_fixed[p.name] > caps_fixed[p.name]:
                    targets = [o for o in ports if usage_fixed[o.name] < caps_fixed[o.name]]
                    items_fx = sorted(assign_fixed[p.name].items(),
                                      key=lambda mv: ((next(m for m in mines if m.name==mv[0]).distance_to_dbct if p.name=='DBCT' else next(m for m in mines if m.name==mv[0]).distance_to_appt) * haulage_rate))
                    for mn, vol in items_fx:
                        if usage_fixed[p.name] <= caps_fixed[p.name]: break
                        move = min(vol, usage_fixed[p.name] - caps_fixed[p.name])
                        assign_fixed[p.name][mn] -= move; usage_fixed[p.name] -= move
                        assign_fixed[targets[0].name][mn] = assign_fixed[targets[0].name].get(mn, 0) + move; usage_fixed[targets[0].name] += move
            haul_fixed = compute_haulage_costs(assign_fixed, haulage_rate, mines)
            cost_fixed = (fixed_chunks['DBCT'] + fixed_chunks['APPT']) * exp_cost + haul_fixed
            # 4) compare one-year haulage penalty vs WACC*exp_cost
            extra_haul = haul_flex - haul_fixed
            threshold = discount_rate * exp_cost
            if extra_haul > threshold:
                # build full fixed chunks
                chunks_to_apply = fixed_chunks
                haul = haul_fixed
            else:
                # keep minimal flex chunks
                chunks_to_apply = alloc
                haul = haul_flex
            # apply expansions permanently
            dbct_cost = chunks_to_apply['DBCT'] * exp_cost
            appt_cost = chunks_to_apply['APPT'] * exp_cost
            for p in ports:
                p.capacity += chunks_to_apply[p.name] * plump
        # record results
        total = dbct_cost + appt_cost + haul
        results['dbct_cost'][year] = dbct_cost
        results['appt_cost'][year] = appt_cost
        results['haulage_cost'][year] = haul
        results['total_cost'][year] = total
    return results

# --- NPV calculation ---
def compute_npv(series, rate):
    return sum(series[t] / ((1 + rate) ** t) for t in series.index)

# --- Streamlit UI ---
st.title("Rail-Port Capacity Simulation")
with st.sidebar:
    RAIL_LENGTH   = st.number_input("Railway length (km)", value=200)
    YEARS         = st.number_input("Simulation horizon (years)", value=20)
    DISCOUNT_RATE = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE  = st.number_input("Haulage cost per unit-km", value=0.1)
    PLUMP         = st.number_input("Capacity chunk (plump)", value=125)
    EXP_COST      = st.number_input("Expansion cost/plump", value=1250)
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
    if st.button("Run simulation"):
        st.session_state.run = True

if "run" not in st.session_state:
    st.session_state.run = False

if st.session_state.run:
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _, r in mines_df.iterrows()]
    for m in mines:
        m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]
    fixed = fixed_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    ports = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]
    flex  = flexible_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    df_f  = pd.DataFrame(fixed)
    df_x  = pd.DataFrame(flex)
    npv_diff = compute_npv(df_f['total_cost'], DISCOUNT_RATE) - compute_npv(df_x['total_cost'], DISCOUNT_RATE)
    st.subheader(f"NPV difference (Fixed - Flexible) at {YEARS} years = {npv_diff:.2f}")
    # Charts
    charts = [
        ("Cumulative DBCT Cost Difference ($)", df_f['dbct_cost'].cumsum() - df_x['dbct_cost'].cumsum(), 'bar'),
        ("Cumulative APPT Cost Difference ($)", df_f['appt_cost'].cumsum() - df_x['appt_cost'].cumsum(), 'bar'),
        ("Cumulative Haulage Cost Difference ($)", df_f['haulage_cost'].cumsum() - df_x['haulage_cost'].cumsum(), 'line'),
        ("PV of Cost Differences Over Time ($)", pd.Series([sum((df_f['total_cost']-df_x['total_cost']).iloc[:t+1] / ((1+DISCOUNT_RATE)**np.arange(t+1))) for t in range(YEARS+1)]), 'line')
    ]
    for title, series, typ in charts:
        st.subheader(title)
        df_plot = pd.DataFrame({"Year": series.index, "Diff": series.values})
        chart = alt.Chart(df_plot).mark_bar() if typ=='bar' else alt.Chart(df_plot).mark_line(point=True)
        st.altair_chart(chart.encode(x='Year:O' if typ=='bar' else 'Year:Q', y='Diff:Q'), use_container_width=True)
    combined = pd.concat([df_f.add_prefix('fixed_'), df_x.add_prefix('flex_')], axis=1)
    st.download_button("Download CSV", combined.to_csv(index=False), file_name='results.csv')
