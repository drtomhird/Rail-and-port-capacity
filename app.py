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
    """Flexible model that tracks its own fixed-trajectory for correct comparisons"""
    results = {'dbct_cost': {}, 'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    # create separate port lists for flexible vs fixed trajectories
    flex_ports = [Port(p.name, p.capacity, p.plump, p.expansion_cost) for p in ports]
    fixed_ports = [Port(p.name, p.capacity, p.plump, p.expansion_cost) for p in ports]

    for year in range(years + 1):
        # base assignment and usage
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name=='DBCT' else m.distance_to_appt) for p in ports}
            choice = min(dists, key=dists.get)
            assignments[choice][m.name] = vol
            usage[choice] += vol

        # record prior capacities
        cap_flex = {p.name: p.capacity for p in flex_ports}
        cap_fix  = {p.name: p.capacity for p in fixed_ports}
        total_demand  = sum(usage.values())
        total_capacity = sum(cap_flex.values())

        # no expansion needed?
        if total_demand <= total_capacity:
            # flexible: no build, no permanent change
            dbct_cost = appt_cost = 0.0
            # haul is simple nearest assignment
            haul_flex = compute_haulage_costs(assignments, haulage_rate, mines)
            alloc_flex = {n: 0 for n in cap_flex}
        else:
            # minimal plumps needed
            N = int(np.ceil((total_demand - total_capacity) / plump))
            # allocate N chunks to flex_ports by excess demand
            excess = {n: usage[n] - cap_flex[n] for n in cap_flex}
            alloc_flex = {n: 0 for n in cap_flex}
            for _ in range(N):
                tgt = max(excess, key=excess.get)
                alloc_flex[tgt] += 1
                excess[tgt] = usage[tgt] - (cap_flex[tgt] + alloc_flex[tgt]*plump)
            # apply flex expansions temporarily
            caps_temp = {n: cap_flex[n] + alloc_flex[n]*plump for n in cap_flex}
            # reroute for flex
            assign_flex = {k: dict(v) for k, v in assignments.items()}
            usage_tmp = usage.copy()
            for p in flex_ports:
                name = p.name
                if usage_tmp[name] > caps_temp[name]:
                    other = [o for o in flex_ports if usage_tmp[o.name] < caps_temp[o.name]][0]
                    items = sorted(assign_flex[name].items(), key=lambda mv: ((next(m for m in mines if m.name==mv[0]).distance_to_dbct if name=='DBCT' else next(m for m in mines if m.name==mv[0]).distance_to_appt) * haulage_rate))
                    for mn, vol in items:
                        if usage_tmp[name] <= caps_temp[name]: break
                        mv = min(vol, usage_tmp[name] - caps_temp[name])
                        assign_flex[name][mn] -= mv; usage_tmp[name] -= mv
                        assign_flex[other.name][mn] = assign_flex[other.name].get(mn,0) + mv; usage_tmp[other.name] += mv
            haul_flex = compute_haulage_costs(assign_flex, haulage_rate, mines)

        # now simulate fixed trajectory step for this year
        # expansion based on fixed_ports
        extra_fix = {n: usage[n] - cap_fix[n] for n in cap_fix}
        alloc_fix = {}
        for p in fixed_ports:
            over = max(0, extra_fix[p.name])
            chunks = int(np.ceil(over / plump))
            alloc_fix[p.name] = chunks
            p.capacity += chunks * plump
        # reroute for fixed
        assign_fix = {k: dict(v) for k, v in assignments.items()}
        usage_f = usage.copy()
        caps_fix = {p.name: p.capacity for p in fixed_ports}
        for p in fixed_ports:
            name = p.name
            if usage_f[name] > caps_fix[name]:
                other = [o for o in fixed_ports if usage_f[o.name] < caps_fix[o.name]][0]
                items = sorted(assign_fix[name].items(), key=lambda mv: ((next(m for m in mines if m.name==mv[0]).distance_to_dbct if name=='DBCT' else next(m for m in mines if m.name==mv[0]).distance_to_appt) * haulage_rate))
                for mn, vol in items:
                    if usage_f[name] <= caps_fix[name]: break
                    mv = min(vol, usage_f[name] - caps_fix[name])
                    assign_fix[name][mn] -= mv; usage_f[name] -= mv
                    assign_fix[other.name][mn] = assign_fix[other.name].get(mn,0) + mv; usage_f[other.name] += mv
        haul_fixed = compute_haulage_costs(assign_fix, haulage_rate, mines)

        # cost metrics
        cost_flex  = sum(alloc_flex.values()) * exp_cost + haul_flex
        cost_fixed = sum(alloc_fix.values()) * exp_cost + haul_fixed
        extra_haul = haul_flex - haul_fixed
        threshold = discount_rate * exp_cost

        # decide which to apply
        if system_demand <= total_capacity:
            # none
            dbct_cost = appt_cost = 0.0
            haul = teardown = haul_flex
        elif cost_fixed < cost_flex and extra_haul > threshold:
            # apply fixed expansions
            dbct_cost = alloc_fix['DBCT'] * exp_cost
            appt_cost = alloc_fix['APPT'] * exp_cost
            haul = haul_fixed
            # update flex_ports to fixed state
            for p in flex_ports:
                p.capacity = caps_fix[p.name]
        else:
            # apply flexible expansions
            dbct_cost = alloc_flex['DBCT'] * exp_cost
            appt_cost = alloc_flex['APPT'] * exp_cost
            haul = haul_flex
            for p in flex_ports:
                p.capacity = caps_temp[p.name]

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
