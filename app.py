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
    results = {'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        # assign to nearest port
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name=='DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        # expand each port independently
        appt_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                if p.name == 'APPT':
                    appt_cost += chunks * exp_cost
        haul = compute_haulage_costs(assignments, haulage_rate, mines)
        total = appt_cost + haul
        results['appt_cost'][year] = appt_cost
        results['haulage_cost'][year] = haul
        results['total_cost'][year] = total
    return results

# --- Flexible model with cost trade-off ---
def flexible_model(mines, ports, years, plump, exp_cost, haulage_rate, discount_rate):
    results = {'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        # base assignment and usage
        outputs = compute_yearly_outputs(mines, year)
        assignments = {p.name: {} for p in ports}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name=='DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        # prior capacities
        cap_prior = {p.name: p.capacity for p in ports}
        system_demand = sum(usage.values())
        system_capacity = sum(cap_prior.values())
        # if no system shortfall
        if system_demand <= system_capacity:
            appt_cost = 0.0
            haul = compute_haulage_costs(assignments, haulage_rate, mines)
        else:
            # minimal plumps
            N = int(np.ceil((system_demand - system_capacity) / plump))
            # allocate minimal N to ports by largest excess demand
            excess = {p.name: usage[p.name] - cap_prior[p.name] for p in ports}
            alloc = {p.name: 0 for p in ports}
            for _ in range(N):
                choice = max(excess, key=excess.get)
                alloc[choice] += 1
                excess[choice] = usage[choice] - (cap_prior[choice] + alloc[choice]*plump)
            # simulate flexible (N)
            caps_flex = cap_prior.copy()
            for p in ports:
                caps_flex[p.name] += alloc[p.name] * plump
            # reroute for flex
            assign_flex = {k: dict(v) for k,v in assignments.items()}
            usage2 = usage.copy()
            for p in ports:
                if usage2[p.name] > caps_flex[p.name]:
                    others = [o for o in ports if usage2[o.name] < caps_flex[o.name]]
                    items = sorted(assign_flex[p.name].items(), key=lambda mv: 
                        ((next(m for m in mines if m.name==mv[0]).distance_to_dbct if p.name=='DBCT' else 
                          next(m for m in mines if m.name==mv[0]).distance_to_appt) * haulage_rate)
                    )
                    for mn,vol in items:
                        if usage2[p.name] <= caps_flex[p.name]: break
                        mv = min(vol, usage2[p.name] - caps_flex[p.name])
                        assign_flex[p.name][mn] -= mv; usage2[p.name] -= mv
                        assign_flex[others[0].name][mn] = assign_flex[others[0].name].get(mn,0)+mv; usage2[others[0].name] += mv
            haul_flex = compute_haulage_costs(assign_flex, haulage_rate, mines)
            cost_flex = N*exp_cost + haul_flex
            # simulate fixed at year
            fixed_chunks = sum(int(np.ceil(max(0, usage[p.name] - cap_prior[p.name]) / plump)) for p in ports)
            # fixed no reroute
            assign_fixed = {k: dict(v) for k,v in assignments.items()}
            haul_fixed = compute_haulage_costs(assign_fixed, haulage_rate, mines)
            cost_fixed = fixed_chunks*exp_cost + haul_fixed
            # compare haulage penalty vs expansion cost threshold
            extra_haul = haul_flex - haul_fixed
            threshold = exp_cost  # full plump cost threshold
            # simulate reroute for fixed to compare apples-to-apples
            # calculate per-port fixed chunks and fixed capacities
            chunks_fixed = {}
            cap_fixed = {}
            for p in ports:
                over = usage[p.name] - cap_prior[p.name]
                c = int(np.ceil(max(0, over) / plump))
                chunks_fixed[p.name] = c
                cap_fixed[p.name] = cap_prior[p.name] + c * plump
            # reroute for fixed scenario
            assign_fixed_r = {k: dict(v) for k, v in assign_fixed.items()}
            usage_fx = usage.copy()
            for p in ports:
                if usage_fx[p.name] > cap_fixed[p.name]:
                    others = [o for o in ports if usage_fx[o.name] < cap_fixed[o.name]]
                    items_fx = sorted(
                        assign_fixed_r[p.name].items(),
                        key=lambda mv: ((next(m for m in mines if m.name == mv[0]).distance_to_dbct if p.name=='DBCT' else next(m for m in mines if m.name == mv[0]).distance_to_appt) * haulage_rate)
                    )
                    for mn, vol in items_fx:
                        if usage_fx[p.name] <= cap_fixed[p.name]:
                            break
                        mv_fx = min(vol, usage_fx[p.name] - cap_fixed[p.name])
                        assign_fixed_r[p.name][mn] -= mv_fx; usage_fx[p.name] -= mv_fx
                        assign_fixed_r[others[0].name][mn] = assign_fixed_r[others[0].name].get(mn, 0) + mv_fx; usage_fx[others[0].name] += mv_fx
            haul_fixed = compute_haulage_costs(assign_fixed_r, haulage_rate, mines)
            if extra_haul > threshold:
                # pick fixed scenario
                appt_cost = sum(chunks_fixed[p] for p in chunks_fixed) * exp_cost
                haul = haul_fixed
            else:
                # keep minimal flexible plumps
                appt_cost = N * exp_cost
                haul = haul_flex
                # pick fixed
                appt_cost = fixed_chunks * exp_cost
                haul = haul_fixed
            else:
                # keep flexible
                appt_cost = N * exp_cost
                haul = haul_flex
        total = appt_cost + haul
        results['appt_cost'][year] = appt_cost
        results['haulage_cost'][year] = haul
        results['total_cost'][year] = total
    return results

# --- NPV calculation ---
def compute_npv(series, rate):
    return sum(series[t] / ((1 + rate) ** t) for t in series.index)

# --- Streamlit App ---
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
    for m in mines: m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]
    fixed = fixed_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    ports = [Port("DBCT", 500, PLUMP, EXP_COST), Port("APPT", 500, PLUMP, EXP_COST)]
    flex  = flexible_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    df_f  = pd.DataFrame(fixed)
    df_x  = pd.DataFrame(flex)
    npv_diff = compute_npv(df_f['total_cost'], DISCOUNT_RATE) - compute_npv(df_x['total_cost'], DISCOUNT_RATE)
    st.subheader(f"NPV difference (Fixed - Flexible) at {YEARS} years = {npv_diff:.2f}")
    st.subheader("Cumulative APPT Cost Difference ($)")
    port_diff = df_f['appt_cost'].cumsum() - df_x['appt_cost'].cumsum()
    port_df = pd.DataFrame({"Year": port_diff.index, "Cost Diff": port_diff.values})
    st.altair_chart(alt.Chart(port_df).mark_bar().encode(x='Year:O', y='Cost Diff:Q'), use_container_width=True)
    st.subheader("Cumulative Haulage Cost Difference ($)")
    haul_diff = df_f['haulage_cost'].cumsum() - df_x['haulage_cost'].cumsum()
    haul_df = pd.DataFrame({"Year": haul_diff.index, "Cost Diff": haul_diff.values})
    st.altair_chart(alt.Chart(haul_df).mark_line(point=True).encode(x='Year:Q', y='Cost Diff:Q'), use_container_width=True)
    st.subheader("PV of Cost Differences Over Time ($)")
    perf = []
    for t in range(YEARS+1):
        pv_p = sum((df_f['appt_cost'] - df_x['appt_cost']).iloc[:t+1] / ((1+DISCOUNT_RATE)**np.arange(t+1)))
        pv_h = sum((df_f['haulage_cost'] - df_x['haulage_cost']).iloc[:t+1] / ((1+DISCOUNT_RATE)**np.arange(t+1)))
        perf.append(pv_p + pv_h)
    pv_df = pd.DataFrame({"Year": range(YEARS+1), "PV Diff": perf})
    st.altair_chart(alt.Chart(pv_df).mark_line(point=True).encode(x='Year:Q', y='PV Diff:Q'), use_container_width=True)
    combined = pd.concat([df_f.add_prefix('fixed_'), df_x.add_prefix('flex_')], axis=1)
    st.download_button("Download CSV", combined.to_csv(index=False), file_name='results.csv')
