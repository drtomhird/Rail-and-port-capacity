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

# Fixed model tracks DBCT, APPT and haulage costs
def fixed_model(mines, ports, years, plump, exp_cost, haulage_rate, discount_rate):
    results = {'dbct_cost': {}, 'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        # Assign to nearest port
        outputs = compute_yearly_outputs(mines, year)
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: m.distance_to_dbct if p.name == 'DBCT' else m.distance_to_appt for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol
        # Expand at each port independently
        dbct_cost = appt_cost = 0.0
        for p in ports:
            if usage[p.name] > p.capacity:
                extra = usage[p.name] - p.capacity
                chunks = int(np.ceil(extra / plump))
                p.capacity += chunks * plump
                cost = chunks * exp_cost
                if p.name == 'DBCT': dbct_cost += cost
                else: appt_cost += cost
        haulage = compute_haulage_costs(assignments, haulage_rate, mines)
        total = dbct_cost + appt_cost + haulage
        results['dbct_cost'][year] = dbct_cost
        results['appt_cost'][year] = appt_cost
        results['haulage_cost'][year] = haulage
        results['total_cost'][year] = total
    return results

# Flexible model with N vs N+1 trade-off and rerouting
def flexible_model(mines, ports, years, plump, exp_cost, haulage_rate, discount_rate):
    """System-wide capacity expansions allocated to minimize haulage"""
    results = {'dbct_cost': {}, 'appt_cost': {}, 'haulage_cost': {}, 'total_cost': {}}
    for year in range(years + 1):
        # 1. Compute outputs and nearest-port assignment
        outputs = compute_yearly_outputs(mines, year)
        assignments = {'DBCT': {}, 'APPT': {}}
        usage = {p.name: 0.0 for p in ports}
        for m in mines:
            vol = outputs[m.name]
            dists = {p.name: (m.distance_to_dbct if p.name == 'DBCT' else m.distance_to_appt) for p in ports}
            chosen = min(dists, key=dists.get)
            assignments[chosen][m.name] = vol
            usage[chosen] += vol

        # 2. Determine system-level expansions (only if demand exceeds capacity)
        cap_prior = {p.name: p.capacity for p in ports}
        system_demand = sum(usage.values())
        system_capacity = sum(cap_prior.values())
        if system_demand <= system_capacity:
            # no expansion
            db_ct = ap_pt = 0.0
        else:
            # number of plumps needed
            N = int(np.ceil((system_demand - system_capacity) / plump))
            # sequentially allocate plumps across ports
            allocated = {p.name: 0 for p in ports}
            for _ in range(N):
                # compute remaining excess per port relative to prior capacity + already allocated
                rem = {p.name: usage[p.name] - (cap_prior[p.name] + allocated[p.name] * plump) for p in ports}
                # choose port with largest excess demand
                choice = max(rem, key=rem.get)
                allocated[choice] += 1
            # apply expansions
            db_ct = ap_pt = 0.0
            for p in ports:
                chunks = allocated[p.name]
                p.capacity += chunks * plump
                if p.name == 'DBCT': db_ct += chunks * exp_cost
                else: ap_pt += chunks * exp_cost

        # 3. Reroute any overload (preserve original rerouting logic)
        # recompute usage after expansions
        usage_after = {p.name: sum(assignments[p.name].values()) for p in ports}
        # reroute from overloaded ports
        for p in ports:
            if usage_after[p.name] > p.capacity:
                other = [o for o in ports if o.name != p.name][0]
                # sort mines by highest per-unit haul cost to this port
                items = sorted(
                    assignments[p.name].items(),
                    key=lambda mv: (
                        (next(m for m in mines if m.name == mv[0]).distance_to_dbct if p.name == 'DBCT'
                         else next(m for m in mines if m.name == mv[0]).distance_to_appt) * haulage_rate
                    ),
                    reverse=True
                )
                for mine_name, vol in items:
                    if usage_after[p.name] <= p.capacity:
                        break
                    move_vol = min(vol, usage_after[p.name] - p.capacity)
                    # move to other port
                    assignments[p.name][mine_name] -= move_vol
                    if assignments[p.name][mine_name] <= 0:
                        del assignments[p.name][mine_name]
                    assignments[other.name][mine_name] = assignments[other.name].get(mine_name, 0) + move_vol
                    usage_after[p.name] -= move_vol
                    usage_after[other.name] += move_vol

        # 4. Compute costs
        haulage = compute_haulage_costs(assignments, haulage_rate, mines)
        total = db_ct + ap_pt + haulage
        results['dbct_cost'][year] = db_ct
        results['appt_cost'][year] = ap_pt
        results['haulage_cost'][year] = haulage
        results['total_cost'][year] = total

    return results

# NPV calculator
def compute_npv(series, rate):
    return sum(series[t]/((1+rate)**t) for t in series.index)

# --- Streamlit App ---
st.title("Rail-Port Capacity Simulation")

# Placeholders to ensure charts persist
chart1_placeholder = st.empty()
chart2_placeholder = st.empty()
chart3_placeholder = st.empty()

with st.sidebar:
    RAIL_LENGTH   = st.number_input("Railway length (km)", value=200)
    YEARS         = st.number_input("Simulation horizon (years)", value=20)
    DISCOUNT_RATE = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE  = st.number_input("Haulage cost per unit-km", value=0.1)
    PLUMP         = st.number_input("Capacity chunk (plump)", value=125)
    EXP_COST      = st.number_input("Expansion cost/plump", value=1250)

                # Two mines scenario: outputs 500 each, 5% growth
    base_df = pd.DataFrame({
        "Name": ["Mine 1", "Mine 2"],
        "Distance to DBCT": [80, 130],  # Updated distances
        "Output0": [500, 500],
        "Growth rate": [0.05, 0.05]
    })
    try:
        mines_df = st.data_editor(base_df, use_container_width=True)
    except AttributeError:
        mines_df = st.experimental_data_editor(base_df, use_container_width=True)

if st.sidebar.button("Run simulation"):
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _,r in mines_df.iterrows()]
    for m in mines: m.distance_to_appt = RAIL_LENGTH - m.distance_to_dbct
    ports = [Port("DBCT",500,PLUMP,EXP_COST),Port("APPT",500,PLUMP,EXP_COST)]

    fixed = fixed_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)
    ports = [Port("DBCT",500,PLUMP,EXP_COST),Port("APPT",500,PLUMP,EXP_COST)]
    flex  = flexible_model(mines, ports, YEARS, PLUMP, EXP_COST, HAULAGE_RATE, DISCOUNT_RATE)

    df_f = pd.DataFrame(fixed)
    df_x = pd.DataFrame(flex)

    npv_diff = compute_npv(df_f['total_cost'],DISCOUNT_RATE) - compute_npv(df_x['total_cost'],DISCOUNT_RATE)
    pv_appt_fixed = compute_npv(df_f['appt_cost'],DISCOUNT_RATE)
    pv_appt_flex  = compute_npv(df_x['appt_cost'],DISCOUNT_RATE)
    pct = npv_diff / pv_appt_fixed * 100
    npv_appt_diff = pv_appt_fixed - pv_appt_flex

    st.subheader(f"Total NPV difference (Fixed - Flexible) at {YEARS} years = {npv_diff:.2f}")
    st.subheader(f"{npv_diff:.2f} as % of PV of APPT port expansion costs = {pct:.2f}%")
    st.subheader(f"NPV of APPT port expansion costs (Fixed - Flexible) = {npv_appt_diff:.2f}")

    st.subheader("Cumulative APPT Port-Expansion Cost Difference ($)")
    port_diff = df_f['appt_cost'].cumsum() - df_x['appt_cost'].cumsum()
    port_df = pd.DataFrame({'Year': port_diff.index, 'Cost Diff': port_diff.values})
    ch1 = alt.Chart(port_df).mark_bar().encode(x='Year:O', y='Cost Diff:Q')
    chart1_placeholder.altair_chart(ch1, use_container_width=True)

    st.subheader("Cumulative Haulage Cost Difference ($)")
    haul_diff = df_f['haulage_cost'].cumsum() - df_x['haulage_cost'].cumsum()
    haul_df = pd.DataFrame({'Year': haul_diff.index, 'Cost Diff': haul_diff.values})
    ch2 = alt.Chart(haul_df).mark_line(point=True).encode(x='Year:Q', y='Cost Diff:Q')
    chart2_placeholder.altair_chart(ch2, use_container_width=True)

    st.subheader("Present Value of Cost Differences Over Time ($)")
    perf = []
    for t in range(YEARS+1):
        pv_port = sum((df_f['appt_cost']-df_x['appt_cost']).iloc[:t+1] / ((1+DISCOUNT_RATE)**np.arange(t+1)))
        pv_h   = sum((df_f['haulage_cost']-df_x['haulage_cost']).iloc[:t+1] / ((1+DISCOUNT_RATE)**np.arange(t+1)))
        perf.append(pv_port + pv_h)
    pv_df = pd.DataFrame({'Year': range(YEARS+1), 'PV Diff': perf})
    ch3 = alt.Chart(pv_df).mark_line(point=True).encode(x='Year:Q', y='PV Diff:Q')
    chart3_placeholder.altair_chart(ch3, use_container_width=True)

    combined = pd.concat([df_f.add_prefix('fixed_'), df_x.add_prefix('flex_')], axis=1)
    st.download_button("Download CSV", combined.to_csv(index=False), file_name='results.csv')
