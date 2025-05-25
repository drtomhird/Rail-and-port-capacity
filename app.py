import streamlit as st
import numpy as np
import pandas as pd

# 1. Data classes
template = """
class Mine:
    def __init__(self, name, distance_to_dbct, output0, growth_rate):
        self.name = name
        self.distance_to_dbct = distance_to_dbct
        self.distance_to_appt = RAIL_LENGTH - distance_to_dbct
        self.output0 = output0
        self.growth_rate = growth_rate

class Port:
    def __init__(self, name, initial_capacity, plump, expansion_cost):
        self.name = name
        self.capacity = initial_capacity
        self.plump = plump
        self.expansion_cost = expansion_cost
"""

# 2. Core model functions
"""
def compute_yearly_outputs(mines, years):
    # returns DataFrame with mine outputs for each year
    pass

def compute_haulage_costs(assignments, haulage_rate):
    # sum over mines → port assignments
    pass

def fixed_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    # always ship to closest port, expand capacity when demand > capacity
    # returns dict with yearly port expansions and haulage costs
    pass

def flexible_model(mines, ports, years, plump, expansion_cost, haulage_rate, discount_rate):
    # central planner: may reroute to delay expansions
    pass

def compute_npv(cash_flows, discount_rate):
    # standard discounting
    return sum(cf / ((1 + discount_rate) ** year) for year, cf in cash_flows.items())
"""

# 3. Streamlit user interface
st.title("Rail-Port Capacity Simulation")

# Sidebar inputs
with st.sidebar:
    RAIL_LENGTH = st.number_input("Length of railway (km)", value=200)
    YEARS = st.number_input("Simulation horizon (years)", value=20, step=1)
    DISCOUNT_RATE = st.number_input("Discount rate", value=0.10)
    HAULAGE_RATE = st.number_input("Haulage cost per unit per km", value=0.1)
    PLUMP = st.number_input("Discrete capacity chunk (plump)", value=125)
    EXPANSION_COST = st.number_input("Cost per plump", value=1250)

    # Input table for mines
    mines_df = st.experimental_data_editor(
        pd.DataFrame({
            "Name": [f"Mine {i+1}" for i in range(10)],
            "Distance to DBCT": [55 + 10*i for i in range(10)],
            "Output0": [100]*10,
            "Growth rate": [0.10]*10
        }), use_container_width=True
    )

if st.sidebar.button("Run simulation"):
    # Instantiate objects
    mines = [Mine(r.Name, r["Distance to DBCT"], r.Output0, r["Growth rate"]) for _, r in mines_df.iterrows()]
    ports = [Port("DBCT", initial_capacity=500, plump=PLUMP, expansion_cost=EXPANSION_COST),
             Port("APPT", initial_capacity=500, plump=PLUMP, expansion_cost=EXPANSION_COST)]

    # Run models
    fixed_results = fixed_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)
    flexible_results = flexible_model(mines, ports, YEARS, PLUMP, EXPANSION_COST, HAULAGE_RATE, DISCOUNT_RATE)

    # Combine results into DataFrame for charting
    df_fixed = pd.DataFrame(fixed_results)
    df_flex = pd.DataFrame(flexible_results)

    # 4. Plots
    st.line_chart(
        df_fixed["cum_port_cost"] - df_flex["cum_port_cost"],
        height=300,
        use_container_width=True
    )
    st.bar_chart(
        pd.DataFrame({
            "Port NPV diff": [compute_npv(df_fixed.port_cost.to_dict(), DISCOUNT_RATE) - compute_npv(df_flex.port_cost.to_dict(), DISCOUNT_RATE)],
            "Haulage NPV diff": [compute_npv(df_fixed.haulage_cost.to_dict(), DISCOUNT_RATE) - compute_npv(df_flex.haulage_cost.to_dict(), DISCOUNT_RATE)]
        })
    )

    # 5. Rerouted volumes bar chart
    st.bar_chart(df_flex.reassigned_volume)

    # 6. Key outcomes table
    outcomes = {
        "NPV fixed - flexible total": compute_npv(df_fixed.total_cost.to_dict(), DISCOUNT_RATE) - compute_npv(df_flex.total_cost.to_dict(), DISCOUNT_RATE),
        "% of flex port capacity cost": (
            compute_npv(df_fixed.total_cost.to_dict(), DISCOUNT_RATE) - compute_npv(df_flex.total_cost.to_dict(), DISCOUNT_RATE)
        ) / compute_npv(df_flex.port_cost.to_dict(), DISCOUNT_RATE)
    }
    st.table(pd.DataFrame.from_dict(outcomes, orient="index", columns=["Value"]))

    # 7. Download CSV
    csv = pd.concat([df_fixed.add_prefix("fixed_"), df_flex.add_prefix("flex_")], axis=1).to_csv(index=False)
    st.download_button("Download full results CSV", data=csv, file_name="simulation_results.csv")

# 4. Requirements file
requirements = """
streamlit
numpy
pandas
"""

st.sidebar.markdown("**requirements.txt**")
st.sidebar.code(requirements)
