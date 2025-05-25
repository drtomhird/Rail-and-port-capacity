import numpy as np
import streamlit as st

@st.cache_data
def compute_fixed_expansions(
    distances, outputs, growth_rates,
    initial_capacities, plump_size, plump_cost,
    years
):
    """
    Fixed model: expands each port by ceil(deficit/plump_size) each year.
    """
    cap_dbct, cap_appt = initial_capacities
    expansions, capacities, demands = [], [], []
    use_dbct = distances[:,0] <= distances[:,1]

    for t in range(1, years+1):
        outputs_t = outputs * ((1 + growth_rates) ** t)
        demand_dbct = outputs_t[use_dbct].sum()
        demand_appt = outputs_t[~use_dbct].sum()

        plumps_dbct = int(np.ceil(max(0, demand_dbct - cap_dbct) / plump_size))
        plumps_appt = int(np.ceil(max(0, demand_appt - cap_appt) / plump_size))

        cap_dbct += plumps_dbct * plump_size
        cap_appt += plumps_appt * plump_size

        expansions.append((plumps_dbct, plumps_appt))
        capacities.append((cap_dbct, cap_appt))
        demands.append((demand_dbct, demand_appt))

    return {
        'expansions': expansions,
        'capacity': capacities,
        'demand': demands
    }

@st.cache_data
def compute_flexible_expansions(
    distances, outputs, growth_rates,
    initial_capacities, plump_size, plump_cost,
    haul_cost_per_unit_km,
    discount_rate,
    years
):
    """
    Flexible model: each year:
      1. Compute total_plumps = ceil((total_demand - total_capacity)/plump_size)
      2. Allocate first total_plumps-1 to meet each port's deficits
      3. Allocate final plump to port with larger remaining deficit
      4. Record diversion and haul costs for any deferred capacity
    """
    cap_dbct, cap_appt = initial_capacities
    expansions, capacities, demands = [], [], []
    diversions, hauling_costs = [], []
    use_dbct = distances[:,0] <= distances[:,1]
    extra_db = distances[:,1] - distances[:,0]
    extra_ap = distances[:,0] - distances[:,1]

    # Precompute per-plump haul costs
    cost_db_plump = plump_size * haul_cost_per_unit_km * (np.min(extra_db[use_dbct]) if np.any(use_dbct) else 0)
    cost_ap_plump = plump_size * haul_cost_per_unit_km * (np.min(extra_ap[~use_dbct]) if np.any(~use_dbct) else 0)

    for t in range(1, years+1):
        outputs_t = outputs * ((1 + growth_rates) ** t)
        demand_dbct = outputs_t[use_dbct].sum()
        demand_appt = outputs_t[~use_dbct].sum()
        total_demand = demand_dbct + demand_appt
        total_capacity = cap_dbct + cap_appt

        shortage = max(0, total_demand - total_capacity)
        total_plumps = int(np.ceil(shortage / plump_size))

        if total_plumps == 0:
            expansions.append((0,0))
            capacities.append((cap_dbct, cap_appt))
            demands.append((demand_dbct, demand_appt))
            diversions.append(0)
            hauling_costs.append(0)
            continue

        fixed_db = int(np.ceil(max(0, demand_dbct - cap_dbct) / plump_size))
        fixed_ap = int(np.ceil(max(0, demand_appt - cap_appt) / plump_size))

        # Step 2: allocate first total_plumps-1
        alloc_db = min(fixed_db, total_plumps-1)
        alloc_ap = min(fixed_ap, total_plumps-1-alloc_db)
        rem = (total_plumps-1) - (alloc_db + alloc_ap)
        if rem > 0:
            db_rem = fixed_db - alloc_db
            ap_rem = fixed_ap - alloc_ap
            if db_rem >= ap_rem:
                add = min(rem, db_rem)
                alloc_db += add
                rem -= add
            if rem > 0:
                alloc_ap += rem

        # Step 3: final plump to larger remaining deficit
        rem_def_db = max(0, demand_dbct - (cap_dbct + alloc_db*plump_size))
        rem_def_ap = max(0, demand_appt - (cap_appt + alloc_ap*plump_size))
        if rem_def_db >= rem_def_ap:
            alloc_db += 1
        else:
            alloc_ap += 1

        # Step 4: record diversions
        diverted_db = max(0, fixed_db-alloc_db)*plump_size
        diverted_ap = max(0, fixed_ap-alloc_ap)*plump_size
        diversion = diverted_db + diverted_ap
        haul_cost = (diverted_db/plump_size)*cost_db_plump + (diverted_ap/plump_size)*cost_ap_plump

        cap_dbct += alloc_db * plump_size
        cap_appt += alloc_ap * plump_size

        expansions.append((alloc_db, alloc_ap))
        capacities.append((cap_dbct, cap_appt))
        demands.append((demand_dbct, demand_appt))
        diversions.append(diversion)
        hauling_costs.append(haul_cost)

    return {
        'expansions': expansions,
        'capacity': capacities,
        'demand': demands,
        'diversions': diversions,
        'hauling_costs': hauling_costs
    }
