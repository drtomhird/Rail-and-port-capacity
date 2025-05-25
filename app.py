import numpy as np

def compute_fixed_expansions(
    distances, outputs, growth_rates,
    initial_capacities, plump_size, plump_cost,
    years
):
    """
    Fixed model: expands ports individually to meet demand.
    """
    cap_dbct, cap_appt = initial_capacities
    expansions, capacities, demands = [], [], []
    use_dbct = distances[:,0] <= distances[:,1]

    for t in range(1, years+1):
        outputs_t = outputs * ((1 + growth_rates) ** t)
        demand_dbct = outputs_t[use_dbct].sum()
        demand_appt = outputs_t[~use_dbct].sum()

        # Deficits
        deficit_dbct = max(0, demand_dbct - cap_dbct)
        deficit_appt = max(0, demand_appt - cap_appt)
        plumps_dbct = int(np.ceil(deficit_dbct / plump_size))
        plumps_appt = int(np.ceil(deficit_appt / plump_size))

        # Apply
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


def compute_flexible_expansions(
    distances, outputs, growth_rates,
    initial_capacities, plump_size, plump_cost,
    haul_cost_per_unit_km,
    discount_rate,
    years
):
    """
    Flexible model: in each year, computes the minimum number of plumps to cover system shortage,
    allocates the first "total_plumps - 1" to fully satisfy port-specific deficits, then assigns the last
    plump to the port that minimizes additional haulage cost.
    """
    cap_dbct, cap_appt = initial_capacities
    expansions, capacities, demands = [], [], []
    diversions, hauling_costs = [], []

    use_dbct = distances[:,0] <= distances[:,1]
    extra_dist_dbct = distances[:,1] - distances[:,0]
    extra_dist_appt = distances[:,0] - distances[:,1]

    for t in range(1, years+1):
        outputs_t = outputs * ((1 + growth_rates) ** t)
        demand_dbct = outputs_t[use_dbct].sum()
        demand_appt = outputs_t[~use_dbct].sum()
        total_demand = demand_dbct + demand_appt
        total_capacity = cap_dbct + cap_appt

        # 1. minimum plumps needed
        shortage = max(0, total_demand - total_capacity)
        total_plumps = int(np.ceil(shortage / plump_size))
        if total_plumps == 0:
            expansions.append((0,0))
            capacities.append((cap_dbct, cap_appt))
            demands.append((demand_dbct, demand_appt))
            diversions.append(0)
            hauling_costs.append(0)
            continue

        # 2. allocate first total_plumps-1 to fully meet each port's own deficit
        fixed_dbct = int(np.ceil(max(0, demand_dbct - cap_dbct) / plump_size))
        fixed_appt = int(np.ceil(max(0, demand_appt - cap_appt) / plump_size))
        alloc_db = min(fixed_dbct, total_plumps - 1)
        alloc_ap = min(fixed_appt, total_plumps - 1 - alloc_db)

        # if still slots remain, allocate to whichever has remaining deficit
        remaining = (total_plumps - 1) - (alloc_db + alloc_ap)
        if remaining > 0:
            if (fixed_dbct - alloc_db) >= (fixed_appt - alloc_ap):
                add_db = min(remaining, fixed_dbct - alloc_db)
                alloc_db += add_db
                remaining -= add_db
            if remaining > 0:
                alloc_ap += remaining

        # compute cost for final plump at each port
        cost_db_last = plump_size * haul_cost_per_unit_km * (
            np.min(extra_dist_dbct[use_dbct]) if np.any(use_dbct) else np.inf)
        cost_ap_last = plump_size * haul_cost_per_unit_km * (
            np.min(extra_dist_appt[~use_dbct]) if np.any(~use_dbct) else np.inf)

                # build the last plump to the port with larger remaining deficit
        remaining_def_db = max(0, demand_dbct - (cap_dbct + alloc_db * plump_size))
        remaining_def_ap = max(0, demand_appt - (cap_appt + alloc_ap * plump_size))
        if remaining_def_db >= remaining_def_ap:
            alloc_db += 1
        else:
            alloc_ap += 1

        # track diversions and haul cost for deferred capacity (optional) (optional)
        diversion = ((fixed_dbct - alloc_db) * plump_size if alloc_db < fixed_dbct else 0) + \
                   ((fixed_appt - alloc_ap) * plump_size if alloc_ap < fixed_appt else 0)
        haul_cost = ((fixed_dbct - alloc_db) * cost_db_last / plump_size if alloc_db < fixed_dbct else 0) + \
                    ((fixed_appt - alloc_ap) * cost_ap_last / plump_size if alloc_ap < fixed_appt else 0)

        # apply expansions
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
