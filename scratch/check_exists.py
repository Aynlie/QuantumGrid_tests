import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import data_loader as dl
import qubo_builder as qb
import disaster_recovery as dr
import power_flow as pf

# Load cached data
bundle = dl.load_all(
    network_csv="network_topology.csv",
    demand_csv="PJME_hourly.csv",
    solar_csv="solar_generation.csv",
    S_base_mva=10.0, V_base_kv=12.66,
)

network = bundle["graph"]
nominal_total_load_pu = sum(b.P_load_pu for b in network.buses.values())
demand_shape = bundle["demand_pu"] / bundle["demand_pu"].mean()
demand_pu = demand_shape * nominal_total_load_pu

PV_CAPACITY_PU = 0.35
solar_shape = bundle["solar_pu"] / bundle["solar_pu"].max()
solar_pu = solar_shape * PV_CAPACITY_PU

import forecasting as fc
features = fc.build_features(demand_pu)
_forecast = fc.train_demand_forecaster(features)
allocation_factors = fc.compute_allocation_factors(network)
demand_by_bus_pu = fc.disaggregate_forecast_series(demand_pu, allocation_factors)

import renewable as rw
PV_BUS = 18
PV_HOSTING_CAP_PU = 0.30
pv_capacity = {bus_id: (PV_CAPACITY_PU if bus_id == PV_BUS else 0.0)
               for bus_id in network.buses}
pv_factors = rw.compute_pv_allocation_factors(
    {k: (v if v > 0 else 1e-9) for k, v in pv_capacity.items()}
)
available_solar = rw.allocate_measured_solar(solar_pu, pv_factors)
hosting_caps = {bus_id: (PV_HOSTING_CAP_PU if bus_id == PV_BUS else 0.0)
                for bus_id in network.buses}
dispatch = rw.apply_hosting_capacity(available_solar, hosting_caps)
net_load_by_bus = rw.compute_net_load(demand_by_bus_pu, dispatch.dispatched_pu)

# Simulate fault
peak_hour = demand_pu.idxmax()
net_injection = {b: net_load_by_bus.loc[peak_hour, b] for b in bundle["graph"].buses}

res = dr.simulate_fault(bundle["graph"], (5, 6), net_injection, root=1)

loops = qb.find_switchable_loops(res.new_dist_graph)
costs = qb.compute_loop_open_costs(res.new_dist_graph, loops, net_injection, root=1)

print("LOOPS:", loops)
print("COSTS:", costs)

default_closed = set(res.new_dist_graph.fixed_edges)
required_switchable, _optional = qb._structurally_required_switchable(res.new_dist_graph)
default_closed.update(required_switchable)

for e in res.new_dist_graph.switchable_edges:
    i_node, j_node = e
    if res.new_dist_graph.graph.edges[i_node, j_node]["s_initial"] == 1:
        default_closed.add(e)

candidate_list = []
for k, loop_edges in enumerate(loops):
    for open_edge in loop_edges:
        loss = costs[k][open_edge]
        
        trial_closed = set(default_closed)
        for e in loop_edges:
            trial_closed.discard(e)
            if e != open_edge:
                trial_closed.add(e)
        trial_closed.discard(open_edge)
        
        flows = pf.compute_tree_flows(res.new_dist_graph, trial_closed, net_injection, root=1)
        q_flows = {key: 0.0 for key in flows}
        v_check = pf.check_voltage_feasibility(res.new_dist_graph, flows, q_flows, root=1)
        
        min_v = min(v_check["voltages_pu"].values()) if v_check["voltages_pu"] else 1.0
        feasible = len(v_check["violations"]) == 0
        
        closed_tie = None
        for e in loop_edges:
            if e != open_edge:
                u, v = e
                if res.new_dist_graph.graph.edges[u, v]["s_initial"] == 0:
                    closed_tie = e
        if open_edge in required_switchable:
            closed_tie = open_edge
        
        is_winner = True
        for e in loop_edges:
            is_closed_in_win = res.new_switch_assignment.get(e, 0)
            if e == open_edge and is_closed_in_win != 0:
                is_winner = False
            elif e != open_edge and is_closed_in_win != 1:
                if e in res.new_switch_assignment and is_closed_in_win != 1:
                    is_winner = False

        print(f"Loop {k}, open_edge {open_edge}: trial_closed = {trial_closed}")
        candidate_list.append({
            "edge": open_edge,
            "closed_tie": closed_tie,
            "loss": loss,
            "min_v": min_v,
            "feasible": feasible,
            "winner": is_winner
        })
