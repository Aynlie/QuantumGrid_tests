"""
disaster_recovery.py
================
Module 7 of QuantumGrid.
Responsibility: simulate a line fault by DELETING the affected edge (not
flipping a switch variable that may not even exist), check whether the
surviving network can structurally be reconnected at all, rerun the
Module 4-6 pipeline on the reduced graph if so, and -- only if the
reconnected network's demand exceeds available supply -- run an exact
priority-weighted 0/1 knapsack to decide which buses to serve.
This module explicitly distinguishes two different failure modes rather
than conflating them:
  1. Structural infeasibility (no physical path exists) -- unavoidable,
     reported honestly, no optimization can fix it.
  2. Capacity shortfall (a path exists, but not enough power) -- THIS is
     where priority weights genuinely apply, via knapsack selection.
"""
from dataclasses import dataclass
import networkx as nx
import data_loader as dl
import network_model as nm
import qubo_builder as qb
import quantum_optimizer as qo
@dataclass
class FaultResult:
    restorable: bool
    stranded_buses: set          # buses with NO possible path (failure mode 1)
    new_dist_graph: object       # DistributionGraph after restoration, or None
    new_switch_assignment: dict  # {edge: 0/1}, or None
    served_buses: set            # buses actually energized after shedding
    shed_buses: set              # buses shed due to capacity shortfall (failure mode 2)
def simulate_fault(network: dl.NetworkGraph, faulted_edge: tuple,
                    net_injection: dict, root: int,
                    available_supply_pu: float = None) -> FaultResult:
    """
    network            : Module 1's NetworkGraph (the pre-fault network).
    faulted_edge        : (i, j) tuple -- the physically failed line.
    net_injection       : {bus_id: P_net,i(t)} from Module 3, for the
                           surviving network's restoration/loss calculation.
    root                : slack bus id.
    available_supply_pu : total power the slack/root can actually deliver
                           post-fault. If None, assumed unlimited (i.e.,
                           only a topological restoration is checked, no
                           capacity-driven shedding).
    """
    i, j = faulted_edge
    # Step 1: delete the faulted edge entirely -- NOT a switch flip.
    reduced_branches = [
        b for b in network.branches
        if not ({b.i, b.j} == {i, j})
    ]
    if len(reduced_branches) == len(network.branches):
        raise ValueError(f"Faulted edge {faulted_edge} not found in network -- "
                          f"cannot simulate a fault on a nonexistent line.")
    reduced_network = dl.NetworkGraph(base=network.base)
    reduced_network.buses = network.buses
    reduced_network.branches = reduced_branches
    # Step 2: structural feasibility check using ALL edges of what remains
    # (fixed AND switchable), since a normally-open tie switch is still a
    # legitimate candidate to recommend restoration switch actions.
    full_graph = nx.Graph()
    full_graph.add_nodes_from(network.buses.keys())
    full_graph.add_edges_from((b.i, b.j) for b in reduced_branches)
    components = list(nx.connected_components(full_graph))
    slack_component = next(c for c in components if root in c)
    stranded = set(network.buses.keys()) - slack_component
    if stranded:
        print(f"[disaster_recovery] UNAVOIDABLE OUTAGE: bus(es) {stranded} have "
              f"NO physical path to the source after this fault, under any "
              f"switch configuration. This is reported as a hard structural "
              f"limit, not something the optimizer can route around.")
        return FaultResult(
            restorable=False, stranded_buses=stranded,
            new_dist_graph=None, new_switch_assignment=None,
            served_buses=slack_component, shed_buses=set(),
        )
    # Step 3: topologically restorable -- rerun the EXACT Module 4-6 pipeline
    # on the reduced graph, rather than inventing new restoration logic.
    dist_graph = nm.build_distribution_graph(reduced_network)
    n_loops = dist_graph.graph.number_of_edges() - dist_graph.graph.number_of_nodes() + 1
    if n_loops == 0:
        # Degenerate case: fixed + switchable edges together form EXACTLY a
        # spanning tree, with zero loops. There is no optimization freedom
        # left -- every switchable edge is structurally required just to
        # keep the network spanning, not a genuine choice. Close them all
        # rather than calling the QUBO builder on an empty problem.
        print("[disaster_recovery] Post-fault topology has zero independent "
              "loops: every switchable edge is now structurally REQUIRED "
              "(not optional) to keep the network connected. Closing all "
              "of them; no QUBO optimization is meaningful here.")
        switch_assignment = {e: 1 for e in dist_graph.switchable_edges}
    else:
        loops = qb.find_switchable_loops(dist_graph)
        costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root)
        Q, var_order = qb.build_qubo(loops, costs)
        switch_assignment, _ = qo.solve_with_classical_sa(Q, var_order)
    # Reconstruct the actual closed-edge set: fixed edges (always closed)
    # plus whichever switchable edges the solver decided to close.
    # Structurally-required switchable edges (needed for connectivity, not
    # QUBO variables) must be included here too -- they never appear in
    # switch_assignment since qb.find_switchable_loops excludes them.
    required_switchable, _optional = qb._structurally_required_switchable(dist_graph)
    closed_edges = set(dist_graph.fixed_edges) | set(required_switchable)
    for e, is_closed in switch_assignment.items():
        if is_closed:
            closed_edges.add(e)
    served_buses = set(network.buses.keys())
    shed_buses = set()
    # Step 4: capacity-driven load shedding, ONLY if a supply limit is given
    # and the reconnected network's total demand actually exceeds it.
    if available_supply_pu is not None:
        total_demand = sum(max(net_injection.get(b, 0.0), 0.0) for b in served_buses)
        if total_demand > available_supply_pu:
            print(f"[disaster_recovery] CAPACITY SHORTFALL: total demand "
                  f"{total_demand:.4f} pu exceeds available supply "
                  f"{available_supply_pu:.4f} pu. Running priority-weighted "
                  f"load shedding.")
            priority_weights = nm.get_priority_weights(dist_graph)
            loads = {b: max(net_injection.get(b, 0.0), 0.0) for b in served_buses}
            served_buses, shed_buses = knapsack_load_shedding(
                loads, priority_weights, available_supply_pu
            )
    return FaultResult(
        restorable=True, stranded_buses=set(),
        new_dist_graph=dist_graph, new_switch_assignment=switch_assignment,
        served_buses=served_buses, shed_buses=shed_buses,
    )
def knapsack_load_shedding(loads: dict, priority_weights: dict,
                            available_supply_pu: float,
                            discretization: float = 0.001):
    """
    Exact 0/1 knapsack via dynamic programming:
        max sum(w_i * served_i * L_i)  s.t.  sum(served_i * L_i) <= capacity
    Loads are continuous, so capacity is discretized into integer "slots"
    at the given resolution (default 0.001 pu) for the DP table -- this is
    an approximation of the true continuous capacity, explicitly named
    rather than silently rounding away demand.
    A greedy priority-only ordering is NOT used here because it is not
    guaranteed optimal for 0/1 knapsack in general -- exact DP is used
    instead so a lower-priority combination is never incorrectly preferred
    over a better-fitting higher-priority one.
    """
    buses = list(loads.keys())
    capacity_slots = int(round(available_supply_pu / discretization))
    weights_slots = {b: int(round(loads[b] / discretization)) for b in buses}
    values = {b: priority_weights.get(b, 1.0) * loads[b] for b in buses}
    n = len(buses)
    # dp[c] = best achievable value using capacity c (rolling 1D DP row)
    dp = [0.0] * (capacity_slots + 1)
    choice = [[False] * (capacity_slots + 1) for _ in range(n)]
    for idx, b in enumerate(buses):
        w, v = weights_slots[b], values[b]
        for c in range(capacity_slots, w - 1, -1):
            candidate = dp[c - w] + v
            if candidate > dp[c]:
                dp[c] = candidate
                choice[idx][c] = True
    # Backtrack to recover which buses were actually served.
    served, shed = set(), set()
    c = capacity_slots
    for idx in range(n - 1, -1, -1):
        b = buses[idx]
        if choice[idx][c]:
            served.add(b)
            c -= weights_slots[b]
        else:
            shed.add(b)
    return served, shed
if __name__ == "__main__":
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    # --- Scenario A: fault is topologically restorable via a tie switch ---
    print("=== Scenario A: restorable fault ===")
    net = dl.NetworkGraph(base=base)
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.3, Q_load_pu=0.1, priority_weight=5_000)
    net.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.2, Q_load_pu=0.05, priority_weight=1_000_000)  # hospital
    net.buses[4] = dl.Bus(id=4, bus_type="PQ", P_load_pu=0.25, Q_load_pu=0.08, priority_weight=100_000)
    net.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),  # will fault
        dl.Branch(i=3, j=4, R_pu=0.05, X_pu=0.08, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=4, j=1, R_pu=0.01, X_pu=0.02, S_max_pu=1.0, is_switchable=True, s_initial=0),  # tie switch
    ]
    net_injection = {1: 0.0, 2: 0.3, 3: 0.2, 4: 0.25}
    result_a = simulate_fault(net, (2, 3), net_injection, root=1)
    print(f"Restorable: {result_a.restorable}")
    print(f"New switch assignment: {result_a.new_switch_assignment}")
    print(f"Served buses: {result_a.served_buses}\n")
    # --- Scenario B: fault strands a bus with no alternate path ---
    print("=== Scenario B: unavoidable outage (no alternate path) ===")
    net2 = dl.NetworkGraph(base=base)
    net2.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net2.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.3, Q_load_pu=0.1)
    net2.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.2, Q_load_pu=0.05)
    net2.buses[5] = dl.Bus(id=5, bus_type="PQ", P_load_pu=0.1, Q_load_pu=0.02, priority_weight=1_000_000)
    net2.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=3, j=5, R_pu=0.03, X_pu=0.05, S_max_pu=0.5, is_switchable=False, s_initial=1),  # dead-end stub, will fault
    ]
    net_injection2 = {1: 0.0, 2: 0.3, 3: 0.2, 5: 0.1}
    result_b = simulate_fault(net2, (3, 5), net_injection2, root=1)
    print(f"Restorable: {result_b.restorable}")
    print(f"Stranded buses (hospital is bus 5!): {result_b.stranded_buses}\n")
    # --- Scenario C: restorable, but capacity-limited -> priority shedding ---
    print("=== Scenario C: restorable but capacity-limited -> priority shedding ===")
    result_c = simulate_fault(net, (2, 3), net_injection, root=1, available_supply_pu=0.5)
    print(f"Served buses: {result_c.served_buses}")
    print(f"Shed buses: {result_c.shed_buses}")