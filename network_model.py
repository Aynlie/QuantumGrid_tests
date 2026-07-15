"""
network_model.py
================
Module 4 of QuantumGrid.
Responsibility: convert Module 1's plain NetworkGraph dataclass into a
networkx graph with the CORRECT attribute placement (voltage on nodes,
never on edges), split edges into fixed vs. switchable sets, and validate
structural feasibility (connectivity, loop count) BEFORE the graph is
handed to the QUBO builder (Module 5/6).
This module does not solve power flow and does not decide switch states --
it only builds and validates the structure those later modules operate on.
"""
from dataclasses import dataclass
import networkx as nx
from data_loader import NetworkGraph  # Module 1
@dataclass
class DistributionGraph:
    """
    Wraps a networkx.Graph with the bookkeeping needed by Modules 5-7.
    graph        : networkx.Graph, nodes = bus ids, edges = branches.
                   Node attrs: bus_type, priority_weight, V_min_pu, V_max_pu.
                   Edge attrs: R_pu, X_pu, S_max_pu, is_switchable, s_initial.
                   NOTE: voltage magnitude V_i is NOT a static edge/node
                   attribute here -- it is a per-timestep POWER FLOW RESULT
                   (Module 5's output), so it is deliberately absent from
                   this static structural model.
    fixed_edges     : list of (i, j) tuples with no switch (always closed).
    switchable_edges: list of (i, j) tuples that are QUBO decision variables.
    """
    graph: nx.Graph
    fixed_edges: list
    switchable_edges: list
def build_distribution_graph(network: NetworkGraph) -> DistributionGraph:
    """
    Build a networkx.Graph from Module 1's NetworkGraph, with attributes
    placed on the correct element (node vs. edge) per the corrected
    mathematical model in Module 4's write-up.
    """
    G = nx.Graph()
    for bus_id, bus in network.buses.items():
        G.add_node(
            bus_id,
            bus_type=bus.bus_type,
            priority_weight=bus.priority_weight,
            V_min_pu=bus.V_min_pu,
            V_max_pu=bus.V_max_pu,
            # P_load_pu is deliberately NOT copied here: Modules 2-3 produce
            # a *time-indexed* per-bus load/generation series, and storing a
            # single static P_load_pu on the graph would silently freeze it
            # at its Module-1 base-case value.
        )
    fixed_edges, switchable_edges = [], []
    for branch in network.branches:
        G.add_edge(
            branch.i, branch.j,
            R_pu=branch.R_pu,
            X_pu=branch.X_pu,
            S_max_pu=branch.S_max_pu,
            is_switchable=branch.is_switchable,
            s_initial=branch.s_initial,
        )
        if branch.is_switchable:
            switchable_edges.append((branch.i, branch.j))
        else:
            fixed_edges.append((branch.i, branch.j))
    dg = DistributionGraph(graph=G, fixed_edges=fixed_edges,
                            switchable_edges=switchable_edges)
    _validate_structure(dg)
    return dg
def _validate_structure(dg: DistributionGraph) -> None:
    """
    Structural checks that must pass BEFORE this graph reaches the QUBO
    builder. Failing loudly here is much cheaper than debugging an
    infeasible or nonsensical QUBO later.
    """
    G = dg.graph
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    # 1. Base topology (every switch closed) must be connected.
    all_closed = G.copy()
    if not nx.is_connected(all_closed):
        components = list(nx.connected_components(all_closed))
        raise ValueError(
            f"Base topology (all switches closed) is NOT connected: "
            f"{len(components)} separate components found. A distribution "
            f"graph that isn't even connected with every switch closed can "
            f"never be made into a single connected radial network."
        )
    # 2. Independent-loop count identity: |E| - |N| + 1 for a connected graph.
    n_loops = n_edges - n_nodes + 1
    if n_loops < 0:
        raise ValueError("Graph is not even a connected structure (negative loop count).")
    # 3. Enough switchable edges to remove all loops and reach a tree.
    if len(dg.switchable_edges) < n_loops:
        raise ValueError(
            f"Structural infeasibility: graph has {n_loops} independent "
            f"loop(s) but only {len(dg.switchable_edges)} switchable "
            f"edge(s). No assignment of switch states can produce a "
            f"radial (loop-free), fully connected operating topology. "
            f"Add more tie-switches or fewer fixed loop-forming edges."
        )
    print(f"[network_model] Validation passed: {n_nodes} buses, {n_edges} "
          f"branches ({len(dg.fixed_edges)} fixed, {len(dg.switchable_edges)} "
          f"switchable), {n_loops} independent loop(s) to resolve via switching.")
def get_priority_weights(dg: DistributionGraph) -> dict:
    """Convenience accessor used by Module 7 (disaster recovery)."""
    return nx.get_node_attributes(dg.graph, "priority_weight")
if __name__ == "__main__":
    import data_loader as dl
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    net = dl.NetworkGraph(base=base)
    # 4-bus loop: 1-2-3-4-1, with bus 1 as slack, plus a fixed radial stub
    # bus 5, to exercise both fixed and switchable edges and a real loop.
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.3, Q_load_pu=0.1)
    net.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.2, Q_load_pu=0.05)
    net.buses[4] = dl.Bus(id=4, bus_type="PQ", P_load_pu=0.25, Q_load_pu=0.08)
    net.buses[5] = dl.Bus(id=5, bus_type="PQ", P_load_pu=0.1, Q_load_pu=0.02,
                           priority_weight=1_000_000)  # e.g. a hospital
    net.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=3, j=4, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=4, j=1, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=True, s_initial=0),  # tie switch closes the loop
        dl.Branch(i=3, j=5, R_pu=0.03, X_pu=0.05, S_max_pu=0.5, is_switchable=False, s_initial=1),
    ]
    dist_graph = build_distribution_graph(net)
    print("Fixed edges:", dist_graph.fixed_edges)
    print("Switchable edges:", dist_graph.switchable_edges)
    print("Priority weights:", get_priority_weights(dist_graph))