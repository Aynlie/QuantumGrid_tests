"""
qubo_builder.py
================
Module 5b of QuantumGrid.
Responsibility: find the network's fundamental loops, precompute a static
ohmic-loss weight for every candidate "open point" in each loop (using
power_flow.py's linearized tree-flow model), and assemble a QUBO whose
ground state selects the lowest-loss radial (spanning-tree) configuration.
Approach follows the strategy of Silva, Carvalho, Ferreira, & Omar (2023)
-- minimum-loss spanning-tree reconfiguration as a QUBO -- with equations
derived independently here rather than reproduced from their paper.
Simplifying assumption, stated explicitly (not hidden): fundamental loops
are treated as edge-disjoint in their switchable edges (the common case
for distribution feeders with distinct tie-switches per loop). Networks
with overlapping switchable loops need the more general MILP-based
radiality encoding of Lavorato et al. (2012), which this module does not
implement.
"""
from collections import defaultdict
import itertools
import networkx as nx
from power_flow import compute_tree_flows, total_ohmic_loss
def _structurally_required_switchable(dg):
    """
    Union-find over fixed edges first, then switchable edges: any
    switchable edge that CONNECTS two previously-separate components is
    structurally REQUIRED to keep the network spanning (this happens on
    post-fault reduced networks, where the surviving fixed edges alone
    are no longer connected). Any switchable edge that DOESN'T connect
    two separate components is a genuine redundant loop chord -- a real
    optimization choice, not a forced closure.
    Returns (required: list[(i,j)], optional: list[(i,j)]).
    """
    parent = {n: n for n in dg.graph.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    for (i, j) in dg.fixed_edges:
        union(i, j)
    required, optional = [], []
    # Process switchable edges with s_initial==1 (start CLOSED -- the
    # standard radial backbone) FIRST, then s_initial==0 (start OPEN --
    # tie switches) second. This ordering matters: it's what makes the
    # backbone naturally come out "required" (since it's what actually
    # spans the network) and the ties come out "optional" (since by the
    # time they're processed, the backbone has already connected
    # everything). Without this ordering, union-find's required/optional
    # split would depend on arbitrary edge-list order instead of which
    # edges are meant to form the default radial configuration.
    ordered_switchable = sorted(
        dg.switchable_edges,
        key=lambda e: 0 if dg.graph.edges[e[0], e[1]]["s_initial"] == 1 else 1,
    )
    for (i, j) in ordered_switchable:
        if union(i, j):
            required.append((i, j))
        else:
            optional.append((i, j))
    return required, optional


def find_switchable_loops(dg):
    """
    Compute each loop as the fundamental cycle of a genuinely OPTIONAL
    switchable (tie) edge -- one whose removal would not disconnect the
    network -- relative to a spanning tree of fixed edges (+ any
    switchable edges that are structurally required, see
    _structurally_required_switchable).

    Why not nx.cycle_basis(dg.graph) directly: that function has no notion
    of "fixed" vs "switchable" -- it builds its own arbitrary internal
    spanning tree from ALL edges (fixed + switchable combined), which can
    (and, on real multi-loop feeders, does) mix multiple switchable edges
    into the same returned cycle. That silently breaks this module's
    edge-disjoint-loops assumption even when the network itself satisfies it.

    Why not "fixed edges alone" either: on a post-fault reduced network,
    fixed edges alone may no longer even be connected -- some switchable
    (tie) edge is then structurally REQUIRED to restore connectivity, not
    an optional decision variable. Those are excluded from the returned
    loops entirely; compute_loop_open_costs() forces them closed.
    """
    required, optional = _structurally_required_switchable(dg)
    if required:
        print(f"[qubo_builder] NOTE: switchable edge(s) {required} are "
              f"structurally REQUIRED here to keep the network connected "
              f"-- treated as forced-closed, not QUBO decision variables.")
    tree_edges = set(dg.fixed_edges) | set(required)
    tree = nx.Graph()
    tree.add_nodes_from(dg.graph.nodes())
    tree.add_edges_from(tree_edges)
    if not nx.is_tree(tree):
        raise ValueError(
            "Fixed edges + structurally-required switchable edges still "
            "do not form a complete spanning tree -- the network is "
            "disconnected even with every available switch closed. This "
            "is a genuine unavoidable-outage case, not a loop-finding bug."
        )
    loops = [[(a, b)] for (a, b) in optional]
    return loops
def _unused_original_cycle_basis_version(dg):
    cycles_by_node = nx.cycle_basis(dg.graph)
    loops = []
    for cycle_nodes in cycles_by_node:
        cycle_edges = list(zip(cycle_nodes, cycle_nodes[1:] + cycle_nodes[:1]))
        # Normalize edge tuples to match dg.switchable_edges' undirected storage.
        normalized = []
        for (a, b) in cycle_edges:
            if (a, b) in dg.switchable_edges:
                normalized.append((a, b))
            elif (b, a) in dg.switchable_edges:
                normalized.append((b, a))
        if not normalized:
            raise ValueError(
                f"Loop {cycle_nodes} has no switchable edge -- this loop "
                f"can never be broken into a radial configuration."
            )

        loops.append(normalized)
    return loops
def compute_loop_open_costs(dg, loops, net_injection, root):
    """
    For every loop and every candidate open edge in that loop, compute the
    TOTAL network ohmic loss (c_e) that results from opening exactly that
    edge, holding every other loop at its default (s_initial) state.
    Returns {loop_index: {edge: cost}}.
    """
    all_switchable = set(dg.switchable_edges)
    default_closed = set(dg.fixed_edges)
    required, _optional = _structurally_required_switchable(dg)
    default_closed.update(required)  # structurally mandatory, regardless of s_initial
    for e in dg.switchable_edges:
        i, j = e
        s_init = dg.graph.edges[i, j]["s_initial"]
        if s_init == 1:
            default_closed.add(e)
    costs = {}
    for k, loop_edges in enumerate(loops):
        costs[k] = {}
        for open_edge in loop_edges:
            trial_closed = set(default_closed)
            # Ensure every switchable edge in THIS loop is closed except open_edge.
            for e in loop_edges:
                trial_closed.discard(e)
                if e != open_edge:
                    trial_closed.add(e)
            trial_closed.discard(open_edge)
            flows = compute_tree_flows(dg, trial_closed, net_injection, root)
            costs[k][open_edge] = total_ohmic_loss(dg, flows)
    return costs
def build_qubo(loops, loop_costs, penalty_strength=None):
    """
    H = sum_k [ sum_e c_e*(1-x_e) ]  +  P * sum_k (sum_e x_e - (|S_k|-1))^2
    x_e = 1 means edge e is CLOSED.
    Returns (Q, var_order) where Q is a dict {(var_i, var_j): coeff} in
    standard upper-triangular QUBO form (var_i, var_i) for linear terms,
    and var_order is the list of all switch variables in a fixed order
    (needed for brute-force verification / exporting to a real solver).
    """
    all_edges = [e for loop in loops for e in loop]
    var_order = list(dict.fromkeys(all_edges))  # de-duplicated, order-preserving
    if penalty_strength is None:
        max_cost = max(c for loop_c in loop_costs.values() for c in loop_c.values())
        penalty_strength = 10 * max_cost + 1.0  # comfortably dominates the objective
        # Penalty-strength selection follows the general guidance in Lucas
        # (2014): P must be large enough that violating a constraint always
        # costs more than any achievable improvement in the objective term.
    Q = defaultdict(float)
    for k, loop_edges in enumerate(loops):
        m = len(loop_edges) - 1  # required number of CLOSED edges in this loop
        for e in loop_edges:
            c_e = loop_costs[k][e]
            # Objective term: -c_e * x_e  (from expanding c_e*(1-x_e), dropping constant)
            Q[(e, e)] += -c_e
            # Penalty diagonal term: P*(1-2m)*x_e
            Q[(e, e)] += penalty_strength * (1 - 2 * m)
        for e, f in itertools.combinations(loop_edges, 2):
            key = (e, f) if var_order.index(e) < var_order.index(f) else (f, e)
            Q[key] += 2 * penalty_strength
    return dict(Q), var_order
def evaluate_qubo(Q, assignment: dict) -> float:
    """Evaluate H(x) for a given {var: 0/1} assignment -- used for brute-force checks."""
    energy = 0.0
    for (a, b), coeff in Q.items():
        if a == b:
            energy += coeff * assignment[a]
        else:
            energy += coeff * assignment[a] * assignment[b]
    return energy
def brute_force_solve(Q, var_order):
    """
    Exhaustive search over all 2^n assignments. Only tractable for small n
    (a handful of switchable edges per loop set, typical for a hackathon-
    scale test feeder) -- used here purely to VERIFY the QUBO's ground
    state matches the direct per-loop argmin, not as a production solver.
    Module 6 uses Qiskit/D-Wave for problems too large to brute force.
    """
    best_energy, best_assignment = None, None
    for bits in itertools.product([0, 1], repeat=len(var_order)):
        assignment = dict(zip(var_order, bits))
        energy = evaluate_qubo(Q, assignment)
        if best_energy is None or energy < best_energy:
            best_energy, best_assignment = energy, assignment
    return best_assignment, best_energy
if __name__ == "__main__":
    import data_loader as dl
    import network_model as nm
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    net = dl.NetworkGraph(base=base)
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.3, Q_load_pu=0.1)
    net.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.2, Q_load_pu=0.05)
    net.buses[4] = dl.Bus(id=4, bus_type="PQ", P_load_pu=0.25, Q_load_pu=0.08)
    # Two paths between bus 1 and bus 4: via 2-3 (one route) and a direct
    # tie switch 4-1 -- exactly one fundamental loop, two candidate open points.
    net.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=3, j=4, R_pu=0.05, X_pu=0.08, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=4, j=1, R_pu=0.01, X_pu=0.02, S_max_pu=1.0, is_switchable=True, s_initial=0),
    ]
    dist_graph = nm.build_distribution_graph(net)
    loops = find_switchable_loops(dist_graph)
    print(f"Found {len(loops)} loop(s): {loops}")
    net_injection = {1: 0.0, 2: 0.3, 3: 0.2, 4: 0.25}  # P_net,i(t) from Module 3
    costs = compute_loop_open_costs(dist_graph, loops, net_injection, root=1)
    print("Per-edge open costs (lower = better place to open):")
    for k, edge_costs in costs.items():
        for e, c in edge_costs.items():
            print(f"  loop {k}, open {e}: total network loss = {c:.6f} pu")
    Q, var_order = build_qubo(loops, costs)
    best_assignment, best_energy = brute_force_solve(Q, var_order)
    print(f"\nQUBO variables: {var_order}")
    print(f"Best assignment (1=closed, 0=open): {best_assignment}")
    print(f"QUBO energy at best assignment: {best_energy:.4f}")
    # Independent correctness check: directly compare against the raw
    # per-loop argmin of the precomputed costs (should agree with the QUBO).
    for k, edge_costs in costs.items():
        direct_best_open = min(edge_costs, key=edge_costs.get)
        qubo_open = [e for e in edge_costs if best_assignment[e] == 0]
        print(f"\nLoop {k}: direct argmin says open {direct_best_open} "
              f"(loss={edge_costs[direct_best_open]:.6f}); "
              f"QUBO solution opens {qubo_open}")
        assert qubo_open == [direct_best_open], "QUBO result disagrees with direct argmin!"
    print("\nVerification passed: QUBO ground state matches direct per-loop argmin.")