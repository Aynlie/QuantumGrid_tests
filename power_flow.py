"""
power_flow.py
================
Module 5a of QuantumGrid.
Responsibility: given a CANDIDATE radial (tree) topology, compute branch
power flows without iterative AC power flow solving, using the linearized
tree-summation model derived in the Module 5 write-up. This is what makes
embedding "loss" into a QUBO possible at all -- it converts a nonlinear
AC power flow problem into a closed-form linear computation, valid once
the topology is fixed to a tree.
Explicitly NOT solved here: exact AC voltage magnitudes under losses. We
use the standard flat-voltage (V=1 pu) approximation for loss estimation,
and a separate linearized voltage-drop check (LinDistFlow) for post-hoc
feasibility -- both approximations are named explicitly, not hidden.
"""
import networkx as nx
def compute_tree_flows(dg, closed_edges, net_injection, root):
    """
    P_(parent(k) -> k) = sum of net_injection over subtree(k)
    dg             : DistributionGraph from Module 4.
    closed_edges   : iterable of (i, j) tuples that are CLOSED in this
                     candidate configuration (fixed edges + chosen switch
                     positions). Must form a spanning tree together with
                     every bus in dg.graph.
    net_injection  : {bus_id: P_net,i(t)} from Module 3 (positive = load,
                     negative = net export).
    root           : slack bus id.
    Returns {(parent, child): flow_pu} for every edge in the tree.
    """
    T = nx.Graph()
    T.add_nodes_from(dg.graph.nodes())
    T.add_edges_from(closed_edges)
    if not nx.is_connected(T):
        raise ValueError("Candidate closed-edge set is not connected -- "
                          "cannot compute tree flows on a disconnected graph.")
    if T.number_of_edges() != T.number_of_nodes() - 1:
        raise ValueError(
            f"Candidate closed-edge set has {T.number_of_edges()} edges and "
            f"{T.number_of_nodes()} nodes -- not a valid tree (expected "
            f"exactly N-1 edges). This configuration is not radial."
        )
    # Post-order traversal from root: process children before parents so
    # each bus's subtree sum already includes all of its descendants.
    order = list(nx.dfs_postorder_nodes(T, source=root))

    bfs_parent = dict(nx.bfs_predecessors(T, source=root))
    subtree_sum = {bus: net_injection.get(bus, 0.0) for bus in T.nodes()}
    flows = {}
    for node in order:
        if node == root:
            continue
        parent = bfs_parent[node]
        # Accumulate this node's (already-complete) subtree sum into its parent.
        subtree_sum[parent] += subtree_sum[node]
        flows[(parent, node)] = subtree_sum[node]
    return flows
def total_ohmic_loss(dg, flows, v_nominal_pu=1.0):
    """
    Total network loss = sum over edges of R_ij * P_ij^2 / V_nominal^2
    Flat-voltage approximation (V_nominal_pu = 1.0 by default) -- explicitly
    an approximation, stated per the Module 5 derivation, not an exact
    AC loss computation.
    """
    total = 0.0
    for (i, j), p_flow in flows.items():
        r_ij = dg.graph.edges[i, j]["R_pu"]
        total += r_ij * (p_flow ** 2) / (v_nominal_pu ** 2)
    return total
def check_voltage_feasibility(dg, flows, q_flows, root, v_root_pu=1.0):
    """
    Post-hoc linearized voltage-drop check (LinDistFlow, Baran & Wu, 1989):
        V_i^2 - V_j^2 ~= 2*(R_ij * P_ij + X_ij * Q_ij)
    This is deliberately NOT embedded in the QUBO -- Silva et al. (2023)
    themselves list adding voltage/current constraints to this class of
    QUBO as future work, so this is implemented here as a classical
    feasibility check run on the QUBO's winning candidate, not as part
    of the combinatorial optimization itself.
    q_flows: reactive power flows, same structure as `flows`. If you only
    have P (active) flows, pass an all-zero dict of the same keys and note
    this in your report -- do not silently omit Q from the check.
    Returns {bus_id: V_pu, violations: [bus_ids outside limits]}.
    """
    v_sq = {root: v_root_pu ** 2}
    # Build the tree from the edges present in `flows` and traverse from
    # the root OUTWARD (pre-order): a parent's voltage must be known before
    # any of its children's can be computed. Iterating `flows` directly in
    # its own (post-order) insertion order would get this backwards.
    T = nx.Graph()
    T.add_edges_from(flows.keys())
    bfs_edges = list(nx.bfs_edges(T, source=root))
    for (parent, child) in bfs_edges:
        key = (parent, child) if (parent, child) in flows else (child, parent)
        p_ij = flows[key]
        q_ij = q_flows.get(key, 0.0)
        r_ij = dg.graph.edges[parent, child]["R_pu"]
        x_ij = dg.graph.edges[parent, child]["X_pu"]
        v_sq[child] = v_sq[parent] - 2 * (r_ij * p_ij + x_ij * q_ij)
    voltages = {bus: v_sq_val ** 0.5 for bus, v_sq_val in v_sq.items()}
    violations = []
    for bus_id, v in voltages.items():
        bus = dg.graph.nodes[bus_id]
        if not (bus["V_min_pu"] <= v <= bus["V_max_pu"]):
            violations.append(bus_id)
    return {"voltages_pu": voltages, "violations": violations}