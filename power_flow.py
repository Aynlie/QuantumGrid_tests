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
from collections import defaultdict
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
def solve_subproblem(dg, closed_edges, net_injection, q_injection, root):
    """
    Benders subproblem (paper Sec 4.3, Eq 35-40), scoped to this repo's
    zero-free-continuous-DOF tree model: once a candidate topology is
    fixed, compute_tree_flows already returns THE unique exact flow
    solution -- there is no LP/NLP left to actually optimize over, so
    this function's real job is "compute the exact flows/Phi for this
    candidate, then check the two hard limits (thermal, voltage) that
    qubo_builder.build_qubo does not enforce in the QUBO itself."
    See benders_loop.py's module docstring for the full honest-scope
    statement this implements.
    q_injection may be {} (reactive power not modeled at this stage,
    matching main.py's existing q_flows convention elsewhere in the
    pipeline) -- Q flows come back all-zero in that case.
    Raises ValueError (propagated directly from compute_tree_flows) if
    closed_edges is not a valid spanning tree -- callers must catch this
    themselves as its own distinct kind of infeasibility (see
    benders_loop.run_benders_qaoa_loop's except ValueError branch).
    Returns {"phi", "flows", "q_flows", "voltages_pu",
             "thermal_violations", "voltage_violations", "feasible"}.
    """
    flows = compute_tree_flows(dg, closed_edges, net_injection, root)
    if q_injection:
        q_flows = compute_tree_flows(dg, closed_edges, q_injection, root)
    else:
        q_flows = {edge: 0.0 for edge in flows}
    phi = total_ohmic_loss(dg, flows)
    voltage_check = check_voltage_feasibility(dg, flows, q_flows, root)
    thermal_violations = []
    for (i, j), p_ij in flows.items():
        q_ij = q_flows.get((i, j), 0.0)
        s_ij = (p_ij ** 2 + q_ij ** 2) ** 0.5
        s_max = dg.graph.edges[i, j]["S_max_pu"]
        if s_ij > s_max:
            thermal_violations.append((i, j))
    voltage_violations = list(voltage_check["violations"])
    return {
        "phi": phi,
        "flows": flows,
        "q_flows": q_flows,
        "voltages_pu": voltage_check["voltages_pu"],
        "thermal_violations": thermal_violations,
        "voltage_violations": voltage_violations,
        "feasible": not thermal_violations and not voltage_violations,
    }
def generate_feasibility_cut(dg, switchable_edges, assignment, sub, cut_penalty):
    """
    Turn one infeasible Benders iteration into {edge: penalty} additions
    for qubo_builder.build_master_qubo's `cuts` argument.
    Only CLOSED switchable edges (assignment[e]==1) are ever penalized
    here -- i.e. this only ever makes closing an edge MORE expensive,
    never cheaper. Two reasons, specific to this repo's single-chord-
    per-loop QUBO (qubo_builder.build_qubo): (1) an OPEN optional edge
    exerts zero influence on tree flows -- it's simply absent from
    closed_edges, so there's nothing physically to "blame" it for; (2)
    build_qubo's own radiality penalty already keeps every optional edge
    at x_e=0 by a wide margin (m = len(loop)-1 = 0 for every singleton
    loop here), so trying to force x_e=1 via a negative cut would have to
    outweigh that dominant term -- which risks exactly the non-spanning-
    -tree failure this module's structural_error branch already exists
    to catch, rather than a controlled, boundable feasibility cut.
    Two cases:
      - sub["structural_error"] set (non-tree candidate): every currently
        -closed optional edge helped create it -- penalize all of them.
      - Physically-violated tree: penalize a closed switchable edge if
        (a) it IS itself a violated thermal line, or (b) a violated
        line/bus lies in the subtree fed through it (i.e. its closure
        routes power through the violated area).
    If no switchable edge is implicated (e.g. the violation is squarely
    on a structurally-required/fixed edge -- see qubo_builder.
    _structurally_required_switchable), returns {}: there is genuinely
    no decision variable that could fix this, and the caller's stall
    detection (benders_loop.py) is what correctly reports that.
    """
    implicated = {}
    if sub.get("structural_error") is not None:
        for e in switchable_edges:
            if assignment.get(e, 0) == 1:
                implicated[e] = cut_penalty
        return implicated
    violated_lines = set(sub.get("thermal_violations", []))
    violated_buses = set(sub.get("voltage_violations", []))
    for e in switchable_edges:
        if assignment.get(e, 0) != 1:
            continue
        if e in violated_lines or (e[1], e[0]) in violated_lines:
            implicated[e] = cut_penalty
    flows = sub.get("flows")
    if flows:
        children = defaultdict(list)
        for (p, c) in flows:
            children[p].append(c)
        def subtree(start):
            seen, stack = {start}, [start]
            while stack:
                n = stack.pop()
                for ch in children.get(n, []):
                    if ch not in seen:
                        seen.add(ch)
                        stack.append(ch)
            return seen
        violated_nodes = set(violated_buses)
        for (u, v) in violated_lines:
            violated_nodes.add(u)
            violated_nodes.add(v)
        for e in switchable_edges:
            if assignment.get(e, 0) != 1 or e in implicated:
                continue
            downstream = subtree(e[1])
            if downstream & violated_nodes:
                implicated[e] = cut_penalty
    return implicated