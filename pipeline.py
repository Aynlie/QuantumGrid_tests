# pipeline.py
"""High‑level helper that runs the full Benders‑QUBO‑QAOA‑MILP pipeline.

The function `run_full_optimization` loads the network, builds the QUBO,
executes the Benders loop (which may internally call a QAOA or SA solver),
optionally runs the MILP verification step, and returns a structured
dictionary that can be JSON‑serialised by a web API.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

import data_loader as dl
import network_model as nm
import qubo_builder as qb
import benders_loop as bl
import quantum_optimizer as qo
import power_flow as pf
import disaster_recovery as dr


def run_full_optimization(
    network_json_path: str = "pilot_requests.json",
    solver: str = "sa",
    cut_penalty: Optional[float] = None,
    max_iters: int = 10,
    seed: int = 0,
    run_milp: bool = True,
) -> Dict[str, Any]:
    """Execute the end‑to‑end optimisation pipeline.

    Parameters
    ----------
    network_json_path: str
        Path to a JSON file that describes the network (same format used by
        ``data_loader`` in the original repo).
    solver: str
        Which backend to use for the master QUBO – ``"sa"`` (simulated
        annealing), ``"dwave"`` or ``"qaoa"``.
    cut_penalty: float, optional
        Custom penalty for feasibility cuts; if ``None`` the default logic in
        ``benders_loop.run_benders_qaoa_loop`` is used.
    max_iters: int
        Maximum number of Benders iterations.
    seed: int
        Random seed for reproducibility.
    run_milp: bool
        If ``True`` the MILP verification (``disaster_recovery``) is executed on
        the final topology.

    Returns
    -------
    dict
        A JSON‑serialisable mapping containing:
        * ``switch_assignment`` – final edge states
        * ``energy`` – master QUBO energy
        * ``phi`` – loss value from the power‑flow sub‑problem
        * ``feasible`` – whether the final topology satisfied all constraints
        * ``milp_status`` – result of the MILP check (or ``None``)
        * ``iterations`` – number of Benders rounds performed
        * ``history`` – per‑iteration diagnostics useful for UI dashboards
    """
    # ---------------------------------------------------------------------
    # 1️⃣ Load the network description
    # ---------------------------------------------------------------------
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    net = dl.NetworkGraph(base=base)
    # The original repository ships a small example network in ``pilot_requests.json``.
    # ``data_loader`` provides a helper to populate ``net`` from that file.
    dl.load_network_from_json(net, network_json_path)

    # ---------------------------------------------------------------------
    # 2️⃣ Build the distribution graph and identify switchable loops
    # ---------------------------------------------------------------------
    dist_graph = nm.build_distribution_graph(net)
    loops = qb.find_switchable_loops(dist_graph)
    # Net injection (real power) – use the values already stored in the network.
    net_injection = {bus.id: bus.P_load_pu for bus in net.buses.values()}
    # Compute loop opening costs (the "loss" contribution of each switchable edge).
    loop_costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root=1)

    # ---------------------------------------------------------------------
    # 3️⃣ Run the Benders‑QUBO loop
    # ---------------------------------------------------------------------
    benders_res = bl.run_benders_qaoa_loop(
        dg=dist_graph,
        loops=loops,
        loop_costs=loop_costs,
        net_injection=net_injection,
        root=1,
        solver=solver,
        cut_penalty=cut_penalty,
        max_iters=max_iters,
        seed=seed,
    )

    # ---------------------------------------------------------------------
    # 4️⃣ Optional MILP verification (disaster recovery)
    # ---------------------------------------------------------------------
    milp_status = None
    if run_milp:
        # ``disaster_recovery`` expects a concrete topology (set of closed edges).
        closed_edges = set(net.fixed_edges)
        closed_edges.update(e for e in benders_res.switch_assignment if benders_res.switch_assignment[e] == 1)
        milp_status = dr.check_feasibility_milp(dist_graph, closed_edges, net_injection)

    # ---------------------------------------------------------------------
    # 5️⃣ Assemble the result dictionary
    # ---------------------------------------------------------------------
    result = {
        "switch_assignment": benders_res.switch_assignment,
        "energy": benders_res.energy,
        "phi": benders_res.subproblem.get("phi"),
        "feasible": benders_res.converged,
        "milp_status": milp_status,
        "iterations": benders_res.iterations,
        "history": benders_res.history,
    }
    return result


if __name__ == "__main__":
    # Simple CLI demo – prints a nicely formatted JSON payload.
    import json
    res = run_full_optimization()
    print(json.dumps(res, indent=2, default=str))
