"""
benders_loop.py
================
Module 6b of QuantumGrid -- Benders-linked QAOA/classical loop, wiring
the paper's Section 4 / Algorithm 1 onto this repo's existing modules
(qubo_builder.py, power_flow.py, quantum_optimizer.py).

HONEST SCOPE STATEMENT (read before using this module):
This repo's radial-reconfiguration QUBO (qubo_builder.py) already
computes the loss objective as an EXACT affine function of x for every
switchable edge (compute_loop_open_costs precomputes it directly), since
power_flow.py's tree-flow model has zero free continuous degrees of
freedom once topology is fixed (see power_flow.solve_subproblem's
docstring). That means there is no "optimality cut" to iterate on here
in the paper's Eq. 41 sense -- a nonconvex loss surface needing
successive linear underestimators -- because the master's objective is
already exact, not an approximation being refined.

What genuinely benefits from an iterative Benders-style loop in THIS
codebase is FEASIBILITY: voltage-band and thermal-rating constraints
(paper Eqs. 9-10) are not encoded in the static QUBO at all -- previously
(main.py's old Stage 5-6) they were only checked once, after the fact,
on whatever the QUBO happened to return, with no mechanism to react if
that candidate turned out to be infeasible. This module closes that gap
by turning it into the loop of Algorithm 1: propose a topology, check
its exact feasibility, and if infeasible, add a feasibility cut
(power_flow.generate_feasibility_cut) to the master and try again --
instead of silently accepting (or crashing on) an infeasible switch
configuration.

What is consequently NOT reproduced from the paper here: the theta
surrogate variable (Eq. 32 -- see qubo_builder.build_master_qubo's
docstring for why), Lagrange-multiplier-weighted optimality cuts
(Eq. 41 -- there are no multipliers to draw on in a zero-DOF tree
model), and a certified UB/LB optimality gap (Section 4.4's UB(t)/LB(t)
-- since Phi is already exact, "LB" here is just the master QUBO energy
and "UB" is the best feasible Phi found; the loop's stopping condition
is feasibility, not a proven optimality gap closing to within delta).
"""
from dataclasses import dataclass, field

import qubo_builder as qb
import power_flow as pf
import quantum_optimizer as qo


@dataclass
class BendersResult:
    switch_assignment: dict      # {edge: 0/1}, 1 = closed -- final accepted topology
    subproblem: dict             # power_flow.solve_subproblem's return dict, at the winner
    energy: float                # master QUBO energy at the winning iteration
    iterations: int              # number of Benders rounds actually run
    converged: bool              # True iff a feasible topology was found within max_iters
    cuts: dict = field(default_factory=dict)      # final accumulated cut set (edge -> penalty)
    history: list = field(default_factory=list)  # per-iteration diagnostics, for dashboard.py


def run_benders_qaoa_loop(dg, loops, loop_costs, net_injection, root,
                           q_injection=None, solver="sa",
                           cut_penalty=None, max_iters=10, seed=0):
    """
    Algorithm 1 (paper Section 4.4), scoped as described in the module
    docstring above.

    dg, loops, loop_costs : outputs of qubo_builder.find_switchable_loops
                             / compute_loop_open_costs, exactly as before.
    net_injection          : {bus_id: P_net,i(t)}, as passed to
                              compute_loop_open_costs.
    q_injection             : {bus_id: Q_net,i(t)}, optional -- pass {}
                               (default) if reactive power isn't modeled
                               at this stage, same convention main.py
                               already uses for q_flows.
    solver                  : "sa" (quantum_optimizer.solve_with_classical_sa,
                               always available), "dwave", or "qaoa" --
                               same (Q, var_order) contract as the rest
                               of the pipeline, so any backend drops in
                               here unchanged.
    cut_penalty              : penalty added per implicated edge per
                                infeasible iteration (see
                                power_flow.generate_feasibility_cut).
                                Defaults to 20x the largest per-edge loop
                                cost -- comfortably dominates the loss
                                objective, same sizing rationale as
                                qubo_builder.build_qubo's own default
                                penalty_strength.
    max_iters                : Tmax (Section 4.4) -- the loop raises
                                RuntimeError if no feasible topology is
                                found within this many rounds, rather
                                than silently returning an infeasible one.

    Returns a BendersResult.
    """
    switchable_edges = [e for loop in loops for e in loop]
    max_cost = max(c for lc in loop_costs.values() for c in lc.values())
    # Mirror build_qubo's own default radiality penalty exactly (rather
    # than letting each call re-derive it independently), so cut sizing
    # below can be guaranteed relative to it.
    radiality_penalty = 10 * max_cost + 1.0

    if cut_penalty is None:
        # MUST stay well below radiality_penalty: a cut is only supposed
        # to make re-selecting the same infeasible edge state LESS
        # attractive among otherwise-radial-feasible alternatives, never
        # strong enough to make the solver break radiality altogether
        # (i.e. return a non-spanning-tree candidate) just to satisfy a
        # cut. 3x the largest per-edge loss swing is comfortably above
        # anything the objective term alone would produce, while staying
        # under a third of radiality_penalty.
        cut_penalty = 3 * max_cost + 1.0
    elif cut_penalty >= radiality_penalty:
        raise ValueError(
            f"cut_penalty={cut_penalty} must be smaller than the master "
            f"QUBO's radiality penalty ({radiality_penalty}), or the "
            f"Benders loop can be driven to return non-spanning-tree "
            f"candidates instead of ever converging."
        )
    # Cap how much a single edge's cumulative cut can grow across many
    # iterations, for the same reason -- without this, an edge that keeps
    # getting implicated could eventually accumulate more penalty than
    # radiality_penalty even though each individual cut was safely sized.
    max_cumulative_cut_per_edge = 0.8 * radiality_penalty

    required, _optional = qb._structurally_required_switchable(dg)

    cuts = {}
    history = []
    prev_assignment = None
    for t in range(1, max_iters + 1):
        Q, var_order = qb.build_master_qubo(loops, loop_costs, cuts=cuts,
                                             penalty_strength=radiality_penalty)

        if solver == "sa":
            assignment, energy = qo.solve_with_classical_sa(Q, var_order, seed=seed + t)
        elif solver == "dwave":
            assignment, energy = qo.solve_with_dwave(Q, var_order)
        elif solver == "qaoa":
            assignment, energy = qo.solve_with_qaoa(Q, var_order)
        else:
            raise ValueError(f"Unknown solver '{solver}' (expected 'sa', 'dwave', or 'qaoa')")

        if t > 1 and assignment == prev_assignment:
            # Cuts accumulated last round had zero effect on the master's
            # choice -- meaning the radiality penalty (correctly) refused
            # to move to any other spanning tree despite the added cost of
            # repeating the same infeasible topology. There is no other
            # radial candidate left to try: continuing to iterate would
            # just repeat this exact outcome up to max_iters for no
            # reason, burning solver calls on a foregone conclusion.
            raise RuntimeError(
                f"Benders search stalled at iteration {t}: the master "
                f"proposed the same topology {assignment} again even after "
                f"a feasibility cut was added against it, which means no "
                f"other spanning-tree candidate exists given this "
                f"network's fixed/switchable edge split (see "
                f"qubo_builder._structurally_required_switchable). This is "
                f"a genuine 'unavoidable violation given the available "
                f"switches' result, not a search failure -- last violation: "
                f"thermal={history[-1]['thermal_violations']}, "
                f"voltage={history[-1]['voltage_violations']}. Restoring "
                f"feasibility here would require either uprating the "
                f"violated line's S_max_pu / voltage limits, or adding a "
                f"sectionalizing switch this feeder's topology doesn't "
                f"currently have."
            )
        prev_assignment = dict(assignment)

        closed_edges = set(dg.fixed_edges)
        closed_edges.update(required)  # structurally forced closed, not QUBO variables
        closed_edges.update(e for e in switchable_edges if assignment.get(e, 0) == 1)

        try:
            sub = pf.solve_subproblem(dg, closed_edges, net_injection, q_injection or {}, root)
        except ValueError as exc:
            # The master's radiality penalty is itself a soft (quadratic
            # penalty) constraint, same caveat the paper states for L(x)
            # in Section 2.4/4.5: a large enough accumulated cut CAN, in
            # principle, outweigh it and produce a non-spanning-tree
            # candidate. Rather than crash, treat that exactly like any
            # other infeasible subproblem outcome (Algorithm 1, line 9-10)
            # and add a feasibility cut against it too.
            sub = {"phi": None, "flows": None, "q_flows": None, "voltages_pu": None,
                   "thermal_violations": [], "voltage_violations": [],
                   "feasible": False, "structural_error": str(exc)}
            history.append({
                "iteration": t, "assignment": dict(assignment), "energy": energy,
                "phi": None, "feasible": False,
                "thermal_violations": [], "voltage_violations": [],
                "structural_error": str(exc),
            })
            new_cut = pf.generate_feasibility_cut(dg, switchable_edges, assignment, sub, cut_penalty)
            for e, p in new_cut.items():
                cuts[e] = min(cuts.get(e, 0.0) + p, max_cumulative_cut_per_edge)
            continue

        history.append({
            "iteration": t,
            "assignment": dict(assignment),
            "energy": energy,
            "phi": sub["phi"],
            "feasible": sub["feasible"],
            "thermal_violations": list(sub["thermal_violations"]),
            "voltage_violations": list(sub["voltage_violations"]),
        })

        if sub["feasible"]:
            return BendersResult(
                switch_assignment=assignment, subproblem=sub, energy=energy,
                iterations=t, converged=True, cuts=dict(cuts), history=history,
            )

        new_cut = pf.generate_feasibility_cut(dg, switchable_edges, assignment, sub, cut_penalty)
        for e, p in new_cut.items():
            cuts[e] = min(cuts.get(e, 0.0) + p, max_cumulative_cut_per_edge)

    raise RuntimeError(
        f"No feasible radial topology found within max_iters={max_iters} "
        f"Benders iterations (paper's Tmax, Section 4.4). Last candidate "
        f"violated: thermal={history[-1]['thermal_violations']}, "
        f"voltage={history[-1]['voltage_violations']}. Consider raising "
        f"max_iters, or check whether S_max_pu / V_min_pu / V_max_pu are "
        f"set unrealistically tight for this network."
    )


if __name__ == "__main__":
    # Same 4-bus fixture used by qubo_builder.py / quantum_optimizer.py,
    # so this module can be sanity-checked in isolation.
    import data_loader as dl
    import network_model as nm

    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    net = dl.NetworkGraph(base=base)
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    net.buses[2] = dl.Bus(id=2, bus_type="PQ", P_load_pu=0.3, Q_load_pu=0.1)
    net.buses[3] = dl.Bus(id=3, bus_type="PQ", P_load_pu=0.2, Q_load_pu=0.05)
    net.buses[4] = dl.Bus(id=4, bus_type="PQ", P_load_pu=0.25, Q_load_pu=0.08)
    net.branches = [
        dl.Branch(i=1, j=2, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=2, j=3, R_pu=0.02, X_pu=0.04, S_max_pu=1.0, is_switchable=False, s_initial=1),
        dl.Branch(i=3, j=4, R_pu=0.05, X_pu=0.08, S_max_pu=1.0, is_switchable=True, s_initial=1),
        dl.Branch(i=4, j=1, R_pu=0.01, X_pu=0.02, S_max_pu=1.0, is_switchable=True, s_initial=0),
    ]
    dist_graph = nm.build_distribution_graph(net)
    loops = qb.find_switchable_loops(dist_graph)
    net_injection = {1: 0.0, 2: 0.3, 3: 0.2, 4: 0.25}
    costs = qb.compute_loop_open_costs(dist_graph, loops, net_injection, root=1)

    print("Case A: capacities as given (should converge in 1 iteration -- "
          "nothing to make infeasible)")
    result = run_benders_qaoa_loop(dist_graph, loops, costs, net_injection, root=1)
    print(f"  converged={result.converged} in {result.iterations} iteration(s)")
    print(f"  switch_assignment={result.switch_assignment}")
    print(f"  phi (loss)={result.subproblem['phi']:.6f}")

    print("\nCase B: artificially tight thermal rating on a backbone edge "
          "in Case A's chosen (all-ties-open) path, to force at least one "
          "feasibility-cut iteration where the master has to try closing "
          "the tie switch instead")
    dist_graph.graph.edges[3, 4]["S_max_pu"] = 1e-6
    try:
        result2 = run_benders_qaoa_loop(dist_graph, loops, costs, net_injection, root=1,
                                         max_iters=5)
        print(f"  converged={result2.converged} in {result2.iterations} iteration(s)")
        print(f"  switch_assignment={result2.switch_assignment}")
        for h in result2.history:
            print(f"    iter {h['iteration']}: feasible={h['feasible']} "
                  f"thermal_violations={h['thermal_violations']}")
    except RuntimeError as exc:
        print(f"  No feasible topology exists once (3,4) is capped this low "
              f"(expected -- it's a required backbone edge, closing the tie "
              f"can't route around it): {exc}")