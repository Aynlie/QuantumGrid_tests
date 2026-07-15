"""
run_quapp_pipeline.py
================
LOCAL orchestration script -- this does NOT run on QuApp. It shows how a
QuApp job result flows into the rest of QuantumGrid, using the same
Q/var_order contract as main.py's SA/brute-force branches.

Everything here runs on your machine:
  1. Build the QUBO (qubo_builder.py)
  2. (Optionally) pre-optimize QAOA angles locally against a statevector
     simulator, OR just pick reasonable fixed defaults for a hackathon demo
  3. Submit gamma/beta/Q/var_order to QuApp as job.input
  4. Download counts
  5. Decode, score, and select the best switch configuration
     (quantum_optimizer.resolve_quapp_job)
  6. Validate with power flow / voltage checks
  7. Hand off to the dashboard
"""
import qubo_builder as qb
import quantum_optimizer as qo
import network_model as nm
import power_flow as pf


def run(dist_graph, loops, costs, net_injection, root,
        quapp_client, gamma=0.5, beta=1.0, p_layers=1, shots=1024):
    # 1. Build the QUBO -- identical to the SA / brute-force / D-Wave paths
    Q, var_order = qb.build_qubo(loops, costs)

    # 2-3. Submit to QuApp (project_id=625, function=quantumgridqaoa)
    job_input = {
        "var_order": [list(v) for v in var_order],  # tuples -> JSON-safe lists
        "Q": {f"{a}|{b}": coeff for (a, b), coeff in Q.items()},
        "gamma": gamma,
        "beta": beta,
        "p_layers": p_layers,
        "shots": shots,
    }
    job_result = quapp_client.submit(
        project_id=625, function_name="quantumgridqaoa", input=job_input,
    )

    # 4. Download counts (handler.py's post_processing() already ran)
    counts = job_result["counts"]

    # 5. Decode + score + select -- everything from here on is identical
    #    to what solve_with_dwave / solve_with_classical_sa produce
    result = qo.resolve_quapp_job(counts, Q, var_order)
    switch_config = result["switch_config"]
    print(f"QuApp selected: {switch_config}  (energy={result['energy']:.4f}, "
          f"frequency={result['frequency']})")

    # 6. Classical validation -- power flow + voltage feasibility
    closed_edges = {e for e, s in switch_config.items() if s == "closed"}
    flows = pf.compute_tree_flows(dist_graph, closed_edges, net_injection, root)
    # Reactive-power injection: pass real q_injection if you have it;
    # an all-zero dict is a named approximation, not a silent omission.
    q_flows = pf.compute_tree_flows(dist_graph, closed_edges,
                                     {b: 0.0 for b in net_injection}, root)
    feasibility = pf.check_voltage_feasibility(dist_graph, flows, q_flows, root)
    if feasibility["violations"]:
        print(f"WARNING: voltage violations at buses {feasibility['violations']}")
    else:
        print("Voltage feasibility check passed.")

    # 7. Ready for dashboard.py's metrics/render step
    return {
        "switch_config": switch_config,
        "flows": flows,
        "feasibility": feasibility,
        "solver": "quapp_qaoa",
        "energy": result["energy"],
    }