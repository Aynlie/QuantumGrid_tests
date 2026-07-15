"""
quantum_optimizer.py
================
Module 6 of QuantumGrid.
Responsibility: take the QUBO (Q, var_order) built in Module 5 and solve it
using one of three interchangeable backends:
  1. Qiskit QAOA (gate-based) -- provided because the original document
     requested it, WITH the honest limitation discussed in the write-up.
  2. D-Wave Ocean SDK (quantum annealing / hybrid) -- the RECOMMENDED path,
     matching the only directly-relevant published precedent for this exact
     problem (Silva et al., 2023).
  3. A from-scratch classical simulated annealing solver -- used as an
     honestly-labeled stand-in whenever qiskit/dimod aren't installed (as
     in this sandbox), and as a sanity baseline in any environment.
All three consume the SAME (Q, var_order) produced by qubo_builder.py, so
switching backends never requires re-deriving the problem.
"""
import math
import random
# ---------------------------------------------------------------------------
# 1. QUBO -> Ising conversion (shared by the QAOA and D-Wave paths)
# ---------------------------------------------------------------------------
def qubo_to_ising(Q: dict, var_order: list):
    """
    x_e = (1 + z_e) / 2,  z_e in {-1, +1}
    Returns (h, J, offset) where H_ising = offset + sum h_e*z_e + sum J_ef*z_e*z_f
    """
    h = {v: 0.0 for v in var_order}
    J = {}
    offset = 0.0
    for (a, b), coeff in Q.items():
        if a == b:
            # coeff * x_a = coeff * (1+z_a)/2 = coeff/2 + (coeff/2) z_a
            offset += coeff / 2
            h[a] += coeff / 2
        else:
            # coeff * x_a * x_b = coeff * (1+z_a)(1+z_b)/4
            #                   = coeff/4 * (1 + z_a + z_b + z_a z_b)
            offset += coeff / 4
            h[a] += coeff / 4
            h[b] += coeff / 4
            J[(a, b)] = J.get((a, b), 0.0) + coeff / 4
    return h, J, offset
# ---------------------------------------------------------------------------
# 2a. Qiskit QAOA path (requires `pip install qiskit qiskit-optimization qiskit-aer`)
# ---------------------------------------------------------------------------
def solve_with_qaoa(Q: dict, var_order: list, reps: int = 1, seed: int = 42):
    """
    Solve the QUBO using Qiskit's QAOA via MinimumEigenOptimizer.
    NOT executed in this sandbox (qiskit is not installed here / no network
    to install it) -- this is complete, real code for the user's own
    environment, not pseudocode, but I could not run it myself to verify
    output in this session. Cross-checked API usage against the current
    qiskit-optimization documentation.
    """
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA
        from qiskit.primitives import StatevectorSampler
    except ImportError as e:
        raise ImportError(
            "Qiskit is not installed. Run: pip install qiskit qiskit-optimization "
            "qiskit-aer --break-system-packages"
        ) from e
    qp = QuadraticProgram()
    for v in var_order:
        qp.binary_var(name=str(v))
    linear = {str(v): Q.get((v, v), 0.0) for v in var_order}
    quadratic = {
        (str(a), str(b)): coeff
        for (a, b), coeff in Q.items() if a != b
    }
    qp.minimize(linear=linear, quadratic=quadratic)
    qaoa_mes = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(),
        reps=reps,
    )
    optimizer = MinimumEigenOptimizer(qaoa_mes)
    result = optimizer.solve(qp)
    assignment = {var_order[i]: int(round(result.x[i])) for i in range(len(var_order))}
    return assignment, result.fval
# ---------------------------------------------------------------------------
# 2a-quapp. QuApp Cloud path -- circuit runs remotely (handler.py), everything
#     below runs LOCALLY on the returned counts. See project deployment notes:
#     QuApp's processing() must return an unexecuted QuantumCircuit, so the
#     QAOA angles are optimized locally first (or accepted as fixed reps=1
#     defaults) and only the final circuit is submitted for execution.
# ---------------------------------------------------------------------------
def bitstring_to_assignment(bitstring: str, var_order: list) -> dict:
    """
    Qiskit counts keys are little-endian (rightmost character = qubit 0).
    Since var_order[i] is mapped to qubit i when the circuit is built,
    the bitstring must be reversed before zipping against var_order.
    """
    bits = bitstring[::-1]
    return {var_order[i]: int(bits[i]) for i in range(len(var_order))}


def decode_quapp_counts(counts: dict, Q: dict, var_order: list) -> list:
    """
    Turn raw QuApp counts ({bitstring: frequency_or_shots}) into decoded,
    energy-scored candidates. Reuses qubo_builder.evaluate_qubo so scoring
    is identical to solve_with_dwave / solve_with_classical_sa / brute_force_solve
    -- one scoring function, never re-derived per backend.
    """
    import qubo_builder as qb
    scored = []
    for bitstring, freq in counts.items():
        assignment = bitstring_to_assignment(bitstring, var_order)
        energy = qb.evaluate_qubo(Q, assignment)
        scored.append({
            "bitstring": bitstring,
            "assignment": assignment,
            "energy": energy,
            "frequency": freq,
        })
    return scored


def select_best_from_quapp(scored: list) -> dict:
    """
    Lowest QUBO energy wins; frequency only breaks ties.
    QAOA sampling noise means the most-frequent bitstring is NOT
    guaranteed to be the true ground state -- always score by energy.
    """
    return sorted(scored, key=lambda r: (r["energy"], -r["frequency"]))[0]


def assignment_to_switch_config(assignment: dict) -> dict:
    """1 = closed, 0 = open -- matches qubo_builder.build_qubo()'s convention."""
    return {edge: ("closed" if state == 1 else "open")
            for edge, state in assignment.items()}


def resolve_quapp_job(counts: dict, Q: dict, var_order: list) -> dict:
    """
    Single entry point: raw QuApp counts in, validated switch config out.
    Call this immediately after downloading the job result. Nothing
    quantum-specific should leak past this function -- everything after
    it (power_flow.py, disaster_recovery.py, dashboard.py) consumes the
    same switch_config dict that solve_with_dwave/solve_with_classical_sa
    already produce, so no downstream module needs to know QuApp was used.
    """
    scored = decode_quapp_counts(counts, Q, var_order)
    best = select_best_from_quapp(scored)
    switch_config = assignment_to_switch_config(best["assignment"])
    return {
        "switch_config": switch_config,
        "energy": best["energy"],
        "frequency": best["frequency"],
        "all_candidates": scored,  # kept for dashboard.py's solver-comparison view
    }
# ---------------------------------------------------------------------------
# 2b. D-Wave Ocean SDK path (requires `pip install dimod dwave-neal` at minimum,
#     or a real D-Wave account + `dwave-system` for hardware access)
# ---------------------------------------------------------------------------
def solve_with_dwave(Q: dict, var_order: list, num_reads: int = 200, use_hardware: bool = False):
    """
    Solve the QUBO using D-Wave's Ocean SDK. Defaults to the free,
    open-source `neal` simulated-annealing sampler (no account needed);
    set use_hardware=True (and configure a D-Wave API token) to run on
    real annealing hardware via dwave-system's EmbeddingComposite.
    NOT executed in this sandbox (dimod/neal not installed, no network) --
    complete, real code for the user's own environment.
    """
    try:
        import dimod
    except ImportError as e:
        raise ImportError(
            "dimod is not installed. Run: pip install dimod dwave-neal "
            "--break-system-packages (add dwave-system for real hardware access)"
        ) from e
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    if use_hardware:
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        from neal import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first
    assignment = {v: int(best.sample[v]) for v in var_order}
    return assignment, best.energy
# ---------------------------------------------------------------------------
# 2c. Classical simulated annealing fallback -- ACTUALLY RUNS in this sandbox
# ---------------------------------------------------------------------------
def solve_with_classical_sa(Q: dict, var_order: list,
                             num_sweeps: int = 2000,
                             initial_temp: float = 5.0,
                             final_temp: float = 0.01,
                             seed: int = 0):
    """
    A from-scratch simulated annealing solver for the QUBO -- the classical
    analogue of quantum annealing, used here as an honestly-labeled stand-in
    when qiskit/dimod aren't available, and as a useful sanity baseline in
    any environment.
    """
    rng = random.Random(seed)
    n = len(var_order)
    idx = {v: i for i, v in enumerate(var_order)}
    # Precompute, for each variable, the list of (other_var, coeff) it interacts with.
    neighbors = {v: [] for v in var_order}
    diag = {v: 0.0 for v in var_order}
    for (a, b), coeff in Q.items():
        if a == b:
            diag[a] += coeff
        else:
            neighbors[a].append((b, coeff))
            neighbors[b].append((a, coeff))
    def energy(state):
        e = 0.0
        for (a, b), coeff in Q.items():
            if a == b:
                e += coeff * state[a]
            else:
                e += coeff * state[a] * state[b]
        return e
    def delta_if_flip(state, v):
        # Change in energy from flipping variable v, computed locally
        # (O(degree) instead of O(n) -- avoids a full energy recomputation
        # per proposed flip).
        old_val = state[v]
        new_val = 1 - old_val
        d = diag[v] * (new_val - old_val)
        for (u, coeff) in neighbors[v]:
            d += coeff * state[u] * (new_val - old_val)
        return d
    state = {v: rng.randint(0, 1) for v in var_order}
    current_energy = energy(state)
    best_state, best_energy = dict(state), current_energy
    for sweep in range(num_sweeps):
        # Geometric cooling schedule from initial_temp to final_temp.
        frac = sweep / max(1, num_sweeps - 1)
        temp = initial_temp * (final_temp / initial_temp) ** frac
        v = var_order[rng.randrange(n)]
        d = delta_if_flip(state, v)
        if d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
            state[v] = 1 - state[v]
            current_energy += d
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = dict(state)
    return best_state, best_energy
if __name__ == "__main__":
    import data_loader as dl
    import network_model as nm
    import qubo_builder as qb
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
    Q, var_order = qb.build_qubo(loops, costs)
    print("Attempting Qiskit QAOA...")
    try:
        assignment, fval = solve_with_qaoa(Q, var_order)
        print(f"  QAOA result: {assignment}, fval={fval:.4f}")
    except ImportError as e:
        print(f"  Skipped (not installed here): {e}")
    print("\nAttempting D-Wave (neal simulated annealing sampler)...")
    try:
        assignment, energy_val = solve_with_dwave(Q, var_order)
        print(f"  D-Wave result: {assignment}, energy={energy_val:.4f}")
    except ImportError as e:
        print(f"  Skipped (not installed here): {e}")
    print("\nRunning classical simulated annealing (always available)...")
    sa_assignment, sa_energy = solve_with_classical_sa(Q, var_order)
    print(f"  SA result: {sa_assignment}, energy={sa_energy:.4f}")
    # Cross-check against Module 5's brute-force ground truth.
    bf_assignment, bf_energy = qb.brute_force_solve(Q, var_order)
    print(f"\nBrute-force ground truth: {bf_assignment}, energy={bf_energy:.4f}")
    assert abs(sa_energy - bf_energy) < 1e-6, "SA did not find the true ground state!"
    print("Verification passed: classical SA matches brute-force ground state.")