import time
import sys
import os
import statistics

# Reconfigure stdout to handle UTF-8 cleanly on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_loader as dl
import network_model as nm
import qubo_builder as qb
import quantum_optimizer as qo

def build_synthetic_network(n):
    """
    Build a synthetic connected radial network with n switchable ties (n loops).
    We have:
      - 1 root node (node 1, slack bus)
      - For each k in 1..n:
        - Node 2k and Node 2k+1 (PQ buses)
        - Branch 1 -> 2k (fixed, closed)
        - Branch 2k -> 2k+1 (fixed, closed)
        - Branch 2k+1 -> 1 (switchable, starts open)
    """
    base = dl.BaseValues(S_base_mva=10.0, V_base_kv=12.66)
    net = dl.NetworkGraph(base=base)
    
    # Root bus
    net.buses[1] = dl.Bus(id=1, bus_type="slack", P_load_pu=0.0, Q_load_pu=0.0)
    
    # Loop components
    for k in range(1, n + 1):
        n1 = 2 * k
        n2 = 2 * k + 1
        
        net.buses[n1] = dl.Bus(id=n1, bus_type="PQ", P_load_pu=0.1, Q_load_pu=0.05)
        net.buses[n2] = dl.Bus(id=n2, bus_type="PQ", P_load_pu=0.1, Q_load_pu=0.05)
        
        net.branches.append(dl.Branch(i=1, j=n1, R_pu=0.05, X_pu=0.05, S_max_pu=1.0, is_switchable=False, s_initial=1))
        net.branches.append(dl.Branch(i=n1, j=n2, R_pu=0.05, X_pu=0.05, S_max_pu=1.0, is_switchable=False, s_initial=1))
        net.branches.append(dl.Branch(i=n2, j=1, R_pu=0.05, X_pu=0.05, S_max_pu=1.0, is_switchable=True, s_initial=0))
        
    dg = nm.build_distribution_graph(net)
    return dg

def run_benchmarks():
    ns = [5, 10, 15, 20]
    results = []
    
    for n in ns:
        print(f"\n==========================================")
        print(f"BENCHMARKING SYSTEM SIZE n = {n}")
        print(f"==========================================")
        
        dg = build_synthetic_network(n)
        loops = qb.find_switchable_loops(dg)
        
        # Prepare net injection input
        net_injection = {1: 0.0}
        for k in range(1, n + 1):
            net_injection[2*k] = 0.1
            net_injection[2*k+1] = 0.1
            
        costs = qb.compute_loop_open_costs(dg, loops, net_injection, root=1)
        Q, var_order = qb.build_qubo(loops, costs)
        
        evaluations = 2 ** n
        
        # 1. Benchmark Brute Force
        bf_times = []
        skip_bf = False
        
        print("\n--- Running Brute Force Solve ---")
        for trial in range(3):
            # If previous trial took too long, skip next trials to avoid hanging
            if sum(bf_times) > 120000.0:  # 120 seconds in ms
                print(f"  Trial {trial+1}: Skipped (cumulative timeout exceeded)")
                skip_bf = True
                break
            
            t_start = time.perf_counter()
            qb.brute_force_solve(Q, var_order)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            bf_times.append(elapsed_ms)
            print(f"  Trial {trial+1}: {elapsed_ms:.4f} ms")
            
        if skip_bf or (bf_times and statistics.median(bf_times) > 120000.0):
            bf_ms_str = "Still Running / Stopped"
        else:
            bf_ms_str = f"{statistics.median(bf_times):.4f}"
            
        # 2. Benchmark Classical SA
        sa_times = []
        print("\n--- Running Classical Simulated Annealing ---")
        for trial in range(3):
            t_start = time.perf_counter()
            qo.solve_with_classical_sa(Q, var_order)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            sa_times.append(elapsed_ms)
            print(f"  Trial {trial+1}: {elapsed_ms:.4f} ms")
            
        sa_ms_str = f"{statistics.median(sa_times):.4f}"
        
        results.append((n, bf_ms_str, sa_ms_str, evaluations))
        
    print(f"\n\n==========================================")
    print(f"FINAL BENCHMARK SUMMARY TABLE")
    print(f"==========================================")
    print(f"{'n':<5} | {'brute_force_ms':<18} | {'classical_sa_ms':<18} | {'brute_force_evaluations':<24}")
    print("-" * 75)
    for n, bf_ms_str, sa_ms_str, evaluations in results:
        print(f"{n:<5} | {bf_ms_str:<18} | {sa_ms_str:<18} | {evaluations:<24}")

if __name__ == "__main__":
    run_benchmarks()
