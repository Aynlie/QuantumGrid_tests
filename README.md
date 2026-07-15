# QuantumGrid

Quantum-assisted distribution network reconfiguration on a real IEEE 33-bus feeder, using real hourly demand and solar data. Given the network's tie-switch layout, it decides which switches to open/close to minimize ohmic loss — solved classically (simulated annealing, brute force) and via QAOA on Quapp's cloud simulator, then cross-checked for agreement. Also simulates a backbone fault and finds the tie-switch reconfiguration needed to restore service.

## What's actually real vs. assumed here

We want judges to be able to verify this quickly, so upfront:

- **Real data:** `network_topology.csv` (IEEE 33-bus feeder — 32 fixed branches + 5 standard tie switches), `PJME_hourly.csv` (Kaggle PJM Interconnection hourly demand), `solar_generation.csv` (Kaggle Plant_1 solar output).
- **Rescaled, not raw:** PJME is a whole utility system (tens of thousands of MW) and Plant_1 is utility-scale solar (~29 MW peak) — both far larger than this 10 MVA feeder. We use only their *shape* (normalized intraday/seasonal pattern), rescaled onto this feeder's own real nominal load and an assumed installed PV capacity. Feeding their raw MW values in directly would be physically meaningless.
- **One placeholder value:** the uploaded feeder dataset has real R/X for the 32 fixed branches, but not for the 5 tie switches (not part of that dataset). We use a standard published placeholder of 0.5 Ω for all five ties. This is noticeably higher than the backbone's real resistance (~0.01–0.09 Ω), which has a real consequence described below.
- **One assumption:** the dataset doesn't specify which bus hosts the solar installation, so we chose bus 18 (a real feeder-end lateral bus) as a representative location. Stated explicitly here rather than buried in code.

## Architecture (Modules 1–8)

| Module | File | Role |
|---|---|---|
| 1 | `data_loader.py` | Loads and normalizes the three CSVs into a per-unit network + time series |
| 2 | `forecasting.py` | Demand forecasting + per-bus disaggregation |
| 3 | `renewable.py` | Solar allocation, hosting-capacity limits, curtailment |
| 4 | `network_model.py` | Builds the distribution graph (fixed + switchable edges) |
| 5 | `power_flow.py` | Radial power flow, ohmic loss, voltage feasibility |
| 6 | `qubo_builder.py` | Builds the loop-reconfiguration QUBO; brute-force solver |
| 7 | `disaster_recovery.py` | Fault simulation + restoration switch search |
| 8 | `dashboard.py` | Metrics, topology figures, solver comparison table, Streamlit UI |
| — | `quantum_optimizer.py` | Classical simulated-annealing QUBO solver |
| — | `quapp_client.py` | Thin wrapper for submitting the QAOA circuit to Quapp Cloud |
| — | `main.py` | Runs Stages 1–8 end-to-end on real data |
| — | `build_demo_materials.py` | Generates `before_fault.png`, `after_fault.png`, `solver_comparison.csv` |

## Setup

```bash
pip install -r requirements.txt   # pandas, numpy, networkx, matplotlib, requests, python-dotenv
```

Place the three CSVs (`network_topology.csv`, `PJME_hourly.csv`, `solar_generation.csv`) in the project root.

### Quapp token (only needed for the QAOA row)

The pipeline runs fully end-to-end **without** Quapp access — QAOA is optional and off by default. To include it:

1. Log in at `functions.quapp.cloud`, open DevTools → Network tab, trigger any API call, and copy the `Authorization: Bearer <token>` value from the request headers.
2. Create a `.env` file next to `main.py`:
   ```
   QUAPP_API_TOKEN=<paste the token here, no "Bearer " prefix>
   ```
3. This token is a session JWT (~12h validity), not a permanent API key — if you get a 401 after a long gap, repeat step 1–2.

## Running it

**Full pipeline (Stages 1–8):**
```bash
python main.py
```

**Full pipeline with QAOA included:**
```bash
export RUN_QAOA_ON_QUAPP=true   # Windows: set RUN_QAOA_ON_QUAPP=true
python main.py
```

**Demo assets (before/after fault figures + solver comparison table):**
```bash
export RUN_QAOA_ON_QUAPP=true   # optional, adds the QAOA row
python build_demo_materials.py
```
Produces `before_fault.png`, `after_fault.png`, `solver_comparison.csv`.

## Results you should expect

At every hour in the dataset, the optimizer finds **all tie switches open** as the loss-minimizing configuration. This is a real, correct result given the 0.5 Ω tie-switch placeholder above — a switch that resistive costs more in extra loss than it saves unless the load imbalance across the loop is severe, and normalized real demand/solar shapes don't produce that kind of imbalance under normal conditions. Classical SA and brute force agree exactly at every hour (confirmed via `assert` in `main.py`); QAOA (p=1, on Quapp's `aer_simulator`) is checked against the same optimum.

**Where switching actually matters:** simulate a fault on the fixed backbone (`disaster_recovery.py`, edge `(5,6)` by default). This removes a fixed branch, and the optimizer correctly identifies that a specific tie switch becomes *structurally required* to keep the network connected — not a QUBO decision anymore, a forced restoration. That's the resilience story this project demonstrates: not everyday reconfiguration, but automated fault response.

## Known limitations

- **Tie-switch resistance is a placeholder**, not measured data (see above). Swapping in literature values (e.g. Baran & Wu 1989 benchmark figures) would let non-fault-scenario reconfiguration occur more readily; we chose not to tune this just to produce a more "interesting" demo result.
- **Reactive power flow is not modeled** at the fault-recovery stage (`q_flows` is zeroed out in `main.py`'s Stage 6) — voltage feasibility checks there use real-flow approximation only.
- **QAOA runs at p=1** (single QAOA layer) on a simulator, not real quantum hardware — solver agreement demonstrates correctness of the QUBO formulation and circuit construction, not quantum advantage. `dashboard.py`'s own Streamlit caption says this explicitly: *"the QUBO is block-diagonal at this scale — agreement between solvers is the EXPECTED result, not evidence of quantum advantage."*
- **No temperature data** was available for demand forecasting (Module 2); the strongest known driver of electricity demand is intentionally left out rather than backfilled with a placeholder, which degrades forecast quality somewhat — stated explicitly in `main.py`'s runtime warning rather than hidden.

## File overview

```
main.py                    entry point, Stages 1-8
build_demo_materials.py    generates before/after fault figures + solver comparison
data_loader.py             Module 1
forecasting.py             Module 2
renewable.py                Module 3
network_model.py           Module 4
power_flow.py               Module 5
qubo_builder.py             Module 6 (QUBO construction, brute-force solver)
quantum_optimizer.py        Module 6 (classical SA solver)
disaster_recovery.py        Module 7
dashboard.py                 Module 8 (metrics, figures, Streamlit UI)
quapp_client.py              Quapp API wrapper (auth, submit, poll)
network_topology.csv         IEEE 33-bus feeder data
PJME_hourly.csv               real hourly demand shape
solar_generation.csv          real solar output shape
.env                          QUAPP_API_TOKEN (not committed — add to .gitignore)
```
