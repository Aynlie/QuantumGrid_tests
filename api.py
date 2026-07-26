# api.py
"""FastAPI server exposing the QuantumGrid optimisation pipeline.

The server provides two endpoints:
* ``POST /run-optimization`` – triggers the full Benders‑QUBO‑QAOA‑MILP flow
  and returns a JSON payload with the results.
* ``GET /health`` – simple health‑check.

Run locally with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

The implementation relies on the ``pipeline.run_full_optimization`` helper
added in ``pipeline.py``.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import the helper we created earlier.
from pipeline import run_full_optimization

app = FastAPI(title="QuantumGrid Optimisation API", version="0.1.0")


class RunRequest(BaseModel):
    network_json_path: Optional[str] = "pilot_requests.json"
    solver: Optional[str] = "sa"
    cut_penalty: Optional[float] = None
    max_iters: Optional[int] = 10
    seed: Optional[int] = 0
    run_milp: Optional[bool] = True


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/run-optimization")
async def run_optimization(req: RunRequest):
    try:
        result = run_full_optimization(
            network_json_path=req.network_json_path,
            solver=req.solver,
            cut_penalty=req.cut_penalty,
            max_iters=req.max_iters,
            seed=req.seed,
            run_milp=req.run_milp,
        )
        return result
    except Exception as exc:
        # Propagate a clear error to the client – the frontend can display it.
        raise HTTPException(status_code=500, detail=str(exc))
