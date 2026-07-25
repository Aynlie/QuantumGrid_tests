---
description: Run FastAPI backend for QuantumGrid optimisation
---
1. Open a terminal in the project root (`QuantumGrid_tests-main`).
2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   .\\venv\\Scripts\\activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
   The server will be reachable at `http://localhost:8000`.
5. Open the frontend page (`frontend/index.html`) in a browser.
   Click **Run optimisation** to invoke the backend and view results.
