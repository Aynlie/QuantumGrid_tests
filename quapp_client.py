"""
quapp_client.py
================
Thin wrapper around Quapp Cloud's job-submission API.

CONFIRMED against real captured traffic (DevTools Network tab, 2026-07-12)
and CONFIRMED WORKING end-to-end (real QUBO submitted, real counts
returned, decoded result matched classical SA/brute-force exactly).

Submit a job:
    POST https://functions.quapp.cloud/api/v1/function/invoke
    Body: {"deviceId": 6, "functionName": "quantumgridqaoa",
           "description": "...", "input": {...}, "shots": 1024}
    Response: {"data": "<job_id-as-plain-string>"}

Fetch a job's result (poll this until status == "DONE"):
    GET https://functions.quapp.cloud/api/v1/jobs/{job_id}/detail
    Response: {"data": {"status": "NEW"|"RUNNING"|"DONE"|..., "jobResult":
               {"counts": {...}}, "shots": ..., ...}}

Auth token lifetime: the Authorization header is a Bearer JWT
(Cognito-style), your logged-in SESSION token, not a permanent API key.
A real captured token showed roughly a 12-hour validity window -- but it
still WILL expire eventually. If you get a 401, grab a fresh one from
DevTools (Network tab -> "invoke" request -> Headers -> Authorization ->
right-click -> Copy value) and update .env.
"""
import base64
import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

QUAPP_API_TOKEN = os.environ.get("QUAPP_API_TOKEN")
PROJECT_ID = 625
FUNCTION_ID = 84
FUNCTION_NAME = "quantumgridqaoa"
DEVICE_ID = 6  # aer_simulator
TENANT_ID = "ws_seaquantathon2026"

SUBMIT_URL = "https://functions.quapp.cloud/api/v1/function/invoke"
DETAIL_URL_TEMPLATE = "https://functions.quapp.cloud/api/v1/jobs/{job_id}/detail"

_TERMINAL_SUCCESS = {"DONE"}
_TERMINAL_FAILURE = {"FAILED", "ERROR", "CANCELLED"}


def _decode_jwt_expiry(token):
    if not token:
        return None
    raw = token.strip().strip('"').strip("'").strip()
    if raw.lower().startswith("bearer "):
        raw = raw[len("bearer "):].strip()
    parts = raw.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return None
    return decoded.get("exp")


def _normalize_token(token):
    raw = token.strip().strip('"').strip("'").strip()
    if raw.lower().startswith("bearer "):
        raw = raw[len("bearer "):].strip()
    return raw


def _headers():
    if not QUAPP_API_TOKEN:
        raise RuntimeError(
            "QUAPP_API_TOKEN not found. Create a .env file next to this "
            "script containing:\n  QUAPP_API_TOKEN=Bearer eyJ...\n"
            "(copy the exact Authorization header value from your "
            "browser's Network tab, right-click -> Copy value -- don't "
            "retype it by hand, that's an easy way to corrupt the JWT.)"
        )
    raw = _normalize_token(QUAPP_API_TOKEN)
    return {
        "Authorization": f"Bearer {raw}",
        "Content-Type": "application/json",
        "X-Project-Id": str(PROJECT_ID),
        "X-Tenant-Id": TENANT_ID,
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"),
        "Origin": "https://functions.quapp.cloud",
        "Referer": f"https://functions.quapp.cloud/functions/{FUNCTION_ID}",
    }


def submit_job(job_input: dict, shots: int = 1024, description: str = "quantumgrid",
               timeout_s: int = 180, poll_interval_s: float = 2.0) -> dict:
    """
    Submit a job to Quapp and poll /detail until it completes.
    Returns the full job_result dict from the "jobResult" field
    (e.g. {"counts": {...}}).
    Raises RuntimeError on missing token, failed submission, job failure,
    or timeout.
    """
    # One clean, single debug line per submission -- not per poll -- so
    # repeated polling doesn't spam the console.
    exp = _decode_jwt_expiry(QUAPP_API_TOKEN)
    exp_str = f", expires at unix={exp}" if exp else ""
    print(f"[quapp_client] Submitting job to Quapp (token present, "
          f"length={len(_normalize_token(QUAPP_API_TOKEN))}{exp_str})")

    payload = {
        "deviceId": DEVICE_ID,
        "functionName": FUNCTION_NAME,
        "description": description,
        "input": job_input,
        "shots": shots,
    }
    resp = requests.post(SUBMIT_URL, json=payload, headers=_headers())
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Quapp submit failed ({resp.status_code}). "
            f"Response body: {resp.text!r}"
        )
    job_id = resp.json().get("data")
    if not job_id:
        raise RuntimeError(f"Quapp submit response had no job id: {resp.text}")
    print(f"[quapp_client] Job submitted: {job_id} -- polling for completion...")

    start = time.time()
    while time.time() - start < timeout_s:
        detail_resp = requests.get(DETAIL_URL_TEMPLATE.format(job_id=job_id), headers=_headers())
        detail_resp.raise_for_status()
        body = detail_resp.json().get("data", {})
        status = body.get("status")
        if status in _TERMINAL_SUCCESS:
            job_result = body.get("jobResult")
            if not job_result or "counts" not in job_result:
                raise RuntimeError(f"Quapp job {job_id} completed but returned no counts: {body}")
            print(f"[quapp_client] Job {job_id} completed successfully.")
            return job_result
        if status in _TERMINAL_FAILURE:
            raise RuntimeError(f"Quapp job {job_id} failed with status {status}: {body}")
        time.sleep(poll_interval_s)
    raise RuntimeError(f"Quapp job {job_id} did not complete within {timeout_s}s.")