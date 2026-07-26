import os
import time
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps
from qbraid.runtime import QbraidProvider

TIMEOUT = 60
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


def build_bell_qasm():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return dumps(qc)


def test_device(provider, device_id, qasm_str):
    """Submit the Bell test to one device and return a result dict."""
    row = {"device_id": device_id, "status": None, "job_id": None,
           "submission_time": None, "round_trip_time": None,
           "error": None, "counts": None}
    try:
        device = provider.get_device(device_id)

        t_start = time.time()
        job = device.run(qasm_str, shots=1024)
        t_submitted = time.time()
        row["submission_time"] = t_submitted - t_start
        row["job_id"] = job.id

        status = job.status()
        while status.name not in TERMINAL_STATUSES and (time.time() - t_submitted) < TIMEOUT:
            time.sleep(2)
            status = job.status()

        row["status"] = status.name
        row["round_trip_time"] = time.time() - t_start

        if status.name == "COMPLETED":
            result = job.result()
            row["counts"] = result.data.get_counts()
        elif status.name not in TERMINAL_STATUSES:
            row["status"] = "TIMEOUT"

    except Exception as e:
        row["status"] = "EXCEPTION"
        row["error"] = f"{type(e).__name__}: {e}"

    return row


def main():
    load_dotenv()
    api_key = os.getenv("QBRAID_API_KEY")
    if not api_key:
        print("ERROR: QBRAID_API_KEY not found in environment or .env file.")
        return

    provider = QbraidProvider(api_key=api_key)
    qasm_str = build_bell_qasm()

    # Pull every device qBraid reports as available to this account.
    # If get_devices() isn't available in 0.12.2, fall back to a manual list.
    try:
        devices = provider.get_devices()
        device_ids = [d.id for d in devices]
    except AttributeError:
        # Manual fallback -- add/remove device IDs as needed
        device_ids = [
            "qbraid:qbraid:sim:qir-sv",
        ]

    print(f"Testing {len(device_ids)} device(s):\n  " + "\n  ".join(device_ids))
    print()

    results = []
    for device_id in device_ids:
        print(f"--- {device_id} ---")
        row = test_device(provider, device_id, qasm_str)
        results.append(row)
        if row["error"]:
            print(f"  EXCEPTION: {row['error']}")
        else:
            print(f"  status={row['status']}  job_id={row['job_id']}  "
                  f"submit={row['submission_time']:.2f}s  "
                  f"round_trip={row['round_trip_time']:.2f}s")
        print()

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'device_id':<40} {'status':<12} {'job_id'}")
    for row in results:
        print(f"{row['device_id']:<40} {str(row['status']):<12} {row['job_id']}")

    failed = [r for r in results if r["status"] not in ("COMPLETED",)]
    passed = [r for r in results if r["status"] == "COMPLETED"]
    print(f"\n{len(passed)} passed, {len(failed)} failed out of {len(results)}")


if __name__ == "__main__":
    main()