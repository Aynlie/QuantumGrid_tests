import os
import time
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps
from qbraid.runtime import QbraidProvider

def main():
    load_dotenv()
    api_key = os.getenv("QBRAID_API_KEY")
    if not api_key:
        print("ERROR: QBRAID_API_KEY not found in environment or .env file.")
        return

    # Build 2-qubit Bell pair circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Convert to OpenQASM 3 string
    qasm_str = dumps(qc)
    print("--- GENERATED OPENQASM 3 ---")
    print(qasm_str)
    print("----------------------------\n")

    # Instantiate provider and get device
    provider = QbraidProvider(api_key=api_key)
    device = provider.get_device("ionq:ionq:sim:simulator")

    # Submit job with OpenQASM 3 string
    t_start = time.time()
    job = device.run(qasm_str, shots=1024)
    t_submitted = time.time()
    submission_time = t_submitted - t_start
    print(f"Submission accepted in {submission_time:.2f} seconds. Job ID: {job.id}")

    # Wait for result with timeout
    print("Polling for job completion...")
    status = job.status()
    timeout = 60
    terminal_statuses = {"COMPLETED", "FAILED", "CANCELLED"}
    while status.name not in terminal_statuses and (time.time() - t_submitted) < timeout:
        time.sleep(2)
        status = job.status()

    t_end = time.time()
    round_trip_time = t_end - t_start

    if status.name not in terminal_statuses:
        print(f"TIMEOUT: Job did not complete within {timeout} seconds. Current status: {status}")
        return

    result = job.result()
    counts = result.data.get_counts()

    print("\n--- TEST RESULTS ---")
    print(f"Submission time:  {submission_time:.2f} s")
    print(f"Round-trip time:  {round_trip_time:.2f} s")
    print(f"Final status:     {status}")
    print(f"Measurement counts: {counts}")

if __name__ == "__main__":
    main()
