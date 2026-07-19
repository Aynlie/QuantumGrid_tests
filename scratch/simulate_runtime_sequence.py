import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamlit.testing.v1 import AppTest

print("--- Step 0: Initial load (Landing Page) ---")
at = AppTest.from_file("dashboard.py")
at.run(timeout=30)

print("\n--- Step 1: Trigger a fault (switching to Shadow-Mode Dashboard) ---")
at.sidebar.radio(key="page").set_value("Shadow-Mode Dashboard").run(timeout=30)

print("\n--- Step 2: Click sidebar navigation to 'Why This Recommendation' ---")
at.sidebar.radio(key="page").set_value("Why This Recommendation").run(timeout=30)

print("\n--- Step 3: Click back to 'Shadow-Mode Dashboard' ---")
at.sidebar.radio(key="page").set_value("Shadow-Mode Dashboard").run(timeout=30)

print("\n--- Step 4: Click 'Acknowledge & Clear Fault' ---")
# Find Acknowledge & Clear Fault button
found = False
for btn in at.button:
    if btn.label == "Acknowledge & Clear Fault":
        btn.click().run(timeout=30)
        found = True
        break
if not found:
    print("WARNING: 'Acknowledge & Clear Fault' button not found by label!")

print("\n--- Step 5: Click any other widget while in normal state (rerun) ---")
at.run(timeout=30)
