import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

def verify_explainability():
    print("Initializing AppTest with dashboard.py...")
    at = AppTest.from_file("dashboard.py")
    at.run(timeout=30)
    
    print("\n=== Navigate to Shadow-Mode Dashboard to initialize simulation ===")
    radio = at.sidebar.radio(key="page")
    radio.set_value("Shadow-Mode Dashboard").run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    print(f"Active Fault: {at.session_state.active_fault}")
    
    print("\n=== Force Page to 'Why This Recommendation' in Session State ===")
    at.session_state.page = "Why This Recommendation"
    at.run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    
    # Check markdown elements on the page
    found_scaling_horizon = False
    print("\nPage Markdown Elements:")
    for md in at.markdown:
        val = md.value.strip()
        if "Solver Scaling Horizon" in val:
            found_scaling_horizon = True
            print("\nFOUND SCALING HORIZON CARD!")
            # Print the card text content
            try:
                print(val)
            except Exception:
                print(val.encode('ascii', errors='replace').decode('ascii'))
                
    if not found_scaling_horizon:
        print("ERROR: Scaling Horizon card not found in markdown!")
        sys.exit(1)
    else:
        print("\nSUCCESS: Scaling Horizon card verified successfully!")

if __name__ == "__main__":
    verify_explainability()
