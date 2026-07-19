import sys
import os

# Ensure we can import from parent directory if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

def test_cache_behavior():
    print("Initializing AppTest with dashboard.py...")
    at = AppTest.from_file("dashboard.py")
    
    # 1. First run (defaults to Landing Page)
    print("\n--- Run 1: Initial load (Landing Page) ---")
    at.run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    print(f"Cache Hits: {at.session_state.cache_hits}")
    print(f"Cache Misses: {at.session_state.cache_misses}")
    
    # 2. Switch to Shadow-Mode Dashboard
    print("\n--- Run 2: Switching to Shadow-Mode Dashboard ---")
    # Set the radio button to 'Shadow-Mode Dashboard'
    radio = at.sidebar.radio(key="page")
    radio.set_value("Shadow-Mode Dashboard").run(timeout=30)
    
    print(f"Current page: {at.session_state.page}")
    print(f"Active Fault: {at.session_state.active_fault}")
    print(f"Cache Hits: {at.session_state.cache_hits}")
    print(f"Cache Misses: {at.session_state.cache_misses}")
    
    # On first load, the auto-trigger simulates a fault. Let's clear the fault.
    # Find the "Acknowledge & Clear Fault" button
    clear_btn = None
    for btn in at.button:
        if btn.label == "Acknowledge & Clear Fault":
            clear_btn = btn
            break
            
    if clear_btn is not None:
        print("\n--- Run 3: Clicking 'Acknowledge & Clear Fault' (Enter Normal State) ---")
        clear_btn.click().run(timeout=30)
    else:
        print("\nWARNING: 'Acknowledge & Clear Fault' button not found!")
        
    print(f"Active Fault: {at.session_state.active_fault}")
    print(f"Cache Hits: {at.session_state.cache_hits}")
    print(f"Cache Misses: {at.session_state.cache_misses}")
    print(f"Last Solver Run Time: {at.session_state.last_solver_time}")
    
    # 3. Trigger a rerun of the page by doing an unrelated interaction (e.g. toggling the radio menu or just calling run() again)
    print("\n--- Run 4: Triggering page rerun (Simulating widget interaction) ---")
    # Let's run again, which simulates a rerun while staying on the same page
    at.run(timeout=30)
    
    print(f"Active Fault: {at.session_state.active_fault}")
    print(f"Cache Hits: {at.session_state.cache_hits}")
    print(f"Cache Misses: {at.session_state.cache_misses}")
    print(f"Last Solver Run Time: {at.session_state.last_solver_time}")
    
    # 4. Trigger another rerun
    print("\n--- Run 5: Triggering another page rerun ---")
    at.run(timeout=30)
    
    print(f"Active Fault: {at.session_state.active_fault}")
    print(f"Cache Hits: {at.session_state.cache_hits}")
    print(f"Cache Misses: {at.session_state.cache_misses}")
    print(f"Last Solver Run Time: {at.session_state.last_solver_time}")

if __name__ == "__main__":
    test_cache_behavior()
