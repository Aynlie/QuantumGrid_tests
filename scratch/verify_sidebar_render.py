import sys
import os

# Set UTF-8 encoding for stdout to prevent Unicode errors on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

def verify_sidebar():
    print("Initializing AppTest with dashboard.py...")
    at = AppTest.from_file("dashboard.py")
    
    print("\n=== Run 1: Initial load (Landing Page) ===")
    at.run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    print("Sidebar Markdown:")
    for md in at.sidebar.markdown:
        try:
            print(f"  - {md.value.strip()}")
        except Exception:
            print(f"  - {md.value.strip().encode('ascii', errors='replace').decode('ascii')}")
        
    print("\n=== Run 2: Switching to Shadow-Mode Dashboard (Fault State) ===")
    radio = at.sidebar.radio(key="page")
    radio.set_value("Shadow-Mode Dashboard").run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    print(f"Active Fault: {at.session_state.active_fault}")
    print("Sidebar Markdown:")
    for md in at.sidebar.markdown:
        try:
            print(f"  - {md.value.strip()}")
        except Exception:
            print(f"  - {md.value.strip().encode('ascii', errors='replace').decode('ascii')}")
        
    # Find the "Acknowledge & Clear Fault" button
    clear_btn = None
    for btn in at.button:
        if btn.label == "Acknowledge & Clear Fault":
            clear_btn = btn
            break
            
    if clear_btn is not None:
        print("\n=== Run 3: Clicking 'Acknowledge & Clear Fault' (Enter Normal State) ===")
        clear_btn.click().run(timeout=30)
        print(f"Active Fault: {at.session_state.active_fault}")
        print("Sidebar Markdown:")
        for md in at.sidebar.markdown:
            try:
                print(f"  - {md.value.strip()}")
            except Exception:
                print(f"  - {md.value.strip().encode('ascii', errors='replace').decode('ascii')}")
    else:
        print("\nWARNING: 'Acknowledge & Clear Fault' button not found!")

if __name__ == "__main__":
    verify_sidebar()
