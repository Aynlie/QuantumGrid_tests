import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest

def test_pilot_submission():
    # Make sure pilot_requests.json is cleared/deleted first so we start clean
    json_path = Path(__file__).resolve().parent.parent / "pilot_requests.json"
    if json_path.exists():
        json_path.unlink()
        print("Cleared existing pilot_requests.json")

    print("Initializing AppTest with dashboard.py...")
    at = AppTest.from_file("dashboard.py")
    
    # Run first load
    at.run(timeout=30)
    print(f"Current page: {at.session_state.page}")
    
    # We should be on the Landing Page. Let's inspect the inputs.
    print(f"Text inputs available: {[t.label for t in at.text_input]}")
    print(f"Number inputs available: {[n.label for n in at.number_input]}")
    
    # Let's set the text inputs
    # Looking at dashboard.py, the order of text_input is:
    # 0. Facilities Manager Name
    # 1. Contact Email Address
    # 2. Industrial Park / Facility Name
    manager_input = at.text_input[0]
    email_input = at.text_input[1]
    facility_input = at.text_input[2]
    peak_load_input = at.number_input[0]
    
    print("Setting Name to 'Jane Doe'")
    manager_input.set_value("Jane Doe")
    print("Setting Email to 'jane.doe@example.com'")
    email_input.set_value("jane.doe@example.com")
    print("Setting Facility to 'Greenfield Power Tech'")
    facility_input.set_value("Greenfield Power Tech")
    print("Setting Peak Load to 15.5")
    peak_load_input.set_value(15.5)
    
    # Find the button
    submit_btn = None
    for btn in at.button:
        if btn.label == "Request Pilot Integration":
            submit_btn = btn
            break
            
    if submit_btn is not None:
        print("Clicking 'Request Pilot Integration'...")
        submit_btn.click().run(timeout=30)
        print("Form submitted successfully via AppTest.")
    else:
        print("ERROR: Submit button not found!")
        
    if json_path.exists():
        print("pilot_requests.json exists. Contents:")
        print(json_path.read_text(encoding="utf-8"))
    else:
        print("ERROR: pilot_requests.json was not created!")

if __name__ == "__main__":
    test_pilot_submission()
