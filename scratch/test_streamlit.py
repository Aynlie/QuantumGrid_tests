import sys
import os
sys.path.insert(0, os.path.abspath("."))
from streamlit.testing.v1 import AppTest

def run_test():
    at = AppTest.from_file("dashboard.py").run()
    if at.exception:
        print(f"Error on load: {at.exception[0]}")
        return
        
    at.session_state.page = "Shadow-Mode Dashboard"
    at.run()
    
    if at.exception:
        print(f"Error on page switch: {at.exception[0]}")
        return
        
    for block in at.markdown:
        if "MAE" in block.value or "RMSE" in block.value or "MAPE" in block.value:
            print(block.value)
            
run_test()
