import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    from app.main import app
    print("App imported successfully")
    
    # Try to access a fairness metric
    from app.ml.fairness.demographic_parity import DemographicParity
    dp = DemographicParity()
    print(f"DemographicParity instantiated: {dp.description}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
