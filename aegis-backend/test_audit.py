import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.api.routes.fairness import run_fairness_audit
from app.models.schemas import FairnessAuditRequest

async def main():
    req = FairnessAuditRequest(
        dataset_name="compas",
        model_type="logistic_regression",
        target_column="two_year_recid",
        sensitive_features=["race", "sex"],
        retrain=True
    )
    try:
        res = await run_fairness_audit(req)
        print("SUCCESS!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
