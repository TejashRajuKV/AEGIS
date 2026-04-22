#!/usr/bin/env python3
"""Run full fairness audit."""

import asyncio
import json

from app.pipeline.pipeline_coordinator import PipelineCoordinator
from app.pipeline.results_aggregator import ResultsAggregator
from app.utils.logger import get_logger


async def main():
    logger = get_logger("run_full_audit")
    logger.info("Starting full AEGIS audit")

    coordinator = PipelineCoordinator()
    aggregator = ResultsAggregator()

    results = await coordinator.run_sequential(
        ["fairness_audit", "drift_detection"],
    )

    report = aggregator.aggregate(**results)
    logger.info(f"Audit complete: {len(results)} pipelines run")

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
