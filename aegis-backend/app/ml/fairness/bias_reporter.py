"""AEGIS Bias Reporter - Generate comprehensive bias reports."""
import numpy as np
from typing import Dict, Any, List, Optional
from .demographic_parity import DemographicParity
from .equalized_odds import EqualizedOdds
from .calibration import CalibrationMetric
import logging

logger = logging.getLogger(__name__)


class BiasReporter:
    """Generate comprehensive bias reports from fairness metric results.

    Compiles all metrics, identifies the most biased attributes,
    and generates actionable recommendations.
    """

    def __init__(self):
        self.dp = DemographicParity()
        self.eo = EqualizedOdds()
        self.calibration = CalibrationMetric()

    def generate_report(
        self,
        metric_results: Dict[str, Dict[str, Any]],
        model_name: str = "unknown",
        dataset_name: str = "unknown",
    ) -> Dict[str, Any]:
        """Generate a comprehensive bias report.

        Args:
            metric_results: Dict mapping attribute name to list of metric result dicts.
            model_name: Name of the model being audited.
            dataset_name: Name of the dataset used.

        Returns:
            Structured report dict with findings and recommendations.
        """
        report = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "overall_fair": True,
            "findings": [],
            "recommendations": [],
            "summary": {
                "total_attributes_audited": 0,
                "biased_attributes": 0,
                "fair_attributes": 0,
                "worst_metric": None,
                "worst_gap": 0.0,
            },
        }

        worst_gap = 0.0
        worst_metric_name = ""

        for attr_name, results_list in metric_results.items():
            if isinstance(results_list, dict):
                results_list = [results_list]
            if isinstance(results_list, list) and len(results_list) > 0 and isinstance(results_list[0], dict):
                pass
            else:
                continue

            report["summary"]["total_attributes_audited"] += 1
            attr_biased = False

            for result in results_list:
                if not isinstance(result, dict):
                    continue
                gap = result.get("gap", 0.0)
                metric_name = result.get("metric_name", "unknown")
                is_fair = result.get("is_fair", True)

                if not is_fair and gap > worst_gap:
                    worst_gap = gap
                    worst_metric_name = f"{attr_name}/{metric_name}"

                if not is_fair:
                    attr_biased = True
                    finding = {
                        "attribute": attr_name,
                        "metric": metric_name,
                        "gap": gap,
                        "threshold": result.get("threshold", 0.1),
                        "group_values": result.get("group_values", {}),
                    }
                    report["findings"].append(finding)

            if attr_biased:
                report["summary"]["biased_attributes"] += 1
                report["overall_fair"] = False
            else:
                report["summary"]["fair_attributes"] += 1

        report["summary"]["worst_metric"] = worst_metric_name
        report["summary"]["worst_gap"] = round(worst_gap, 6)

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        logger.info(
            f"Bias report: {report['summary']['biased_attributes']}/{report['summary']['total_attributes_audited']} "
            f"attributes biased, overall_fair={report['overall_fair']}"
        )

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        if not report["findings"]:
            recommendations.append("No significant bias detected. Continue monitoring.")
            return recommendations

        # Group findings by attribute
        attrs_with_bias = set(f["attribute"] for f in report["findings"])

        for attr in attrs_with_bias:
            attr_findings = [f for f in report["findings"] if f["attribute"] == attr]
            gaps = [f["gap"] for f in attr_findings]
            max_gap = max(gaps)

            if max_gap > 0.3:
                recommendations.append(
                    f"CRITICAL: '{attr}' shows severe bias (gap={max_gap:.1%}). "
                    f"Consider removing or carefully re-encoding this attribute."
                )
            elif max_gap > 0.1:
                recommendations.append(
                    f"MODERATE: '{attr}' shows notable bias (gap={max_gap:.1%}). "
                    f"Consider reweighting or adversarial debiasing."
                )
            else:
                recommendations.append(
                    f"MILD: '{attr}' shows slight bias (gap={max_gap:.1%}). "
                    f"Monitor and consider pre-processing adjustments."
                )

        # Check for equalized odds issues
        eo_findings = [f for f in report["findings"] if f["metric"] == "equalized_odds"]
        if eo_findings:
            recommendations.append(
                "Equalized odds violations detected. Consider using AEGIS RL autopilot "
                "to automatically optimize fairness constraints."
            )

        return recommendations


def generate_bias_report(
    metric_results: Dict[str, Any],
    model_name: str = "unknown",
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """Module-level convenience wrapper around BiasReporter.generate_report().

    Args:
        metric_results: Dict mapping attribute name → metric result(s).
        model_name: Name of the model being audited.
        dataset_name: Name of the dataset used.

    Returns:
        Structured report dict with findings and recommendations.
    """
    reporter = BiasReporter()
    return reporter.generate_report(metric_results, model_name, dataset_name)
