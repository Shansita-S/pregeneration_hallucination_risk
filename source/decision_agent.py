import os


def choose_route(risk_score: float) -> str:
    threshold = float(os.getenv("RISK_THRESHOLD", "0.60"))
    if risk_score >= min(0.95, threshold + 0.15):
        return "high_risk_evidence_first"
    if risk_score >= threshold:
        return "medium_risk_cautious_generation"
    return "low_risk_direct_generation"
