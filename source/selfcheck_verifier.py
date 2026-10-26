from .claim_verifier import extract_claims
from .provider_factory import generate_with_provider


def check_consistency(question: str, first_answer: str, samples=2):
    """Compare multiple answer samples. Consistency is not proof of correctness."""
    answers = [first_answer]
    for _ in range(max(0, samples - 1)):
        try:
            answers.append(generate_with_provider(
                "Answer this question briefly and independently. If unsure, say so.\n"
                f"Question: {question}", temperature=0.7
            ))
        except Exception as exc:
            return {
                "available": False,
                "consistent": None,
                "samples": answers,
                "note": f"Additional samples unavailable: {exc}",
            }
    normalized = [set(a.lower().split()) for a in answers]
    similarities = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            union = normalized[i] | normalized[j]
            similarities.append(len(normalized[i] & normalized[j]) / len(union) if union else 1.0)
    avg = sum(similarities) / len(similarities) if similarities else 1.0
    return {
        "available": True,
        "method": "SelfCheckGPT-inspired prompted sampling with token-set comparison",
        "consistent": avg >= 0.35,
        "average_token_jaccard": round(avg, 3),
        "claims_checked": extract_claims(first_answer),
        "samples": answers,
        "note": "Agreement does not establish truth; disagreement indicates uncertainty, not falsity. Samples from one model are not independent evidence.",
    }
