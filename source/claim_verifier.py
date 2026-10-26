import re

from .schemas import ClaimVerification, EvidenceItem, VerificationResult


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def extract_claims(answer: str, max_claims: int = 8) -> list[str]:
    """Extract sentence-sized factual units for transparent baseline checking."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    claims = []
    for sentence in sentences:
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", sentence).strip()
        if len(_tokens(cleaned)) >= 2:
            claims.append(cleaned)
        if len(claims) >= max_claims:
            break
    return claims


def _evidence_item(item: dict) -> EvidenceItem:
    return EvidenceItem(
        title=item["title"],
        text=item["text"],
        score=float(item.get("score", 0.0)),
        source_id=item.get("source_id"),
        source_url=item.get("source_url"),
    )


def _has_negation(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|without|cannot|can't|isn't|doesn't)\b", text.lower()))


def verify_claims(answer: str, evidence: list[dict]) -> list[ClaimVerification]:
    claims = extract_claims(answer)
    if not claims:
        return []
    results = []
    for claim in claims:
        claim_tokens = _tokens(claim)
        candidates = []
        for item in evidence:
            overlap = len(claim_tokens & _tokens(item["text"])) / max(len(claim_tokens), 1)
            if overlap >= 0.3:
                candidates.append((item, overlap))
        if not candidates:
            results.append(ClaimVerification(
                claim=claim, verdict="INSUFFICIENT_EVIDENCE",
                explanation="No retrieved passage contains enough related information to assess this claim.",
            ))
            continue
        best_item, best_overlap = max(candidates, key=lambda pair: pair[1])
        best_evidence = [_evidence_item(best_item)]
        if _has_negation(claim) != _has_negation(best_item["text"]) and best_overlap >= 0.5:
            results.append(ClaimVerification(
                claim=claim, verdict="CONTRADICTED",
                contradicting_evidence=best_evidence,
                explanation="The claim's negation pattern conflicts with the closest retrieved passage; review the context manually.",
            ))
        else:
            results.append(ClaimVerification(
                claim=claim, verdict="SUPPORTED",
                supporting_evidence=best_evidence,
                explanation="The claim is broadly consistent with the closest retrieved passage, but this lexical baseline is not semantic entailment.",
            ))
    return results


def verify_answer(answer: str, evidence: list[dict]) -> VerificationResult:
    if not evidence:
        return VerificationResult(
            status="no_evidence", score=0.0,
            explanation="No relevant local evidence was retrieved; factual support cannot be assessed.",
            evidence=[], claims=[
                ClaimVerification(
                    claim=claim, verdict="INSUFFICIENT_EVIDENCE",
                    explanation="No retrieved evidence is available for this claim.",
                )
                for claim in extract_claims(answer)
            ],
        )
    claims = verify_claims(answer, evidence)
    supported = sum(c.verdict == "SUPPORTED" for c in claims)
    contradicted = sum(c.verdict == "CONTRADICTED" for c in claims)
    score = supported / max(len(claims), 1)
    if contradicted:
        status = "conflicting"
    elif supported == len(claims) and claims:
        status = "supported"
    elif supported:
        status = "partially_supported"
    else:
        status = "unsupported"
    return VerificationResult(
        status=status, score=round(float(score), 3),
        explanation=(
            "Claim-level baseline using retrieved passages and token overlap. "
            "It is not semantic entailment or an independent fact-check."
        ),
        evidence=[_evidence_item(item) for item in evidence],
        claims=claims,
    )
