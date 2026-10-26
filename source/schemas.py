from typing import Any, Literal
from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class QueryFeatures(BaseModel):
    question_length: int
    word_count: int
    question_mark_count: int
    has_named_entity_hint: int
    uncertainty_word_count: int
    broad_question_hint: int


class RiskPrediction(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    model_used: str
    explanation: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    text: str
    score: float = 0.0
    source_id: str | None = None
    source_url: str | None = None


class PreGenerationCheck(BaseModel):
    question: str
    risk: RiskPrediction
    route: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    healthcare_question: bool = False
    generation_guidance: list[str] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal[
        "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE", "NOT_CHECKED",
    ]
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: str
    uncertain: bool = True


class VerificationResult(BaseModel):
    status: Literal[
        "supported", "partially_supported", "unsupported", "no_evidence",
        "conflicting",
    ]
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[ClaimVerification] = Field(default_factory=list)


class AnswerResult(BaseModel):
    question: str
    features: QueryFeatures
    risk: RiskPrediction
    route: str
    route_explanation: str
    answer: str
    verification: VerificationResult
    consistency: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
