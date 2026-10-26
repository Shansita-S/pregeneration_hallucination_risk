from .schemas import AnswerResult, PreGenerationCheck, RiskPrediction
from .feature_extractor import extract_features
from .risk_model import RiskClassifier
from .decision_agent import choose_route
from .retrieval_agent import RetrievalAgent
from .claim_verifier import verify_answer
from .provider_factory import configured_provider, generate_with_provider


class Orchestrator:
    def __init__(self, risk_model=None, retriever=None):
        self.risk_model = risk_model or RiskClassifier.load()
        self.retriever = retriever or RetrievalAgent()

    def assess_risk(self, question: str) -> RiskPrediction:
        features = extract_features(question)
        score, model_name = self.risk_model.predict(question)
        healthcare_question = self.is_healthcare_question(question)
        normalized = question.lower()
        medication_action = (
            any(term in normalized for term in ("medication", "medicine", "drug", "dose", "dosage"))
            and any(
                term in normalized
                for term in (
                    "change", "increase", "decrease", "stop", "start",
                    "double", "skip", "adjust",
                )
            )
        )
        if medication_action:
            score = max(score, 0.75)
        elif healthcare_question:
            score = max(score, 0.60)
        level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
        explanations = []
        if medication_action:
            explanations.append(
                "Medication or dose-change wording requires evidence-first handling and clinician/pharmacist review."
            )
        elif healthcare_question:
            explanations.append(
                "Healthcare-related questions require evidence-aware handling even when the language model estimates low lexical risk."
            )
        if features.uncertainty_word_count:
            explanations.append("Question contains uncertainty/speculation-related wording.")
        if features.broad_question_hint:
            explanations.append("Question contains broad or absolute wording.")
        if not explanations:
            explanations.append("No simple high-risk wording indicators were detected.")
        return RiskPrediction(
            risk_score=round(score, 4), risk_level=level,
            model_used=model_name, explanation=explanations,
        )

    def retrieve_evidence(self, question: str, top_k=3):
        return self.retriever.search(question, top_k=top_k)

    @staticmethod
    def is_healthcare_question(question: str) -> bool:
        terms = {
            "symptom", "diagnosis", "disease", "medicine", "medication",
            "drug", "dose", "dosage", "vaccine", "antibiotic", "pregnant",
            "pregnancy", "diabetes", "blood pressure", "mental health",
            "therapy", "treatment", "doctor", "clinician", "medical",
            "health", "pain", "fever", "allergy",
        }
        normalized = question.lower()
        return any(term in normalized for term in terms)

    def check_before_generation(self, question: str, top_k: int = 5) -> PreGenerationCheck:
        risk = self.assess_risk(question)
        route = choose_route(risk.risk_score)
        evidence = self.retrieve_evidence(question, top_k=max(1, min(top_k, 5)))
        healthcare_question = self.is_healthcare_question(question)
        guidance = [
            "Use only the returned evidence for factual claims and cite source_id values.",
            "If evidence is insufficient or conflicting, say that explicitly instead of filling gaps.",
        ]
        if healthcare_question:
            guidance.extend([
                "Do not diagnose, prescribe, recommend a dose, or tell the user to start or stop treatment.",
                "For severe, urgent, or immediately dangerous symptoms, direct the user to local emergency services.",
                "State that the information is general education and recommend a qualified clinician or pharmacist for personal advice.",
            ])
        if not evidence:
            guidance.append("No relevant local evidence was found; do not generate an evidence-backed medical answer.")
        return PreGenerationCheck(
            question=question, risk=risk, route=route, evidence=evidence,
            healthcare_question=healthcare_question,
            generation_guidance=guidance,
        )

    def process(self, question: str, use_gemini=True, run_consistency=False,
                consistency_samples: int = 2) -> AnswerResult:
        features = extract_features(question)
        pre_generation = self.check_before_generation(question, top_k=5)
        risk = pre_generation.risk  # Must occur before answer generation.
        route = choose_route(risk.risk_score)
        route_explanation = {
            "low_risk_direct_generation": "The question has a lower estimated risk, so the pipeline can generate directly while still checking the result.",
            "medium_risk_cautious_generation": "The question is routed through retrieval and cautious generation because its estimated risk is moderate.",
            "high_risk_evidence_first": "The question is routed through evidence-first processing because its estimated risk is high.",
        }[route]
        evidence = [item.model_dump() for item in pre_generation.evidence]
        evidence_context = "\n".join(
            f"- [{item.get('source_id', item['title'])}] {item['title']}: {item['text']}"
            for item in evidence
        ) or "No relevant evidence found in the local demo corpus."
        warnings = []
        if not evidence:
            warnings.append("No relevant evidence was retrieved from the local demonstration corpus.")
        if use_gemini:
            provider = configured_provider()
            prompt = (
                "Answer the question carefully. Use only retrieved evidence for factual claims. "
                "Cite sources exactly using [source_id] when relevant. If evidence is insufficient, "
                "say so explicitly. Do not add numbers, nutrient values, medical benefits, or "
                "preparation claims that are not stated in the evidence. Never invent sources, "
                "URLs, or citations. Include a brief medical-safety caveat for health questions "
                "and do not diagnose or replace a clinician. For urgent symptoms, direct the user "
                "to local emergency services rather than attempting triage.\n\n"
                f"Question: {question}\nRisk route: {route}\nEvidence:\n{evidence_context}\n"
            )
            try:
                answer = generate_with_provider(prompt)
            except Exception as exc:
                answer = (
                    f"Demo response only: I could not call {provider.upper()}. "
                    f"Reason: {exc}. Retrieved evidence: {evidence_context}"
                )
                warnings.append(f"{provider.upper()} generation failed; the displayed text is not an LLM answer.")
        else:
            answer = (
                "Demo response (not an LLM-generated answer). Retrieved evidence:\n"
                + evidence_context
                + ("\nNo relevant local evidence was found." if not evidence else "")
            )
        verification = verify_answer(answer, evidence)
        consistency = None
        if run_consistency and use_gemini:
            from .selfcheck_verifier import check_consistency
            consistency = check_consistency(
                question, answer, samples=max(1, min(consistency_samples, 3))
            )
        elif run_consistency:
            warnings.append("Consistency checking requires an LLM provider and was not run in demo mode.")
        return AnswerResult(
            question=question, features=features, risk=risk, route=route,
            route_explanation=route_explanation, answer=answer,
            verification=verification, consistency=consistency, warnings=warnings,
        )
