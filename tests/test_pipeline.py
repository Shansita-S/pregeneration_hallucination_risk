from fastapi.testclient import TestClient

from source.app import app
from source.feature_extractor import extract_features
from source.risk_model import RiskClassifier
from source.orchestrator import Orchestrator
from source.retrieval_agent import RetrievalAgent
from source.claim_verifier import extract_claims, verify_answer
from source.provider_factory import generate_with_provider
from source.ingest_knowledge import build_knowledge_base


def test_features_are_question_only():
    f = extract_features("What is the latest guaranteed prediction?")
    assert f.word_count > 0
    assert f.broad_question_hint == 1


def test_risk_prediction_in_range():
    score, name = RiskClassifier().predict("What will definitely happen in the future?")
    assert 0.0 <= score <= 1.0
    assert name


def test_retrieval_returns_relevant_document():
    results = RetrievalAgent().search("What is photosynthesis?")
    assert results
    assert results[0]["title"] == "Photosynthesis"
    assert results[0]["source_id"] == "demo:photosynthesis"


def test_retrieval_chunks_documents_with_metadata():
    chunks = RetrievalAgent.chunk_documents(
        [{"title": "Guide", "text": "abcdefghij", "source_id": "docs:guide"}],
        chunk_size=6, overlap=2,
    )
    assert len(chunks) == 2
    assert all(chunk["source_id"] == "docs:guide" for chunk in chunks)


def test_verifier_reports_no_evidence():
    result = verify_answer("Some claim.", [])
    assert result.status == "no_evidence"
    assert result.claims[0].verdict == "INSUFFICIENT_EVIDENCE"


def test_claim_verifier_reports_supported_and_contradicted():
    evidence = [{
        "title": "Fact", "source_id": "test:fact",
        "text": "Water boils at 100 degrees Celsius at sea level.",
    }]
    supported = verify_answer("Water boils at 100 degrees Celsius at sea level.", evidence)
    assert supported.claims[0].verdict == "SUPPORTED"
    contradicted = verify_answer("Water does not boil at 100 degrees Celsius at sea level.", evidence)
    assert contradicted.claims[0].verdict == "CONTRADICTED"


def test_claim_extraction_is_bounded():
    assert extract_claims("One fact. Another fact.") == ["One fact.", "Another fact."]


def test_orchestrator_demo_mode():
    result = Orchestrator().process("What is photosynthesis?", use_gemini=False)
    assert result.risk.risk_level in {"low", "medium", "high"}
    assert result.answer
    assert result.verification.status in {
        "supported", "partially_supported", "unsupported", "no_evidence"
    }


def test_health_and_homepage():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "Hallucination Risk Intelligence" in client.get("/").text


def test_knowledge_base_ingestion(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "guide.txt").write_text("A trusted fact.", encoding="utf-8")
    output = tmp_path / "knowledge_base.json"
    assert build_knowledge_base(source_dir, output) == 1
    loaded = RetrievalAgent.from_json(output)
    assert loaded.documents[0]["source_id"].startswith("file:guide:")


def test_risk_endpoint_does_not_need_llm():
    response = TestClient(app).post("/risk", json={"question": "What is photosynthesis?"})
    assert response.status_code == 200
    assert 0 <= response.json()["risk_score"] <= 1


def test_healthcare_risk_uses_safety_floor():
    response = TestClient(app).post(
        "/risk",
        json={"question": "Can I change my blood pressure medication dose?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_score"] >= 0.75
    assert body["risk_level"] == "high"
    assert any("Medication or dose-change" in item for item in body["explanation"])


def test_pre_generation_check_returns_healthcare_guidance():
    response = TestClient(app).post(
        "/pre-generation-check",
        json={"question": "Can I change my medication dose for high blood pressure?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["healthcare_question"] is True
    assert any("diagnose" in item for item in body["generation_guidance"])
    assert body["evidence"]


def test_empty_question_is_rejected():
    response = TestClient(app).post("/risk", json={"question": ""})
    assert response.status_code == 422


def test_groq_missing_key_is_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    try:
        generate_with_provider("hello")
    except RuntimeError as exc:
        assert "GROQ_API_KEY is not set" in str(exc)
    else:
        raise AssertionError("Expected missing Groq key error")
