from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .orchestrator import Orchestrator

app = FastAPI(
    title="Pre-Generation Hallucination Risk Prediction",
    description="Prototype API for pre-generation risk scoring and evidence-aware answering.",
    version="1.0.0",
)
engine = Orchestrator()
INDEX_FILE = Path(__file__).parent / "static" / "index.html"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    run_consistency: bool = False
    use_groq: bool = True
    consistency_samples: int = Field(default=2, ge=1, le=3)


class PreGenerationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=5)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "hallucination-risk-engine",
        "mcp_endpoint": "/mcp",
        "knowledge_base": len(engine.retriever.documents),
    }


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(INDEX_FILE)


@app.post("/risk")
def risk(request: AnswerRequest):
    try:
        return engine.assess_risk(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pre-generation-check")
def pre_generation_check(request: PreGenerationRequest):
    try:
        return engine.check_before_generation(request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/answer")
def answer(request: AnswerRequest):
    try:
        return engine.process(
            request.question,
            use_gemini=request.use_groq,
            run_consistency=request.run_consistency,
            consistency_samples=request.consistency_samples,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("source.app:app", host="0.0.0.0", port=8000, reload=False)
