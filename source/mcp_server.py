"""Official MCP SDK server using Streamable HTTP transport."""
import os
from mcp.server.fastmcp import FastMCP
from .orchestrator import Orchestrator

mcp = FastMCP(
    "hallucination-risk-engine",
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8001")),
)
engine = Orchestrator()


@mcp.tool()
def predict_hallucination_risk(question: str) -> dict:
    """Estimate hallucination risk from the question before generation."""
    return engine.assess_risk(question).model_dump()


@mcp.tool()
def retrieve_evidence(question: str, top_k: int = 3) -> list[dict]:
    """Retrieve relevant passages from the small local demo evidence corpus."""
    return engine.retrieve_evidence(question, top_k=max(1, min(top_k, 5)))


@mcp.tool()
def check_before_generation(question: str, top_k: int = 5) -> dict:
    """Assess risk and retrieve evidence before another AI generates an answer."""
    return engine.check_before_generation(
        question, top_k=max(1, min(top_k, 5))
    ).model_dump()


@mcp.tool()
def verify_answer(answer: str, question: str) -> dict:
    """Check an answer against locally retrieved evidence using a lexical heuristic."""
    evidence = engine.retrieve_evidence(question, top_k=3)
    from .claim_verifier import verify_answer as verify
    return verify(answer, evidence).model_dump()


@mcp.tool()
def generate_verified_answer(
    question: str, use_groq: bool = True, run_consistency: bool = False,
    consistency_samples: int = 2,
) -> dict:
    """Run pre-generation risk scoring, retrieval, generation, and heuristic verification."""
    return engine.process(
        question, use_gemini=use_groq, run_consistency=run_consistency,
        consistency_samples=consistency_samples,
    ).model_dump()


if __name__ == "__main__":
    # Streamable HTTP endpoint, typically http://localhost:8001/mcp
    mcp.run(transport="streamable-http")
