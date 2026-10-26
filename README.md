# Pre-Generation Hallucination Risk Prediction

A small, beginner-friendly prototype that estimates **the risk that a question may lead to an unreliable answer before answer generation**, then routes the request through retrieval, Gemini (optional), and lightweight evidence checks.

## What is included

- Pre-generation feature extraction and ML risk classification.
- Train/test evaluation for a Logistic Regression baseline and Random Forest.
- A transparent demo dataset generator. **Synthetic demo data is not a benchmark result** and must not be described as TruthfulQA/HaluEval data.
- Optional Gemini or Groq generation using provider-specific environment variables.
- Local evidence retrieval and simple claim/evidence checks.
- A post-generation multi-sample consistency check (inspired by SelfCheckGPT; not proof of truth).
- Official MCP Python SDK server with tools for risk prediction, evidence search, answer verification, and end-to-end processing.
- A healthcare-focused, source-linked evidence set and a pre-generation check that
  downstream AI clients can call before drafting an answer.
- Tests and Docker/Render configuration.

## Important scientific limitation

The included `data/demo_risk_data.csv` is synthetic, intended only to verify that the pipeline runs. It does **not** establish real-world accuracy. To report meaningful performance, replace it with a properly labelled dataset (for example, an appropriately licensed and schema-checked subset of TruthfulQA, HaluEval, or another suitable dataset). Record the source, label definitions, preprocessing, and split strategy. Do not invent metrics or claim that synthetic examples are benchmark results.

Risk is estimated from the **question before generation**. Claim verification and multi-sample consistency checks happen **after generation** and are separate signals. Lexical overlap is a heuristic, not a fact-checking system.

## Quick start (local or Colab)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add the key for the provider you want, or leave it unset for demo mode.
python -m source.train_model
python -m source.app
```

Open `http://127.0.0.1:8000/` for the web interface. The REST API remains
available at `http://127.0.0.1:8000/docs`.

For Groq generation, set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and
`GROQ_MODEL` in `.env`. A Groq key must not be placed in `GEMINI_API_KEY`.
The homepage sends requests to the real `/answer` endpoint and displays risk,
routing, local evidence, the generated answer, verification, and optional
consistency output.

For Colab, install requirements with `!pip install -r requirements.txt`, set the selected provider's key using Colab Secrets (or `os.environ` for a temporary session), and run the training command. Never print or commit your API key.

### Using Groq instead of Gemini

A Groq key cannot be pasted into `GEMINI_API_KEY`; the providers use different
endpoints and authentication. To use Groq, set these values in `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key
GROQ_MODEL=openai/gpt-oss-120b
```

The default remains Gemini. If no valid provider key is configured, `/answer`
returns the existing explicit demo response rather than failing silently.

## Run tests

```bash
pytest -q
```

## Run the MCP server

```bash
python -m source.mcp_server
```

The MCP server uses the official Python `mcp` SDK and Streamable HTTP transport.
By default it listens locally on port `8001` at `/mcp`. Test it with an
MCP-compatible client or MCP Inspector. A local server is **not** a public
endpoint. Render deployment is documented in [docs/mcp_setup.md](docs/mcp_setup.md)
and creates a separate MCP service. Do not claim a public endpoint until the
deployed URL has been tested.

## Public deployment

A `Dockerfile` and `render.yaml` are included as a starting point. Connect this repository to your hosting provider and deploy it. Configure the selected provider's API key (`GEMINI_API_KEY` or `GROQ_API_KEY`) and
`LLM_PROVIDER` as secret environment variables in the hosting dashboard. Free
hosting can sleep or have usage limits. After deployment, verify the actual URL
and MCP tools using a remote client before sharing it.

## Project flow

1. Validate and analyze the question.
2. Extract question-only features.
3. Predict pre-generation risk.
4. Route: lower-risk requests can generate directly; higher-risk requests use evidence-aware handling and a cautious response policy.
5. Retrieve from the small local evidence store.
6. Generate with Gemini if configured; otherwise use a deterministic demo response.
7. Check claims against retrieved evidence using a simple lexical heuristic.
8. Optionally compare multiple generated samples for consistency.
9. Expose key capabilities as MCP tools.

## Retrieval, claims, and consistency

The starter retriever is a local TF-IDF RAG baseline. Documents can be loaded
from a JSON list containing `title`, `text`, and optional `source_id` and
`source_url` fields with `RetrievalAgent.from_json(...)`; long documents are
chunked while retaining their source metadata. See
[data/knowledge_base.example.json](data/knowledge_base.example.json). This
avoids requiring a vector database for the first working version. It is not
semantic search and the small included corpus is not comprehensive.

The included [data/knowledge_base.json](data/knowledge_base.json) is a curated
cross-domain starter corpus with source-attributed summaries covering health,
food safety, programming, web development, Git, Docker, security, science,
education, finance basics, travel, weather, emergency preparedness, and MCP.
These are summaries with links, not copied source documents. They improve
common-question coverage but do not contain all world knowledge; current,
niche, legal, medical, financial, and location-specific questions require
additional authoritative sources and professional judgment.

### Expanding the knowledge base

No finite file can answer every question on every topic. Add reliable,
licensed, topic-specific documents and preserve their provenance. The project
supports JSON, JSONL, TXT, and Markdown ingestion:

```powershell
New-Item -ItemType Directory -Force data\sources
# Put your permitted source files in data\sources
python -m source.ingest_knowledge data\sources data\knowledge_base.json
```

For JSON, use objects with `title`, `text`, and preferably `source_id` and
`source_url`. The ingestion command deduplicates text, validates that every
document has content, and writes a reproducible JSON corpus. The service
automatically chunks long documents and preserves source metadata. It does not
automatically search the internet, because search credentials, licensing,
freshness, and source quality require an explicit policy.

After changing the corpus, restart the REST and MCP processes so each process
rebuilds its in-memory TF-IDF index.

Generated text is split into bounded sentence-sized claims. Each claim is
reported as `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE` against
retrieved passages. This is a transparent lexical baseline, not semantic
entailment or independent ground truth; high overlap must not be interpreted as
proof.

The optional consistency check is **SelfCheckGPT-inspired prompted sampling**:
it compares a bounded number of additional answers from the same model. It is
not the original SelfCheckGPT implementation. Agreement does not prove truth,
and disagreement does not prove falsity.

LangChain/Chroma are intentionally not required in this baseline: the existing
TF-IDF implementation is easier to run offline and preserves source metadata.
They can be added later behind the same retriever interface if a larger
knowledge base requires semantic retrieval.

## API endpoints

- `GET /health`
- `POST /risk`
- `POST /pre-generation-check`
- `POST /answer`

## Example

```json
{"question":"What is photosynthesis?"}
```

### Pre-generation connector workflow

Use `POST /pre-generation-check` or the MCP tool `check_before_generation`
before asking another model to draft an answer. It returns the risk score,
route, retrieved evidence with source URLs, whether the question appears
healthcare-related, and generation guardrails. The downstream model should use
that response as a gate: cite only returned sources, disclose insufficient or
conflicting evidence, and never turn general healthcare information into a
diagnosis or prescription.

Healthcare-related questions receive a minimum operational risk score even if
the generic trained classifier finds no unusual wording. Medication and dosage
change requests receive high-risk evidence-first routing. This safety floor is
an explicit policy guard, not a medical diagnosis or a claim that the model's
probability estimate is calibrated.

The healthcare corpus prioritizes MedlinePlus/NLM, NIH, CDC, FDA, WHO, ADA,
USDA, NHS, and Poison Control pages. These are source-linked summaries, not
copies of the pages and not a replacement for live clinical guidance. Refresh
time-sensitive content and verify current official recommendations before
production use.

No external benchmark, API, or hosting service is accessed automatically by this project.
