# Remote MCP setup

This project exposes MCP separately from the REST API. The MCP server uses the
official Python SDK and Streamable HTTP transport; `/docs` is REST
documentation, not an MCP endpoint.

## Local testing

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m source.mcp_server
```

The local MCP endpoint is `http://127.0.0.1:8001/mcp`. Use an MCP-compatible
client or MCP Inspector to discover and call:

- `predict_hallucination_risk`
- `retrieve_evidence`
- `check_before_generation`
- `verify_answer`
- `generate_verified_answer`

The tools call the same orchestrator as the REST API. Set `LLM_PROVIDER=groq`
and `GROQ_API_KEY` in `.env` before calling the generation tool. The default model
is `openai/gpt-oss-120b`; use `GROQ_MODEL` for another active model. Risk,
retrieval, and verification tools do not require a Groq key.

### Using it as a pre-generation connector

Call `check_before_generation` first with the user's question. Pass its
`risk`, `evidence`, and `generation_guidance` to the answer model as trusted
instructions/context. Only call `generate_verified_answer` when this service
should also perform generation. For healthcare questions, the check adds
guardrails against diagnosis, prescribing, unsupported dosing, and filling
gaps when no authoritative local evidence was retrieved.

The equivalent REST request is:

```powershell
Invoke-RestMethod `
  -Uri https://<api-service>/pre-generation-check `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What should I know about antibiotics?","top_k":5}'
```

In a custom ChatGPT or Claude integration, the host application should:

1. Send the user question to `check_before_generation`.
2. Provide the returned evidence and guardrails to the answer model.
3. Require source IDs/URLs in the answer and preserve the warning text.
4. Send the drafted answer and original question to `verify_answer`.
5. Show the verification status and warnings to the user; do not silently
   present `SUPPORTED` as proof of medical truth.

This is an MCP/REST integration pattern, not a universal connector switch.
The AI product must support remote MCP or permit an HTTPS REST action.

## Render deployment

The included `render.yaml` defines two services:

1. `hallucination-risk-api`, which serves the homepage and REST API.
2. `hallucination-risk-mcp`, which runs `python -m source.mcp_server`.

Connect the repository to Render and deploy the blueprint. Add
`GROQ_API_KEY` as a secret when generation is needed. Render supplies `PORT`;
the MCP server reads it automatically. After deployment, copy the MCP
service's HTTPS URL and append `/mcp`, for example:

```text
https://your-mcp-service.onrender.com/mcp
```

This is only an example. The actual endpoint does not exist until your service
is deployed and tested. Do not publish an endpoint based on this example.

### Exact deployment steps

1. Put your expanded knowledge base in `data/knowledge_base.json`, or build it
   from a source folder:

   ```powershell
   New-Item -ItemType Directory -Force data\sources
   # Add licensed JSON, JSONL, TXT, or Markdown files to data\sources
   python -m source.ingest_knowledge data\sources data\knowledge_base.json
   ```

   JSON documents should contain `title`, `text`, and preferably `source_id`
   and `source_url`. Do not scrape or redistribute material without permission.
2. Commit the knowledge-base JSON and source-code changes to GitHub. Do not
   commit `.env`, API keys, private documents, or copyrighted material.
3. In Render, choose **New > Blueprint**, select the repository, and deploy
   `render.yaml`.
4. Enter `GROQ_API_KEY` as a Render secret for both services. Keep
   `LLM_PROVIDER=groq` and use an active `GROQ_MODEL`.
5. Wait for both services to become healthy. The REST URL is the API/UI
   service URL; the MCP URL is the separate MCP service URL plus `/mcp`.
6. Test `https://<api-service>/health` and open
   `https://<api-service>/docs`.
7. Test MCP tool discovery from an MCP client against
   `https://<mcp-service>/mcp`. A service that merely loads in a browser is not
   proof that MCP initialization and tool calls work.

The current Render configuration exposes an unauthenticated MCP service. Do
not use it for private data or unrestricted paid operations until you add
authentication, rate limiting, logging, and a host/network policy. Public
deployment also means anyone who discovers the endpoint may invoke tools.

## Client connection and verification

Configure the URL in a client that explicitly supports remote Streamable HTTP
MCP servers. Each client has its own connection format, authentication rules,
and approval process; not every AI application accepts arbitrary remote MCP
servers. Use that client's official documentation for the exact configuration.

For Claude, use its custom remote MCP connector flow and provide the deployed
HTTPS `/mcp` URL. Claude's connector availability, plan permissions, approval
steps, and authentication requirements are controlled by Anthropic; follow
Anthropic's current connector documentation.

For ChatGPT, use a product/workspace feature that explicitly supports custom
remote MCP connectors if it is available to your account. ChatGPT products and
plans may impose separate approval, authentication, or allow-list
requirements. An arbitrary REST URL or `/docs` URL is not a ChatGPT connector.

There is no universal “connect to every AI” switch. Each client must support
the MCP transport, accept your server URL, and authorize the tools. If a
client does not support remote MCP, use its supported integration method or
call this project's REST API directly.

Verify the deployment in this order:

1. Open the REST service URL and confirm the homepage loads.
2. Open the REST service's `/health`.
3. Use a compatible MCP client to initialize against the MCP URL.
4. Confirm tool discovery lists the four tools.
5. Call `check_before_generation` with `{"question":"What is photosynthesis?"}`.
6. Call `generate_verified_answer` only after setting the Groq secret.

If connection fails, check that the URL ends in `/mcp`, the MCP service is
running (not the REST service URL), the deployment logs show the expected
`PORT`, and the client supports Streamable HTTP. Provider errors usually mean
the Groq key is missing, invalid, rate-limited, or the configured model is not
available.

The included local corpus is a small demonstration corpus. Retrieval uses
TF-IDF and claim verification uses a bounded lexical baseline; neither is a
comprehensive fact-checking service or semantic entailment model.
