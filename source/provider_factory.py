import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _require_key(name: str) -> str:
    api_key = os.getenv(name, "").strip()
    if not api_key:
        raise RuntimeError(f"{name} is not set. Add it to .env or use demo mode.")
    return api_key


def _generate_with_gemini(prompt: str, temperature: float) -> str:
    api_key = _require_key("GEMINI_API_KEY")
    from google import genai

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def _generate_with_groq(prompt: str, temperature: float) -> str:
    api_key = _require_key("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hallucination-risk-engine/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            detail = "The Groq API key was rejected. Check that it is active and copied correctly."
        elif exc.code == 403:
            detail = (
                "Groq denied this request (HTTP 403). Check the key/account permissions "
                "and whether a proxy, VPN, firewall, or network policy is blocking api.groq.com. "
                f"Provider detail: {detail.strip()}"
            )
        elif exc.code == 429:
            detail = "Groq rate limit reached. Wait and retry, or review your Groq limits."
        elif exc.code == 404:
            detail = (
                f"Groq could not find or allow model '{model}'. Set GROQ_MODEL to an "
                "active model from https://api.groq.com/openai/v1/models. "
                f"Provider detail: {detail.strip()}"
            )
        raise RuntimeError(f"Groq request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Groq request failed: {exc.reason}") from exc

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an unexpected response.") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Groq returned an empty response.")
    return text.strip()


def generate_with_provider(prompt: str, temperature: float = 0.2) -> str:
    """Generate text with the configured provider."""
    _load_environment()
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        return _generate_with_gemini(prompt, temperature)
    if provider == "groq":
        return _generate_with_groq(prompt, temperature)
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'gemini' or 'groq'."
    )


def configured_provider() -> str:
    _load_environment()
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def generate_with_gemini(prompt: str, temperature: float = 0.2) -> str:
    """Backward-compatible Gemini entry point."""
    _load_environment()
    return _generate_with_gemini(prompt, temperature)
