"""Build a validated JSON knowledge base from local source files."""
import argparse
import hashlib
import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".txt", ".md"}


def _source_id(path: Path, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"file:{path.stem}:{digest}"


def _read_file(path: Path) -> list[dict]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a document object or list")
        return value
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    text = path.read_text(encoding="utf-8").strip()
    return [{"title": path.stem, "text": text}]


def build_knowledge_base(input_dir: Path, output_path: Path) -> int:
    documents = []
    seen = set()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        for document in _read_file(path):
            if not isinstance(document, dict) or not document.get("text"):
                raise ValueError(f"{path} contains a document without text")
            text = str(document["text"]).strip()
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            documents.append({
                "title": str(document.get("title") or path.stem),
                "source_id": str(document.get("source_id") or _source_id(path, text)),
                "source_url": document.get("source_url"),
                "text": text,
            })
    if not documents:
        raise ValueError(f"No supported documents found in {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG knowledge base")
    parser.add_argument("input_dir", type=Path, help="Directory containing JSON, JSONL, TXT, or Markdown sources")
    parser.add_argument("output_path", type=Path, help="Output JSON knowledge-base path")
    args = parser.parse_args()
    count = build_knowledge_base(args.input_dir, args.output_path)
    print(f"Wrote {count} documents to {args.output_path}")


if __name__ == "__main__":
    main()
