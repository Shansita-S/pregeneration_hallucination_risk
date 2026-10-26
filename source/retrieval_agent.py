import json
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Small starter corpus for demonstrating retrieval. Replace with sourced documents
# appropriate to your use case; each entry should preserve its provenance.
DEFAULT_DOCUMENTS = [
    {"title": "Photosynthesis", "source_id": "demo:photosynthesis", "source_url": None, "text": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy."},
    {"title": "Cloud computing", "source_id": "demo:cloud-computing", "source_url": None, "text": "Cloud computing provides on-demand access to computing resources such as servers, storage, databases, and software over a network."},
    {"title": "Machine learning", "source_id": "demo:machine-learning", "source_url": None, "text": "Machine learning is a field of computing that develops methods that learn patterns from data to make predictions or decisions."},
    {"title": "Water cycle", "source_id": "demo:water-cycle", "source_url": None, "text": "The water cycle describes the continuous movement of water between Earth's surface and the atmosphere."},
    {"title": "India", "source_id": "demo:india", "source_url": None, "text": "India is a country in South Asia. New Delhi is the capital of India."},
]

DEFAULT_KNOWLEDGE_BASE_PATH = Path(__file__).parents[1] / "data" / "knowledge_base.json"


def load_default_documents():
    configured_path = Path(
        os.getenv("KNOWLEDGE_BASE_PATH", str(DEFAULT_KNOWLEDGE_BASE_PATH))
    )
    if configured_path.exists():
        return DEFAULT_DOCUMENTS + json.loads(
            configured_path.read_text(encoding="utf-8")
        )
    return DEFAULT_DOCUMENTS


class RetrievalAgent:
    def __init__(self, documents=None, chunk_size: int = 800, overlap: int = 100):
        self.documents = documents if documents is not None else load_default_documents()
        self.documents = self.chunk_documents(self.documents, chunk_size, overlap)
        self.documents = [
            {"source_id": f"local:{i}", "source_url": None, **document}
            if "source_id" not in document else document
            for i, document in enumerate(self.documents)
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.documents])

    @staticmethod
    def chunk_documents(documents, chunk_size: int = 800, overlap: int = 100):
        """Split documents while preserving source metadata for citations."""
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
        chunks = []
        for document in documents:
            text = document["text"].strip()
            start = 0
            part = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                chunks.append({
                    **document,
                    "title": f"{document['title']} (part {part + 1})" if len(text) > chunk_size else document["title"],
                    "text": text[start:end],
                    "source_id": document.get("source_id", f"local:{part}"),
                })
                if end == len(text):
                    break
                start = end - overlap
                part += 1
        return chunks

    @classmethod
    def from_json(cls, path, chunk_size: int = 800, overlap: int = 100):
        """Load [{title, text, source_id?, source_url?}] documents from JSON."""
        documents = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            raise ValueError("Knowledge-base JSON must contain a list of documents")
        required = {"title", "text"}
        if any(not required.issubset(document) for document in documents):
            raise ValueError("Every knowledge-base document needs title and text")
        return cls(documents, chunk_size, overlap)

    def search(self, query: str, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indexes = scores.argsort()[::-1][:top_k]
        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in indexes if scores[i] > 0
        ]
