from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_extractor import feature_vector

COLUMNS = [
    "question_length", "word_count", "question_mark_count",
    "has_named_entity_hint", "uncertainty_word_count", "broad_question_hint",
]


def create_demo_dataset(path="data/demo_risk_data.csv", n=600, seed=42):
    """Create synthetic data for smoke-testing only; not a real benchmark."""
    import random
    rng = random.Random(seed)
    safe_templates = [
        "What is the capital of {place}?",
        "How does {topic} work?",
        "Define {topic} in simple terms.",
        "What are the main parts of {topic}?",
    ]
    risky_templates = [
        "What is the exact latest guaranteed prediction about {topic}?",
        "Why does everyone always agree that {topic} is the best?",
        "Prove that {topic} will never fail in the future.",
        "What is the completely certain answer about unknown {topic}?",
    ]
    topics = ["photosynthesis", "cloud computing", "quantum physics", "climate",
              "artificial intelligence", "the economy", "history", "medicine"]
    places = ["France", "Japan", "Canada", "India", "Brazil"]
    rows = []
    for i in range(n):
        label = i % 2
        template = rng.choice(risky_templates if label else safe_templates)
        q = template.format(topic=rng.choice(topics), place=rng.choice(places))
        f = feature_vector(q)
        rows.append(dict(zip(COLUMNS, f), label=label, question=q))
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_and_split(path="data/demo_risk_data.csv", test_size=0.25, seed=42):
    p = Path(path)
    if not p.exists():
        create_demo_dataset(path)
    df = pd.read_csv(p)
    missing = set(COLUMNS + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    X = df[COLUMNS].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
