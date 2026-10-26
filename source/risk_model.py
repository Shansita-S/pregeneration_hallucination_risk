from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extractor import feature_vector

FEATURE_NAMES = [
    "question_length", "word_count", "question_mark_count",
    "has_named_entity_hint", "uncertainty_word_count", "broad_question_hint",
]


class RiskClassifier:
    def __init__(self, model=None, model_name="heuristic-fallback"):
        self.model = model
        self.model_name = model_name

    def predict(self, question: str) -> tuple[float, str]:
        if self.model is None:
            features = feature_vector(question)
            # Transparent fallback heuristic, not a trained model.
            score = min(0.95, 0.15 + 0.025 * features[4] + 0.20 * features[5]
                        + 0.08 * int(features[1] > 25)
                        + 0.10 * int("?" in question and features[1] < 4))
            return float(max(0.05, score)), self.model_name
        probability = float(self.model.predict_proba([feature_vector(question)])[0][1])
        return probability, self.model_name

    @classmethod
    def load(cls, path="models/risk_model.joblib"):
        p = Path(path)
        if not p.exists():
            return cls()
        payload = joblib.load(p)
        return cls(payload["model"], payload.get("model_name", "trained-model"))


def build_models():
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=42, min_samples_leaf=2
        ),
    }


def save_model(model, model_name: str, path="models/risk_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_name": model_name}, path)
