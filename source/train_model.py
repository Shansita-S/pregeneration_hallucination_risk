from pathlib import Path
import json
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from .prepare_data import load_and_split
from .risk_model import build_models, save_model


def run_training_pipeline(dataset_path="data/demo_risk_data.csv",
                          output_path="models/risk_model.joblib"):
    X_train, X_test, y_train, y_test = load_and_split(dataset_path)
    results = {}
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        try:
            prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob))
        except Exception:
            auc = None
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = model

    # Select based on F1, with deterministic tie-breaking.
    best_name = max(results, key=lambda n: (results[n]["f1"], n == "logistic_regression"))
    save_model(fitted[best_name], best_name, output_path)
    Path("reports").mkdir(exist_ok=True)
    report = {
        "dataset": str(dataset_path),
        "dataset_note": "Synthetic demo data unless you replaced the file with a labelled real dataset.",
        "best_model": best_name,
        "metrics": results,
    }
    Path("reports/model_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_training_pipeline()
