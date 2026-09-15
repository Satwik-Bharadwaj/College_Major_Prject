"""Random Forest workload classifier: train, evaluate, save, load, predict.

We compare Random Forest against a few baselines so the "best model" choice is
justified with numbers (as in the report), then persist the winner.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from . import FEATURE_NAMES, WORKLOAD_CLASSES, __version__

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "trainos_rf.joblib"
)


def _candidate_models() -> Dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_select(
    X: List[List[float]], y: List[str], test_size: float = 0.25
) -> Tuple[object, Dict[str, Dict]]:
    """Train candidates, pick the best by macro F1. Returns (best_model, report)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results: Dict[str, Dict] = {}
    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in _candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {
            "accuracy": round(float(acc), 6),
            "macro_f1": round(float(f1), 6),
        }
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    # detailed diagnostics for the winner
    preds = best_model.predict(X_test)
    report = {
        "chosen_model": best_name,
        "comparison": results,
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": list(WORKLOAD_CLASSES),
            "matrix": confusion_matrix(
                y_test, preds, labels=list(WORKLOAD_CLASSES)
            ).tolist(),
        },
    }

    # leakage / sanity check: feature importances (RF/tree/GB expose these)
    if hasattr(best_model, "feature_importances_"):
        report["feature_importances"] = {
            name: round(float(imp), 6)
            for name, imp in zip(FEATURE_NAMES, best_model.feature_importances_)
        }

    return best_model, report


def save(model: object, report: Dict, path: str = DEFAULT_MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(FEATURE_NAMES),
            "classes": list(WORKLOAD_CLASSES),
            "version": __version__,
        },
        path,
    )
    with open(path + ".report.json", "w") as fh:
        json.dump(report, fh, indent=2)


def load(path: str = DEFAULT_MODEL_PATH):
    bundle = joblib.load(path)
    return bundle


class Classifier:
    """Thin inference wrapper used by the scheduler loop."""

    def __init__(self, path: str = DEFAULT_MODEL_PATH):
        bundle = load(path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.classes = bundle["classes"]

    def predict(self, feature_vector: List[float]) -> str:
        return str(self.model.predict([feature_vector])[0])

    def predict_batch(self, vectors: List[List[float]]) -> List[str]:
        if not vectors:
            return []
        return [str(p) for p in self.model.predict(vectors)]
