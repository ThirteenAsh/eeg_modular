from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from eeg_emotion.models.base import ModelAdapter


@dataclass(frozen=True)
class SVMConfig:
    # if param_grid is provided, we run GridSearchCV; otherwise we train a single model
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 5
    n_jobs: int = -1
    probability: bool = True  # needed for predict_proba


class SVMClassifier(ModelAdapter):
    def __init__(self, cfg: SVMConfig):
        self.cfg = cfg
        self.model: Optional[SVC] = None
        self.best_params_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "SVMClassifier":
        if self.cfg.param_grid:
            grid = GridSearchCV(
                SVC(probability=self.cfg.probability),
                self.cfg.param_grid,
                refit=True,
                cv=self.cfg.cv,
                n_jobs=self.cfg.n_jobs,
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.best_params_ = dict(grid.best_params_)
        else:
            self.model = SVC(probability=self.cfg.probability, **kwargs)
            self.model.fit(X, y)
            self.best_params_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SVM model is not fit.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            raise RuntimeError("SVM model is not fit.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, out_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("SVM model is not fit.")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(
            {"cfg": self.cfg, "model": self.model, "best_params": self.best_params_},
            os.path.join(out_dir, "svm.joblib"),
        )

    @classmethod
    def load(cls, in_dir: str) -> "SVMClassifier":
        payload = joblib.load(os.path.join(in_dir, "svm.joblib"))
        obj = cls(payload["cfg"])
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params")
        return obj
