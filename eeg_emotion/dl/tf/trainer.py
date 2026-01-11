from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf

from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix

@dataclass(frozen=True)
class TFTrainConfig:
    epochs: int = 50
    batch_size: int = 32
    val_split: float = 0.2

def train_and_eval_classifier(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    X_test: np.ndarray,
    y_test_int: np.ndarray,
    class_names: List[str],
    out_dir: str,
    cfg: TFTrainConfig,
    callbacks: Optional[list] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    model.fit(
        X_train, y_train_onehot,
        validation_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks or [],
        verbose=2
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    m = classification_metrics(y_test_int, y_pred, class_names=class_names)

    save_confusion_matrix(
        y_true=y_test_int,
        y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(figs_dir, "confusion_matrix.png"),
        normalize="true",
        title="Confusion Matrix (Normalized)"
    )

    out = {"accuracy": m["accuracy"], "report": m["report"], "best_params": {"epochs": cfg.epochs, "batch_size": cfg.batch_size}}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out
