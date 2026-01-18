from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split

from eeg_emotion.config.loader import ConfigError, load_config, get, require
from eeg_emotion.features.csv_stats import DEFAULT_CSV_FILES, build_tabular_dataset
from eeg_emotion.models.sklearn.mlp import MLPAdapter, MLPConfig
from eeg_emotion.models.sklearn.rf import RFAdapter, RFConfig
from sklearn.model_selection import GridSearchCV
import joblib

from eeg_emotion.models.sklearn.svm import SVMModel, from_dict

from eeg_emotion.preprocess.tabular import TabularPreprocessConfig, TabularPreprocessor
from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.utils.seed import set_seed
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix

#通用 GridSearch 包装器
class SklearnSearchAdapter:
    def __init__(self, search):
        self.search = search
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y):
        self.search.fit(X, y)
        self.best_params_ = getattr(self.search, "best_params_", None)
        self.best_score_ = getattr(self.search, "best_score_", None)
        return self

    def predict(self, X):
        return self.search.predict(X)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.search, os.path.join(out_dir, "model.joblib"))


def build_model(model_cfg: Dict[str, Any]):
    mtype = require(model_cfg, "type", str).lower()

    # Common grid keys are passed verbatim to GridSearchCV
    param_grid = model_cfg.get("param_grid")
    if mtype == "svm":
        # 兼容两种写法：
        # A) legacy: model: {type: svm, param_grid: {...}, probability: true, ...}
        # B) new:    model: {type: svm, svm: {...}, param_grid: {...}}
        svm_block = model_cfg.get("svm")
        svm_params = svm_block if isinstance(svm_block, dict) else model_cfg

        # 用你现在 svm.py 的 from_dict 解析（支持 kernel/C/gamma/max_iter/tol/solver 等）
        cfg = from_dict(svm_params)

        base_estimator = SVMModel(cfg).model  # 拿到真正的 sklearn estimator (SVC/LinearSVC)

        if param_grid:
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
            )
            return SklearnSearchAdapter(search)

        # 不做网格搜索就直接返回一个轻量适配器，提供 fit/predict/save
        class _SVMNoSearchAdapter:
            def __init__(self, est, cfg):
                self.est = est
                self.cfg = cfg

            def fit(self, X, y):
                self.est.fit(X, y)
                return self

            def predict(self, X):
                return self.est.predict(X)

            def save(self, out_dir: str):
                os.makedirs(out_dir, exist_ok=True)
                joblib.dump({"cfg": self.cfg, "model": self.est}, os.path.join(out_dir, "model.joblib"))

        return _SVMNoSearchAdapter(base_estimator, cfg)

    if mtype == "mlp":
        return MLPAdapter(
            MLPConfig(
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
                max_iter=int(model_cfg.get("max_iter", 500)),
                random_state=int(model_cfg.get("random_state", 42)),
            )
        )
    if mtype == "rf":
        return RFAdapter(
            RFConfig(
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
                random_state=int(model_cfg.get("random_state", 42)),
            )
        )

    raise ConfigError(f"Unsupported model.type: {mtype} (supported: svm/mlp/rf)")


def build_preprocess(pp_cfg: Dict[str, Any]) -> TabularPreprocessor:
    cfg = TabularPreprocessConfig(
        impute_strategy=str(pp_cfg.get("impute_strategy", "mean")),
        scale=bool(pp_cfg.get("scale", True)),
        select_k_best=pp_cfg.get("select_k_best", None),
        pca_n_components=pp_cfg.get("pca_n_components", None),
        augment=bool(pp_cfg.get("augment", True)),
        noise_std=float(pp_cfg.get("noise_std", 0.01)),
        time_jitter=float(pp_cfg.get("time_jitter", 0.02)),
    )
    return TabularPreprocessor(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True, help="Path to YAML/JSON config file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(get(cfg, "seed", 42))
    set_seed(seed)

    data_dir = str(require(cfg, "data_dir", str))
    emotions = require(cfg, "emotions", list)
    if not all(isinstance(x, str) for x in emotions):
        raise ConfigError("emotions must be a list of strings.")
    csv_files = list(get(cfg, "csv_files", list(DEFAULT_CSV_FILES)))

    split_cfg = get(cfg, "split", {})
    test_size = float(split_cfg.get("test_size", 0.30))
    val_size = float(split_cfg.get("val_size", 0.10))
    random_state = int(split_cfg.get("random_state", seed))

    out_cfg = get(cfg, "output", {})
    base_dir = str(out_cfg.get("base_dir", "outputs"))
    run_name = out_cfg.get("run_name", None)

    run = make_run_paths(base_dir=base_dir, run_name=run_name)
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))

    model = build_model(require(cfg, "model", dict))
    pp = build_preprocess(require(cfg, "preprocess", dict))

    # -------------------- dataset -------------------- #
    logger.info("🔎 Building dataset from %s", data_dir)
    X_all, y_all, skipped = build_tabular_dataset(data_dir=data_dir, emotions=emotions, csv_files=csv_files)
    logger.info("✅ Samples: %d | Features: %d", X_all.shape[0], X_all.shape[1])
    logger.info("📊 Label distribution: %s", dict(Counter(y_all)))
    if skipped:
        logger.info("⚠️ Skipped samples: %d (first 5 shown)", len(skipped))
        for s in skipped[:5]:
            logger.info("   - %s", s)

    # -------------------- split -------------------- #
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    # val_size is proportion of temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    logger.info("Split: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # -------------------- preprocess -------------------- #
    X_train_t, y_train_t = pp.fit_transform_train(X_train, y_train)
    X_val_t = pp.transform(X_val)
    X_test_t = pp.transform(X_test)

    # -------------------- fit -------------------- #
    logger.info("⏳ Training model...")
    model.fit(X_train_t, y_train_t)

    best_params = getattr(model, "best_params_", None)
    if best_params:
        logger.info("✅ Best params: %s", best_params)

    # -------------------- evaluate (test) -------------------- #
    y_pred = model.predict(X_test_t)
    m = classification_metrics(y_test, y_pred, class_names=emotions)
    logger.info("🎯 Test accuracy: %.4f", m["accuracy"])

    # -------------------- save -------------------- #
    model.save(run.models_dir)
    pp.save(run.artifacts_dir)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=emotions,
        save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
        normalize="true",
        title="Confusion Matrix (Normalized)",
    )

    out = {
        "accuracy": m["accuracy"],
        "report": m["report"],
        "best_params": best_params,
        "emotions": emotions,
        "seed": seed,
        "config_path": os.path.abspath(args.config),
    }
    with open(os.path.join(run.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved to: %s", run.run_dir)


if __name__ == "__main__":
    main()
