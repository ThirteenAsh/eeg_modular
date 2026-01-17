from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model

from eeg_emotion.config.loader import load_config, require, get
from eeg_emotion.features.sequence.extract import SequenceFeatureConfig, extract_all_features
from eeg_emotion.features.sequence.augment import (
    augment_class_samples, mixup_augment, compute_sample_stats, apply_gaussian_noise_batch
)
from eeg_emotion.preprocess.sequence import SequencePreprocessConfig, SequencePreprocessor
from eeg_emotion.models.tf.lstm_ae import LSTMAEConfig, build_lstm_autoencoder
from eeg_emotion.models.tf.lstm_clf import (
    build_encoded_mlp_classifier, build_sequence_bilstm_classifier, BiLSTMClassifierConfig
)
from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()


def plot_training_curves(history, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    if "accuracy" in history.history:
        ax2.plot(history.history["accuracy"], label="train_acc")
    if "val_accuracy" in history.history:
        ax2.plot(history.history["val_accuracy"], label="val_acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_cfg = get(cfg, "output", {})
    run = make_run_paths(base_dir=str(out_cfg.get("base_dir", "outputs")), run_name=out_cfg.get("run_name"))
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))

    data_dir = str(require(cfg, "data_dir", str))
    emotions = require(cfg, "emotions", list)
    csv_files = require(cfg, "csv_files", list)
    time_steps = int(get(cfg, "time_steps", 128))

    X, y = extract_all_features(SequenceFeatureConfig(
        data_dir=data_dir,
        emotions=list(emotions),
        csv_files=list(csv_files),
        time_steps=time_steps,
        min_cols_per_file=int(get(cfg, "min_cols_per_file", 10)),
    ))
    logger.info("Extracted X=%s y=%s labels=%s", X.shape, y.shape, dict(Counter(y)))

    split_cfg = get(cfg, "split", {})
    test_size = float(split_cfg.get("test_size", 0.30))
    seed = int(split_cfg.get("seed", 42))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    logger.info("Split train=%s test=%s", X_train.shape, X_test.shape)

    aug_cfg = get(cfg, "augment", {})
    aug_noise = get(aug_cfg, "noise", {}) or {}
    noise_mean = float(aug_noise.get("mean", 0.0))
    noise_std = float(aug_noise.get("std", 0.01))

    if bool(aug_cfg.get("enabled", True)):
        sad_times = int(aug_cfg.get("sad_times", 3))
        other_times = int(aug_cfg.get("other_times", 3))
        X_train, y_train = augment_class_samples(
            X_train, y_train, target_labels=[1],
            augment_times=sad_times,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        X_train, y_train = augment_class_samples(
            X_train, y_train, target_labels=[0, 2],
            augment_times=other_times,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        logger.info("After train-only augmentation labels=%s", dict(Counter(y_train)))

    pp_cfg = get(cfg, "preprocess", {})
    pp = SequencePreprocessor(SequencePreprocessConfig(
        impute_strategy=str(pp_cfg.get("impute_strategy", "mean")),
        scale=bool(pp_cfg.get("scale", True)),
        pca_n_components=pp_cfg.get("pca_n_components", 64),
    ))
    X_train = pp.fit_transform(X_train)
    X_test = pp.transform(X_test)
    pp.save(run.artifacts_dir)
    logger.info("After preprocess train=%s test=%s", X_train.shape, X_test.shape)

    num_classes = len(emotions)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    # Gaussian noise injection (train-only, after preprocess)
    noise_cfg = get(cfg, "gaussian_noise", {}) or {}
    noise_enabled = bool(noise_cfg.get("enabled", False))
    noise_apply_to = set(noise_cfg.get("apply_to", ["ae", "clf"]))
    noise_mean2 = float(noise_cfg.get("mean", 0.0))
    noise_std2 = float(noise_cfg.get("std", 0.01))

    X_train_for_ae = X_train
    X_train_for_clf_seq = X_train
    if noise_enabled and noise_std2 > 0:
        if "ae" in noise_apply_to:
            X_train_for_ae = apply_gaussian_noise_batch(X_train_for_ae, mean=noise_mean2, std=noise_std2)
        if "clf" in noise_apply_to:
            X_train_for_clf_seq = apply_gaussian_noise_batch(X_train_for_clf_seq, mean=noise_mean2, std=noise_std2)
        logger.info("Applied gaussian noise: mean=%.4f std=%.4f apply_to=%s", noise_mean2, noise_std2, sorted(list(noise_apply_to)))

    # AE (BiLSTM + configurable dropout)
    ae_cfg = get(cfg, "autoencoder", {})
    ae_epochs = int(ae_cfg.get("epochs", 100))
    latent_dim = int(ae_cfg.get("latent_dim", 128))

    autoencoder = build_lstm_autoencoder(
        (time_steps, X_train_for_ae.shape[2]),
        LSTMAEConfig(
            latent_dim=latent_dim,
            enc_units=ae_cfg.get("enc_units"),
            enc_layers=int(ae_cfg.get("enc_layers", 2)),
            enc_dropout=float(ae_cfg.get("enc_dropout", ae_cfg.get("dropout_rate", 0.2))),
            use_bidirectional_decoder=bool(ae_cfg.get("use_bidirectional_decoder", True)),
            dec_units=ae_cfg.get("dec_units"),
            dec_dropout=float(ae_cfg.get("dec_dropout", ae_cfg.get("dropout_rate", 0.2))),
            lr=float(ae_cfg.get("lr", 1e-3)),
        )
    )

    ae_callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=int(ae_cfg.get("lr_patience", 8)), min_lr=float(ae_cfg.get("min_lr", 1e-6)), verbose=1),
        EarlyStopping(monitor="val_loss", patience=int(ae_cfg.get("early_stop_patience", 20)), restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(run.models_dir, "best_autoencoder.weights.h5"), save_best_only=True, save_weights_only=True, monitor="val_loss", verbose=1),
    ]

    logger.info("Training AE...")
    history_ae = autoencoder.fit(
        X_train_for_ae, X_train_for_ae,
        epochs=ae_epochs,
        batch_size=int(ae_cfg.get("batch_size", 32)),
        validation_split=float(ae_cfg.get("val_split", 0.1)),
        verbose=2,
        callbacks=ae_callbacks,
    )
    plot_training_curves(history_ae, os.path.join(run.figures_dir, "training_curves_ae.png"))

    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("encoder_output").output)
    encoder_model.save(os.path.join(run.models_dir, "trained_encoder.keras"))

    X_train_enc = encoder_model.predict(X_train, verbose=0)
    X_test_enc = encoder_model.predict(X_test, verbose=0)

    stats_cfg = get(cfg, "sample_stats", {})
    if bool(stats_cfg.get("enabled", True)):
        X_train_enc = np.concatenate([X_train_enc, compute_sample_stats(X_train)], axis=1)
        X_test_enc = np.concatenate([X_test_enc, compute_sample_stats(X_test)], axis=1)

    # Classifier: BiLSTM (sequence) OR MLP (encoded)
    clf_cfg = get(cfg, "classifier", {})
    clf_mode = str(clf_cfg.get("mode", "bilstm")).lower()
    clf_epochs = int(clf_cfg.get("epochs", 200))
    clf_batch = int(clf_cfg.get("batch_size", 16))
    initial_lr = float(clf_cfg.get("initial_lr", 1e-3))
    use_cosine_decay = bool(clf_cfg.get("use_cosine_decay", True))

    if use_cosine_decay:
        steps_per_epoch = max(1, int(np.ceil((X_train_for_clf_seq.shape[0] if clf_mode == "bilstm" else X_train_enc.shape[0]) / clf_batch)))
        decay_steps = int(clf_epochs * steps_per_epoch)
        lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        lr_is_schedule = True
    else:
        lr = initial_lr
        lr_is_schedule = False

    mix_cfg = get(cfg, "mixup", {}) or {}
    mix_ratio = float(mix_cfg.get("augment_ratio", 1.0))
    mix_alpha = float(mix_cfg.get("alpha", 0.3))
    use_mixup = bool(mix_cfg.get("enabled", True)) and mix_ratio > 0

    if clf_mode == "bilstm":
        bilstm_cfg = BiLSTMClassifierConfig(
            lstm_units=int(clf_cfg.get("lstm_units", 128)),
            num_layers=int(clf_cfg.get("num_layers", 2)),
            dropout=float(clf_cfg.get("dropout", 0.3)),
            recurrent_dropout=float(clf_cfg.get("recurrent_dropout", 0.0)),
            pooling=str(clf_cfg.get("pooling", "avgmax")),
            label_smoothing=float(clf_cfg.get("label_smoothing", 0.05)),
        )
        Xtr, Ytr = X_train_for_clf_seq, y_train_cat
        if use_mixup:
            Xtr, Ytr = mixup_augment(Xtr, Ytr, alpha=mix_alpha, augment_ratio=mix_ratio)
            logger.info("After mixup (seq) X=%s y=%s", Xtr.shape, Ytr.shape)
        classifier = build_sequence_bilstm_classifier((time_steps, X_train_for_clf_seq.shape[2]), num_classes, lr=lr, cfg=bilstm_cfg)
        X_test_input = X_test
        y_eval = y_test
    else:
        Xtr, Ytr = X_train_enc, y_train_cat
        if use_mixup:
            Xtr, Ytr = mixup_augment(Xtr, Ytr, alpha=mix_alpha, augment_ratio=mix_ratio)
            logger.info("After mixup (enc) X=%s y=%s", Xtr.shape, Ytr.shape)
        classifier = build_encoded_mlp_classifier(X_train_enc.shape[1], num_classes, lr=lr)
        X_test_input = X_test_enc
        y_eval = y_test

    use_class_weight = (not use_mixup) or float(mix_cfg.get("augment_ratio", 0)) == 0
    class_weight_dict = None
    if use_class_weight:
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train), cw)}

    clf_callbacks = [
        EarlyStopping(monitor="val_loss", patience=int(clf_cfg.get("early_stop_patience", 30)), restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(run.models_dir, "best_classifier.weights.h5"), save_best_only=True, save_weights_only=True, monitor="val_loss", verbose=1),
    ]
    if not lr_is_schedule:
        clf_callbacks.insert(0, ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(clf_cfg.get("lr_patience", 10)),
            min_lr=float(clf_cfg.get("min_lr", 1e-6)),
            verbose=1,
        ))

    logger.info("Training classifier... mode=%s use_cosine_decay=%s", clf_mode, use_cosine_decay)
    history_clf = classifier.fit(
        Xtr, Ytr,
        validation_split=float(clf_cfg.get("val_split", 0.2)),
        epochs=clf_epochs,
        batch_size=clf_batch,
        class_weight=class_weight_dict,
        callbacks=clf_callbacks,
        verbose=2,
    )
    plot_training_curves(history_clf, os.path.join(run.figures_dir, "training_curves_clf.png"))

    classifier.load_weights(os.path.join(run.models_dir, "best_classifier.weights.h5"))
    y_pred = np.argmax(classifier.predict(X_test_input, verbose=0), axis=1)

    m = classification_metrics(y_eval, y_pred, class_names=list(emotions))
    logger.info("Test accuracy: %.4f", m["accuracy"])

    save_confusion_matrix(
        y_true=y_eval,
        y_pred=y_pred,
        class_names=list(emotions),
        save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
        normalize="true",
        title="Confusion Matrix (Normalized)",
    )

    out = {
        "accuracy": m["accuracy"],
        "report": m["report"],
        "best_params": {
            "classifier_mode": clf_mode,
            "time_steps": time_steps,
            "latent_dim": latent_dim,
            "pca_n_components": pp_cfg.get("pca_n_components", 64),
            "mixup_alpha": mix_alpha,
            "mixup_ratio": mix_ratio,
            "augment": bool(aug_cfg.get("enabled", True)),
            "gaussian_noise": {"enabled": noise_enabled, "apply_to": sorted(list(noise_apply_to)), "std": noise_std2},
            "use_cosine_decay": use_cosine_decay,
        },
        "config_path": os.path.abspath(args.config),
    }
    with open(os.path.join(run.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved to %s", run.run_dir)


if __name__ == "__main__":
    main()
