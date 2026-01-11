from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


@dataclass(frozen=True)
class LSTMClassifierConfig:
    dropout1: float = 0.4
    dropout2: float = 0.3
    dropout3: float = 0.2
    label_smoothing: float = 0.05


def build_lstm_classifier(input_dim: int, num_classes: int, lr) -> tf.keras.Model:
    cfg = LSTMClassifierConfig()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(cfg.dropout1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(cfg.dropout2),
        Dense(64, activation="relu"),
        Dropout(cfg.dropout3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
        metrics=["accuracy"],
    )
    return model
