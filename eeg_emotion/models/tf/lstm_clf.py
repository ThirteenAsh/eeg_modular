from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


@dataclass(frozen=True)
class MLPClassifierConfig:
    dropout1: float = 0.4
    dropout2: float = 0.3
    dropout3: float = 0.2
    label_smoothing: float = 0.05


@dataclass(frozen=True)
class BiLSTMClassifierConfig:
    lstm_units: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    recurrent_dropout: float = 0.0
    pooling: str = "avgmax"  # 'avg' | 'max' | 'avgmax'
    label_smoothing: float = 0.05


def build_encoded_mlp_classifier(input_dim: int, num_classes: int, lr) -> tf.keras.Model:
    cfg = MLPClassifierConfig()
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


def build_sequence_bilstm_classifier(input_shape, num_classes: int, lr, cfg: BiLSTMClassifierConfig | None = None) -> tf.keras.Model:
    cfg = cfg or BiLSTMClassifierConfig()
    t, f = input_shape

    inputs = Input(shape=(t, f), name="seq_input")
    x = inputs
    for i in range(int(cfg.num_layers)):
        x = Bidirectional(
            LSTM(
                int(cfg.lstm_units),
                return_sequences=True,
                activation="tanh",
                recurrent_dropout=float(cfg.recurrent_dropout),
            ),
            name=f"bilstm_{i+1}",
        )(x)
        if cfg.dropout and cfg.dropout > 0:
            x = Dropout(float(cfg.dropout), name=f"drop_{i+1}")(x)

    if cfg.pooling == "avg":
        x = GlobalAveragePooling1D(name="gap")(x)
    elif cfg.pooling == "max":
        x = GlobalMaxPooling1D(name="gmp")(x)
    else:
        x_avg = GlobalAveragePooling1D(name="gap")(x)
        x_max = GlobalMaxPooling1D(name="gmp")(x)
        x = Concatenate(name="avgmax")([x_avg, x_max])

    x = Dense(128, activation="relu", name="head_dense")(x)
    x = Dropout(float(cfg.dropout), name="head_drop")(x)
    outputs = Dense(num_classes, activation="softmax", name="out")(x)

    model = Model(inputs, outputs, name="sequence_bilstm_classifier")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=float(cfg.label_smoothing)),
        metrics=["accuracy"],
    )
    return model
