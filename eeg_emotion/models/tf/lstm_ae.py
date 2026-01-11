from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout,
    Bidirectional, Masking, BatchNormalization, Lambda
)
from tensorflow.keras.optimizers import Adam


@dataclass(frozen=True)
class LSTMAEConfig:
    latent_dim: int = 128
    dropout_rate: float = 0.2
    lr: float = 1e-3


def build_lstm_autoencoder(input_shape, cfg: LSTMAEConfig) -> Model:
    time_steps, features = input_shape
    inputs = Input(shape=(time_steps, features), name="ae_input")
    x = Masking(mask_value=0.0, name="masking")(inputs)

    x = Bidirectional(LSTM(cfg.latent_dim // 2, activation="tanh", return_sequences=True), name="enc_bi_1")(x)
    x = Dropout(cfg.dropout_rate, name="enc_drop_1")(x)
    x = Bidirectional(LSTM(cfg.latent_dim // 2, activation="tanh", return_sequences=True), name="enc_bi_2")(x)
    x = Dropout(cfg.dropout_rate, name="enc_drop_2")(x)

    e = Dense(1, activation="tanh", name="att_score_dense")(x)
    e = Lambda(lambda z: tf.squeeze(z, axis=-1), name="att_squeeze")(e)
    alphas = Lambda(lambda z: tf.nn.softmax(z, axis=1), name="attention_weights")(e)
    alphas_exp = Lambda(lambda z: tf.expand_dims(z, axis=-1), name="alphas_exp")(alphas)
    context = Lambda(lambda args: tf.reduce_sum(args[0] * args[1], axis=1), name="att_context")([x, alphas_exp])

    encoded = Dense(cfg.latent_dim, activation="tanh", name="encoder_output")(context)

    y = RepeatVector(time_steps, name="repeat_vec")(encoded)
    y = LSTM(cfg.latent_dim, activation="tanh", return_sequences=True, name="decoder_lstm")(y)
    outputs = TimeDistributed(Dense(features), name="decoder_out")(y)

    autoencoder = Model(inputs, outputs, name="lstm_autoencoder_attention")
    autoencoder.compile(optimizer=Adam(cfg.lr), loss="mse", metrics=["mae"])
    return autoencoder
