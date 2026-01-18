from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout,
    Bidirectional, Masking, Lambda
)
from tensorflow.keras.optimizers import Adam


@dataclass(frozen=True)
class LSTMAEConfig:
    latent_dim: int = 128
    enc_units: int | None = None
    enc_layers: int = 2
    enc_dropout: float = 0.2

    use_bidirectional_decoder: bool = True
    dec_units: int | None = None
    dec_dropout: float = 0.2

    lr: float = 1e-3


def build_lstm_autoencoder(input_shape, cfg: LSTMAEConfig) -> Model:
    time_steps, features = input_shape
    inputs = Input(shape=(time_steps, features), name="ae_input")
    x = Masking(mask_value=0.0, name="masking")(inputs)

    enc_units = int(cfg.enc_units) if cfg.enc_units is not None else int(max(8, cfg.latent_dim // 2))
    for i in range(int(cfg.enc_layers)):
        x = Bidirectional(
            LSTM(enc_units, activation="tanh", return_sequences=True),
            name=f"enc_bi_{i+1}",
        )(x)
        if cfg.enc_dropout and cfg.enc_dropout > 0:
            x = Dropout(float(cfg.enc_dropout), name=f"enc_drop_{i+1}")(x)

    e = Dense(1, activation="tanh", name="att_score_dense")(x)
    e = Lambda(lambda z: tf.squeeze(z, axis=-1), name="att_squeeze")(e)
    alphas = Lambda(lambda z: tf.nn.softmax(z, axis=1), name="attention_weights")(e)
    alphas_exp = Lambda(lambda z: tf.expand_dims(z, axis=-1), name="alphas_exp")(alphas)
    context = Lambda(lambda args: tf.reduce_sum(args[0] * args[1], axis=1), name="att_context")([x, alphas_exp])

    encoded = Dense(int(cfg.latent_dim), activation="tanh", name="encoder_output")(context)

    y = RepeatVector(time_steps, name="repeat_vec")(encoded)
    dec_units = int(cfg.dec_units) if cfg.dec_units is not None else int(cfg.latent_dim)

    if bool(cfg.use_bidirectional_decoder):
        y = Bidirectional(
            LSTM(max(8, dec_units // 2), activation="tanh", return_sequences=True),
            name="dec_bi_lstm",
        )(y)
    else:
        y = LSTM(dec_units, activation="tanh", return_sequences=True, name="dec_lstm")(y)

    if cfg.dec_dropout and cfg.dec_dropout > 0:
        y = Dropout(float(cfg.dec_dropout), name="dec_drop")(y)

    outputs = TimeDistributed(Dense(features), name="decoder_out")(y)

    autoencoder = Model(inputs, outputs, name="lstm_autoencoder_attention_bilstm")
    autoencoder.compile(optimizer=Adam(cfg.lr), loss="mse", metrics=["mae"])
    return autoencoder
