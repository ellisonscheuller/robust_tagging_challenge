from __future__ import annotations

import keras
import numpy as np
from pathlib import Path


def load_encoder(path: str | Path):
    keras.config.enable_unsafe_deserialization()
    return keras.models.load_model(path, compile=False)


def truncate_at_cls_hidden(encoder):
    latent = encoder.get_layer("cls_hidden").input
    new_input = encoder.input
    classifier = keras.Model(inputs=new_input, outputs=latent)
    return classifier


def clean_model_for_hls4ml(model):
    new_input = keras.layers.Input(shape=model.input_shape[1:])
    outputs = model(new_input, training=False)
    clean = keras.Model(inputs=new_input, outputs=outputs)
    clean.set_weights(model.get_weights())
    return clean


def save_model_surgery_report(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
