import keras
import tensorflow as tf

from .projector import build_projector


def info_nce_loss(embeddings, labels, temperature=0.07):
    embeddings = embeddings / (keras.ops.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = keras.ops.matmul(embeddings, keras.ops.transpose(embeddings)) / temperature
    N = keras.ops.shape(sim)[0]
    mask_self = 1 - keras.ops.eye(N)
    sim = sim * mask_self - 1e9 * (1 - mask_self)
    labels_row = keras.ops.expand_dims(labels, 1)
    labels_col = keras.ops.expand_dims(labels, 0)
    pos_mask = keras.ops.cast(labels_row == labels_col, "float32") * mask_self
    log_sm = sim - keras.ops.logsumexp(sim, axis=1, keepdims=True)
    loss = -keras.ops.sum(log_sm * pos_mask, axis=1) / (keras.ops.sum(pos_mask, axis=1) + 1e-8)
    return keras.ops.mean(loss)


class ContrastiveHGQEncoder(keras.Model):
    def __init__(
        self,
        encoder,
        proj_dim,
        mse_weight=0.05,
        temperature=0.07,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projector = build_projector(None, proj_dim)
        self.mse_weight = mse_weight
        self.temperature = temperature
        self.nce_weight = 0.0

        self._metric_ce = keras.metrics.Mean(name="train_ce")
        self._metric_nce = keras.metrics.Mean(name="train_nce")

    @property
    def layers(self):
        return super().layers + self.encoder.layers

    def save(self, filepath, **kwargs):
        self.encoder.save(filepath, **kwargs)

    def call(self, inputs, training=False):
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        return self.encoder(x, training=training)["latent"]

    @property
    def metrics(self):
        base = super().metrics
        return base + [self._metric_ce, self._metric_nce]

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        x_in, x_aug = x
        labels = keras.ops.cast(y, "int32")

        out_in = self.encoder(x_in, training=True)
        out_aug = self.encoder(x_aug, training=True)

        latent = out_in["latent"]
        latent_aug = out_aug["latent"]
        class_probs = out_in["class_probs"]

        embeddings = self.projector(latent, training=True)

        loss_nce = info_nce_loss(embeddings, labels, self.temperature)

        class_weights = tf.constant([1.0, 1.0, 50.0 / 400.0, 1.0, 1.0], dtype=tf.float32)
        loss_ce = keras.ops.mean(
            keras.losses.sparse_categorical_crossentropy(labels, class_probs, from_logits=False)
            * keras.ops.take(class_weights, labels)
        )

        loss_mse = keras.ops.mean(keras.ops.sum((latent - latent_aug) ** 2, axis=-1))

        loss_hgq = keras.ops.sum(self.losses)

        self._metric_ce.update_state(loss_ce)
        self._metric_nce.update_state(loss_nce)

        return self.nce_weight * loss_nce + loss_ce + self.mse_weight * loss_mse + loss_hgq
