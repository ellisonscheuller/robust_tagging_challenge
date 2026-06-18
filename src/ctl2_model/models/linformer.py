from hgq.layers import QDense, QAdd, QSum, QLinformerAttention
from hgq.constraints import MinMax
from hgq.regularizers import MonoL1
from hgq.config import QuantizerConfigScope, LayerConfigScope
import keras


def get_linformer(
    conf=None,
    n_constituents=100,
    n_features=12,
    latent_dim=12,
    n_classes=5,
    ff_dim=32,
    num_heads=2,
    proj_k=8,
    beta0=0.0,
    pt_eta_phi=False,
):
    if conf is not None:
        n_constituents = getattr(conf, 'n_constituents', n_constituents)
        n_features     = getattr(conf, 'n_features', n_features)
        latent_dim     = getattr(conf, 'latent_dim', latent_dim)
        pt_eta_phi     = getattr(conf, 'pt_eta_phi', pt_eta_phi)
    n = 3 if pt_eta_phi else n_features
    N = n_constituents

    scope0 = QuantizerConfigScope(k0=1, b0=8, i0=1, br=MonoL1(1e-8), overflow_mode="WRAP")
    scope1 = QuantizerConfigScope(
        place="datalane", k0=1, f0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8)
    )
    mhaconf = QuantizerConfigScope(
        k0=1, i0=1, f0=6, round_mode="RND", overflow_mode="SAT", bc=MinMax(1, 8)
    )
    betascope = LayerConfigScope(enable_ebops=True, beta0=beta0)

    with betascope, scope0, scope1:
        inp = keras.layers.Input((N, n))

        emb = QDense(ff_dim, activation="relu")(inp)

        with mhaconf:
            x = QLinformerAttention(
                num_heads,
                key_dim=ff_dim // num_heads,
                lin_kv_proj_dim=proj_k,
                name="attention1",
                dropout=0,
            )(emb, emb, emb)
        res = QAdd()([x, emb])

        with mhaconf:
            x = QLinformerAttention(
                num_heads,
                key_dim=ff_dim // num_heads,
                lin_kv_proj_dim=proj_k,
                name="attention2",
                dropout=0,
            )(res, res, res)
        res = QAdd()([x, res])

        x = QSum(axes=1)(res) / N

        x = QDense(ff_dim, activation="relu")(x)
        x = QDense(ff_dim, activation="relu")(x)

        latent = QDense(latent_dim, name="latent")(x)
        cls_hid = QDense(ff_dim, activation="relu", name="cls_hidden")(latent)
        cls_out = QDense(n_classes, activation="softmax", name="class_probs")(cls_hid)

    return keras.Model(inputs=inp, outputs={"latent": latent, "class_probs": cls_out})
