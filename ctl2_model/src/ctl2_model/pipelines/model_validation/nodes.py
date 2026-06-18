"""model_validation pipeline nodes."""
import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctl2_model.evaluation.anomaly_score import (
    ClassifierAnomalyScore,
    get_rate_curve,
    efficiency_at_rate,
)
from ctl2_model.evaluation.plots import plot_rate_curves
from ctl2_model.utils.plotting import (
    plot_anomaly_score_distributions,
    plot_efficiency_table,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
})

MINBIAS_COLOR = "#1B3A5C"
SIGNAL_COLORS = [
    "#D62728", "#2CA02C", "#FF7F0E", "#9467BD",
    "#8C564B", "#17BECF", "#E377C2", "#BCBD22",
]
BG_COLORS = {"qcd": "#1f77b4", "top": "#2ca02c", "dy": "#9467bd"}


def _run_pca(latents, n_components=2):
    return PCA(n_components=n_components).fit_transform(latents)


def _plot_pca_train(fig, latents_train, y_train, label_names):
    pca_all = _run_pca(latents_train)
    ax = fig.subplots()
    classes = np.unique(y_train)
    for c in classes:
        mask = y_train == c
        name = label_names.get(str(c), f"class_{c}")
        color = BG_COLORS.get(name, SIGNAL_COLORS[c % len(SIGNAL_COLORS)])
        ax.scatter(
            pca_all[mask, 0], pca_all[mask, 1],
            s=4, alpha=0.4, label=name, color=color, edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent Space PCA — Training Set")
    ax.legend(framealpha=0.3, markerscale=3)
    return fig


def _plot_pca_signals(fig, latents_dict, signal_names):
    n = len(signal_names)
    axes = fig.subplots(1, n, squeeze=False)[0]
    for ax, sig_name in zip(axes, signal_names):
        sig_lat = latents_dict.get(sig_name)
        mb_lat = latents_dict.get("minbias")
        if sig_lat is None or mb_lat is None:
            continue
        combined = np.vstack([mb_lat, sig_lat])
        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(combined)
        n_mb = len(mb_lat)
        ax.scatter(
            pca_out[:n_mb, 0], pca_out[:n_mb, 1],
            s=12, alpha=0.7, label="minbias", color=MINBIAS_COLOR, edgecolors="none",
        )
        ax.scatter(
            pca_out[n_mb:, 0], pca_out[n_mb:, 1],
            s=4, alpha=0.3, label=sig_name, color=SIGNAL_COLORS[signal_names.index(sig_name) % len(SIGNAL_COLORS)],
            edgecolors="none",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"minbias vs {sig_name}")
        ax.legend(framealpha=0.3, markerscale=2)
    return fig


def validate_model(
    trained_model,
    processed_X_train,
    processed_y_train,
    processed_X_val,
    processed_y_val,
    processed_X_eval,
    processed_y_eval,
    model_validation,
):
    rate_threshold = model_validation.get("rate_threshold", 16.0)
    label_names = {str(k): v for k, v in model_validation.get("label_names", {"0": "qcd", "1": "top", "2": "minbias", "3": "dy"}).items()}
    eval_names = {str(k): v for k, v in model_validation.get("eval_names", {1: "suep", 2: "hh4b", 4: "zprime", 5: "svj_m250", 3: "vbf_hinv"}).items()}

    scorer = ClassifierAnomalyScore(trained_model, normal_class_idx=2)

    scores_val = scorer(processed_X_val)
    minbias_mask_val = processed_y_val == 2
    scores_bg = scores_val[minbias_mask_val]

    metrics = {}
    scores_dict = {"minbias": scores_bg}
    signal_names = []
    latents_dict = {}

    train_mask = processed_y_val != 2
    for sig_label in np.unique(processed_y_val[train_mask]):
        sig_mask = processed_y_val == sig_label
        scores_sig = scores_val[sig_mask]
        sig_name = eval_names.get(str(sig_label), label_names.get(str(sig_label), f"class_{sig_label}"))
        eff = efficiency_at_rate(scores_sig, scores_bg, rate_threshold)
        metrics[f"{sig_name}_eff@{rate_threshold}Hz"] = float(eff)
        scores_dict[sig_name] = scores_sig
        signal_names.append(sig_name)

    eval_labels = np.unique(processed_y_eval)
    for sig_label in eval_labels:
        sig_mask = processed_y_eval == sig_label
        scores_sig = scorer(processed_X_eval[sig_mask])
        sig_name = eval_names.get(str(sig_label), f"eval_class_{sig_label}")
        eff = efficiency_at_rate(scores_sig, scores_bg, rate_threshold)
        metrics[f"{sig_name}_eff@{rate_threshold}Hz"] = float(eff)
        scores_dict[sig_name] = scores_sig
        signal_names.append(sig_name)

    fig_rate = plot_rate_curves(
        [scores_dict],
        signal_names,
        scores_normal="minbias",
        save_path="data/05_validation/rate_curves.png",
    )

    latents_train = trained_model.predict(processed_X_train, batch_size=2048, verbose=0)["latent"]
    minbias_mask_train = processed_y_train == 2
    mb_lat = latents_train[minbias_mask_train]

    fig_pca_train = plt.figure(figsize=(7, 6))
    _plot_pca_train(fig_pca_train, latents_train, processed_y_train, label_names)
    fig_pca_train.savefig("data/05_validation/pca_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pca_train)

    sig_to_plot = ["suep", "zprime", "hh4b", "svj_m250"]
    latents_for_pca = {"minbias": mb_lat}
    for sig_name in sig_to_plot:
        sig_label = next((int(k) for k, v in eval_names.items() if v == sig_name), None)
        if sig_label is None:
            continue
        mask = processed_y_eval == sig_label
        if mask.sum() == 0:
            continue
        sig_lat = trained_model.predict(processed_X_eval[mask], batch_size=2048, verbose=0)["latent"]
        latents_for_pca[sig_name] = sig_lat

    fig_pca_sigs = plt.figure(figsize=(7 * len(sig_to_plot), 6))
    _plot_pca_signals(fig_pca_sigs, latents_for_pca, sig_to_plot)
    fig_pca_sigs.savefig("data/05_validation/pca_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pca_sigs)

    for k, v in metrics.items():
        safe_k = k.replace("@", "_at_").replace(" ", "_").replace("/", "_per_")
        mlflow.log_metric(safe_k, v)

    val_dir = Path("data/05_validation")
    val_dir.mkdir(parents=True, exist_ok=True)

    score_dist_path = plot_anomaly_score_distributions(
        scores_dict, signal_names, scores_normal="minbias",
        save_path=str(val_dir / "anomaly_scores.png"),
    )
    eff_table_path = plot_efficiency_table(
        metrics,
        save_path=str(val_dir / "efficiency_table.png"),
    )

    mlflow.log_metric("n_signals_evaluated", len(signal_names))
    mlflow.log_artifact(score_dist_path)
    mlflow.log_artifact(eff_table_path)
    mlflow.log_artifact(str(val_dir / "pca_train.png"))
    mlflow.log_artifact(str(val_dir / "pca_signals.png"))
    mlflow.log_artifact(str(val_dir / "rate_curves.png"))

    with open(val_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(str(val_dir / "metrics.json"))

    rate_curves_serial = {}
    for name, scores in scores_dict.items():
        rate_curves_serial[name] = scores.tolist() if hasattr(scores, "tolist") else scores

    return metrics, rate_curves_serial, fig_pca_train, fig_pca_sigs
