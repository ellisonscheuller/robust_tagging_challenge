import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def plot_multiplicity(mults: dict, save_path: str = "multiplicity_plot.png"):
    plt.figure(figsize=(10, 6))
    colors = {"QCD": SIGNAL_COLORS[0], "Top": SIGNAL_COLORS[1],
              "W+Jets": SIGNAL_COLORS[2], "DY": SIGNAL_COLORS[3],
              "SUEP": SIGNAL_COLORS[4]}

    for label, data in mults.items():
        c = MINBIAS_COLOR if label.lower() == "minbias" else colors.get(label)
        plt.hist(data, bins=np.arange(0, 101, 1), alpha=0.8, label=label,
                 density=True, histtype="step", linewidth=2, color=c)

    plt.xlabel("Particle Multiplicity")
    plt.ylabel("Probability Density")
    plt.title("Particle Multiplicity Discrimination")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


def plot_pca_latents(latents_dict, save_path: str = "pca_latents.png"):
    embeddings_list, slice_map, start_idx = [], {}, 0
    for key, emb in latents_dict.items():
        embeddings_list.append(emb)
        N = emb.shape[0]
        slice_map[key] = (start_idx, start_idx + N)
        start_idx += N

    from sklearn.decomposition import PCA
    combined = np.vstack(embeddings_list)
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    plt.figure(figsize=(8, 6))
    color_iter = iter(SIGNAL_COLORS)
    for (key, (start, end)) in slice_map.items():
        emb_2d = combined_2d[start:end]
        if key.lower() == "minbias":
            c, a, s = MINBIAS_COLOR, 0.7, 8
        else:
            c, a, s = next(color_iter), 0.3, 3
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=s, alpha=a, label=key, color=c, edgecolors="none")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Latent Representations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")
    return plt.gcf()


def plot_rate_curves(
    score_dicts: list,
    score_names: list,
    scores_normal,
    save_path: str = "rate_curves.png",
    target_rate_khz: float = 10.0,
):
    n_plots = len(score_dicts)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))

    from ctl2_model.evaluation.anomaly_score import get_rate_curve, BUNCH_CROSSING_RATE, FILLED_BUNCHES

    if n_plots == 1:
        axes = [axes]

    for ax, score_dict, title in zip(axes, score_dicts, score_names):
        s_normal = score_dict[scores_normal]
        thresholds = np.percentile(s_normal, np.linspace(0, 100, 10000))
        colors = SIGNAL_COLORS[:len(score_dict) - 1]

        for (key, s), color in zip(
            [(k, v) for k, v in score_dict.items() if k != scores_normal], colors
        ):
            bg_rate_khz, sig_eff = get_rate_curve(s, s_normal, thresholds)
            order = np.argsort(bg_rate_khz)
            ax.plot(bg_rate_khz[order], sig_eff[order], label=key, color=color, lw=2)

            target_fpr = target_rate_khz * 1e3 / (BUNCH_CROSSING_RATE * FILLED_BUNCHES)
            threshold = thresholds[
                np.argmin(
                    np.abs(
                        np.array([(s_normal > t).mean() for t in thresholds])
                        - target_fpr
                    )
                )
            ]
            eff = (s > threshold).mean()
            ax.annotate(
                f"{eff:.3f}",
                xy=(target_rate_khz, eff),
                xytext=(target_rate_khz + 2, eff),
                color=color,
                fontsize=9,
                va="center",
            )

        ax.axvline(target_rate_khz, color="gray", linestyle="--", lw=1.2, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Background Rate (kHz)", fontsize=12)
        ax.set_ylabel("Signal Efficiency", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.1, 40e3)
        ax.set_ylim(1e-3, 1)
        ax.legend(framealpha=0.3, fontsize=10)
        ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")
    return fig
