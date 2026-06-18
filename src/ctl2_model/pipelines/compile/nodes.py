"""compile pipeline nodes."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from hgq.utils import trace_minmax

from ctl2_model.synthesis.model_surgery import (
    truncate_at_cls_hidden,
    clean_model_for_hls4ml,
    save_model_surgery_report,
)
from ctl2_model.synthesis.trace_and_synth import (
    trace_comb,
    generate_rtl,
    generate_hls_hls4ml,
)
from ctl2_model.synthesis.validate import compare_predictions
from ctl2_model.evaluation.anomaly_score import (
    ClassifierAnomalyScore,
    efficiency_at_rate,
)
from ctl2_model.utils.plotting import (
    plot_efficiency_table,
)


def _log_physics_metrics(model, X_val, y_val, X_eval, y_eval, model_validation):
    rate_threshold = model_validation.get("rate_threshold", 16.0)
    label_names = {str(k): v for k, v in model_validation.get("label_names", {"0": "qcd", "1": "top", "2": "minbias", "3": "dy"}).items()}
    eval_names = {str(k): v for k, v in model_validation.get("eval_names", {1: "suep", 2: "hh4b", 4: "zprime", 5: "svj_m250", 3: "vbf_hinv"}).items()}

    scorer = ClassifierAnomalyScore(model, normal_class_idx=2)
    scores_val = scorer(X_val)
    minbias_mask_val = y_val == 2
    scores_bg = scores_val[minbias_mask_val]

    metrics = {}
    scores_dict = {"minbias": scores_bg}
    signal_names = []

    train_mask = y_val != 2
    for sig_label in np.unique(y_val[train_mask]):
        sig_mask = y_val == sig_label
        sig_name = eval_names.get(str(sig_label), label_names.get(str(sig_label), f"val_class_{sig_label}"))
        scores_sig = scores_val[sig_mask]
        eff = efficiency_at_rate(scores_sig, scores_bg, rate_threshold)
        metrics[f"{sig_name}_eff@{rate_threshold}Hz"] = float(eff)
        scores_dict[sig_name] = scores_sig
        signal_names.append(sig_name)

    for sig_label in np.unique(y_eval):
        sig_mask = y_eval == sig_label
        scores_sig = scorer(X_eval[sig_mask])
        sig_name = eval_names.get(str(sig_label), f"eval_{sig_label}")
        eff = efficiency_at_rate(scores_sig, scores_bg, rate_threshold)
        metrics[f"{sig_name}_eff@{rate_threshold}Hz"] = float(eff)
        scores_dict[sig_name] = scores_sig
        signal_names.append(sig_name)

    for k, v in metrics.items():
        safe_k = k.replace("@", "_at_").replace(" ", "_").replace("/", "_per_")
        mlflow.log_metric(f"phys_{safe_k}", v)

    report_dir = Path("data/05_validation")
    report_dir.mkdir(parents=True, exist_ok=True)

    eff_table_path = plot_efficiency_table(
        metrics,
        save_path=str(report_dir / "compile_efficiency_table.png"),
    )
    if eff_table_path and Path(eff_table_path).exists():
        mlflow.log_artifact(eff_table_path)

    with open(report_dir / "compile_physics_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(str(report_dir / "compile_physics_metrics.json"))
    mlflow.log_metric("phys_n_signals", len(signal_names))

    return metrics


def compile_model(
    trained_model,
    processed_X_train,
    processed_y_train,
    processed_X_val,
    processed_y_val,
    processed_X_eval,
    processed_y_eval,
    compile,
    model_validation,
):
    fpga_part = compile.get("fpga_part", "xcvu13p-flga2577-2-e")
    clock_period = compile.get("clock_period", 5.556)
    io_type = compile.get("io_type", "io_parallel")
    backend = compile.get("backend", "Vitis")

    output_dir = Path("data/06_compile")
    rtl_dir = output_dir / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    classifier = truncate_at_cls_hidden(trained_model)

    x_sample = processed_X_train[:1].astype(np.float32)
    comb_logic = trace_comb(classifier, x_sample)

    rtl_model = generate_rtl(
        comb_logic,
        prj_name="ctl2_embedding",
        output_dir=str(rtl_dir),
        part_name=fpga_part,
        clock_period=clock_period,
    )

    rtl_model.write()

    print(f'Estimated LUT : {round(rtl_model._pipe.cost):,}')
    print(f'Estimated FF  : {round(rtl_model._pipe.reg_bits):,}')

    x_val = processed_X_train[1:1001].astype(np.float32)
    compare_predictions(
        classifier, comb_logic, x_val,
        save_path=str(output_dir / "keras_vs_verilog.png"),
    )

    classifier_clean = clean_model_for_hls4ml(classifier)
    save_model_surgery_report(classifier_clean, output_dir)

    try:
        hls_dir = output_dir / "hls4ml"
        hls_model = generate_hls_hls4ml(
            classifier_clean,
            output_dir=str(hls_dir),
            part_name=fpga_part,
            io_type=io_type,
            clock_period=clock_period,
        )
        hls_generated = True
    except Exception as e:
        print(f"  hls4ml skipped: {e}")
        hls_generated = False
        hls_dir = None

    report = {
        "fpga_part": fpga_part,
        "clock_period": clock_period,
        "io_type": io_type,
        "backend": backend,
        "rtl_dir": str(rtl_dir),
        "hls_dir": str(hls_dir) if hls_dir else None,
        "hls_generated": hls_generated,
    }
    if hasattr(rtl_model, "_pipe"):
        report["estimated_luts"] = getattr(rtl_model._pipe, "cost", None)
        report["estimated_ffs"] = getattr(rtl_model._pipe, "reg_bits", None)

    with open(output_dir / "synthesis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    mlflow.log_artifact(str(output_dir / "synthesis_report.json"))
    res_keys = ["estimated_luts", "estimated_ffs"]
    res_vals = [report.get(k) for k in res_keys]
    if any(v is not None for v in res_vals):
        fig, ax = plt.subplots(figsize=(4, 3))
        valid = [(k, v) for k, v in zip(res_keys, res_vals) if v is not None]
        if valid:
            names, vals = zip(*valid)
            colors = ["#1f77b4", "#ff7f0e"][:len(names)]
            ax.bar(names, vals, color=colors, edgecolor="white")
            ax.set_ylabel("count")
            ax.set_title("Firmware Resources")
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.02, f"{v:,}", ha="center", fontsize=9)
            src_path = output_dir / "firmware_resources.png"
            fig.savefig(str(src_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(src_path))

    mlflow.log_artifact(str(output_dir / "keras_vs_verilog.png"))
    for tag, d in [("rtl", rtl_dir), ("hls", hls_dir)]:
        if d is not None and d.exists() and any(d.iterdir()):
            archive = output_dir / f"{tag}.zip"
            shutil.make_archive(str(output_dir / tag), "zip", d)
            mlflow.log_artifact(str(archive))
    for k, v in report.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"compile_{k}", v)

    print("  Computing physics performance metrics...")
    physics_metrics = _log_physics_metrics(
        trained_model,
        processed_X_val, processed_y_val,
        processed_X_eval, processed_y_eval,
        model_validation,
    )

    return rtl_dir, hls_dir, report, physics_metrics
