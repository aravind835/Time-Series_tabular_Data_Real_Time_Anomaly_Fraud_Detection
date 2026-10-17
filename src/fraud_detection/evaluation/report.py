from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fraud_detection.artifacts import ArtifactPaths
from fraud_detection.utils import ensure_dir, read_json


def _format_float(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown_report(payload: Dict, thresholds: Dict, metadata: Dict) -> str:
    metrics = payload["metrics"]
    lines: List[str] = []
    lines.append("# Fraud Detection Evaluation Report")
    lines.append("")
    summary = metadata.get("data_summary", {})
    test = summary.get("test", {})
    lines.append(
        f"Test set: {test.get('total', 0)} transactions, "
        f"{test.get('positives', 0)} fraudulent "
        f"({test.get('positive_rate', 0.0) * 100:.3f} percent positive)."
    )
    lines.append("")
    lines.append(
        "| Model | PR-AUC | ROC-AUC | Precision | Recall | F-beta | Threshold | p95 ms | Train s |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, values in metrics.items():
        latency = values.get("latency", {})
        lines.append(
            "| {model} | {pr} | {roc} | {prec} | {rec} | {fbeta} | {thr} | {p95} | {train} |".format(
                model=name,
                pr=_format_float(values.get("pr_auc")),
                roc=_format_float(values.get("roc_auc")),
                prec=_format_float(values.get("precision"), 3),
                rec=_format_float(values.get("recall"), 3),
                fbeta=_format_float(values.get("fbeta"), 3),
                thr=_format_float(values.get("threshold"), 4),
                p95=_format_float(latency.get("p95_ms"), 3),
                train=_format_float(values.get("train_seconds"), 2),
            )
        )
    lines.append("")
    best = max(metrics.items(), key=lambda item: item[1].get("pr_auc", 0.0))
    lines.append(f"Best model by PR-AUC: {best[0]} ({_format_float(best[1].get('pr_auc'))}).")
    lines.append("")
    imbalance = metadata.get("imbalance", {})
    lines.append(
        f"Imbalance handling: {imbalance.get('method')} "
        f"(strategy {imbalance.get('sampling_strategy')}), "
        f"train positives after resample {imbalance.get('train_positives_after_resample')}."
    )
    lines.append(
        f"Threshold policy: maximize precision at target recall "
        f"{metadata.get('target_recall')} with F-beta reference beta {metadata.get('beta')}."
    )
    return "\n".join(lines) + "\n"


def plot_pr_curves(curves: Dict[str, List[Dict[str, float]]], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, points in curves.items():
        if not points:
            continue
        recalls = [point["recall"] for point in points]
        precisions = [point["precision"] for point in points]
        axis.plot(recalls, precisions, label=name, linewidth=1.5)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curves")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_reports(artifacts_dir: str | Path, reports_dir: str | Path) -> Dict[str, str]:
    paths = ArtifactPaths(Path(artifacts_dir))
    payload = read_json(paths.metrics)
    thresholds = read_json(paths.thresholds)
    metadata = read_json(paths.metadata)

    reports = ensure_dir(reports_dir)
    markdown = build_markdown_report(payload, thresholds, metadata)
    report_path = reports / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    plot_path = reports / "pr_curves.png"
    plot_pr_curves(payload.get("curves", {}), plot_path)

    return {"report": str(report_path), "plot": str(plot_path)}
