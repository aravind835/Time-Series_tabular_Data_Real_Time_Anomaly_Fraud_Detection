from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from fraud_detection.serving.predictor import Predictor


@dataclass
class StreamSummary:
    processed: int
    frauds: int
    alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    duration_seconds: float
    throughput_per_second: float
    latency: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def run_stream(
    predictor: Predictor,
    records: Sequence[Dict[str, float]],
    labels: Optional[Sequence[int]] = None,
    rate_per_second: int = 200,
    duration_seconds: Optional[float] = None,
    report_interval: float = 1.0,
    reporter: Optional[Callable[[Dict[str, object]], None]] = None,
) -> StreamSummary:
    interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
    label_list: List[int] = list(labels) if labels is not None else [0] * len(records)

    processed = frauds = alerts = tp = fp = fn = 0
    start = time.perf_counter()
    next_report = start + report_interval

    for index, record in enumerate(records):
        label = int(label_list[index]) if index < len(label_list) else 0
        result = predictor.predict(record)
        processed += 1
        is_alert = bool(result["is_fraud"])
        if label == 1:
            frauds += 1
        if is_alert:
            alerts += 1
            if label == 1:
                tp += 1
            else:
                fp += 1
        elif label == 1:
            fn += 1

        now = time.perf_counter()
        if duration_seconds is not None and now - start >= duration_seconds:
            break
        if reporter is not None and now >= next_report:
            reporter(
                _running_snapshot(predictor, processed, frauds, alerts, tp, fp, fn, now - start)
            )
            next_report += report_interval

        target = start + (index + 1) * interval
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    duration = time.perf_counter() - start
    snapshot = predictor.latency.snapshot()
    return StreamSummary(
        processed=processed,
        frauds=frauds,
        alerts=alerts,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=_safe_ratio(tp, tp + fp),
        recall=_safe_ratio(tp, tp + fn),
        duration_seconds=duration,
        throughput_per_second=processed / duration if duration > 0 else 0.0,
        latency=snapshot,
    )


def _running_snapshot(
    predictor: Predictor,
    processed: int,
    frauds: int,
    alerts: int,
    tp: int,
    fp: int,
    fn: int,
    elapsed: float,
) -> Dict[str, object]:
    snapshot = predictor.latency.snapshot()
    return {
        "elapsed_seconds": elapsed,
        "processed": processed,
        "throughput_per_second": processed / elapsed if elapsed > 0 else 0.0,
        "frauds": frauds,
        "alerts": alerts,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "latency_p95_ms": snapshot.get("p95_ms", 0.0),
    }


def records_from_frame(frame, feature_names: Sequence[str]) -> List[Dict[str, float]]:
    return frame[list(feature_names)].to_dict(orient="records")
