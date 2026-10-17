from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from fraud_detection.config import Config, load_config


def _resolve_config(path: Optional[str]) -> Config:
    if path:
        return load_config(path)
    default = Path("config.yaml")
    if default.exists():
        return load_config(str(default))
    return Config()


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to config.yaml")


def command_generate(args: argparse.Namespace) -> None:
    from fraud_detection.data.loader import load_or_generate, split_dataframe

    config = _resolve_config(args.config)
    frame = load_or_generate(config)
    split = split_dataframe(frame, config)
    print(f"Generated {len(frame)} transactions in {config.data.output_dir}/{config.data.filename}")
    print(f"Overall fraud rate: {frame['is_fraud'].mean() * 100:.4f} percent")
    for name, counts in split.summary().items():
        if name == "features":
            continue
        print(f"  {name}: {counts}")


def command_train(args: argparse.Namespace) -> None:
    from fraud_detection.training import train_and_save

    config = _resolve_config(args.config)
    if args.artifacts:
        config.serving.artifacts_dir = args.artifacts
    results = train_and_save(config, artifacts_dir=config.serving.artifacts_dir)
    print("")
    print(f"Artifacts written to {config.serving.artifacts_dir}")
    best = max(results["metrics"].items(), key=lambda item: item[1].get("pr_auc", 0.0))
    print(f"Best model by PR-AUC: {best[0]} ({best[1]['pr_auc']:.4f})")


def command_evaluate(args: argparse.Namespace) -> None:
    from fraud_detection.evaluation.report import write_reports

    config = _resolve_config(args.config)
    artifacts_dir = args.artifacts or config.serving.artifacts_dir
    outputs = write_reports(artifacts_dir, config.evaluation.reports_dir)
    report_text = Path(outputs["report"]).read_text(encoding="utf-8")
    print(report_text)
    print(f"Report written to {outputs['report']}")
    print(f"Precision-recall plot written to {outputs['plot']}")


def command_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from fraud_detection.serving.app import create_app

    config = _resolve_config(args.config)
    artifacts_dir = args.artifacts or config.serving.artifacts_dir
    model_name = args.model or config.serving.model
    host = args.host or config.serving.host
    port = args.port or config.serving.port
    app = create_app(artifacts_dir, model_name)
    print(f"Serving model '{model_name}' from {artifacts_dir} at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def command_stream(args: argparse.Namespace) -> None:
    from fraud_detection.data.loader import load_or_generate, split_dataframe
    from fraud_detection.serving.predictor import Predictor
    from fraud_detection.streaming import records_from_frame, run_stream

    config = _resolve_config(args.config)
    artifacts_dir = args.artifacts or config.serving.artifacts_dir
    model_name = args.model or config.streaming.model

    frame = load_or_generate(config)
    split = split_dataframe(frame, config)
    predictor = Predictor(artifacts_dir, model_name)
    records = records_from_frame(split.test_frame, split.feature_names)
    labels = split.test_frame["is_fraud"].tolist()

    def reporter(snapshot: dict) -> None:
        print(
            f"  t={snapshot['elapsed_seconds']:.1f}s "
            f"processed={snapshot['processed']} "
            f"throughput={snapshot['throughput_per_second']:.0f}/s "
            f"alerts={snapshot['alerts']} "
            f"precision={snapshot['precision']:.3f} "
            f"recall={snapshot['recall']:.3f} "
            f"p95={snapshot['latency_p95_ms']:.2f}ms"
        )

    print(f"Streaming {len(records)} test transactions through model '{model_name}'")
    summary = run_stream(
        predictor,
        records,
        labels,
        rate_per_second=args.rate or config.streaming.rate_per_second,
        duration_seconds=args.duration if args.duration is not None else config.streaming.duration_seconds,
        reporter=reporter,
    )
    print("")
    print("Stream summary:")
    print(f"  processed: {summary.processed}")
    print(f"  frauds: {summary.frauds}  alerts: {summary.alerts}")
    print(f"  precision: {summary.precision:.3f}  recall: {summary.recall:.3f}")
    print(f"  throughput: {summary.throughput_per_second:.0f} transactions/second")
    print(
        f"  latency p50/p95/p99 ms: "
        f"{summary.latency['p50_ms']:.3f} / "
        f"{summary.latency['p95_ms']:.3f} / "
        f"{summary.latency['p99_ms']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", description="Real-time fraud detection toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate synthetic transactions")
    _add_config_argument(generate)
    generate.set_defaults(func=command_generate)

    train = subparsers.add_parser("train", help="train all models and save artifacts")
    _add_config_argument(train)
    train.add_argument("--artifacts", default=None, help="artifacts output directory")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="render evaluation report and plots")
    _add_config_argument(evaluate)
    evaluate.add_argument("--artifacts", default=None, help="artifacts directory")
    evaluate.set_defaults(func=command_evaluate)

    serve = subparsers.add_parser("serve", help="serve predictions over HTTP")
    _add_config_argument(serve)
    serve.add_argument("--artifacts", default=None)
    serve.add_argument("--model", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=command_serve)

    stream = subparsers.add_parser("stream", help="replay the test set as a live stream")
    _add_config_argument(stream)
    stream.add_argument("--artifacts", default=None)
    stream.add_argument("--model", default=None)
    stream.add_argument("--rate", type=int, default=None, help="transactions per second")
    stream.add_argument("--duration", type=float, default=None, help="seconds to run")
    stream.set_defaults(func=command_stream)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def generate_main() -> None:
    main(["generate", *sys.argv[1:]])


def train_main() -> None:
    main(["train", *sys.argv[1:]])


def evaluate_main() -> None:
    main(["evaluate", *sys.argv[1:]])


def serve_main() -> None:
    main(["serve", *sys.argv[1:]])


def stream_main() -> None:
    main(["stream", *sys.argv[1:]])


if __name__ == "__main__":
    main()
