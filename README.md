## Real-Time Anomaly and Fraud Detection

Detect anomalous transactions and system faults in streaming financial or IoT
sensor data. The project targets the hardest part of the problem in industry:
extreme class imbalance where positive labels are below 0.1 percent of the data.

It combines unsupervised anomaly detection with supervised classification, treats
imbalance explicitly, tunes decision thresholds on the precision-recall curve, and
serves predictions behind a latency-aware API and a streaming simulator.

### Why it stands out

Accuracy is meaningless at 0.1 percent prevalence: a model that predicts "never
fraud" scores 99.9 percent accurate and catches nothing. This project is built
around the metrics and mechanics that actually matter in production fraud systems:

- The precision-recall tradeoff instead of accuracy or ROC alone.
- Resampling with SMOTE that is fit only on training data, never leaking into
  validation or test.
- A time-ordered split with injected concept drift, so the test set is genuinely
  harder and reflects a non-stationary stream.
- Threshold selection tied to a business recall target rather than a default 0.5.
- Per-request latency budgets with rolling p50/p95/p99 measurement.

### Results

Trained on 200,000 synthetic transactions with a 0.1 percent fraud rate
(`config.yaml` defaults), evaluated on a held-out, time-ordered test set of 40,000
transactions containing 34 fraud cases. A random baseline scores a PR-AUC of about
0.00085 (the positive rate).

| Model            | PR-AUC | ROC-AUC | Precision | Recall | p95 latency |
| ---------------- | ------ | ------- | --------- | ------ | ----------- |
| XGBoost          | 0.845  | 0.997   | 0.643     | 0.794  | ~1.0 ms     |
| Autoencoder      | 0.051  | 0.975   | 0.015     | 0.912  | ~0.4 ms     |
| Isolation Forest | 0.034  | 0.977   | 0.015     | 0.882  | ~15 ms      |

XGBoost is the production model: a PR-AUC of 0.845 is roughly a 990x lift over the
random baseline. The unsupervised detectors need no labels and reach high recall at
low precision, which makes them useful complements for catching novel fraud patterns
the supervised model has never seen.

### Architecture

```
src/fraud_detection/
  config.py            typed configuration loaded from config.yaml
  data/                synthetic streaming generator and time-ordered splitter
  features/            scaling and log-amount preprocessing, saved with the model
  training/            SMOTE and class weighting, plus the training pipeline
  models/              Isolation Forest, XGBoost, and a PyTorch autoencoder
  evaluation/          PR-AUC metrics, threshold optimization, report generation
  serving/             latency-aware predictor and FastAPI app
  streaming/           real-time replay simulator with live metrics
  artifacts.py         artifact paths and bundle loading
  cli.py               command line entry points
```

Each detector shares one interface: `fit`, `score` (returns a fraud probability in
[0, 1] where higher means more anomalous), `save`, and `load`. The supervised and
unsupervised paths are kept distinct on purpose:

- XGBoost trains on the SMOTE-resampled set with `scale_pos_weight`.
- Isolation Forest trains unsupervised on all transactions with a contamination prior.
- The autoencoder trains on normal transactions only and scores by reconstruction error.

### Class imbalance handling

- SMOTE oversamples the minority class only on the training fold, with `k_neighbors`
  clamped to the available minority count so it degrades gracefully on tiny slices.
- `scale_pos_weight` and balanced class weights are computed from the label counts.
- The evaluation never looks at accuracy. Thresholds are chosen on validation by
  maximizing precision subject to a target recall, with F-beta as a reference point.

### Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

This installs the package and the `fraud` command. Every command also runs as
`python -m fraud_detection <command>` or `python scripts/fraud.py <command>`.

### Usage

Generate the synthetic dataset:

```
python -m fraud_detection generate --config config.yaml
```

Train all three models and write artifacts, thresholds, and metrics:

```
python -m fraud_detection train --config config.yaml
```

Render the evaluation report and precision-recall plot into `reports/`:

```
python -m fraud_detection evaluate --config config.yaml
```

Serve predictions over HTTP:

```
python -m fraud_detection serve --config config.yaml
```

Then call the API:

```
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"amount": 250.0, "hour": 2, "day_of_week": 5, "f0": 1.8}}'
curl http://127.0.0.1:8000/metrics
```

Endpoints: `GET /health`, `GET /info`, `GET /metrics`, `POST /predict`,
`POST /predict/batch`. Missing features default to zero, so partial records are
scored without error.

Replay the held-out test set as a live stream and watch rolling metrics:

```
python -m fraud_detection stream --config config.yaml --rate 2000 --duration 10
```

### Configuration

All parameters live in `config.yaml`: dataset size and fraud rate, drift strength,
the train/validation/test split, the SMOTE strategy, per-model hyperparameters, the
target recall for threshold selection, and serving and streaming settings. Pass a
different file with `--config` to run experiments without touching code.

### Testing

```
pytest
```

The suite covers the generator and split, preprocessing and resampling, all three
models with save/load roundtrips, the evaluation metrics and threshold logic, the
serving endpoints, and the streaming simulator.
