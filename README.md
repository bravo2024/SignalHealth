# SignalHealth

> Industrial IoT sensor anomaly detection for manufacturing and power plant monitoring.

Generates synthetic multivariate sensor time-series data (temperature, pressure, vibration, current, voltage, flow rate) with injected anomaly events (thermal runaway, pressure spike, bearing wear, etc.). Trains a classifier to detect anomalies and provides a real-time monitoring dashboard with sensor readouts, anomaly timeline, alert configurations, and root cause analysis via correlation heatmaps.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 1.000 |
| Accuracy | 1.000 |

Trained on 750 samples, tested on 250. 6 sensor channels + 9 anomaly event types.

## Features

| Component | What it does |
|---|---|
| **Live Monitor** | Real-time sensor gauges, streaming anomaly score, event log |
| **Anomaly Timeline** | Historical anomaly detection timeline, event clustering |
| **Sensor Analysis** | Per-sensor distributions, correlation matrix, pairwise scatter plots |
| **Alert Config** | Per-sensor threshold configuration, alert sensitivity tuning |
| **Root Cause** | Anomaly pattern fingerprinting, sensor contribution analysis |

## Repo Structure

```
SignalHealth/
  train.py     model training
  app.py       Streamlit dashboard (820 lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic industrial IoT dataset: 6 sensor channels sampled over time with 9 anomaly event types (thermal runaway, pressure spike, bearing wear, vibration anomaly, current surge, voltage sag, flow blockage, sensor drift, multi-sensor cascade).

## License

MIT
