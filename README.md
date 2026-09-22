# Sensor degradation → twin state → decisions (dairy-cow digital twin)

Code for the article *"From sensor faults to farm decisions: how wearable-sensor degradation
propagates through a dairy-cow digital twin"* (target: Computers and Electronics in Agriculture).

**Research question.** How does progressive sensor degradation propagate through dairy-cow
digital-twin state estimation to downstream management decisions, and which safeguards contain it?

## Pipeline

| Step | Module | What it does |
|---|---|---|
| 1 | `dtq1/preprocess.py` | MmCows raw files → clean native-rate streams (IMU 1 Hz summary of 10 Hz, UWB 15 s, CBT/leg/THI 60 s), minute behaviour labels |
| 2 | `dtq1/degrade.py` | 6 fault types (noise, dropout, drift, delay, failure, stuck-at) × 4 severities × 5 modalities + all, applied at native rate |
| 3 | `dtq1/data.py` | observation model: per-minute features, carry-forward, availability and staleness per modality |
| 4 | `dtq1/models.py`, `dtq1/gbm.py` | state estimators: LightGBM, GRU, causal Transformer, Mamba-2-style selective SSM (multi-task: behaviour, lying, CBT) |
| 5 | `dtq1/train.py` | leave-two-cows-out CV, clean vs degradation-aware training, 5-seed deep ensembles |
| 6 | `dtq1/evaluate.py` | replays 14 days of every held-out cow through all 145 conditions; direct-sensor baseline; sensor-health indicators |
| 7 | `dtq1/tsfm.py` | Chronos-Bolt zero-shot virtual CBT sensor |
| 8 | `dtq1/decisions.py` | hourly rules: heat stress (CBT ≥ 39 °C), lying < 10 h/24 h, feeding drop < 80 % of own 72-h mean, barn cooling (THI ≥ 72) |
| 9 | `dtq1/analysis.py` | decision disagreement / missed actions / regret vs clean twin and vs oracle; abstention policies |
| 10 | `dtq1/figures.py` | figures (`figures/`) and tables (`results/`) |

## Reproduce

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch numpy pandas scipy scikit-learn lightgbm matplotlib pyarrow chronos-forecasting
# MmCows sensor_data (only the parts used; ~7.7 GB) -> data/raw/sensor_data/
bash scripts/run_all.sh
```

`scripts/cpu_pool.sh` can be started alongside `run_all.sh` to train GRU and Transformer models on the CPU
while the GPU trains the state-space models; seed-level lock files keep the two pools from duplicating work.

Runtime on an Apple M4 Pro (48 GB): about 4 h of training (200 models) and about 5 h for the 145-condition
evaluation grid (inference of the selective state-space ensembles dominates).

Note: LightGBM and PyTorch are never imported into the same process. On macOS their two OpenMP
runtimes crash intermittently when loaded together.
