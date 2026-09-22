"""Global configuration for the dairy-cow digital-twin degradation study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sensor_data"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

COWS = [f"C{i:02d}" for i in range(1, 11)]  # cows with wearable sensors (tags T01..T10)
TAG = {c: "T" + c[1:] for c in COWS}
DAYS = ["0721", "0722", "0723", "0724", "0725", "0726", "0727", "0728",
        "0729", "0730", "0731", "0801", "0802", "0803", "0804"]
LABEL_DAY = "0725"

# Study window (unix seconds, from CBT/THI grid): 2023-07-21 17:30 UTC .. 2023-08-04 12:00 UTC
T0 = 1689960600
T1 = 1691150400
STEP = 60  # twin update period (s)
N_MIN = (T1 - T0) // STEP + 1

# MmCows behaviour codes -> twin behaviour classes
# 0 unknown, 1 walking, 2 standing, 3 feeding head up, 4 feeding head down, 5 licking, 6 drinking, 7 lying
# Walking and drinking last < 1 min per bout and occupy only 3-24 min/cow/day at the 1-min twin
# resolution, so they are merged into 'standing/other'; lying and feeding drive the decision layer.
BEHAVIOURS = ["lying", "standing", "feeding"]
RAW2CLS = {7: 0, 2: 1, 5: 1, 1: 1, 6: 1, 3: 2, 4: 2}  # 0 (unknown) -> ignored

# Sensor modalities that can be degraded (name -> channels in native stream)
MODALITIES = ["imu", "uwb", "cbt", "ankle", "thi"]

# Cross-validation: leave-two-cows-out, 5 folds
FOLDS = [["C01", "C02"], ["C03", "C04"], ["C05", "C06"], ["C07", "C08"], ["C09", "C10"]]

SEEDS = [0, 1, 2, 3, 4]
