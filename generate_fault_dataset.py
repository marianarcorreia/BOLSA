"""
Synthetic multi-sensor fault dataset generator.

Simulates a process with 3 sensors that normally hover around a baseline and
occasionally drifts into one of several fault states (Markov chain + smoothed
ramp, like a real sensor would move rather than jump). Produces ONLY the raw
dataset as JSON -- no clustering/analysis here, that's a separate step.

Every run uses a fresh random seed, so the dataset is different each time you
run this script. The actual seed used is recorded in the output file's
"meta" block, so a specific run can be reproduced later if needed.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG -- tune these
# ---------------------------------------------------------------------------
N_SAMPLES        = 2000                    # total time steps
VAR_NAMES        = ["sensor_1", "sensor_2", "sensor_3"]  # keep this at 3 variables
VAR_RANGE        = (0.0, 50.0)             # nominal operating range per variable
CLIP_TO_RANGE    = True                    # clip readings back into VAR_RANGE

NOISE_STD        = 5.0                     # baseline gaussian noise std dev
SMOOTHING_ALPHA  = 0.15                    # how fast readings drift toward a new target (0-1)

START_TIME       = datetime(2026, 1, 1, 0, 0, 0)
SAMPLE_INTERVAL  = timedelta(minutes=15)   # spacing between readings

P_ENTER_FAULT    = 0.01                    # per-step prob. of leaving "normal"
P_RECOVER        = 0.05                    # per-step prob. of recovering to "normal"

MISSING_RATE     = 0.005                   # fraction of cells set to null
OUTLIER_RATE     = 0.005                   # fraction of cells set to extreme spikes
OUTLIER_MAGNITUDE = (8.0, 15.0)            # |spike| range added on top of a reading

# One entry per state.
#   "center"      -- target [sensor_1, sensor_2, sensor_3] the signal ramps toward
#   "noise_scale" -- multiplies NOISE_STD while in this state.
#                    1.0 = same noise as normal (an "easy", clearly describable fault).
#                    >1.0 = noisier (mean shift gets buried -> harder/impossible for
#                    k-means to describe cleanly, which is the thing you want to test).
STATES = {
    "normal":              {"center": [5.0, 5.0, 5.0], "noise_scale": 1.0},
    "fault_overheat":      {"center": [8.5, 6.5, 2.0], "noise_scale": 1.0},
    "fault_vibration":     {"center": [3.0, 9.0, 6.5], "noise_scale": 1.0},
    "fault_pressure_drop": {"center": [4.5, 2.0, 8.5], "noise_scale": 1.0},
}
FAULT_NAMES = [s for s in STATES if s != "normal"]

OUTPUT_PATH = Path(__file__).parent / "fault_detection_dataset.json"

assert len(VAR_NAMES) == 3, "this generator is set up for exactly 3 variables"
assert all(len(s["center"]) == len(VAR_NAMES) for s in STATES.values()), \
    "every STATES center must have one value per VAR_NAMES"

# No fixed seed -> a different dataset every run. The entropy is captured so
# this exact run can be replayed later via np.random.default_rng(seed).
seed_sequence = np.random.SeedSequence()
rng = np.random.default_rng(seed_sequence)
assert isinstance(seed_sequence.entropy, int)
seed = seed_sequence.entropy


# ---------------------------------------------------------------------------
# SIMULATE THE STATE SEQUENCE (Markov chain) + SMOOTHED SIGNAL
# ---------------------------------------------------------------------------
def generate_dataset():
    state = "normal"
    current_level = np.array(STATES["normal"]["center"], dtype=float)

    records = []
    timestamp = START_TIME

    for _ in range(N_SAMPLES):
        # Markov transition
        if state == "normal":
            if rng.random() < P_ENTER_FAULT:
                state = rng.choice(FAULT_NAMES)
        else:
            if rng.random() < P_RECOVER:
                state = "normal"

        target = np.array(STATES[state]["center"], dtype=float)
        # exponential smoothing = gradual ramp toward the new state, not a jump
        current_level = current_level + SMOOTHING_ALPHA * (target - current_level)

        noise_std = NOISE_STD * STATES[state]["noise_scale"]
        reading = current_level + rng.normal(0, noise_std, size=len(VAR_NAMES))

        if CLIP_TO_RANGE:
            reading = np.clip(reading, VAR_RANGE[0], VAR_RANGE[1])

        record: dict[str, Any] = {"timestamp": timestamp.isoformat()}
        record.update(zip(VAR_NAMES, reading.tolist()))
        # The underlying simulation still drifts toward one of several fault
        # centers (so the signal moves through distinct regions rather than
        # jumping to one spot), but the label exposed in the dataset only
        # distinguishes "normal" vs "not_normal" -- the fault flavor itself
        # is not something the dataset is meant to reveal.
        record["state"] = "normal" if state == "normal" else "not_normal"
        record["is_fault"] = state != "normal"
        records.append(record)

        timestamp += SAMPLE_INTERVAL

    # --- inject sensor glitches: missing values ---
    n_missing = int(MISSING_RATE * N_SAMPLES * len(VAR_NAMES))
    for _ in range(n_missing):
        r = int(rng.integers(0, N_SAMPLES))
        c = str(rng.choice(VAR_NAMES))
        records[r][c] = None

    # --- inject sensor glitches: outlier spikes (not clipped -- glitches can exceed sensor range) ---
    n_outliers = int(OUTLIER_RATE * N_SAMPLES * len(VAR_NAMES))
    for _ in range(n_outliers):
        r = int(rng.integers(0, N_SAMPLES))
        c = str(rng.choice(VAR_NAMES))
        if records[r][c] is None:
            continue  # skip cells already blanked out as missing
        spike = float(rng.choice([-1, 1]) * rng.uniform(*OUTLIER_MAGNITUDE))
        records[r][c] += spike

    return records


if __name__ == "__main__":
    records = generate_dataset()
    n_fault = sum(r["is_fault"] for r in records)

    dataset = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "seed": seed,
            "n_samples": N_SAMPLES,
            "variables": VAR_NAMES,
            "var_range": VAR_RANGE,
            "noise_std": NOISE_STD,
            "smoothing_alpha": SMOOTHING_ALPHA,
            "sample_interval_minutes": SAMPLE_INTERVAL.total_seconds() / 60,
            "p_enter_fault": P_ENTER_FAULT,
            "p_recover": P_RECOVER,
            "missing_rate": MISSING_RATE,
            "outlier_rate": OUTLIER_RATE,
            "states": STATES,
            "fault_sample_fraction": n_fault / N_SAMPLES,
        },
        "records": records,
    }

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))

    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print(f"Seed for this run: {seed}")
    print(f"Fault samples: {n_fault} ({n_fault / N_SAMPLES:.1%})")




