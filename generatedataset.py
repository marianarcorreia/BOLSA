#the three sensors are; furnace temperature, flue-gas O2 and furnace draft pressure.
from typing import Optional
import os
import json
from datetime import datetime
from zipfile import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import argparse


#fault types
class FaultType(Enum):
    NONE = "none"
    BIAS = "bias"   #miscalibration
    DRIFT = "drift" #fouling, thermocouple aging, oxidation
    STUCK = "stuck"     # frozen transmitter, blocked impulse line, software latch
    DROPOUT = "dropout" #wiring break, power loss, loose connector
    NOISE = "noise"    #degraded insulation, EMI, loose contact
    SPIKE = "spike" #intermittent connection, electrical interference

#sensor model
@dataclass
class Sensor:
    name: str
    unit: str
    true_signal: np.ndarray
    noise_std: float
    measured: Optional[np.ndarray] = field(default=None)

    def measure(self, rng: np.random.Generator) -> np.ndarray:
        """Healthy measurement = true process value + sensor noise"""
        noise = rng.normal(0, self.noise_std, size= self.true_signal.shape)
        self.measured = self.true_signal + noise
        return self.measured


#fault injection model
def inject_fault(signal: np.ndarray, fault_type: FaultType, t_start_idx: int,
                 rng: np.random.Generator, bias_value:float = 0.0,
                 drift_rate: float = 0.0, extra_noise_std: float = 0.0,
                 spike_prob: float = 0.2, spike_magnitude: float = 0.0) -> np.ndarray:

    """Return a copy of 'signal" with a single fault type applied from index t_start_idx onwards"""
    faulty = signal.copy()
    n = len(signal)

    if fault_type == FaultType.NONE:
        return faulty

    elif fault_type is FaultType.BIAS:
        faulty[t_start_idx:] += bias_value

    elif fault_type is FaultType.DRIFT:
        ramp = np.arange(n - t_start_idx, dtype=float) * drift_rate
        faulty[t_start_idx:] += ramp

    elif fault_type is FaultType.STUCK:
        stuck_value = faulty[t_start_idx]
        faulty[t_start_idx:] = stuck_value

    elif fault_type is FaultType.DROPOUT:
        faulty[t_start_idx:] = 0.0

    elif fault_type is FaultType.NOISE:
        extra = rng.normal(0, extra_noise_std, size=n - t_start_idx)
        faulty[t_start_idx:] += extra

    elif fault_type is FaultType.SPIKE:
        for i in range(t_start_idx, n):
            if rng.random() < spike_prob:
                faulty[i] += rng.choice([-1, 1]) * spike_magnitude

    return faulty

#sensor-specific scales, used to keep randomly generated fault severities
#physically plausible for each signal (a 60-degree bias makes sense for the
#temperature sensor but would be absurd on the O2 sensor)
@dataclass
class SensorFaultRanges:
    noise_std: float
    bias: tuple[float, float]
    drift: tuple[float, float]
    extra_noise: tuple[float, float]
    spike_mag: tuple[float, float]

SENSOR_INFO: dict[str, SensorFaultRanges] = {
    "TEMP":     SensorFaultRanges(noise_std=2.0,  bias=(20.0, 60.0), drift=(0.01, 0.10),
                                   extra_noise=(2.0, 10.0), spike_mag=(20.0, 100.0)),
    "O2":       SensorFaultRanges(noise_std=0.15, bias=(0.5, 2.0),   drift=(0.005, 0.02),
                                   extra_noise=(0.1, 0.5),  spike_mag=(1.0, 4.0)),
    "PRESSURE": SensorFaultRanges(noise_std=0.5,  bias=(5.0, 20.0),  drift=(0.01, 0.08),
                                   extra_noise=(0.5, 3.0),  spike_mag=(5.0, 30.0)),
}
SENSOR_NAMES = list(SENSOR_INFO.keys())

#NONE is excluded here -- every random trial injects exactly one fault so it
#has a ground-truth label to check the detectors against
INJECTABLE_FAULTS = [FaultType.BIAS, FaultType.DRIFT, FaultType.STUCK,
                     FaultType.DROPOUT, FaultType.NOISE, FaultType.SPIKE]

#fault configuration: which sensor, which fault type, when it starts, and how severe it is
def fault_config(rng: np.random.Generator, n_samples: int, margin: int = 100,
                 sensor_name: Optional[str] = None,
                 fault_type: Optional[FaultType] = None) -> dict:
    """Pick which sensor faults, which fault type, how severe it is,
    and when it starts. `sensor_name`/`fault_type` force a fixed choice
    instead of a random one (used when the user selects them via CLI args).
    `margin` keeps the onset away from the very start/end of the run so
    there's baseline before it and room to detect it after."""
    if sensor_name is None:
        sensor_name = SENSOR_NAMES[rng.integers(len(SENSOR_NAMES))]
    if fault_type is None:
        fault_type = INJECTABLE_FAULTS[rng.integers(len(INJECTABLE_FAULTS))]
    start = int(rng.integers(margin, n_samples - margin))
    info = SENSOR_INFO[sensor_name]
    sign = float(rng.choice([-1.0, 1.0]))

    params = {}
    if fault_type is FaultType.BIAS:
        params["bias_value"] = sign * rng.uniform(*info.bias)
    elif fault_type is FaultType.DRIFT:
        params["drift_rate"] = sign * rng.uniform(*info.drift)
    elif fault_type is FaultType.NOISE:
        params["extra_noise_std"] = float(rng.uniform(*info.extra_noise))
    elif fault_type is FaultType.SPIKE:
        params["spike_prob"] = float(rng.uniform(0.05, 0.3))
        params["spike_magnitude"] = float(rng.uniform(*info.spike_mag))
    #STUCK and DROPOUT have no severity knob -- they're binary states

    return {"sensor": sensor_name, "fault_type": fault_type, "start": start, "params": params}

#train/val/test + fault/no-fault split, by run
def assign_split(run_id: int, n_runs: int, val_runs_per_group: int = 1,
                 test_runs_per_group: int = 1) -> tuple[str, str]:
    """First half of runs is fault-free (`no_fault`), second half has a
    fault injected (`fault`) -- a 50/50 pool. Val and test are each held to
    a FIXED size regardless of `n_runs` (default 1 run per group, so 2 runs
    total each): the last `test_runs_per_group` runs of each half go to
    test, the `val_runs_per_group` runs just before those go to val, and
    everything else (the bulk) goes to train. Val/test are meant only as a
    small sanity check ("is the model good or not"), not a statistically
    robust estimate -- train is where the volume lives. Applying the split
    within each group separately (rather than across the whole pool) means
    train/val/test each end up with both a fault part and a no-fault part."""
    half = n_runs // 2
    no_fault_group_size = half
    fault_group_size = n_runs - half

    if run_id < half:
        group = "no_fault"
        idx_in_group = run_id
        group_size = no_fault_group_size
    else:
        group = "fault"
        idx_in_group = run_id - half
        group_size = fault_group_size

    test_start = group_size - test_runs_per_group
    val_start = test_start - val_runs_per_group

    if idx_in_group >= test_start:
        split = "test"
    elif idx_in_group >= val_start:
        split = "val"
    else:
        split = "train"

    return split, group


def run_trial(rng: np.random.Generator, n_samples: int, dt: float, run_id: int,
              save_signals: bool = True, sensor_name: Optional[str] = None,
              fault_type: Optional[FaultType] = None) -> dict:
    """Simulate one randomized trial: fresh sensor noise + one fault,
    then check what each detector does with it. Returns a labeled record.
    `sensor_name`/`fault_type` force a fixed choice instead of a random one."""
    t, temp_true, o2_true, pressure_true = simulate_process(n_samples=n_samples, dt=dt, rng=rng)

    sensors = {
        "TEMP": Sensor(name="Furnace Temperature", unit="°C", true_signal=temp_true,
                       noise_std=SENSOR_INFO["TEMP"].noise_std),
        "O2": Sensor(name="Flue-gas O2", unit="%", true_signal=o2_true,
                     noise_std=SENSOR_INFO["O2"].noise_std),
        "PRESSURE": Sensor(name="Furnace Draft Pressure", unit="Pa", true_signal=pressure_true,
                           noise_std=SENSOR_INFO["PRESSURE"].noise_std),
    }
    signals = {name: sensor.measure(rng) for name, sensor in sensors.items()}

    cfg = fault_config(rng, n_samples, sensor_name=sensor_name, fault_type=fault_type)
    picked_sensor, picked_fault, start, params = cfg["sensor"], cfg["fault_type"], cfg["start"], cfg["params"]
    signals[picked_sensor] = inject_fault(signals[picked_sensor], picked_fault, start, rng, **params)
    faulty_signal = signals[picked_sensor]

    #`start` is always picked by fault_config even for NONE trials (it just
    #goes unused by inject_fault). For ground-truth labeling/evaluation we
    #need a cutoff that reflects whether a fault actually happened: for a
    #NONE trial that's "never" (n_samples), so downstream `faulty` columns
    #come out all-zero and false-alarm counts cover the whole run.
    fault_occurred = picked_fault is not FaultType.NONE
    eval_start = start if fault_occurred else n_samples

    flags_roll, _ = detect_fault(faulty_signal)
    flags_fixed, _ = detect_fault_fixed_baseline(faulty_signal)
    flags_variance, _ = detect_noise_fault(faulty_signal)

    def summarize(flags: np.ndarray):
        false_alarms = int(np.sum(flags[:eval_start]))
        post = flags[eval_start:]
        if post.any():
            return True, int(np.argmax(post)), false_alarms
        return False, None, false_alarms

    detected_roll, delay_roll, fa_roll = summarize(flags_roll)
    detected_fixed, delay_fixed, fa_fixed = summarize(flags_fixed)
    detected_variance, delay_variance, fa_variance = summarize(flags_variance)

    record = {
        "run_id": run_id,
        "faulty_sensor": picked_sensor,
        "fault_type": picked_fault.value,
        "fault_occurred": fault_occurred,
        "fault_start": eval_start,
        "fault_params": params,
        "rolling_detector": {"detected": detected_roll, "delay": delay_roll,
                             "false_alarms_before_onset": fa_roll},
        "fixed_baseline_detector": {"detected": detected_fixed, "delay": delay_fixed,
                                    "false_alarms_before_onset": fa_fixed},
        "variance_detector": {"detected": detected_variance, "delay": delay_variance,
                              "false_alarms_before_onset": fa_variance},
    }
    if save_signals:
        record["signals"] = {name: np.round(sig, 4).tolist() for name, sig in signals.items()}

    return record

#process simulation
def simulate_process(n_samples: int=1000, dt:float = 1.0,
                     rng: Optional[np.random.Generator] = None, seed: int = 42):
    if rng is None:
        rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    #baseline operating points for the three sensors -- drawn randomly each
    #run (seeded, so reproducible) but clipped to physically plausible
    #ranges, so they vary run-to-run without ever landing on an absurd value
    temp_baseline = rng.uniform(800.0, 900.0)       #°C, typical furnace setpoint band
    o2_baseline = rng.uniform(4.0, 6.0)              #%, normal flue-gas O2 band
    pressure_baseline = rng.uniform(-60.0, -40.0)    #Pa, normal furnace draft band

    #furnace temperature: sinusoidal with noise
    temp_true = temp_baseline - (temp_baseline - 200) * np.exp(-t/80)
    temp_true += 5.0 * np.sin(2 * np.pi * t / 150)

    #flue-gas O2: sinusoidal with noise
    o2_true = o2_baseline + 0.5 * np.sin(2 * np.pi * t / 200.0 + 1.0)
    o2_true = np.clip(o2_true, 2.0, 12.0)

    #furnace draft pressure: sinusoidal with noise
    pressure_true = pressure_baseline + 3.0 * np.sin(2 * np.pi * t / 100.0 + 0.5)

    return t, temp_true, o2_true, pressure_true

def appropriate_dataset(temp_final, pressure_final, o2_final, fault_start: int,
                        run_id: Optional[int] = None,
                        split: Optional[str] = None,
                        group: Optional[str] = None,
                        out_path: Optional[str] = None) -> pd.DataFrame:
    """Per-sample table: temperature, pressure, o2, faulty (0/1).
    `faulty` is 1 from `fault_start` onward, 0 before -- labels the
    sample/timestep, regardless of which single sensor was faulted.
    Pass `fault_start = len(signal)` for a fault-free run so every row
    comes out faulty=0. `run_id` tags every row with its trial id, and
    `split`/`group` tag it train/test and no_fault/fault (all omitted if
    None), so per-run tables can be concatenated into one dataset.
    `sample_index` (position within the run, 0..n-1) is included so a row
    can still be identified after `run_id`+`sample_index` are split off into
    a features-only file, with `faulty` split off separately -- see the
    val_X/val_y and test_X/test_y files written in __main__."""
    n = len(temp_final)
    faulty = (np.arange(n) >= fault_start).astype(int)

    data = {
        "sample_index": np.arange(n),
        "temperature": np.float32(temp_final),
        "pressure": np.float32(pressure_final),
        "o2": np.float32(o2_final),
        "faulty": faulty,
    }
    if group is not None:
        data = {"group": group, **data}
    if split is not None:
        data = {"split": split, **data}
    if run_id is not None:
        data = {"run_id": run_id, **data}

    df = pd.DataFrame(data)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_json(out_path, orient="records", indent=2)

    return df

#simple statistical fault detector
def detect_fault(signal: np.ndarray, window: int = 20, threshold: float = 4.0,
                 persistence: int = 5):
    """
    Flags samples where the signal deviates too far from a recent robust
    baseline. Uses rolling median + MAD (median absolute deviation)
    instead of mean/std so the detector isn't itself fooled by outliers.

    `persistence` requires `persistence` consecutive raw flags before an
    alarm is confirmed -- a simple form of hysteresis/debouncing. Without
    it, ordinary sensor noise occasionally crosses the threshold and
    generates isolated false alarms; real fault-detection logic (CUSUM,
    SPRT, alarm-shelving, etc.) always includes something like this.

    This is a simple univariate example; real systems usually combine
    several such residuals (see write-up below)."""

    n = len(signal)
    residuals = np.zeros(n)
    raw_flags = np.zeros(n, dtype=bool)

    for i in range(window, n):
        w = signal[i-window:i]
        med = np.median(w)
        mad_raw = np.median(np.abs(w - med))
        mad = mad_raw + 1e-6
        residuals[i] = (signal[i] - med) / (1.4826*mad)
        #a stuck/frozen sensor repeats the exact same value, so the window's
        #own MAD collapses to 0 -- residual-from-median can't see that (the
        #frozen value IS the median), so a flatlined window is flagged directly
        raw_flags[i] = (np.abs(residuals[i]) > threshold) or (mad_raw == 0.0)

    confirmed = np.zeros(n, dtype=bool)
    run = 0
    for i in range(n):
        run = run +1 if raw_flags[i] else 0
        confirmed[i] = run >= persistence

    return confirmed, residuals
#detector that uses a fixed baseline instead of a rolling window
def detect_fault_fixed_baseline(signal, baseline_end: int = 100,
                                threshold: float = 4.0, persistence: int = 2):
    """
    Alternative detector: statistics are computed ONCE from a known-healthy
    baseline period (e.g. commissioning / start-up data) and then held
    fixed, rather than being recomputed from a sliding window.

    Why have both? A rolling-window detector (detect_fault above) adapts
    to whatever the recent signal looks like -- great for catching abrupt
    faults (bias, stuck, dropout, spikes) against a changing process, but
    it can be blind to SLOW DRIFT: the window slides along with the drift
    almost as fast as the drift happens, so the residual never grows.
    A fixed baseline has no such blind spot for drift, at the cost of
    being less adaptive if the healthy process itself changes over time
    (load changes, fuel switching, etc.). Real systems typically combine
    both ideas (e.g. CUSUM on top of a slowly-updated baseline).
    """
    n = len(signal)
    baseline = signal[:baseline_end]
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med)) + 1e-6

    residuals = (signal - med) / (1.4826*mad)
    raw_flags = np.abs(residuals) > threshold

    confirmed = np.zeros(n, dtype=bool)
    run = 0

    for i in range(n):
        run = run + 1 if raw_flags[i] else 0
        confirmed[i] = run >= persistence

    return confirmed, residuals

#F-test-style variance-ratio detector -- targets NOISE, which the two detectors
#above are structurally blind to
def detect_noise_fault(signal, window: int = 20, ref_window: int = 100,
                       threshold: float = 2.5, persistence: int = 6):
    """
    detect_fault and detect_fault_fixed_baseline both flag samples whose
    VALUE is far from a center (median) -- i.e. they detect a shift in
    location. A NOISE fault doesn't shift the location, it widens the
    spread, so both are structurally blind to it (see the write-up this
    was derived from). This detector instead compares the SPREAD of a
    recent window against the spread of an earlier reference window --
    an F-test-style variance ratio -- so it reacts to widening scatter
    even when the mean/median never moves.

    Two things make this work where a naive version wouldn't:
    - The reference window ends *before* the recent window starts
      (`signal[i-ref_window-window : i-window]`), instead of reusing the
      same window for both roles. That avoids the failure mode of
      `detect_fault`'s rolling MAD, which recomputes its own yardstick
      from the same samples it's judging -- once a noise fault starts,
      that yardstick inflates right along with the fault and the ratio
      never grows.
    - Both scales are computed on the FIRST DIFFERENCE of the signal,
      not the raw value. Differencing is a high-pass filter: it removes
      slow deterministic structure (the sinusoidal process signals here)
      while barely touching sample-to-sample noise, so a fast periodic
      signal (e.g. the draft-pressure sinusoid) doesn't get mistaken for
      "spread" and mask the actual noise-variance change.

    Threshold is a ratio (recent scale / reference scale), not a z-score,
    which is what makes one threshold usable across sensors with very
    different native noise levels (TEMP/O2/PRESSURE) -- same reasoning as
    using MAD instead of mean/std elsewhere in this file.

    Tuned against the generated NOISE datasets to hold false alarms at
    ~0 (matching the other two detectors on this data) while recovering
    most of the detections they miss entirely.
    """
    n = len(signal)
    diffed = np.diff(signal, prepend=signal[0])
    ratio = np.zeros(n)
    raw_flags = np.zeros(n, dtype=bool)

    for i in range(ref_window + window, n):
        recent = diffed[i-window:i]
        reference = diffed[i-ref_window-window:i-window]

        recent_scale = 1.4826 * np.median(np.abs(recent - np.median(recent)))
        ref_scale = 1.4826 * np.median(np.abs(reference - np.median(reference))) + 1e-6

        ratio[i] = recent_scale / ref_scale
        raw_flags[i] = ratio[i] > threshold

    confirmed = np.zeros(n, dtype=bool)
    run = 0
    for i in range(n):
        run = run + 1 if raw_flags[i] else 0
        confirmed[i] = run >= persistence

    return confirmed, ratio

#background color per split, used so a glance at the PNG says which stretch is which
SPLIT_BG = {"train": "#eaf3fb", "val": "#eafbea", "test": "#fdf0e6"}  #light blue / light green / light peach

def plot_illustrative_examples(split_records: dict, n_samples: int, out_path: str) -> Optional[str]:
    """Plot one train-split trial and one test-split trial back-to-back on the
    SAME axes per signal -- train stretch shaded light blue, test stretch
    shaded light peach (via axvspan), divided by a solid vertical line -- so
    a single graph shows what "training data" vs "test data" looks like.
    `split_records` maps "train"/"val"/"test" -> record dict (with `signals`
    populated); a split with no record available is simply skipped."""
    order = [s for s in ("train", "val", "test") if split_records.get(s) is not None]
    if not order:
        return None

    row_labels = ["Furnace Temperature", "Flue-gas O2", "Furnace Draft Pressure"]
    row_ylabels = ["Temp (°C)", "O2 (%)", "Pressure (Pa)"]
    row_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    row_keys = ["TEMP", "O2", "PRESSURE"]

    #build one continuous x-axis by laying each split's samples end to end
    segments = []  #(split, x_start, x_end, record, faulty_signal, flags_roll, residual_roll, flags_fixed, residual_fixed)
    offset = 0
    for split in order:
        record = split_records[split]
        faulty_sensor = record["faulty_sensor"]
        fault_start = record["fault_start"]
        faulty_signal = np.array(record["signals"][faulty_sensor])

        flags_roll, residual_roll = detect_fault(faulty_signal)
        flags_fixed, residual_fixed = detect_fault_fixed_baseline(faulty_signal)
        flags_variance, residual_variance = detect_noise_fault(faulty_signal)

        def report(name, flags, split=split, fault_start=fault_start):
            if flags.any():
                i = np.argmax(flags)
                print(f" [{split}] {name} alarm at sample {i} "
                      f"({i - fault_start:+d} samples relative to fault onset)")
            else:
                print(f" [{split}] {name} alarm: no alarm raised")

        print(f"[{split}] Fault injected in: {faulty_sensor} sensor, type: {record['fault_type']}, "
              f"start at sample: {fault_start}, params: {record['fault_params']}")
        report("Rolling-window detector", flags_roll)
        report("Fixed-baseline detector", flags_fixed)
        report("Variance-ratio detector", flags_variance)

        segments.append({
            "split": split, "offset": offset, "record": record,
            "faulty_sensor": faulty_sensor, "fault_start_abs": offset + fault_start,
            "flags_roll": flags_roll, "residual_roll": residual_roll,
            "flags_fixed": flags_fixed, "residual_fixed": residual_fixed,
            "flags_variance": flags_variance, "residual_variance": residual_variance,
        })
        offset += n_samples

    fig, axes = plt.subplots(3, 1, figsize=(6 + 5 * len(order), 9), sharex=True)

    for ax in axes:
        for seg in segments:
            ax.axvspan(seg["offset"], seg["offset"] + n_samples, color=SPLIT_BG[seg["split"]], zorder=0)
        for boundary in range(n_samples, offset, n_samples):
            ax.axvline(boundary, color="k", lw=1.5, alpha=0.8)

    for row, key in enumerate(row_keys):
        for seg in segments:
            x = np.arange(n_samples) + seg["offset"]
            y = np.array(seg["record"]["signals"][key])
            axes[row].plot(x, y, color=row_colors[row], lw=1)
            axes[row].axvline(seg["fault_start_abs"], color="k", ls="--", lw=1, alpha=0.6)
        axes[row].set_ylabel(row_ylabels[row])
        axes[row].set_title(row_labels[row])

    axes[-1].set_xlabel("Sample index")

    for i, seg in enumerate(segments):
        mid = (i + 0.5) / len(segments)
        axes[0].text(mid, 1.18, f"{seg['split'].upper()} SPLIT", transform=axes[0].transAxes,
                     ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Illustrative example: training data vs. test data", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined {'/'.join(order)} example figure to {out_path}")
    return out_path

#args to choose sensor and fault type from command line
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic sensor-fault dataset.")
    parser.add_argument("--sensor", choices=SENSOR_NAMES + ["ALL"], default=None,
                        help="Sensor to inject the fault into. 'ALL' cycles evenly through "
                             "every sensor across the runs (default: random each run)")
    parser.add_argument("--fault-type", dest="fault_type",
                        choices=[f.value for f in INJECTABLE_FAULTS], default=None,
                        help="Fault type to inject (default: random each run)")
    parser.add_argument("--runs", type=int, default=200,
                        help="Total number of random trials to generate, split 50/50 into "
                             "no_fault/fault (default: 200). Val and test are each held to a "
                             "fixed 1 run per group (2 runs each, regardless of --runs) as a small "
                             "sanity check; everything else goes to train. See assign_split()")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples per run (default: 1000)")
    return parser.parse_args()

#run
if __name__ == "__main__":
    args = parse_args()

    N_SAMPLES = args.samples
    N_RUNS = args.runs            #TOTAL trials across all splits, 50/50 no_fault/fault
    TOTAL_RUNS = N_RUNS            #passed to assign_split, which carves fixed-size val/test off the tail and gives the rest to train
    SAVE_SIGNALS = True           #include full time series in the dataset (bigger file)
    #None -> random sensor picked per trial; "ALL" -> round-robin through every sensor
    SENSOR_NAME = None if args.sensor == "ALL" else args.sensor
    SENSOR_CYCLE = SENSOR_NAMES if args.sensor == "ALL" else None
    FAULT_TYPE_ARG = FaultType(args.fault_type) if args.fault_type else None  #None -> random fault picked per trial

    #a fresh, unpredictable seed every run -- but captured in the output so
    #this exact dataset can be regenerated later via np.random.default_rng(seed)
    seed_sequence = np.random.SeedSequence()
    master_rng = np.random.default_rng(seed_sequence)
    assert isinstance(seed_sequence.entropy, int)
    seed = seed_sequence.entropy

    t, temp_true, o2_true, pressure_true = simulate_process(n_samples=N_SAMPLES, dt=1.0)

    #--- batch of random trials -> labeled dataset, split train/val/test x no_fault/fault ---
    #run_id order within the no_fault half [0, half): train, then 1 val run,
    #then 1 test run. Same pattern in the fault half [half, TOTAL_RUNS).
    #Val and test are fixed at 1 run per group regardless of TOTAL_RUNS;
    #train gets everything else. See assign_split().
    records = []
    for run_id in range(TOTAL_RUNS):
        trial_rng = np.random.default_rng(master_rng.integers(0, 2**63))
        run_sensor = SENSOR_CYCLE[run_id % len(SENSOR_CYCLE)] if SENSOR_CYCLE else SENSOR_NAME
        split, group = assign_split(run_id, TOTAL_RUNS)
        #no_fault quarters get FaultType.NONE forced regardless of --fault-type;
        #fault quarters use whatever the caller asked for (or random, if nothing was forced)
        run_fault_type = FaultType.NONE if group == "no_fault" else FAULT_TYPE_ARG
        record = run_trial(trial_rng, N_SAMPLES, 1.0, run_id, save_signals=SAVE_SIGNALS,
                           sensor_name=run_sensor, fault_type=run_fault_type)
        record["split"] = split
        record["group"] = group
        records.append(record)

    def detection_rates(subset: list) -> dict:
        """Detection rate only makes sense over runs that actually had a
        fault; no_fault runs contribute to the false-alarm count instead."""
        fault_runs = [r for r in subset if r["fault_occurred"]]
        no_fault_runs = [r for r in subset if not r["fault_occurred"]]
        n_fault = len(fault_runs)
        n_no_fault = len(no_fault_runs)
        return {
            "n_runs": len(subset),
            "n_fault_runs": n_fault,
            "n_no_fault_runs": n_no_fault,
            "rolling_detector_detection_rate": (sum(r["rolling_detector"]["detected"] for r in fault_runs) / n_fault) if n_fault else None,
            "fixed_baseline_detection_rate": (sum(r["fixed_baseline_detector"]["detected"] for r in fault_runs) / n_fault) if n_fault else None,
            "variance_detector_detection_rate": (sum(r["variance_detector"]["detected"] for r in fault_runs) / n_fault) if n_fault else None,
            "rolling_detector_avg_false_alarms_per_no_fault_run": (sum(r["rolling_detector"]["false_alarms_before_onset"] for r in no_fault_runs) / n_no_fault) if n_no_fault else None,
            "fixed_baseline_avg_false_alarms_per_no_fault_run": (sum(r["fixed_baseline_detector"]["false_alarms_before_onset"] for r in no_fault_runs) / n_no_fault) if n_no_fault else None,
            "variance_detector_avg_false_alarms_per_no_fault_run": (sum(r["variance_detector"]["false_alarms_before_onset"] for r in no_fault_runs) / n_no_fault) if n_no_fault else None,
        }

    sensor_tag = "ALL" if SENSOR_CYCLE else (SENSOR_NAME or "random")
    fault_tag = FAULT_TYPE_ARG.value if FAULT_TYPE_ARG else "random"
    out_dir = f"test_train_val_test/{fault_tag}/{sensor_tag}"
    os.makedirs(out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        split_records = [r for r in records if r["split"] == split]
        dataset = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "seed": int(seed),
                "split": split,
                "n_samples": N_SAMPLES,
                "forced_sensor": "ALL (round-robin)" if SENSOR_CYCLE else SENSOR_NAME,
                "forced_fault_type": FAULT_TYPE_ARG.value if FAULT_TYPE_ARG else None,
                "fault_types": [f.value for f in INJECTABLE_FAULTS],
                "sensors": list(SENSOR_INFO.keys()),
                **detection_rates(split_records),
            },
            "records": split_records,
        }
        out_path = (f"{out_dir}/{split}_detectionfault_dataset_{fault_tag}_{sensor_tag}_"
                    f"runs{N_RUNS}_samples{N_SAMPLES}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Wrote {len(split_records)} {split} trials "
              f"({dataset['meta']['n_no_fault_runs']} no-fault / {dataset['meta']['n_fault_runs']} fault) "
              f"to {out_path}")

    print(f"Seed for this run: {seed}")

    #--- per-sample table (temperature, pressure, o2, faulty), every run
    #concatenated as-is -- so this file has all runs, both splits, and both
    #groups (no_fault/fault) present, each with its full N_SAMPLES rows ---
    if SAVE_SIGNALS:
        per_run_tables = [
            appropriate_dataset(r["signals"]["TEMP"], r["signals"]["PRESSURE"], r["signals"]["O2"],
                                r["fault_start"], run_id=r["run_id"], split=r["split"], group=r["group"])
            for r in records
        ]
        all_runs_df = pd.concat(per_run_tables, ignore_index=True)

        combined_out_path = f"{out_dir}/{fault_tag}_{sensor_tag}_appropriate_dataset_all_runs.json"
        all_runs_df.to_json(combined_out_path, orient="records", indent=2)
        counts = all_runs_df.groupby(["split", "group"])["run_id"].nunique()
        print(f"Wrote combined per-sample dataset ({len(all_runs_df)} rows across "
              f"{all_runs_df['run_id'].nunique()} runs) to {combined_out_path}")
        print(counts.to_string())

        #--- val/test as separate features (X) / labels (y) files, so the
        #model can never be fed the answer by accident (see write-up on
        #leakage). Neither file carries `group` -- for test in particular,
        #nothing here tells you which runs were the no_fault one and which
        #was the fault one; `faulty` in y is the only ground truth exposed,
        #and only after predictions are made. Train is left as the single
        #combined file above since a training run legitimately needs
        #features and labels together. ---
        feature_cols = ["run_id", "sample_index", "temperature", "pressure", "o2"]
        label_cols = ["run_id", "sample_index", "faulty"]
        for split in ("val", "test"):
            split_df = all_runs_df.loc[all_runs_df["split"] == split]
            x_path = f"{out_dir}/{split}_X_{fault_tag}_{sensor_tag}_runs{N_RUNS}_samples{N_SAMPLES}.json"
            y_path = f"{out_dir}/{split}_y_{fault_tag}_{sensor_tag}_runs{N_RUNS}_samples{N_SAMPLES}.json"
            split_df[feature_cols].to_json(x_path, orient="records", indent=2)
            split_df[label_cols].to_json(y_path, orient="records", indent=2)
            print(f"Wrote {split} features (no faulty, no group) to {x_path}")
            print(f"Wrote {split} labels ({len(split_df)} rows) to {y_path}")

    #--- illustrative example, plotted -- one train-split trial, one val-split
    #trial, and one test-split trial side by side in a single PNG (train=light
    #blue, val=light green, test=light peach), so the background reflects
    #which data was really used ---
    def pick_example(split: str) -> Optional[dict]:
        candidates = [r for r in records if r["split"] == split and r["fault_occurred"]]
        if not candidates:
            candidates = [r for r in records if r["split"] == split]
        return candidates[-1] if candidates else None

    split_examples = {split: pick_example(split) for split in ("train", "val", "test")}
    example_out_path = f"{out_dir}/illustrative_{fault_tag}_{sensor_tag}.png"
    plot_illustrative_examples(split_examples, N_SAMPLES, example_out_path)
