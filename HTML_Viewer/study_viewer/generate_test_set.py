from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "test_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_PATH = Path("/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/outputs/scalar_regression_run/experiment_20260129_150545/data_splits_fatassigned.json")

BINS = [0.0, 0.05, 0.15, 0.25, 1.0]
BIN_LABELS = ["0-0.05", "0.05-0.15", "0.15-0.25", "0.25-1.0"]
MIN_BIN_COUNTS = {"0.15-0.25": 10, "0.25-1.0": 10}
TARGET_N = 150
SEED = 1337
MAX_TRIES = 10000


def load_splits(path: Path) -> Tuple[List[str], Dict[str, float]]:
    with path.open() as f:
        data = json.load(f)
    test_ids = list(data.get("test", []))
    medians = data.get("median_ff", {})
    return test_ids, {k: float(v) for k, v in medians.items()}


def bin_label(value: float) -> str:
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        if i == len(BINS) - 2:
            if lo <= value <= hi:
                return BIN_LABELS[i]
        if lo <= value < hi:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


def sample_cases(test_ids: List[str], medians: Dict[str, float]) -> List[str]:
    valid = [pid for pid in test_ids if pid in medians]
    rng = random.Random(SEED)

    bins: Dict[str, List[str]] = {label: [] for label in BIN_LABELS}
    for pid in valid:
        bins[bin_label(medians[pid])].append(pid)

    # Ensure we can satisfy constraints
    for label, minimum in MIN_BIN_COUNTS.items():
        if len(bins[label]) < minimum:
            raise RuntimeError(f"Not enough samples in bin {label}: {len(bins[label])} < {minimum}")

    # Randomly pick minimum required from constrained bins, then fill the rest
    selected = []
    for label, minimum in MIN_BIN_COUNTS.items():
        picked = rng.sample(bins[label], k=minimum)
        selected.extend(picked)

    remaining = [pid for pid in valid if pid not in set(selected)]
    need = TARGET_N - len(selected)
    if need < 0:
        raise RuntimeError("Minimum bin constraints exceed target sample size.")
    selected.extend(rng.sample(remaining, k=need))

    # Shuffle for random order
    rng.shuffle(selected)
    return selected


def write_yaml(payload: dict, path: Path) -> None:
    if yaml is None:
        # minimal YAML writer fallback
        def _dump(obj, indent=0):
            sp = "  " * indent
            if isinstance(obj, dict):
                lines = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{sp}{k}:")
                        lines.extend(_dump(v, indent + 1))
                    else:
                        lines.append(f"{sp}{k}: {v}")
                return lines
            if isinstance(obj, list):
                lines = []
                for v in obj:
                    if isinstance(v, (dict, list)):
                        lines.append(f"{sp}-")
                        lines.extend(_dump(v, indent + 1))
                    else:
                        lines.append(f"{sp}- {v}")
                return lines
            return [f"{sp}{obj}"]

        path.write_text("\n".join(_dump(payload)) + "\n")
    else:
        with path.open("w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)


def plot_distributions(all_vals: List[float], sample_vals: List[float]) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(all_vals, bins=40, alpha=0.6, label="test set")
    plt.hist(sample_vals, bins=40, alpha=0.6, label="selected 150")
    plt.axvline(0.05, color="#999", linestyle="--", linewidth=1)
    plt.axvline(0.15, color="#999", linestyle="--", linewidth=1)
    plt.axvline(0.25, color="#999", linestyle="--", linewidth=1)
    plt.title("Median FF distribution")
    plt.xlabel("Median FF")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ff_distribution_overlay.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(sample_vals, bins=40, alpha=0.8, color="#c55a11")
    plt.axvline(0.05, color="#999", linestyle="--", linewidth=1)
    plt.axvline(0.15, color="#999", linestyle="--", linewidth=1)
    plt.axvline(0.25, color="#999", linestyle="--", linewidth=1)
    plt.title("Selected 150 median FF distribution")
    plt.xlabel("Median FF")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ff_distribution_selected.png", dpi=150)
    plt.close()


def main() -> None:
    test_ids, medians = load_splits(SPLITS_PATH)
    selected = sample_cases(test_ids, medians)

    cases = []
    bin_counts = {label: 0 for label in BIN_LABELS}
    for pid in selected:
        val = medians[pid]
        label = bin_label(val)
        bin_counts[label] += 1
        cases.append({
            "patient_id": pid,
            "median_ff": round(val, 6),
            "bin": label,
        })

    payload = {
        "source": str(SPLITS_PATH),
        "seed": SEED,
        "total_cases": len(cases),
        "bins": BINS,
        "bin_labels": BIN_LABELS,
        "bin_counts": bin_counts,
        "cases": cases,
    }

    out_yaml = OUTPUT_DIR / "study_test_set.yaml"
    write_yaml(payload, out_yaml)

    all_vals = [medians[pid] for pid in test_ids if pid in medians]
    sample_vals = [medians[pid] for pid in selected]
    plot_distributions(all_vals, sample_vals)

    print(f"Wrote {out_yaml}")
    print(f"Bin counts: {bin_counts}")


if __name__ == "__main__":
    main()
