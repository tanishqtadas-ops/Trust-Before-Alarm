"""Precompute and cache causal feature tables for train and val splits.

Runs causal feature extraction using ProcessPoolExecutor on observable tables.
Filters output to scored rows (timestamp >= 480 and timestamp % 5 == 0).
Saves resulting parquet files to data/features/<split>/.
Strictly isolates observables (never touches ground truth or test split).
"""

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engine import (
    FeatureEngine,
    SENSOR_METADATA_COLS,
    SENSOR_FEATURE_NAMES,
    GROUP_METADATA_COLS,
    GROUP_FEATURE_NAMES,
)


def _process_single_scenario(args: Tuple[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Worker function to compute features for a single scenario and slice to scored rows."""
    scen_id, sub_df = args
    fe = FeatureEngine()
    s_feat, g_feat = fe.compute_scenario_features(sub_df)

    # Filter to scored rows: timestamp >= 480 and timestamp % 5 == 0
    s_scored = s_feat[(s_feat["timestamp"] >= 480) & (s_feat["timestamp"] % 5 == 0)].copy()
    g_scored = g_feat[(g_feat["timestamp"] >= 480) & (g_feat["timestamp"] % 5 == 0)].copy()

    # Cast float64 features to float32 to save memory and disk space
    for col in SENSOR_FEATURE_NAMES:
        if col in s_scored.columns:
            s_scored[col] = s_scored[col].astype("float32")
    for col in GROUP_FEATURE_NAMES:
        if col in g_scored.columns:
            g_scored[col] = g_scored[col].astype("float32")

    return s_scored, g_scored


def build_cache_for_split(
    split: str,
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/features"),
    max_workers: int = 12,
    batch_size: int = 50,
) -> None:
    """Extract features for all scenarios in a split and save scored parquet tables."""
    if split not in ("train", "val"):
        raise ValueError(f"Only 'train' and 'val' splits are permitted. Got: '{split}'")

    split_raw_file = raw_dir / split / "observables.parquet"
    if not split_raw_file.exists():
        raise FileNotFoundError(f"Observables file not found: {split_raw_file}")

    split_out_dir = out_dir / split
    split_out_dir.mkdir(parents=True, exist_ok=True)
    sensor_out_file = split_out_dir / "sensor_features_scored.parquet"
    group_out_file = split_out_dir / "group_features_scored.parquet"

    print(f"[{split.upper()}] Loading observables from {split_raw_file}...")
    t0 = time.time()
    obs = pd.read_parquet(split_raw_file)
    load_time = time.time() - t0
    scenarios = obs["scenario_id"].unique()
    n_scens = len(scenarios)
    print(f"[{split.upper()}] Loaded {len(obs):,} observable rows across {n_scens} scenarios in {load_time:.2f}s")

    # Group observables by scenario
    print(f"[{split.upper()}] Preparing scenario partitions...")
    scen_groups = [
        (scen_id, obs[obs["scenario_id"] == scen_id].copy())
        for scen_id in scenarios
    ]

    all_sensor_dfs: List[pd.DataFrame] = []
    all_group_dfs: List[pd.DataFrame] = []

    print(f"[{split.upper()}] Extracting causal features using {max_workers} worker processes...")
    t_start = time.time()
    done_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for s_df, g_df in executor.map(_process_single_scenario, scen_groups, chunksize=2):
            all_sensor_dfs.append(s_df)
            all_group_dfs.append(g_df)
            done_count += 1
            if done_count % 50 == 0 or done_count == n_scens:
                elapsed = time.time() - t_start
                rate = done_count / elapsed
                remaining = (n_scens - done_count) / rate if rate > 0 else 0
                print(f"[{split.upper()}] Processed {done_count}/{n_scens} scenarios ({done_count/n_scens*100:.1f}%) in {elapsed:.1f}s (~{remaining:.1f}s remaining)")

    print(f"[{split.upper()}] Concatenating feature tables...")
    combined_sensor = pd.concat(all_sensor_dfs, ignore_index=True)
    combined_group = pd.concat(all_group_dfs, ignore_index=True)

    print(f"[{split.upper()}] Sensor scored table: {combined_sensor.shape} ({combined_sensor.memory_usage().sum() / 1e6:.1f} MB)")
    print(f"[{split.upper()}] Group scored table:  {combined_group.shape} ({combined_group.memory_usage().sum() / 1e6:.1f} MB)")

    print(f"[{split.upper()}] Writing to {sensor_out_file}...")
    combined_sensor.to_parquet(sensor_out_file, index=False)
    print(f"[{split.upper()}] Writing to {group_out_file}...")
    combined_group.to_parquet(group_out_file, index=False)

    total_time = time.time() - t_start
    print(f"[{split.upper()}] Completed feature caching in {total_time:.1f}s ({total_time / n_scens:.2f}s per scenario)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute and cache causal feature tables.")
    parser.add_argument("--split", type=str, choices=["val", "train", "both"], default="val", help="Split to process")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel workers")
    args = parser.parse_args()

    if args.split in ("val", "both"):
        build_cache_for_split("val", max_workers=args.workers)
    if args.split in ("train", "both"):
        build_cache_for_split("train", max_workers=args.workers)
