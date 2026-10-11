"""Deterministic batch dataset generation, downsampling, and Parquet export.

Implements Phase P1.9 for Trust Before Alarm:
- Batch generation of exactly 1,500 scenarios deterministically via Generator A.
- Decision 47 Stratified Biproportional Apportionment (900 train, 300 val, 300 test).
- Test set partitioning into test_full_resolution (150, stride 1) and test_downsampled (150, stride 5)
  with canonical difficulty-tier parity.
- Row-stride downsampling of time series tables (observables, ground_truth_sensors, ground_truth_events).
- Deterministic Parquet export under data/raw/ and data/ground_truth/ with strict schema preservation.
- Dataset reader and round-trip validation.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.sim.scenario import ScenarioSimulator, VirtualScenario
from src.sim.strata import (
    ScenarioDescriptor,
    apportion_splits,
    extract_scenario_descriptor,
    partition_test_subsets,
)
from src.utils.config import get_config_dir, load_yaml


# Canonical column definitions
OBSERVABLES_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"
]

GROUND_TRUTH_SENSORS_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "timestamp", "true_value",
    "sensor_untrusted", "f_spike", "f_offset_drift", "f_gain_error",
    "f_frozen", "f_high_noise", "f_dropout", "f_slow_degradation",
    "fault_severity", "true_health", "fault_id"
]

GROUND_TRUTH_EVENTS_COLUMNS = [
    "scenario_id", "group_id", "timestamp", "genuine_event",
    "event_below_detectability", "event_id", "event_magnitude_sigma",
    "true_event_breadth", "true_group_value"
]

FAULT_EPISODES_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "fault_id", "fault_type",
    "start_step", "end_step", "duration", "max_severity", "shape_params"
]

EVENT_EPISODES_COLUMNS = [
    "scenario_id", "event_id", "event_type", "scope", "start_step",
    "end_step", "start_time_minutes", "magnitude", "abs_magnitude",
    "sign", "affected_sensors", "affected_groups", "sensor_attenuations",
    "sensor_delays", "shape_params", "is_detectable"
]

SCENARIOS_COLUMNS = [
    "scenario_id", "split", "generator_version", "seed", "scenario_type",
    "difficulty", "hard_case_type", "sensor_count", "group_count",
    "group_sizes", "length", "sampling", "profile", "quantisation",
    "clip_limits", "unit_scale", "faulty_sensors_count", "faulty_fraction",
    "has_events", "event_count", "has_shared_cause"
]


# Explicit PyArrow schemas for empty table preservation
SCHEMA_FAULT_EPISODES = pa.schema([
    ("scenario_id", pa.string()),
    ("group_id", pa.string()),
    ("sensor_id", pa.string()),
    ("fault_id", pa.string()),
    ("fault_type", pa.string()),
    ("start_step", pa.int64()),
    ("end_step", pa.int64()),
    ("duration", pa.int64()),
    ("max_severity", pa.float64()),
    ("shape_params", pa.string()),
])

SCHEMA_EVENT_EPISODES = pa.schema([
    ("scenario_id", pa.string()),
    ("event_id", pa.string()),
    ("event_type", pa.string()),
    ("scope", pa.string()),
    ("start_step", pa.int64()),
    ("end_step", pa.int64()),
    ("start_time_minutes", pa.float64()),
    ("magnitude", pa.float64()),
    ("abs_magnitude", pa.float64()),
    ("sign", pa.int64()),
    ("affected_sensors", pa.string()),
    ("affected_groups", pa.string()),
    ("sensor_attenuations", pa.string()),
    ("sensor_delays", pa.string()),
    ("shape_params", pa.string()),
    ("is_detectable", pa.bool_()),
])


def _empty_fault_episodes_dataframe() -> pd.DataFrame:
    """Return an empty fault_episodes DataFrame with defined dtypes."""
    return pd.DataFrame({
        "scenario_id": pd.Series(dtype="str"),
        "group_id": pd.Series(dtype="str"),
        "sensor_id": pd.Series(dtype="str"),
        "fault_id": pd.Series(dtype="str"),
        "fault_type": pd.Series(dtype="str"),
        "start_step": pd.Series(dtype="int64"),
        "end_step": pd.Series(dtype="int64"),
        "duration": pd.Series(dtype="int64"),
        "max_severity": pd.Series(dtype="float64"),
        "shape_params": pd.Series(dtype="str"),
    })[FAULT_EPISODES_COLUMNS]


def _empty_event_episodes_dataframe() -> pd.DataFrame:
    """Return an empty event_episodes DataFrame with defined dtypes."""
    return pd.DataFrame({
        "scenario_id": pd.Series(dtype="str"),
        "event_id": pd.Series(dtype="str"),
        "event_type": pd.Series(dtype="str"),
        "scope": pd.Series(dtype="str"),
        "start_step": pd.Series(dtype="int64"),
        "end_step": pd.Series(dtype="int64"),
        "start_time_minutes": pd.Series(dtype="float64"),
        "magnitude": pd.Series(dtype="float64"),
        "abs_magnitude": pd.Series(dtype="float64"),
        "sign": pd.Series(dtype="int64"),
        "affected_sensors": pd.Series(dtype="str"),
        "affected_groups": pd.Series(dtype="str"),
        "sensor_attenuations": pd.Series(dtype="str"),
        "sensor_delays": pd.Series(dtype="str"),
        "shape_params": pd.Series(dtype="str"),
        "is_detectable": pd.Series(dtype="bool"),
    })[EVENT_EPISODES_COLUMNS]


# Top-level multiprocessing worker functions
def _worker_generate_descriptor(seed: int, config: Optional[Dict[str, Any]] = None) -> ScenarioDescriptor:
    """Worker function to simulate a scenario and extract its descriptor."""
    sim = ScenarioSimulator(config=config)
    scenario_id = f"scenario_{seed:04d}"
    scen = sim.simulate_scenario(scenario_id=scenario_id, seed=seed)
    return extract_scenario_descriptor(scen)


def _worker_generate_and_export(
    task: Tuple[int, str, int, Optional[Dict[str, Any]]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Worker function to simulate scenario and export all 6 tables.
    
    task tuple: (seed, split, stride, config)
    """
    seed, split, stride, config = task
    sim = ScenarioSimulator(config=config)
    scenario_id = f"scenario_{seed:04d}"
    scen = sim.simulate_scenario(scenario_id=scenario_id, seed=seed)

    df_obs = scen.to_observables_dataframe(stride=stride)
    df_gt_s = scen.to_ground_truth_sensors_dataframe(stride=stride)
    df_gt_e = scen.to_ground_truth_events_dataframe(stride=stride)
    df_f_ep = scen.to_fault_episodes_dataframe()
    df_e_ep = scen.to_event_episodes_dataframe()
    df_scen = scen.to_scenario_dataframe(split=split)

    return df_obs, df_gt_s, df_gt_e, df_f_ep, df_e_ep, df_scen


class BatchDatasetGenerator:
    """Orchestrator for batch generation, stratification, splitting, and Parquet export."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        master_seed: int = 42,
        total_scenarios: int = 1500,
        num_workers: Optional[int] = None,
    ):
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

        self.master_seed = master_seed
        self.total_scenarios = total_scenarios
        # Default workers to min(12, os.cpu_count() or 4)
        if num_workers is None:
            cpus = os.cpu_count() or 4
            self.num_workers = max(1, min(12, cpus))
        else:
            self.num_workers = max(1, num_workers)

    def generate_all_descriptors(self) -> List[ScenarioDescriptor]:
        """Generate all 1,500 scenario descriptors deterministically."""
        seeds = list(range(1, self.total_scenarios + 1))
        tasks = [(s, self.config) for s in seeds]

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Use helper wrapper
                futures = executor.map(_worker_generate_descriptor_wrapper, tasks)
                descriptors = list(futures)
        else:
            sim = ScenarioSimulator(config=self.config)
            descriptors = [
                extract_scenario_descriptor(sim.simulate_scenario(f"scenario_{s:04d}", seed=s))
                for s in seeds
            ]

        descriptors.sort(key=lambda d: d.seed)
        return descriptors

    def plan_splits(
        self, descriptors: Optional[List[ScenarioDescriptor]] = None
    ) -> Tuple[Dict[str, str], Dict[str, str], List[ScenarioDescriptor]]:
        """Plan whole-scenario assignments to splits and test subsets.
        
        Returns:
            (scenario_splits, test_subsets, descriptors)
            - scenario_splits: scenario_id -> 'train', 'val', or 'test'
            - test_subsets: scenario_id -> 'test_full_resolution' or 'test_downsampled'
        """
        if descriptors is None:
            descriptors = self.generate_all_descriptors()

        # Step 2: Apportion splits following Decision 47
        scenario_splits = apportion_splits(descriptors, master_seed=self.master_seed)

        # Step 3: Partition test set into test_full_resolution and test_downsampled
        test_descriptors = [d for d in descriptors if scenario_splits[d.scenario_id] == "test"]
        test_subsets = partition_test_subsets(test_descriptors, master_seed=self.master_seed)

        return scenario_splits, test_subsets, descriptors

    def build_and_export_dataset(
        self,
        output_dir: Union[str, Path] = "data",
        descriptors: Optional[List[ScenarioDescriptor]] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Generate and export the complete 1,500 scenario dataset to Parquet.
        
        Directory layout:
        - data/raw/{train, val, test_full_resolution, test_downsampled}/observables.parquet
        - data/ground_truth/{train, val, test_full_resolution, test_downsampled}/
            ground_truth_sensors.parquet
            ground_truth_events.parquet
            fault_episodes.parquet
            event_episodes.parquet
            scenarios.parquet
        - data/ground_truth/scenarios.parquet (all 1500 scenarios)
        - data/manifest.json
        """
        out_path = Path(output_dir)
        raw_path = out_path / "raw"
        gt_path = out_path / "ground_truth"

        scenario_splits, test_subsets, descriptors = self.plan_splits(descriptors)
        desc_by_id = {d.scenario_id: d for d in descriptors}

        # Partition definitions: name -> (scenario_ids, split_label, stride)
        partitions: Dict[str, Tuple[List[str], str, int]] = {
            "train": (
                [d.scenario_id for d in descriptors if scenario_splits[d.scenario_id] == "train"],
                "train",
                5,
            ),
            "val": (
                [d.scenario_id for d in descriptors if scenario_splits[d.scenario_id] == "val"],
                "val",
                5,
            ),
            "test_full_resolution": (
                [d.scenario_id for d in descriptors if test_subsets.get(d.scenario_id) == "test_full_resolution"],
                "test",
                1,
            ),
            "test_downsampled": (
                [d.scenario_id for d in descriptors if test_subsets.get(d.scenario_id) == "test_downsampled"],
                "test",
                5,
            ),
        }

        # Prepare directories
        for part_name in partitions:
            (raw_path / part_name).mkdir(parents=True, exist_ok=True)
            (gt_path / part_name).mkdir(parents=True, exist_ok=True)

        manifest_stats: Dict[str, Any] = {
            "total_scenarios": self.total_scenarios,
            "master_seed": self.master_seed,
            "partitions": {},
            "table_row_counts": {},
        }

        all_scenarios_dfs: List[pd.DataFrame] = []

        # Process each partition sequentially to maintain bounded RAM usage
        for part_name, (scen_ids, split_label, stride) in partitions.items():
            tasks = [
                (desc_by_id[sid].seed, split_label, stride, self.config)
                for sid in scen_ids
            ]

            obs_list: List[pd.DataFrame] = []
            gt_s_list: List[pd.DataFrame] = []
            gt_e_list: List[pd.DataFrame] = []
            f_ep_list: List[pd.DataFrame] = []
            e_ep_list: List[pd.DataFrame] = []
            scen_list: List[pd.DataFrame] = []

            # Execute in batches if large
            if self.num_workers > 1:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    for batch_start in range(0, len(tasks), batch_size):
                        batch = tasks[batch_start : batch_start + batch_size]
                        results = list(executor.map(_worker_generate_and_export, batch))
                        for o, gs, ge, f, e, s in results:
                            obs_list.append(o)
                            gt_s_list.append(gs)
                            gt_e_list.append(ge)
                            if len(f) > 0:
                                f_ep_list.append(f)
                            if len(e) > 0:
                                e_ep_list.append(e)
                            scen_list.append(s)
            else:
                for t in tasks:
                    o, gs, ge, f, e, s = _worker_generate_and_export(t)
                    obs_list.append(o)
                    gt_s_list.append(gs)
                    gt_e_list.append(ge)
                    if len(f) > 0:
                        f_ep_list.append(f)
                    if len(e) > 0:
                        e_ep_list.append(e)
                    scen_list.append(s)

            # Concatenate partition dataframes
            df_obs_part = pd.concat(obs_list, ignore_index=True)[OBSERVABLES_COLUMNS] if obs_list else pd.DataFrame(columns=OBSERVABLES_COLUMNS)
            df_gts_part = pd.concat(gt_s_list, ignore_index=True)[GROUND_TRUTH_SENSORS_COLUMNS] if gt_s_list else pd.DataFrame(columns=GROUND_TRUTH_SENSORS_COLUMNS)
            df_gte_part = pd.concat(gt_e_list, ignore_index=True)[GROUND_TRUTH_EVENTS_COLUMNS] if gt_e_list else pd.DataFrame(columns=GROUND_TRUTH_EVENTS_COLUMNS)
            df_fep_part = pd.concat(f_ep_list, ignore_index=True)[FAULT_EPISODES_COLUMNS] if f_ep_list else _empty_fault_episodes_dataframe()
            df_eep_part = pd.concat(e_ep_list, ignore_index=True)[EVENT_EPISODES_COLUMNS] if e_ep_list else _empty_event_episodes_dataframe()
            df_scen_part = pd.concat(scen_list, ignore_index=True)[SCENARIOS_COLUMNS] if scen_list else pd.DataFrame(columns=SCENARIOS_COLUMNS)

            # Export to Parquet files safely
            obs_file = raw_path / part_name / "observables.parquet"
            gts_file = gt_path / part_name / "ground_truth_sensors.parquet"
            gte_file = gt_path / part_name / "ground_truth_events.parquet"
            fep_file = gt_path / part_name / "fault_episodes.parquet"
            eep_file = gt_path / part_name / "event_episodes.parquet"
            scen_file = gt_path / part_name / "scenarios.parquet"

            _safe_write_parquet(df_obs_part, obs_file)
            _safe_write_parquet(df_gts_part, gts_file)
            _safe_write_parquet(df_gte_part, gte_file)
            _safe_write_parquet(df_fep_part, fep_file, schema=SCHEMA_FAULT_EPISODES)
            _safe_write_parquet(df_eep_part, eep_file, schema=SCHEMA_EVENT_EPISODES)
            _safe_write_parquet(df_scen_part, scen_file)

            all_scenarios_dfs.append(df_scen_part)

            # Record stats
            part_stats = {
                "scenario_count": len(scen_ids),
                "split_label": split_label,
                "stride": stride,
                "row_counts": {
                    "observables": len(df_obs_part),
                    "ground_truth_sensors": len(df_gts_part),
                    "ground_truth_events": len(df_gte_part),
                    "fault_episodes": len(df_fep_part),
                    "event_episodes": len(df_eep_part),
                    "scenarios": len(df_scen_part),
                },
                "difficulty_distribution": df_scen_part["difficulty"].value_counts().to_dict(),
            }
            manifest_stats["partitions"][part_name] = part_stats

        # Write overall scenarios.parquet under data/ground_truth/
        df_all_scenarios = pd.concat(all_scenarios_dfs, ignore_index=True).sort_values("seed")[SCENARIOS_COLUMNS]
        all_scen_file = gt_path / "scenarios.parquet"
        _safe_write_parquet(df_all_scenarios, all_scen_file)

        manifest_stats["total_scenarios_written"] = len(df_all_scenarios)
        manifest_stats["global_split_distribution"] = df_all_scenarios["split"].value_counts().to_dict()

        # Write manifest.json
        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as mf:
            json.dump(manifest_stats, mf, indent=2)

        # Validate round-trip immediately
        self.validate_dataset(out_path)

        return manifest_stats

    @staticmethod
    def validate_dataset(dataset_dir: Union[str, Path]) -> None:
        """Validate all exported Parquet tables by reading them back and verifying schemas."""
        reader = DatasetReader(dataset_dir)
        manifest = reader.load_manifest()

        # Validate all partitions
        for part_name, meta in manifest["partitions"].items():
            df_obs = reader.load_observables(part_name)
            df_gts = reader.load_ground_truth_sensors(part_name)
            df_gte = reader.load_ground_truth_events(part_name)
            df_fep = reader.load_fault_episodes(part_name)
            df_eep = reader.load_event_episodes(part_name)
            df_scen = reader.load_scenarios(part_name)

            assert list(df_obs.columns) == OBSERVABLES_COLUMNS
            assert list(df_gts.columns) == GROUND_TRUTH_SENSORS_COLUMNS
            assert list(df_gte.columns) == GROUND_TRUTH_EVENTS_COLUMNS
            assert list(df_fep.columns) == FAULT_EPISODES_COLUMNS
            assert list(df_eep.columns) == EVENT_EPISODES_COLUMNS
            assert list(df_scen.columns) == SCENARIOS_COLUMNS

            assert len(df_obs) == meta["row_counts"]["observables"]
            assert len(df_gts) == meta["row_counts"]["ground_truth_sensors"]
            assert len(df_gte) == meta["row_counts"]["ground_truth_events"]
            assert len(df_fep) == meta["row_counts"]["fault_episodes"]
            assert len(df_eep) == meta["row_counts"]["event_episodes"]
            assert len(df_scen) == meta["row_counts"]["scenarios"]

            # Observables and ground truth timestamp match
            if len(df_obs) > 0 and len(df_gts) > 0:
                s0 = df_obs["sensor_id"].iloc[0]
                sc0 = df_obs["scenario_id"].iloc[0]
                t_obs = df_obs[(df_obs["sensor_id"] == s0) & (df_obs["scenario_id"] == sc0)]["timestamp"].values
                t_gts = df_gts[(df_gts["sensor_id"] == s0) & (df_gts["scenario_id"] == sc0)]["timestamp"].values
                assert np.array_equal(t_obs, t_gts), f"Timestamp mismatch in {part_name} for sensor {s0}"

        # Validate master scenarios table
        df_all_scen = reader.load_scenarios(partition=None)
        assert list(df_all_scen.columns) == SCENARIOS_COLUMNS
        assert len(df_all_scen) == manifest["total_scenarios"]
        assert set(df_all_scen["split"].unique()).issubset({"train", "val", "test"})


def _worker_generate_descriptor_wrapper(task: Tuple[int, Optional[Dict[str, Any]]]) -> ScenarioDescriptor:
    seed, config = task
    return _worker_generate_descriptor(seed, config)


def _safe_write_parquet(df: pd.DataFrame, file_path: Path, schema: Optional[pa.Schema] = None) -> None:
    """Safely write DataFrame to Parquet, replacing any existing file cleanly."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(".tmp.parquet")
    if tmp_path.exists():
        tmp_path.unlink()

    if schema is not None:
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, tmp_path)
    else:
        df.to_parquet(tmp_path, index=False)

    if file_path.exists():
        file_path.unlink()
    shutil.move(str(tmp_path), str(file_path))


class DatasetReader:
    """Reader interface for loading exported Parquet dataset partitions."""

    def __init__(self, dataset_dir: Union[str, Path] = "data"):
        self.dataset_dir = Path(dataset_dir)
        self.raw_dir = self.dataset_dir / "raw"
        self.gt_dir = self.dataset_dir / "ground_truth"

    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest.json metadata."""
        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_observables(self, partition: str) -> pd.DataFrame:
        """Load observables table for a specific partition."""
        p = self.raw_dir / partition / "observables.parquet"
        return pd.read_parquet(p)

    def load_ground_truth_sensors(self, partition: str) -> pd.DataFrame:
        """Load ground_truth_sensors table for a specific partition."""
        p = self.gt_dir / partition / "ground_truth_sensors.parquet"
        return pd.read_parquet(p)

    def load_ground_truth_events(self, partition: str) -> pd.DataFrame:
        """Load ground_truth_events table for a specific partition."""
        p = self.gt_dir / partition / "ground_truth_events.parquet"
        return pd.read_parquet(p)

    def load_fault_episodes(self, partition: str) -> pd.DataFrame:
        """Load fault_episodes table for a specific partition."""
        p = self.gt_dir / partition / "fault_episodes.parquet"
        return pd.read_parquet(p)

    def load_event_episodes(self, partition: str) -> pd.DataFrame:
        """Load event_episodes table for a specific partition."""
        p = self.gt_dir / partition / "event_episodes.parquet"
        return pd.read_parquet(p)

    def load_scenarios(self, partition: Optional[str] = None) -> pd.DataFrame:
        """Load scenarios table for a specific partition or all 1,500 scenarios if partition is None."""
        if partition is None:
            p = self.gt_dir / "scenarios.parquet"
        else:
            p = self.gt_dir / partition / "scenarios.parquet"
        return pd.read_parquet(p)
