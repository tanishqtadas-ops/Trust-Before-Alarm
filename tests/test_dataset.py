"""Tests for P1.9: Deterministic Batch Dataset Generation, Splitting, Downsampling, and Parquet Export."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sim import (
    APPROVED_FAULT_PROFILES,
    BatchDatasetGenerator,
    DatasetReader,
    ScenarioDescriptor,
    ScenarioSimulator,
    VirtualScenario,
    apportion_splits,
    classify_scenario_stratum,
    extract_scenario_descriptor,
    partition_test_subsets,
    solve_decision_47_apportionment,
)
from src.sim.dataset import (
    EVENT_EPISODES_COLUMNS,
    FAULT_EPISODES_COLUMNS,
    GROUND_TRUTH_EVENTS_COLUMNS,
    GROUND_TRUTH_SENSORS_COLUMNS,
    OBSERVABLES_COLUMNS,
    SCENARIOS_COLUMNS,
    SCHEMA_EVENT_EPISODES,
    SCHEMA_FAULT_EPISODES,
    _empty_event_episodes_dataframe,
    _empty_fault_episodes_dataframe,
    _safe_write_parquet,
)


class TestStratumClassification:
    """Verify stratum classification and 22 fault profiles."""

    def test_22_approved_fault_profiles_exist(self):
        assert len(APPROVED_FAULT_PROFILES) == 22
        assert "healthy" in APPROVED_FAULT_PROFILES
        assert sum(1 for p in APPROVED_FAULT_PROFILES if p.startswith("single_sensor_")) == 7
        assert sum(1 for p in APPROVED_FAULT_PROFILES if p.startswith("shared_cause_")) == 7
        assert sum(1 for p in APPROVED_FAULT_PROFILES if p.startswith("multi_independent_")) == 7

    def test_classify_healthy_scenario(self):
        sim = ScenarioSimulator()
        scen = sim.simulate_scenario(scenario_id="scenario_0001", seed=1)
        tier, profile = classify_scenario_stratum(scen)
        assert tier in {"easy", "medium", "hard"}
        assert profile in APPROVED_FAULT_PROFILES
        if scen.metadata["faulty_sensor_count"] == 0:
            assert profile == "healthy"

    def test_classify_single_sensor_scenario(self):
        # Find a seed that produces k_actual == 1
        sim = ScenarioSimulator()
        for seed in range(1, 50):
            scen = sim.simulate_scenario(scenario_id=f"scenario_{seed:04d}", seed=seed)
            if scen.metadata["faulty_sensor_count"] == 1:
                tier, profile = classify_scenario_stratum(scen)
                assert profile.startswith("single_sensor_")
                assert profile in APPROVED_FAULT_PROFILES
                break

    def test_classify_shared_cause_scenario(self):
        # Find a seed that produces shared cause
        sim = ScenarioSimulator()
        found = False
        for seed in range(1, 100):
            scen = sim.simulate_scenario(scenario_id=f"scenario_{seed:04d}", seed=seed)
            if scen.metadata.get("has_shared_cause", False):
                tier, profile = classify_scenario_stratum(scen)
                assert profile.startswith("shared_cause_")
                assert profile in APPROVED_FAULT_PROFILES
                found = True
                break
        assert found, "Should find at least one shared-cause scenario in 100 seeds"

    def test_classify_multi_independent_scenario(self):
        # Find a seed that produces k_actual >= 2 without shared cause
        sim = ScenarioSimulator()
        found = False
        for seed in range(1, 100):
            scen = sim.simulate_scenario(scenario_id=f"scenario_{seed:04d}", seed=seed)
            if scen.metadata["faulty_sensor_count"] >= 2 and not scen.metadata.get("has_shared_cause", False):
                tier, profile = classify_scenario_stratum(scen)
                assert profile.startswith("multi_independent_")
                assert profile in APPROVED_FAULT_PROFILES
                found = True
                break
        assert found, "Should find at least one multi-independent scenario in 100 seeds"


class TestDecision47Apportionment:
    """Verify Decision 47 integer apportionment and controlled rounding."""

    def test_exact_totals_and_controlled_rounding(self):
        rng = np.random.default_rng(42)
        # 30 strata summing to 1500
        counts = rng.multinomial(1500, [1 / 30] * 30)
        names = [f"stratum_{i:02d}" for i in range(30)]

        alloc = solve_decision_47_apportionment(
            stratum_names=names,
            stratum_counts=counts,
            target_totals=(900, 300, 300),
            weights=(3, 1, 1),
        )

        assert alloc.shape == (30, 3)
        # Exact column totals
        assert np.array_equal(alloc.sum(axis=0), [900, 300, 300])
        # Exact row sums
        assert np.array_equal(alloc.sum(axis=1), counts)

        # Controlled rounding check: |x_{j, c} - T_{j, c}| < 1
        for j in range(30):
            for c in range(3):
                T = counts[j] * [3, 1, 1][c] / 5.0
                assert abs(alloc[j, c] - T) < 1.0

    def test_apportion_splits_with_1500_descriptors(self):
        # Generate 1500 mock descriptors
        descriptors = []
        rng = np.random.default_rng(123)
        tiers = ["easy", "medium", "hard"]
        for i in range(1, 1501):
            tier = tiers[i % 3]
            profile = APPROVED_FAULT_PROFILES[i % 22]
            stratum = f"{tier}__{profile}"
            descriptors.append(
                ScenarioDescriptor(
                    scenario_id=f"scenario_{i:04d}",
                    seed=i,
                    difficulty_tier=tier,
                    fault_profile=profile,
                    stratum_name=stratum,
                )
            )

        splits = apportion_splits(descriptors, master_seed=42)
        assert len(splits) == 1500
        counts = {
            "train": sum(1 for v in splits.values() if v == "train"),
            "val": sum(1 for v in splits.values() if v == "val"),
            "test": sum(1 for v in splits.values() if v == "test"),
        }
        assert counts == {"train": 900, "val": 300, "test": 300}

        # Determinism check: repeated run must yield identical splits
        splits2 = apportion_splits(descriptors, master_seed=42)
        assert splits == splits2


class TestTestSubsetPartitioning:
    """Verify test set partitioning and difficulty parity."""

    def test_even_distribution_parity(self):
        descriptors = []
        for i in range(100):
            descriptors.append(ScenarioDescriptor(f"s_e_{i}", i, "easy", "healthy", "easy__healthy"))
        for i in range(100):
            descriptors.append(ScenarioDescriptor(f"s_m_{i}", 100 + i, "medium", "healthy", "medium__healthy"))
        for i in range(100):
            descriptors.append(ScenarioDescriptor(f"s_h_{i}", 200 + i, "hard", "healthy", "hard__healthy"))

        subsets = partition_test_subsets(descriptors, master_seed=42)
        assert len(subsets) == 300
        n_full = sum(1 for v in subsets.values() if v == "test_full_resolution")
        n_down = sum(1 for v in subsets.values() if v == "test_downsampled")
        assert n_full == 150
        assert n_down == 150

        for t in ["easy", "medium", "hard"]:
            t_full = sum(1 for d in descriptors if d.difficulty_tier == t and subsets[d.scenario_id] == "test_full_resolution")
            t_down = sum(1 for d in descriptors if d.difficulty_tier == t and subsets[d.scenario_id] == "test_downsampled")
            assert t_full == 50
            assert t_down == 50

    def test_odd_distribution_parity(self):
        # 91 easy, 119 medium, 90 hard
        descriptors = []
        for i in range(91):
            descriptors.append(ScenarioDescriptor(f"s_e_{i}", i, "easy", "healthy", "easy__healthy"))
        for i in range(119):
            descriptors.append(ScenarioDescriptor(f"s_m_{i}", 100 + i, "medium", "healthy", "medium__healthy"))
        for i in range(90):
            descriptors.append(ScenarioDescriptor(f"s_h_{i}", 300 + i, "hard", "healthy", "hard__healthy"))

        subsets = partition_test_subsets(descriptors, master_seed=42)
        assert len(subsets) == 300
        n_full = sum(1 for v in subsets.values() if v == "test_full_resolution")
        n_down = sum(1 for v in subsets.values() if v == "test_downsampled")
        assert n_full == 150
        assert n_down == 150

        # Canonical parity rule: first odd tier (easy) gets +1 in full, second odd tier (medium) gets +1 in down
        e_full = sum(1 for d in descriptors if d.difficulty_tier == "easy" and subsets[d.scenario_id] == "test_full_resolution")
        e_down = sum(1 for d in descriptors if d.difficulty_tier == "easy" and subsets[d.scenario_id] == "test_downsampled")
        m_full = sum(1 for d in descriptors if d.difficulty_tier == "medium" and subsets[d.scenario_id] == "test_full_resolution")
        m_down = sum(1 for d in descriptors if d.difficulty_tier == "medium" and subsets[d.scenario_id] == "test_downsampled")
        h_full = sum(1 for d in descriptors if d.difficulty_tier == "hard" and subsets[d.scenario_id] == "test_full_resolution")
        h_down = sum(1 for d in descriptors if d.difficulty_tier == "hard" and subsets[d.scenario_id] == "test_downsampled")

        assert e_full == 46 and e_down == 45
        assert m_full == 59 and m_down == 60
        assert h_full == 45 and h_down == 45


class TestRowStrideAndAlignment:
    """Verify row stride downsampling and timestamp alignment."""

    def test_observables_and_ground_truth_stride_5(self):
        sim = ScenarioSimulator()
        scen = sim.simulate_scenario(scenario_id="scenario_0001", seed=1)

        df_obs_5 = scen.to_observables_dataframe(stride=5)
        df_gts_5 = scen.to_ground_truth_sensors_dataframe(stride=5)
        df_gte_5 = scen.to_ground_truth_events_dataframe(stride=5)

        total_steps = scen.environment.total_steps  # 1920
        expected_steps = total_steps // 5  # 384
        sensors_count = scen.total_sensors

        assert len(df_obs_5) == expected_steps * sensors_count
        assert len(df_gts_5) == expected_steps * sensors_count

        # Check timestamp alignment
        s0 = scen.sensor_ids[0]
        t_obs = df_obs_5[df_obs_5["sensor_id"] == s0]["timestamp"].values
        t_gts = df_gts_5[df_gts_5["sensor_id"] == s0]["timestamp"].values
        assert np.array_equal(t_obs, t_gts)
        assert np.isclose(t_obs[1] - t_obs[0], 5.0)

    def test_stride_1_full_resolution(self):
        sim = ScenarioSimulator()
        scen = sim.simulate_scenario(scenario_id="scenario_0001", seed=1)

        df_obs_1 = scen.to_observables_dataframe(stride=1)
        df_gts_1 = scen.to_ground_truth_sensors_dataframe(stride=1)
        df_gte_1 = scen.to_ground_truth_events_dataframe(stride=1)

        total_steps = scen.environment.total_steps
        sensors_count = scen.total_sensors

        assert len(df_obs_1) == total_steps * sensors_count
        assert len(df_gts_1) == total_steps * sensors_count

        s0 = scen.sensor_ids[0]
        t_obs = df_obs_1[df_obs_1["sensor_id"] == s0]["timestamp"].values
        assert np.isclose(t_obs[1] - t_obs[0], 1.0)


class TestParquetExportAndRoundTrip:
    """Verify Parquet export, schema preservation, and reader round-trip."""

    def test_empty_episode_tables_preserve_schemas(self):
        df_empty_f = _empty_fault_episodes_dataframe()
        df_empty_e = _empty_event_episodes_dataframe()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            f_file = tmp_path / "empty_faults.parquet"
            e_file = tmp_path / "empty_events.parquet"

            _safe_write_parquet(df_empty_f, f_file, schema=SCHEMA_FAULT_EPISODES)
            _safe_write_parquet(df_empty_e, e_file, schema=SCHEMA_EVENT_EPISODES)

            df_f_read = pd.read_parquet(f_file)
            df_e_read = pd.read_parquet(e_file)

            assert list(df_f_read.columns) == FAULT_EPISODES_COLUMNS
            assert list(df_e_read.columns) == EVENT_EPISODES_COLUMNS
            assert len(df_f_read) == 0
            assert len(df_e_read) == 0

    def test_mini_batch_export_and_reader(self):
        # Test batch export on a 4-scenario mini dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_data"
            gen = BatchDatasetGenerator(total_scenarios=4, num_workers=1)

            # Mock descriptors for 4 scenarios
            mock_desc = [
                ScenarioDescriptor("scenario_0001", 1, "easy", "healthy", "easy__healthy"),
                ScenarioDescriptor("scenario_0002", 2, "medium", "healthy", "medium__healthy"),
                ScenarioDescriptor("scenario_0003", 3, "hard", "healthy", "hard__healthy"),
                ScenarioDescriptor("scenario_0004", 4, "easy", "healthy", "easy__healthy"),
            ]

            # Override plan_splits to test all 4 partitions
            splits = {
                "scenario_0001": "train",
                "scenario_0002": "val",
                "scenario_0003": "test",
                "scenario_0004": "test",
            }
            test_subsets = {
                "scenario_0003": "test_full_resolution",
                "scenario_0004": "test_downsampled",
            }
            gen.plan_splits = lambda descriptors=None: (splits, test_subsets, mock_desc)

            stats = gen.build_and_export_dataset(output_dir=out_dir, descriptors=mock_desc)
            assert stats["total_scenarios"] == 4

            # Verify files exist
            reader = DatasetReader(out_dir)
            manifest = reader.load_manifest()
            assert "train" in manifest["partitions"]
            assert "val" in manifest["partitions"]
            assert "test_full_resolution" in manifest["partitions"]
            assert "test_downsampled" in manifest["partitions"]

            # Load tables
            df_obs_train = reader.load_observables("train")
            assert list(df_obs_train.columns) == OBSERVABLES_COLUMNS
            assert df_obs_train["scenario_id"].iloc[0] == "scenario_0001"

            df_scen = reader.load_scenarios(partition=None)
            assert len(df_scen) == 4
            assert list(df_scen.columns) == SCENARIOS_COLUMNS
            assert set(df_scen["split"].unique()).issubset({"train", "val", "test"})

    def test_full_dataset_in_data_dir(self):
        data_dir = Path("data")
        if not (data_dir / "manifest.json").exists():
            pytest.skip("Full dataset in data/ not yet generated")

        reader = DatasetReader(data_dir)
        manifest = reader.load_manifest()

        assert manifest["total_scenarios"] == 1500
        assert manifest["master_seed"] == 42
        assert manifest["global_split_distribution"] == {"train": 900, "val": 300, "test": 300}

        # Check partitions
        parts = manifest["partitions"]
        assert set(parts.keys()) == {"train", "val", "test_full_resolution", "test_downsampled"}
        assert parts["train"]["scenario_count"] == 900
        assert parts["val"]["scenario_count"] == 300
        assert parts["test_full_resolution"]["scenario_count"] == 150
        assert parts["test_downsampled"]["scenario_count"] == 150

        # Master scenarios table
        df_all = reader.load_scenarios(partition=None)
        assert len(df_all) == 1500
        assert list(df_all.columns) == SCENARIOS_COLUMNS
        assert len(df_all["scenario_id"].unique()) == 1500
        assert len(df_all["seed"].unique()) == 1500
        assert set(df_all["split"].unique()) == {"train", "val", "test"}

        # Check all partition tables and column schemas
        for p_name in parts:
            df_obs = reader.load_observables(p_name)
            df_gts = reader.load_ground_truth_sensors(p_name)
            df_gte = reader.load_ground_truth_events(p_name)
            df_fep = reader.load_fault_episodes(p_name)
            df_eep = reader.load_event_episodes(p_name)
            df_sc = reader.load_scenarios(p_name)

            assert list(df_obs.columns) == OBSERVABLES_COLUMNS
            assert list(df_gts.columns) == GROUND_TRUTH_SENSORS_COLUMNS
            assert list(df_gte.columns) == GROUND_TRUTH_EVENTS_COLUMNS
            assert list(df_fep.columns) == FAULT_EPISODES_COLUMNS
            assert list(df_eep.columns) == EVENT_EPISODES_COLUMNS
            assert list(df_sc.columns) == SCENARIOS_COLUMNS

            # Observables has no label leakage
            forbidden = {"true_value", "sensor_untrusted", "genuine_event", "fault_severity", "true_health"}
            assert not any(col in df_obs.columns for col in forbidden)

