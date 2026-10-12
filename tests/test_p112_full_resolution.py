"""Test suite for Phase P1.12 full 1-minute resolution dataset export.

Verifies:
1. For ALL scenarios in downsampled partitions ('train', 'val', 'test_downsampled'),
   the every-5th rows (timestamp % 5 == 0) of the new full-resolution observables,
   ground_truth_sensors, and ground_truth_events equal the backup stride-5 files bit for bit.
2. For 'test_full_resolution', all rows equal the backup stride-1 files bit for bit.
3. Episode tables ('fault_episodes', 'event_episodes') and scenario tables ('scenarios')
   in all partitions, as well as the master scenarios table, are identical to the backup bit for bit.
"""

from pathlib import Path
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


BACKUP_DIR = Path("data_backup_p111")
NEW_DIR = Path("data")


@pytest.fixture(scope="module")
def backup_available():
    """Ensure the backup dataset is present for regression testing."""
    if not BACKUP_DIR.exists() or not (BACKUP_DIR / "ground_truth" / "scenarios.parquet").exists():
        pytest.skip("data_backup_p111 directory not available for bit-for-bit regression test.")


def test_master_scenarios_identical(backup_available):
    """Verify master scenarios.parquet is identical to the backup bit for bit."""
    old_master = pd.read_parquet(BACKUP_DIR / "ground_truth" / "scenarios.parquet")
    new_master = pd.read_parquet(NEW_DIR / "ground_truth" / "scenarios.parquet")
    assert len(new_master) == 1500
    assert_frame_equal(old_master, new_master, check_exact=True)


@pytest.mark.parametrize("partition", ["train", "val", "test_downsampled", "test_full_resolution"])
def test_partition_scenario_tables_identical(partition, backup_available):
    """Verify scenarios.parquet within each partition is identical to the backup bit for bit."""
    old_scen = pd.read_parquet(BACKUP_DIR / "ground_truth" / partition / "scenarios.parquet")
    new_scen = pd.read_parquet(NEW_DIR / "ground_truth" / partition / "scenarios.parquet")
    assert_frame_equal(old_scen, new_scen, check_exact=True)


@pytest.mark.parametrize("partition", ["train", "val", "test_downsampled", "test_full_resolution"])
def test_partition_episode_tables_identical(partition, backup_available):
    """Verify fault_episodes and event_episodes are identical to the backup bit for bit."""
    # Fault episodes
    old_fep = pd.read_parquet(BACKUP_DIR / "ground_truth" / partition / "fault_episodes.parquet")
    new_fep = pd.read_parquet(NEW_DIR / "ground_truth" / partition / "fault_episodes.parquet")
    assert_frame_equal(old_fep, new_fep, check_exact=True)

    # Event episodes
    old_eep = pd.read_parquet(BACKUP_DIR / "ground_truth" / partition / "event_episodes.parquet")
    new_eep = pd.read_parquet(NEW_DIR / "ground_truth" / partition / "event_episodes.parquet")
    assert_frame_equal(old_eep, new_eep, check_exact=True)


def test_test_full_resolution_identical(backup_available):
    """Verify test_full_resolution partition matches the backup bit for bit on all rows."""
    part = "test_full_resolution"

    old_obs = pd.read_parquet(BACKUP_DIR / "raw" / part / "observables.parquet")
    new_obs = pd.read_parquet(NEW_DIR / "raw" / part / "observables.parquet")
    assert_frame_equal(old_obs, new_obs, check_exact=True)

    old_gts = pd.read_parquet(BACKUP_DIR / "ground_truth" / part / "ground_truth_sensors.parquet")
    new_gts = pd.read_parquet(NEW_DIR / "ground_truth" / part / "ground_truth_sensors.parquet")
    assert_frame_equal(old_gts, new_gts, check_exact=True)

    old_gte = pd.read_parquet(BACKUP_DIR / "ground_truth" / part / "ground_truth_events.parquet")
    new_gte = pd.read_parquet(NEW_DIR / "ground_truth" / part / "ground_truth_events.parquet")
    assert_frame_equal(old_gte, new_gte, check_exact=True)


@pytest.mark.parametrize("partition", ["train", "val", "test_downsampled"])
def test_downsampled_partitions_every_5th_row_equality(partition, backup_available):
    """Verify downsampled partitions match the backup bit for bit on every 5th row (t % 5 == 0)."""
    # 1. Observables
    old_obs = pd.read_parquet(BACKUP_DIR / "raw" / partition / "observables.parquet")
    new_obs = pd.read_parquet(NEW_DIR / "raw" / partition / "observables.parquet")

    # Stride-5 slice by timestamp rule
    new_obs_5 = new_obs[new_obs["timestamp"] % 5 == 0].reset_index(drop=True)
    assert len(new_obs_5) == len(old_obs)
    assert_frame_equal(old_obs, new_obs_5, check_exact=True)

    # 2. Ground truth sensors
    old_gts = pd.read_parquet(BACKUP_DIR / "ground_truth" / partition / "ground_truth_sensors.parquet")
    new_gts = pd.read_parquet(NEW_DIR / "ground_truth" / partition / "ground_truth_sensors.parquet")

    new_gts_5 = new_gts[new_gts["timestamp"] % 5 == 0].reset_index(drop=True)
    assert len(new_gts_5) == len(old_gts)
    assert_frame_equal(old_gts, new_gts_5, check_exact=True)

    # 3. Ground truth events
    old_gte = pd.read_parquet(BACKUP_DIR / "ground_truth" / partition / "ground_truth_events.parquet")
    new_gte = pd.read_parquet(NEW_DIR / "ground_truth" / partition / "ground_truth_events.parquet")

    new_gte_5 = new_gte[new_gte["timestamp"] % 5 == 0].reset_index(drop=True)
    assert len(new_gte_5) == len(old_gte)
    assert_frame_equal(old_gte, new_gte_5, check_exact=True)
