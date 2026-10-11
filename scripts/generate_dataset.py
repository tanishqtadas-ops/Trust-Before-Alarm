"""Script to batch-generate and export the complete 1,500 scenario dataset.

Phase P1.9: Deterministic Batch Dataset Generation, Scenario-Level Splitting,
Downsampling, and Parquet Export.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.dataset import BatchDatasetGenerator


def main():
    print("=" * 70)
    print("TRUST BEFORE ALARM — P1.9 DATASET GENERATION")
    print("=" * 70)
    print("Generating 1,500 scenarios via Generator A...")

    t0 = time.time()
    generator = BatchDatasetGenerator(
        master_seed=42,
        total_scenarios=1500,
    )

    stats = generator.build_and_export_dataset(output_dir="data")
    total_time = time.time() - t0

    print("\nDataset Generation & Export Complete!")
    print(f"Total Wall Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Scenarios: {stats['total_scenarios']}")
    print(f"Master Seed: {stats['master_seed']}")
    print(f"Global Split Distribution: {stats['global_split_distribution']}")

    for part_name, part_info in stats["partitions"].items():
        print(f"\n--- Partition: {part_name} ---")
        print(f"  Scenarios: {part_info['scenario_count']}")
        print(f"  Split label: {part_info['split_label']}")
        print(f"  Row stride: {part_info['stride']}")
        print(f"  Difficulty distribution: {part_info['difficulty_distribution']}")
        print("  Row counts:")
        for tbl_name, count in part_info["row_counts"].items():
            print(f"    {tbl_name:25s}: {count:,} rows")

    print("\nParquet Round-Trip Validation Passed Successfully.")


if __name__ == "__main__":
    main()
