#!/usr/bin/env python3
"""
Results JSONL to CSV Grouping Statistics Script

This script recursively parses results.jsonl files from model evaluation results
and aggregates them into a unified CSV summary file, grouped by:
  - Entry directory (parent experiment folder)
  - Checkpoint (model save point)
  - Dataset type (e.g., eyeq_restoration_test)
  - Degrade type (e.g., blur, illumination+spot)

Usage:
    python parse_results_jsonl_to_csv.py --input_dirs <dir1> [<dir2> ...] [--output OUTPUT] [--verbose]

Example:
    python parse_results_jsonl_to_csv.py \
        --input_dirs /path/to/stage1_train_pixel_loss_l2_50_eye_0_5_max_T_lr_2_5e-6_pixel_weight_10_ver2 \
        --output results_grouped.csv
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


# ============================================================================
# Configuration Constants
# ============================================================================

# Default input directories (used if --input_dirs is not specified)
# Add your experiment directories here
DEFAULT_INPUT_DIRECTORIES = [
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_train_pixel_loss_l2_50_eye_0_5_max_T_lr_2_5e-6_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_train_pixel_loss_l2_50_eye_0_5_max_T_lr_2_5e-6_pixel_weight_10_ver2",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_train_pixel_loss_l2_50_eye_0_5_max_T_lr_2_5e-6_pixel_weight_10_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_train_50_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_test_pixel_loss_l2_50_eye_ver1_ctu_stage1_comb_v1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_test_1000_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_test_50_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_test_50_eye_ver1",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/results/stage1_test_50_eye_ctu_stage1_comb_v1",
]


# Default output filename
DEFAULT_OUTPUT_FILENAME = 'results_grouped.csv'

# Default output directory (where this script is located)
DEFAULT_OUTPUT_DIR = Path(__file__).parent

# Default logging format
DEFAULT_LOG_FORMAT = '%(levelname)s: %(message)s'

# CSV column names
CSV_COLUMNS = [
    'entry_dir',
    'checkpoint',
    'dataset_type',
    'main_task_type',
    'degrade_type',
    'count',
    'psnr_mean',
    'psnr_std',
    'ssim_mean',
    'ssim_std'
]

# ============================================================================
# Functions
# ============================================================================


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT
    )


def normalize_degrade_type(degrade_type) -> str:
    """
    Normalize degrade_type to a consistent string format.

    - If list/array: sort alphabetically and join with '+'
    - If string: return as-is

    Args:
        degrade_type: Either a list of strings or a single string

    Returns:
        Normalized string representation

    Examples:
        ["illumination", "blur"] -> "blur+illumination"
        ["blur"] -> "blur"
        "low quality fundus image" -> "low quality fundus image"
    """
    if isinstance(degrade_type, list):
        # Sort alphabetically and join with '+'
        return '+'.join(sorted(degrade_type))
    elif isinstance(degrade_type, str):
        return degrade_type
    else:
        return str(degrade_type)


def find_results_jsonl_files(entry_dirs: List[Path]) -> List[Dict]:
    """
    Recursively find all results.jsonl files in the given entry directories.

    Args:
        entry_dirs: List of entry directory paths to search

    Returns:
        List of dicts with file info: {path, entry_dir, checkpoint, dataset_type}
    """
    all_files = []

    for entry_dir in entry_dirs:
        if not entry_dir.exists():
            logging.warning(f"Directory does not exist: {entry_dir}")
            continue

        if not entry_dir.is_dir():
            logging.warning(f"Not a directory: {entry_dir}")
            continue

        # Find all results.jsonl files recursively
        jsonl_files = list(entry_dir.rglob('**/results.jsonl'))
        logging.info(f"Found {len(jsonl_files)} results.jsonl files in {entry_dir.name}")

        for jsonl_file in jsonl_files:
            # Extract metadata from path
            # Structure: entry_dir/checkpoint/dataset_type/results.jsonl
            dataset_type = jsonl_file.parent.name  # e.g., eyeq_restoration_test
            checkpoint = jsonl_file.parent.parent.name  # e.g., stage1_medq_..._0012000

            all_files.append({
                'path': jsonl_file,
                'entry_dir': entry_dir.name,
                'checkpoint': checkpoint,
                'dataset_type': dataset_type
            })

    return all_files


def parse_results_jsonl(file_path: Path) -> List[Dict]:
    """
    Parse a results.jsonl file and extract relevant fields.

    Args:
        file_path: Path to the results.jsonl file

    Returns:
        List of dicts with sample data: {main_task_type, degrade_type, psnr, ssim}
    """
    records = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract required fields
                    main_task_type = data.get('main_task_type', 'unknown')
                    degrade_type = data.get('degrade_type', 'unknown')
                    psnr = data.get('psnr')
                    ssim = data.get('ssim')

                    # Skip if psnr or ssim is missing
                    if psnr is None or ssim is None:
                        logging.debug(f"Skipping line {line_num} in {file_path}: missing psnr or ssim")
                        continue

                    # Normalize degrade_type
                    normalized_degrade_type = normalize_degrade_type(degrade_type)

                    records.append({
                        'main_task_type': main_task_type,
                        'degrade_type': normalized_degrade_type,
                        'psnr': float(psnr),
                        'ssim': float(ssim)
                    })

                except json.JSONDecodeError as e:
                    logging.warning(f"Malformed JSON at line {line_num} in {file_path}: {e}")
                    continue

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

    return records


def group_and_aggregate(all_records: List[Dict]) -> pd.DataFrame:
    """
    Group records by (entry_dir, checkpoint, dataset_type, main_task_type, degrade_type)
    and compute statistics.

    Args:
        all_records: List of dicts with full record info

    Returns:
        DataFrame with grouped statistics
    """
    if not all_records:
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(all_records)

    # Group by the hierarchical keys
    group_keys = ['entry_dir', 'checkpoint', 'dataset_type', 'main_task_type', 'degrade_type']

    # Aggregate statistics
    grouped = df.groupby(group_keys, as_index=False).agg(
        count=('psnr', 'count'),
        psnr_mean=('psnr', 'mean'),
        psnr_std=('psnr', 'std'),
        ssim_mean=('ssim', 'mean'),
        ssim_std=('ssim', 'std')
    )

    # Handle NaN std (when count=1, std is NaN)
    grouped['psnr_std'] = grouped['psnr_std'].fillna(0.0)
    grouped['ssim_std'] = grouped['ssim_std'].fillna(0.0)

    # Sort by entry_dir, checkpoint, dataset_type, then degrade_type
    grouped = grouped.sort_values(
        by=['entry_dir', 'checkpoint', 'dataset_type', 'degrade_type']
    ).reset_index(drop=True)

    return grouped


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Aggregate results.jsonl files into a grouped CSV summary by degrade_type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories (defined in DEFAULT_INPUT_DIRECTORIES)
  python parse_results_jsonl_to_csv.py

  # Single entry directory
  python parse_results_jsonl_to_csv.py --input_dirs /path/to/experiment_dir

  # Multiple entry directories
  python parse_results_jsonl_to_csv.py --input_dirs dir1 dir2 dir3 --output combined_results.csv

  # With verbose logging
  python parse_results_jsonl_to_csv.py --input_dirs /path/to/experiment_dir --verbose
        """
    )

    parser.add_argument(
        '-i', '--input_dirs',
        nargs='+',
        type=str,
        default=None,
        help='One or more entry directories containing checkpoint subdirectories with results.jsonl files (default: use DEFAULT_INPUT_DIRECTORIES)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=f'Output CSV file path (default: {DEFAULT_OUTPUT_FILENAME} in script directory)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logging.info("=" * 60)
    logging.info("Results JSONL to CSV Grouping Statistics")
    logging.info("=" * 60)

    # Use default directories if not specified
    if args.input_dirs:
        entry_dirs = [Path(d) for d in args.input_dirs]
    else:
        entry_dirs = [Path(d) for d in DEFAULT_INPUT_DIRECTORIES]
        logging.info(f"Using default directories ({len(entry_dirs)} directories)")

    logging.info(f"Input directories: {len(entry_dirs)}")

    # Determine output path
    if args.output:
        output_csv = Path(args.output)
    else:
        # Default: save to the script's directory
        output_csv = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME

    # Step 1: Find all results.jsonl files
    logging.info("Step 1: Finding results.jsonl files...")
    file_infos = find_results_jsonl_files(entry_dirs)

    if not file_infos:
        logging.error("No results.jsonl files found!")
        return

    logging.info(f"Total files found: {len(file_infos)}")

    # Step 2: Parse all JSONL files and collect records
    logging.info("Step 2: Parsing JSONL files...")
    all_records = []

    for file_info in file_infos:
        records = parse_results_jsonl(file_info['path'])

        # Add metadata to each record
        for record in records:
            record['entry_dir'] = file_info['entry_dir']
            record['checkpoint'] = file_info['checkpoint']
            record['dataset_type'] = file_info['dataset_type']

        all_records.extend(records)
        logging.debug(f"Parsed {len(records)} records from {file_info['path'].name}")

    if not all_records:
        logging.error("No valid records found in JSONL files!")
        return

    logging.info(f"Total records parsed: {len(all_records)}")

    # Step 3: Group and aggregate
    logging.info("Step 3: Grouping and aggregating statistics...")
    result_df = group_and_aggregate(all_records)

    logging.info(f"Generated {len(result_df)} grouped rows")

    # Step 4: Write to CSV
    logging.info(f"Step 4: Writing results to {output_csv}...")
    result_df.to_csv(output_csv, index=False)

    logging.info("=" * 60)
    logging.info(f"SUCCESS: Wrote {len(result_df)} rows to {output_csv}")
    logging.info("=" * 60)

    # Print summary
    print("\n--- Summary ---")
    print(f"Entry directories processed: {result_df['entry_dir'].nunique()}")
    print(f"Checkpoints found: {result_df['checkpoint'].nunique()}")
    print(f"Dataset types: {result_df['dataset_type'].nunique()}")
    print(f"Unique degrade types: {result_df['degrade_type'].nunique()}")
    print(f"Total grouped rows: {len(result_df)}")
    print(f"Output file: {output_csv}")


if __name__ == '__main__':
    main()
