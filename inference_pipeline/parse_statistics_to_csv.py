#!/usr/bin/env python3
"""
Statistics JSON to CSV Aggregation Script

This script recursively parses statistics.json files from model evaluation results
and aggregates them into a unified CSV summary file.

Usage:
    python parse_statistics_to_csv.py <directory1> [<directory2> ...] [--output OUTPUT] [--verbose]

Example:
    python parse_statistics_to_csv.py /path/to/checkpoint_dir1 /path/to/checkpoint_dir2 -o summary.csv
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


# ============================================================================
# Configuration Constants (Default Values)
# ============================================================================

# Default input directories to search (used if --directories is not specified)
# Modify these paths according to your project structure
DEFAULT_INPUT_DIRECTORIES = [
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/stage1_test_1000_ver1/stage1_medq_2nodes_unif_combined_v1_0004000",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/stage1_test_1000_ver1/stage1_medq_2nodes_unif_combined_v1_0008000",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/stage1_test_1000_ver1/stage1_medq_2nodes_unif_combined_v1_0012000",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/stage1_test_1000_ver1/stage1_medq_2nodes_unif_combined_v1_0016000",
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/stage1_test_1000_ver1/stage1_medq_2nodes_unif_combined_v1_0020000"
]



# Default output filename when not specified
DEFAULT_OUTPUT_FILENAME = 'summary.csv'

# Default logging format
DEFAULT_LOG_FORMAT = '%(levelname)s: %(message)s'

# JSON field names (for easy reference and modification)
JSON_FIELD_OVERALL = 'overall'
JSON_FIELD_BY_TASK_TYPE = 'by_task_type'
JSON_FIELD_TIMESTAMP = 'timestamp'

# CSV base column names (excluding task-specific columns)
DEFAULT_BASE_COLUMNS = [
    'model_id',
    'split',
    'total_samples',
    'psnr_mean',
    'psnr_std',
    'ssim_mean',
    'ssim_std',
    'avg_inference_time'
]

# Task-specific column suffixes (will be prefixed with task1_, task2_, etc.)
DEFAULT_TASK_COLUMN_SUFFIXES = [
    'type',
    'count',
    'psnr_mean',
    'psnr_std',
    'ssim_mean',
    'ssim_std'
]

# Overall metrics field mapping (JSON field -> CSV column)
DEFAULT_OVERALL_METRICS = {
    'total_samples': 'total_samples',
    'psnr_mean': 'psnr_mean',
    'psnr_std': 'psnr_std',
    'ssim_mean': 'ssim_mean',
    'ssim_std': 'ssim_std',
    'avg_inference_time': 'avg_inference_time'
}

# Task metrics field mapping (JSON field -> column suffix)
DEFAULT_TASK_METRICS = {
    'count': 'count',
    'psnr_mean': 'psnr_mean',
    'psnr_std': 'psnr_std',
    'ssim_mean': 'ssim_mean',
    'ssim_std': 'ssim_std'
}

# Empty value placeholder for missing data
DEFAULT_EMPTY_VALUE = ''

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


def find_statistics_files(root_dirs: List[Path]) -> List[Path]:
    """
    Recursively find all statistics.json files in the given root directories.

    Args:
        root_dirs: List of root directory paths to search

    Returns:
        List of paths to statistics.json files
    """
    all_json_files = []

    for root_dir in root_dirs:
        if not root_dir.exists():
            logging.warning(f"Directory does not exist: {root_dir}")
            continue

        if not root_dir.is_dir():
            logging.warning(f"Not a directory: {root_dir}")
            continue

        # Find all statistics.json files recursively
        json_files = list(root_dir.rglob('**/statistics.json'))
        logging.info(f"Found {len(json_files)} statistics.json files in {root_dir}")
        all_json_files.extend(json_files)

    return all_json_files


def extract_metadata_from_path(json_path: Path, root_dirs: List[Path]) -> Tuple[str, str]:
    """
    Extract model_id and split from the statistics.json file path.

    Args:
        json_path: Path to the statistics.json file
        root_dirs: List of root directories (to identify model_id)

    Returns:
        Tuple of (model_id, split)

    Example:
        Input: /path/to/stage1_medq_2nodes_unif_combined_v1_0016000/AAPM-CT-MAR_test/statistics.json
        Output: ('stage1_medq_2nodes_unif_combined_v1_0016000', 'AAPM-CT-MAR_test')
    """
    # split is the immediate parent directory of statistics.json
    split = json_path.parent.name

    # model_id is the checkpoint directory name
    # Try to find which root_dir this file belongs to
    model_id = None
    for root_dir in root_dirs:
        try:
            # Check if json_path is relative to root_dir
            relative_path = json_path.relative_to(root_dir)
            # The first part of the relative path should be the split directory
            # So model_id is the root_dir name itself
            model_id = root_dir.name
            break
        except ValueError:
            # json_path is not relative to this root_dir, try next
            continue

    # If we couldn't find the model_id from root_dirs, use parent's parent
    if model_id is None:
        # Fallback: assume structure is model_id/split/statistics.json
        model_id = json_path.parent.parent.name

    return model_id, split


def determine_max_task_count(all_json_files: List[Path]) -> int:
    """
    First pass: scan all statistics.json files to determine the maximum number of task types.

    Args:
        all_json_files: List of paths to statistics.json files

    Returns:
        Maximum number of task types found across all files
    """
    max_task_count = 0

    for json_file in all_json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if by_task_type exists and is not empty
            if JSON_FIELD_BY_TASK_TYPE in data and data[JSON_FIELD_BY_TASK_TYPE]:
                task_count = len(data[JSON_FIELD_BY_TASK_TYPE])
                max_task_count = max(max_task_count, task_count)
                logging.debug(f"{json_file.name}: {task_count} task types")

        except json.JSONDecodeError as e:
            logging.warning(f"Malformed JSON in {json_file}: {e}")
            continue
        except Exception as e:
            logging.warning(f"Error reading {json_file}: {e}")
            continue

    logging.info(f"Maximum number of task types: {max_task_count}")
    return max_task_count


def parse_statistics_json(
    json_path: Path,
    model_id: str,
    split: str,
    max_tasks: int
) -> Optional[Dict]:
    """
    Parse a statistics.json file and create a row dictionary with flattened task metrics.

    Args:
        json_path: Path to the statistics.json file
        model_id: Model identifier
        split: Data split name
        max_tasks: Maximum number of task types (for column alignment)

    Returns:
        Dictionary representing a row in the CSV, or None if file should be skipped
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Skip if no by_task_type or it's empty
        if JSON_FIELD_BY_TASK_TYPE not in data or not data[JSON_FIELD_BY_TASK_TYPE]:
            logging.warning(f"Skipping {json_path}: no task types in {JSON_FIELD_BY_TASK_TYPE}")
            return None

        # Check if required fields exist in overall section
        if JSON_FIELD_OVERALL not in data:
            logging.warning(f"Skipping {json_path}: missing '{JSON_FIELD_OVERALL}' section")
            return None

        overall = data[JSON_FIELD_OVERALL]

        # Build the row dictionary with metadata
        row = {
            'model_id': model_id,
            'split': split,
        }

        # Add overall metrics using the defined mapping
        for json_field, csv_column in DEFAULT_OVERALL_METRICS.items():
            row[csv_column] = overall.get(json_field, DEFAULT_EMPTY_VALUE)

        # Flatten task types (sorted alphabetically for consistency)
        task_types = sorted(data[JSON_FIELD_BY_TASK_TYPE].keys())

        for i, task_type in enumerate(task_types, start=1):
            task_data = data[JSON_FIELD_BY_TASK_TYPE][task_type]
            row[f'task{i}_type'] = task_type

            # Add task metrics using the defined mapping
            for json_field, column_suffix in DEFAULT_TASK_METRICS.items():
                row[f'task{i}_{column_suffix}'] = task_data.get(json_field, DEFAULT_EMPTY_VALUE)

        # Fill remaining task columns with empty values
        for i in range(len(task_types) + 1, max_tasks + 1):
            row[f'task{i}_type'] = DEFAULT_EMPTY_VALUE
            for column_suffix in DEFAULT_TASK_METRICS.values():
                row[f'task{i}_{column_suffix}'] = DEFAULT_EMPTY_VALUE

        # Add timestamp
        row[JSON_FIELD_TIMESTAMP] = data.get(JSON_FIELD_TIMESTAMP, DEFAULT_EMPTY_VALUE)

        logging.debug(f"Parsed {json_path.name}: {len(task_types)} task types")
        return row

    except json.JSONDecodeError as e:
        logging.error(f"Malformed JSON in {json_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return None


def generate_column_names(max_tasks: int) -> List[str]:
    """
    Generate column names for the CSV based on the maximum number of task types.

    Args:
        max_tasks: Maximum number of task types

    Returns:
        List of column names
    """
    # Start with base columns
    columns = list(DEFAULT_BASE_COLUMNS)

    # Add task1 columns first (if max_tasks >= 1)
    if max_tasks >= 1:
        for suffix in DEFAULT_TASK_COLUMN_SUFFIXES:
            columns.append(f'task1_{suffix}')

        # Add timestamp immediately after task1
        columns.append(JSON_FIELD_TIMESTAMP)

    # Add remaining task columns (task2 onwards)
    for i in range(2, max_tasks + 1):
        for suffix in DEFAULT_TASK_COLUMN_SUFFIXES:
            columns.append(f'task{i}_{suffix}')

    return columns


def aggregate_to_csv(root_dirs: List[Path], output_csv: Path):
    """
    Main orchestration function: aggregates all statistics.json files into a CSV.

    Args:
        root_dirs: List of root directories to search
        output_csv: Output CSV file path
    """
    logging.info("=" * 60)
    logging.info("Statistics JSON to CSV Aggregation")
    logging.info("=" * 60)

    # Step 1: Find all statistics.json files
    logging.info(f"Searching for statistics.json files in {len(root_dirs)} director{'y' if len(root_dirs) == 1 else 'ies'}...")
    all_json_files = find_statistics_files(root_dirs)

    if not all_json_files:
        logging.error("No statistics.json files found!")
        return

    logging.info(f"Total files found: {len(all_json_files)}")

    # Step 2: First pass - determine max task count
    logging.info("Pass 1: Determining maximum number of task types...")
    max_task_count = determine_max_task_count(all_json_files)

    if max_task_count == 0:
        logging.error("No valid statistics.json files with task types found!")
        return

    # Step 3: Generate column names
    columns = generate_column_names(max_task_count)
    logging.info(f"CSV will have {len(columns)} columns")

    # Step 4: Second pass - parse and build rows
    logging.info("Pass 2: Parsing files and building rows...")
    rows = []

    for json_file in all_json_files:
        model_id, split = extract_metadata_from_path(json_file, root_dirs)
        row = parse_statistics_json(json_file, model_id, split, max_task_count)

        if row is not None:
            rows.append(row)

    if not rows:
        logging.error("No valid rows generated!")
        return

    logging.info(f"Successfully parsed {len(rows)} files")

    # Step 5: Write to CSV
    logging.info(f"Writing results to {output_csv}...")
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    logging.info("=" * 60)
    logging.info(f"SUCCESS: Wrote {len(rows)} rows to {output_csv}")
    logging.info("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Aggregate statistics.json files into a unified CSV summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories (defined in script)
  python parse_statistics_to_csv.py

  # Single directory
  python parse_statistics_to_csv.py --directories /path/to/checkpoint_dir

  # Multiple directories
  python parse_statistics_to_csv.py --directories dir1 dir2 dir3 --output combined_summary.csv

  # With verbose logging
  python parse_statistics_to_csv.py --directories /path/to/checkpoint_dir --verbose
        """
    )

    parser.add_argument(
        '-d', '--directories',
        nargs='+',
        type=str,
        default=None,
        help=f'One or more directories containing statistics.json files (default: use DEFAULT_INPUT_DIRECTORIES from script)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=f'Output CSV file path (default: {DEFAULT_OUTPUT_FILENAME} in first input directory)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging first
    setup_logging(args.verbose)

    # Use default directories if not specified
    if args.directories:
        root_dirs = [Path(d) for d in args.directories]
    else:
        # Use default directories from configuration
        root_dirs = [Path(d) for d in DEFAULT_INPUT_DIRECTORIES]
        logging.info(f"Using default directories: {DEFAULT_INPUT_DIRECTORIES}")

    # Determine output path using default constant
    if args.output:
        output_csv = Path(args.output)
    else:
        # Default: DEFAULT_OUTPUT_FILENAME in the first input directory
        output_csv = root_dirs[0] / DEFAULT_OUTPUT_FILENAME

    # Run the aggregation
    aggregate_to_csv(root_dirs, output_csv)


if __name__ == '__main__':
    main()
