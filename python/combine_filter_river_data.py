import pandas as pd
import os
import logging
from utils import load_config, project_path
import numpy as np
from data_quality import validate_combined_file
from lineage import publish_atomic_output, resolve_input_manifest, temporary_output_path
from raw_ingestion import DEFAULT_SOURCE_FILE_PREFIXES, load_active_raw_files


logger = logging.getLogger(__name__)


def combine_filter_river_data(config_file_path: str, is_full_data: bool):
    """
    Loops through CSV files in a directory and combines filtered data into a single CSV using pandas with chunking.

    Args:
        config_file_path (str): Path to config file.
        is_full_data (bool): Flag to switch between full data load and filtered data load.
    """
    temporary_output = None
    try:
        config_params = load_config(config_file_path)
        download_folder = project_path(config_params['RAW_FOLDER'])
        staging_folder = project_path(config_params['BRONZE_FOLDER'])
        column_filter = config_params.get('COLUMN_FILTER', {})
        row_filter = config_params.get('ROW_FILTER', {})
        fields_to_retrieve = config_params.get('FIELDS_TO_RETRIEVE', [])
        source_file_prefixes = tuple(
            config_params.get('RAW_SOURCE_FILE_PREFIXES', DEFAULT_SOURCE_FILE_PREFIXES)
        )

        if is_full_data:
            output_file_name = "combined_data.csv"
        else:
            output_file_name = "combined_filtered_data.csv"
        
        output_file_path = os.path.join(staging_folder, output_file_name)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        temporary_output = temporary_output_path(output_file_path)
        print(f"Writing unpublished bronze output to: {temporary_output}")

        header_written = False
        matched_files = 0
        rows_written = 0
        input_files = []

        manifest_path = resolve_input_manifest(config_file_path)
        active_files = load_active_raw_files(
            manifest_path,
            download_folder,
            filename_prefixes=source_file_prefixes,
        )
        print(f"Using active raw snapshot: {manifest_path}")
        for input_file_path in active_files:
            filename = input_file_path.name
            matched_files += 1
            input_files.append(input_file_path)
            print(f"Processing active raw file: {input_file_path}")

            chunk_iterator = pd.read_csv(
                input_file_path,
                sep=';',
                usecols=fields_to_retrieve,
                chunksize=10000,
            )

            for i, chunk in enumerate(chunk_iterator):
                chunk = chunk.replace('NULL', np.nan)

                if is_full_data:
                    filtered_chunk = chunk
                else:
                    filtered_chunk = chunk[
                        chunk[column_filter].astype('string').str.contains(
                            row_filter,
                            regex=False,
                            na=False,
                        )
                    ]

                filtered_chunk.to_csv(
                    temporary_output,
                    mode='a',
                    header=(not header_written),
                    index=False,
                    encoding='utf-8'
                )
                rows_written += len(filtered_chunk)
                if not header_written:
                    header_written = True # Set flag after writing header
                print(f"  - Processed chunk {i+1} from {filename}")

        if matched_files == 0:
            raise FileNotFoundError(
                f"No active raw files found in manifest '{manifest_path}'"
            )
        if rows_written == 0:
            raise ValueError(
                f"No rows were written to '{output_file_path}'. "
                "Check the source data and configured filters."
            )

        validation_kwargs = {}
        if not is_full_data:
            validation_kwargs = {
                "non_null_columns": ("Controle_Date", column_filter),
                "at_least_one_non_null_columns": (config_params["INDICATOR"],),
                "filter_column": column_filter,
                "filter_value": row_filter,
            }
        report = validate_combined_file(
            temporary_output,
            fields_to_retrieve,
            **validation_kwargs,
        )
        lineage_path = publish_atomic_output(
            temporary_output,
            output_file_path,
            config_file_path,
            "bronze",
            input_files=input_files,
            extra={"table": output_file_name, "row_count": report["rows"]},
        )

        print(f"\nCombined data quality passed: {report}")
        print(f"Combined data written to: {output_file_path}")
        print(f"Lineage metadata written to: {lineage_path}")
    except Exception:
        logger.exception("Failed to combine river data")
        raise
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    config_file_path = project_path("data_lake", "config.json")
    combine_filter_river_data(config_file_path, True)
    combine_filter_river_data(config_file_path, False)
