import requests
import logging
from utils import load_config, project_path
from data_quality import validate_raw_directory
from raw_ingestion import DEFAULT_SOURCE_FILE_PREFIXES, ingest_resources


logger = logging.getLogger(__name__)


def download_dataset_resources(config_file_path: str):
    """
    Fetches all resources for a given dataset and downloads them to a specified folder in config.
    
    Args:
        config_file_path (str): Path to config file.
    """
    try:
        config_params = load_config(config_file_path)
        download_folder = project_path(config_params['RAW_FOLDER'])
        manifest_folder = project_path(config_params['MANIFEST_FOLDER'])
        source_file_prefixes = tuple(
            config_params.get('RAW_SOURCE_FILE_PREFIXES', DEFAULT_SOURCE_FILE_PREFIXES)
        )
        dataset_name = config_params.get('DATASET_NAME', {})
        api_base_url = config_params.get('API_BASE_URL', {})

        print(f"Raw directory: {download_folder}")

        # Construct the API URL to get dataset metadata
        package_show_url = f"{api_base_url}package_show"
        params = {"id": dataset_name}
        
        print(f"Fetching dataset metadata for '{dataset_name}'...")
        response = requests.get(package_show_url, params=params, timeout=(30, 60))
        response.raise_for_status()  # Raise an HTTPError for bad responses
        
        dataset_data = response.json()
        
        if not dataset_data.get("success"):
            raise RuntimeError(
                "Could not retrieve dataset metadata: "
                f"{dataset_data.get('error', {})}"
            )
            
        resources = dataset_data["result"]["resources"]
        
        if not resources:
            raise RuntimeError(
                f"No resources (files) found for dataset '{dataset_name}'."
            )
            
        print(f"Found {len(resources)} resources. Starting immutable ingestion...")
        manifest_path = ingest_resources(
            resources,
            raw_folder=download_folder,
            manifest_folder=manifest_folder,
            dataset_name=dataset_name,
            source_filename_prefixes=source_file_prefixes,
        )
        print(f"Raw ingestion manifest written to: {manifest_path}")

        report = validate_raw_directory(download_folder, config_params["FIELDS_TO_RETRIEVE"])
        print(f"Raw data quality passed: {report}")
        return manifest_path

    except requests.exceptions.RequestException:
        logger.exception("A request error occurred while extracting river data")
        raise
    except Exception:
        logger.exception("Failed to extract river data")
        raise


if __name__ == "__main__":
    config_file_path = project_path("data_lake", "config.json")
    # Keep the manifest path as the final stdout line so an Airflow
    # BashOperator can safely capture it as the task output/XCom value.
    manifest_path = download_dataset_resources(config_file_path)
    print(manifest_path)
