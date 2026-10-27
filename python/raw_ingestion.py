"""Immutable raw-data ingestion helpers.

Raw files are never opened for writing after publication. Downloads are first
written to a temporary file, checksummed, and then moved to a new destination.
Each ingestion creates a new append-only manifest under ``data_lake/manifests``.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from uuid import uuid4

import requests


MANIFEST_SCHEMA_VERSION = 1
ACTIVE_STATUSES = frozenset(
    {"downloaded", "already_ingested", "rechecked_unchanged", "legacy_existing"}
)
DEFAULT_SOURCE_FILE_PREFIXES = ("monitoring", "monitiring")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def sha256_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(value: str) -> str:
    """Convert a remote name to a safe single-level CSV filename."""
    filename = Path(value).name.strip()
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"[^\w.()\-]+", "_", filename, flags=re.UNICODE)
    if not filename:
        filename = "resource"
    if Path(filename).suffix.lower() != ".csv":
        filename = f"{filename}.csv"
    return filename


def resource_filename(resource: dict[str, Any]) -> str:
    original_name = resource.get("name") or resource.get("id")
    if not original_name and resource.get("url"):
        original_name = Path(urlparse(resource["url"]).path).name
    return safe_filename(str(original_name or "resource"))


def resource_id(resource: dict[str, Any]) -> str | None:
    value = resource.get("id")
    return str(value) if value is not None else None


def remote_fingerprint(resource: dict[str, Any]) -> str | None:
    for key in ("hash", "last_modified", "revision_id", "metadata_modified"):
        value = resource.get(key)
        if value:
            return str(value)
    return None


def manifest_directory(manifest_folder: str | Path) -> Path:
    directory = Path(manifest_folder) / "raw_ingestion"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_previous_records(manifest_folder: str | Path) -> dict[str, dict[str, Any]]:
    """Load the newest known record for each resource/file without mutating it."""
    records: dict[str, dict[str, Any]] = {}
    directory = manifest_directory(manifest_folder)
    manifests = sorted(
        directory.glob("raw_ingestion_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in manifest.get("records", []):
            keys = (
                record.get("resource_id"),
                record.get("source_url"),
                record.get("stored_filename"),
            )
            for key in keys:
                if key:
                    records[str(key)] = record
    return records


def active_records(
    manifest: dict[str, Any],
    filename_prefixes: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the logical source snapshot represented by one manifest.

    New manifests contain an explicit ``active_records`` section. The fallback
    keeps older manifests readable while excluding legacy files that have no
    source/resource provenance.
    """
    explicit = manifest.get("active_records")
    if explicit is not None:
        records = list(explicit)
    else:
        records = [
            record
            for record in manifest.get("records", [])
            if record.get("stored_filename")
            and record.get("status") in ACTIVE_STATUSES
        ]

    if filename_prefixes is not None:
        prefixes = tuple(prefix.lower() for prefix in filename_prefixes)
        records = [
            record
            for record in records
            if str(record.get("stored_filename", "")).lower().startswith(prefixes)
        ]

    seen_resources: set[str] = set()
    seen_files: set[str] = set()
    for record in records:
        stored_filename = record.get("stored_filename")
        if not stored_filename:
            raise ValueError("Active manifest record lacks stored_filename")
        normalized_filename = Path(stored_filename).name
        if normalized_filename != stored_filename:
            raise ValueError(
                f"Active raw filename must be a single filename: {stored_filename!r}"
            )
        if stored_filename in seen_files:
            raise ValueError(f"Active manifest repeats raw file: {stored_filename}")
        seen_files.add(stored_filename)
        resource = record.get("resource_id")
        if resource:
            resource = str(resource)
            if resource in seen_resources:
                raise ValueError(
                    f"Active manifest repeats resource_id: {resource}"
                )
            seen_resources.add(resource)
    if not records:
        raise ValueError("Manifest contains no active raw source records")
    return records


def active_snapshot_checksum(
    manifest: dict[str, Any],
    filename_prefixes: Iterable[str] | None = None,
) -> str:
    """Hash logical active inputs, independent of manifest timestamps/run ID."""
    snapshot = [
        {
            "resource_id": record.get("resource_id"),
            "source_url": record.get("source_url"),
            "original_filename": record.get("original_filename"),
            "stored_filename": record.get("stored_filename"),
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        }
        for record in active_records(manifest, filename_prefixes)
    ]
    serialized = json.dumps(
        sorted(snapshot, key=lambda item: (str(item["resource_id"]), item["stored_filename"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_active_raw_files(
    manifest_path: str | Path,
    raw_folder: str | Path,
    *,
    filename_prefixes: Iterable[str] | None = None,
) -> list[Path]:
    """Resolve and checksum exactly the files selected by one raw manifest."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read raw-ingestion manifest: {path}") from exc

    raw_directory = Path(raw_folder).resolve()
    files: list[Path] = []
    for record in active_records(manifest, filename_prefixes):
        stored_filename = str(record["stored_filename"])
        file_path = (raw_directory / stored_filename).resolve()
        if file_path.parent != raw_directory:
            raise ValueError(f"Active raw file escapes raw directory: {stored_filename}")
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Active raw file listed by manifest is missing: {file_path}"
            )
        expected_checksum = record.get("sha256")
        if not expected_checksum:
            raise ValueError(f"Active raw record lacks sha256: {stored_filename}")
        actual_checksum = sha256_file(file_path)
        if actual_checksum != expected_checksum:
            raise ValueError(f"Active raw checksum mismatch: {file_path}")
        files.append(file_path)
    return files


def previous_record(
    records: dict[str, dict[str, Any]],
    resource: dict[str, Any],
    stored_filename: str,
) -> dict[str, Any] | None:
    for key in (resource_id(resource), resource.get("url"), stored_filename):
        if key and str(key) in records:
            return records[str(key)]
    return None


def legacy_timestamp(file_path: Path) -> str:
    return utc_iso(datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc))


def unique_destination(
    raw_folder: Path,
    preferred_filename: str,
    resource: dict[str, Any],
    checksum: str,
) -> Path:
    preferred = raw_folder / preferred_filename
    if not preferred.exists():
        return preferred

    stem = preferred.stem
    identifier = resource_id(resource) or "resource"
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)
    candidate = raw_folder / f"{stem}__{identifier}__{checksum[:12]}.csv"
    if not candidate.exists():
        return candidate

    suffix = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return raw_folder / f"{stem}__{identifier}__{checksum[:12]}__{suffix}.csv"


def existing_destination(raw_folder: Path, preferred_filename: str) -> Path:
    """Find a legacy filename variant without rewriting it."""
    preferred = raw_folder / preferred_filename
    if preferred.exists():
        return preferred

    if preferred_filename.startswith("monitoring-"):
        legacy = raw_folder / preferred_filename.replace(
            "monitoring-", "monitiring-", 1
        )
        if legacy.exists():
            return legacy
    return preferred


def download_to_temp(url: str, raw_folder: Path) -> tuple[Path, str, int]:
    """Download to a hidden temporary file and return path, checksum, size."""
    temporary_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".download-",
                suffix=".tmp",
                dir=raw_folder,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary.write(chunk)
        return temporary_path, sha256_file(temporary_path), temporary_path.stat().st_size
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_manifest(
    manifest_folder: str | Path,
    dataset_name: str,
    raw_folder: Path,
    records: Iterable[dict[str, Any]],
    source_filename_prefixes: Iterable[str] | None = None,
) -> Path:
    timestamp = utc_now()
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}__{uuid4().hex[:12]}"
    record_list = list(records)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_name": dataset_name,
        "created_at_utc": utc_iso(timestamp),
        "raw_folder": str(raw_folder),
        "records": record_list,
    }
    manifest["active_records"] = active_records(manifest)
    manifest["active_snapshot_checksum"] = active_snapshot_checksum(manifest)
    if source_filename_prefixes is not None:
        manifest["active_source_snapshot_checksum"] = active_snapshot_checksum(
            manifest,
            source_filename_prefixes,
        )

    directory = manifest_directory(manifest_folder)
    final_path = directory / f"raw_ingestion_{run_id}.json"
    temporary_path = directory / f".{final_path.name}.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(final_path)
    return final_path


def ingest_resources(
    resources: Iterable[dict[str, Any]],
    *,
    raw_folder: str | Path,
    manifest_folder: str | Path,
    dataset_name: str,
    source_filename_prefixes: Iterable[str] | None = None,
) -> Path:
    """Ingest resources without overwriting any existing raw file."""
    raw_directory = Path(raw_folder)
    raw_directory.mkdir(parents=True, exist_ok=True)
    records = load_previous_records(manifest_folder)
    ingestion_records: list[dict[str, Any]] = []

    for resource in resources:
        source_url = resource.get("url")
        original_filename = str(
            resource.get("name")
            or (Path(urlparse(source_url).path).name if source_url else "resource")
        )
        preferred_filename = resource_filename(resource)
        prior = previous_record(records, resource, preferred_filename)
        if prior and prior.get("stored_filename"):
            destination = raw_directory / str(prior["stored_filename"])
        else:
            destination = existing_destination(raw_directory, preferred_filename)
        observed_at = utc_iso()
        fingerprint = remote_fingerprint(resource)

        if not source_url:
            ingestion_records.append(
                {
                    "resource_id": resource_id(resource),
                    "source_url": None,
                    "original_filename": original_filename,
                    "stored_filename": None,
                    "downloaded_at_utc": None,
                    "observed_at_utc": observed_at,
                    "sha256": None,
                    "size_bytes": None,
                    "remote_fingerprint": fingerprint,
                    "status": "skipped_no_url",
                }
            )
            continue

        if destination.exists() and prior is not None:
            remote_changed = (
                fingerprint is not None
                and prior.get("remote_fingerprint") is not None
                and fingerprint != prior.get("remote_fingerprint")
            )
            fingerprint_confirmed_unchanged = (
                fingerprint is not None
                and prior.get("remote_fingerprint") is not None
                and not remote_changed
            )
            if fingerprint_confirmed_unchanged:
                checksum = sha256_file(destination)
                ingestion_records.append(
                    {
                        "resource_id": resource_id(resource),
                        "source_url": source_url,
                        "original_filename": original_filename,
                        "stored_filename": destination.name,
                        "downloaded_at_utc": prior.get("downloaded_at_utc")
                        or legacy_timestamp(destination),
                        "observed_at_utc": observed_at,
                        "sha256": checksum,
                        "size_bytes": destination.stat().st_size,
                        "remote_fingerprint": fingerprint,
                        "status": "already_ingested",
                    }
                )
                continue

            # If the provider does not expose a stable fingerprint, download
            # and compare bytes. This preserves immutability while still
            # detecting an update behind an unchanged URL/resource ID.
            temporary_path, checksum, size_bytes = download_to_temp(
                source_url, raw_directory
            )
            existing_checksum = sha256_file(destination)
            if checksum == existing_checksum:
                temporary_path.unlink(missing_ok=True)
                ingestion_records.append(
                    {
                        "resource_id": resource_id(resource),
                        "source_url": source_url,
                        "original_filename": original_filename,
                        "stored_filename": destination.name,
                        "downloaded_at_utc": prior.get("downloaded_at_utc")
                        or legacy_timestamp(destination),
                        "observed_at_utc": observed_at,
                        "sha256": existing_checksum,
                        "size_bytes": destination.stat().st_size,
                        "remote_fingerprint": fingerprint,
                        "status": "rechecked_unchanged",
                    }
                )
                continue

            try:
                destination = unique_destination(
                    raw_directory,
                    preferred_filename,
                    resource,
                    checksum,
                )
                if destination.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing raw file: {destination}"
                    )
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)

            ingestion_records.append(
                {
                    "resource_id": resource_id(resource),
                    "source_url": source_url,
                    "original_filename": original_filename,
                    "stored_filename": destination.name,
                    "downloaded_at_utc": observed_at,
                    "observed_at_utc": observed_at,
                    "sha256": checksum,
                    "size_bytes": size_bytes,
                    "remote_fingerprint": fingerprint,
                    "status": "downloaded",
                }
            )
            continue

        if destination.exists() and prior is None:
            checksum = sha256_file(destination)
            ingestion_records.append(
                {
                    "resource_id": resource_id(resource),
                    "source_url": source_url,
                    "original_filename": original_filename,
                    "stored_filename": destination.name,
                    "downloaded_at_utc": legacy_timestamp(destination),
                    "observed_at_utc": observed_at,
                    "sha256": checksum,
                    "size_bytes": destination.stat().st_size,
                    "remote_fingerprint": fingerprint,
                    "status": "legacy_existing",
                }
            )
            continue

        temporary_path, checksum, size_bytes = download_to_temp(source_url, raw_directory)
        try:
            destination = unique_destination(
                raw_directory,
                preferred_filename,
                resource,
                checksum,
            )
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing raw file: {destination}"
                )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)

        ingestion_records.append(
            {
                "resource_id": resource_id(resource),
                "source_url": source_url,
                "original_filename": original_filename,
                "stored_filename": destination.name,
                "downloaded_at_utc": observed_at,
                "observed_at_utc": observed_at,
                "sha256": checksum,
                "size_bytes": size_bytes,
                "remote_fingerprint": fingerprint,
                "status": "downloaded",
            }
        )

    recorded_filenames = {
        record.get("stored_filename")
        for record in ingestion_records
        if record.get("stored_filename")
    }
    for legacy_file in sorted(raw_directory.glob("*.csv")):
        if legacy_file.name in recorded_filenames:
            continue
        ingestion_records.append(
            {
                "resource_id": None,
                "source_url": None,
                "original_filename": legacy_file.name,
                "stored_filename": legacy_file.name,
                "downloaded_at_utc": legacy_timestamp(legacy_file),
                "observed_at_utc": utc_iso(),
                "sha256": sha256_file(legacy_file),
                "size_bytes": legacy_file.stat().st_size,
                "remote_fingerprint": None,
                "status": "legacy_unmatched",
            }
        )

    return write_manifest(
        manifest_folder,
        dataset_name,
        raw_directory,
        ingestion_records,
        source_filename_prefixes,
    )
