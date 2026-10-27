"""Lineage metadata for files produced after raw ingestion.

Every published downstream artifact gets a small JSON sidecar named
``<artifact>.lineage.json``.  The sidecar identifies the raw-ingestion run
that supplied the artifact and records checksums for the input and output
files.  ``WATER_ANALYZER_INPUT_MANIFEST`` can be set by Airflow so every task
in a DAG run uses the exact same raw manifest instead of discovering a newer
manifest from the filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LINEAGE_SCHEMA_VERSION = 1
INPUT_MANIFEST_ENV = "WATER_ANALYZER_INPUT_MANIFEST"


def sha256_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _configured_path(config_path: Path, configured_value: str) -> Path:
    value = Path(configured_value)
    if value.is_absolute():
        return value
    return config_path.parent.parent.joinpath(value)


def _repository_relative(path: Path, config_path: Path) -> str:
    repository_root = config_path.parent.parent
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_candidates(config_path: Path) -> list[Path]:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    manifest_folder = _configured_path(config_path, config["MANIFEST_FOLDER"])
    candidates = list((manifest_folder / "raw_ingestion").glob("raw_ingestion_*.json"))
    # The filename timestamp has one-second precision and the suffix is a
    # random UUID. Use filesystem publication order as the primary fallback
    # when Airflow has not pinned WATER_ANALYZER_INPUT_MANIFEST.
    return sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_input_manifest(config_path: str | Path) -> Path:
    """Resolve the raw manifest for this pipeline run.

    Airflow should provide ``WATER_ANALYZER_INPUT_MANIFEST``.  Falling back to
    the newest manifest keeps standalone CLI execution convenient.
    """
    config_file = Path(config_path).resolve()
    configured_manifest = os.getenv(INPUT_MANIFEST_ENV)
    if configured_manifest:
        manifest_path = Path(configured_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Configured input manifest does not exist: {manifest_path}"
            )
        return manifest_path

    candidates = _manifest_candidates(config_file)
    if not candidates:
        raise FileNotFoundError(
            "No raw-ingestion manifest found. Run raw ingestion before a "
            "downstream stage."
        )
    return candidates[-1].resolve()


def load_lineage_context(config_path: str | Path) -> dict[str, Any]:
    """Return the immutable lineage identity for the current pipeline run."""
    config_file = Path(config_path).resolve()
    manifest_path = resolve_input_manifest(config_file)
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read input manifest: {manifest_path}") from exc

    run_id = manifest.get("run_id")
    if not run_id:
        raise ValueError(f"Input manifest lacks run_id: {manifest_path}")

    return {
        "run_id": str(run_id),
        "input_manifest_path": _repository_relative(manifest_path, config_file),
        "input_manifest_checksum": sha256_file(manifest_path),
    }


def _file_records(paths: Iterable[str | Path], config_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lineage input file does not exist: {path}")
        records.append(
            {
                "path": _repository_relative(path, config_path),
                "sha256": sha256_file(path),
            }
        )
    return records


def lineage_sidecar_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.name}.lineage.json")


def temporary_output_path(output_path: str | Path) -> Path:
    """Create a same-directory temporary path for an unpublished artifact."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _build_output_lineage(
    output_path: Path,
    checksum_path: Path,
    config_path: Path,
    stage: str,
    input_files: Iterable[str | Path],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": utc_now(),
        **load_lineage_context(config_path),
        "output": {
            "path": _repository_relative(output_path, config_path),
            "sha256": sha256_file(checksum_path),
            "size_bytes": checksum_path.stat().st_size,
        },
        "input_files": _file_records(input_files, config_path),
    }
    if extra:
        metadata["metadata"] = extra
    return metadata


def _write_lineage_json(sidecar: Path, metadata: dict[str, Any]) -> None:
    temporary = sidecar.with_name(
        f".{sidecar.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def write_output_lineage(
    output_path: str | Path,
    config_path: str | Path,
    stage: str,
    *,
    input_files: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Publish lineage metadata after an output has been successfully written."""
    config_file = Path(config_path).resolve()
    output = Path(output_path).resolve()
    if not output.is_file():
        raise FileNotFoundError(f"Cannot write lineage for missing output: {output}")

    sidecar = lineage_sidecar_path(output)
    _write_lineage_json(
        sidecar,
        _build_output_lineage(
            output,
            output,
            config_file,
            stage,
            input_files,
            extra,
        ),
    )
    return sidecar


def publish_atomic_output(
    temporary_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    stage: str,
    *,
    input_files: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Publish an output and lineage sidecar after successful validation.

    The temporary artifact must be in the same filesystem as ``output_path``
    so ``os.replace`` is atomic.  The old final artifact remains untouched if
    lineage construction or sidecar writing fails before publication.
    """
    temporary = Path(temporary_path).resolve()
    output = Path(output_path).resolve()
    config_file = Path(config_path).resolve()
    if not temporary.is_file():
        raise FileNotFoundError(f"Temporary output does not exist: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = _build_output_lineage(
        output,
        temporary,
        config_file,
        stage,
        input_files,
        extra,
    )
    final_sidecar = lineage_sidecar_path(output)
    temporary_sidecar = final_sidecar.with_name(
        f".{final_sidecar.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        os.replace(temporary_sidecar, final_sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return final_sidecar
