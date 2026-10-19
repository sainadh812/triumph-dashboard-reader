"""Public Release download and strict hash-verified extraction; no Torch import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from .contracts import TASKS, ARCHITECTURE_ID, PREPROCESSING_ID
from .release import RELEASE_TAG, WEIGHTS_FILENAME, WEIGHTS_URL, WEIGHTS_SHA256

EXPECTED_PATHS = {f"{task}/best_model.pt" for task in TASKS}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def validate_manifest(document):
    if (
        document.get("schema_version") != 1
        or document.get("release_tag") != RELEASE_TAG
    ):
        raise ValueError("Unsupported weights manifest/release")
    if (
        document.get("architecture_id") != ARCHITECTURE_ID
        or document.get("preprocessing_id") != PREPROCESSING_ID
    ):
        raise ValueError(
            "Weights architecture/preprocessing does not match this reader"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", document.get("dataset_manifest_sha256", "")):
        raise ValueError("Missing shared training dataset identity")
    entries = document.get("files", [])
    if (
        not isinstance(entries, list)
        or len(entries) != 3
        or {v.get("path") for v in entries} != EXPECTED_PATHS
    ):
        raise ValueError("Expected exactly the three selected model paths")
    for entry in entries:
        if (
            not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
            or type(entry.get("size")) is not int
            or not 0 < entry["size"] <= 64 * 1024 * 1024
        ):
            raise ValueError("Invalid model hash/size")
    if document.get("selected_epochs") != {"localizer": 5, "speed": 0, "rpm": 14}:
        raise ValueError("Unexpected selected model epochs for release v0.1.0")
    return entries


def extract_verified(archive, destination, expected_sha256=WEIGHTS_SHA256):
    """Only the three named files are written; archive paths are never trusted."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            "A verified release SHA256 is required before downloading/loading weights"
        )
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != expected_sha256:
        raise ValueError("Weights archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != (
            EXPECTED_PATHS | {"weights_manifest.json"}
        ):
            raise ValueError("Unexpected or duplicate archive members")
        info = z.getinfo("weights_manifest.json")
        if info.file_size > 1024 * 1024:
            raise ValueError("Oversized model manifest")
        document = json.loads(z.read(info))
        entries = validate_manifest(document)
        # Verify all members first; corrupt archives cannot publish partial sets.
        payloads = {}
        for entry in entries:
            info = z.getinfo(entry["path"])
            if (
                info.file_size != entry["size"]
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Invalid model archive member")
            data = z.read(info)
            if (
                len(data) != entry["size"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise ValueError("Model member SHA256/size mismatch")
            payloads[entry["path"]] = data
    for name, data in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == hashlib.sha256(data).hexdigest():
            continue
        partial = target.with_suffix(".pt.part")
        partial.write_bytes(data)
        partial.replace(target)
    return {task: destination / task / "best_model.pt" for task in TASKS}, document


def download_weights(cache_dir):
    """Download public assets without OAuth; reuse only exact verified content."""
    if not re.fullmatch(r"[0-9a-f]{64}", WEIGHTS_SHA256):
        raise ValueError("Published release digest is not configured")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / WEIGHTS_FILENAME
    if not archive.is_file() or sha256(archive) != WEIGHTS_SHA256:
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(
            WEIGHTS_URL, headers={"User-Agent": "triumph-dashboard-reader/0.1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            partial.open("wb") as stream,
        ):
            total = 0
            for data in iter(lambda: response.read(1024 * 1024), b""):
                total += len(data)
                if total > 200 * 1024 * 1024:
                    raise ValueError("Unexpected release archive size")
                stream.write(data)
        if sha256(partial) != WEIGHTS_SHA256:
            raise ValueError("Downloaded release checksum failed")
        partial.replace(archive)
    return extract_verified(archive, cache / "selected_models")
