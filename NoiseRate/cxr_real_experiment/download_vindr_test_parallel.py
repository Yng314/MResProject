#!/usr/bin/env python3
"""Download and verify the VinDr-CXR consensus-labelled test cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import netrc
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

BASE_URL = "https://physionet.org/files/vindr-cxr/1.0.0/"
EXPECTED_TEST_IMAGES = 3000
METADATA_PATHS = (
    "annotations/image_labels_test.csv",
    "LICENSE.txt",
    "SHA256SUMS.txt",
)


class DicomLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and urlparse(href).path.lower().endswith(".dicom"):
            self.links.append(href)


@dataclass(frozen=True)
class DownloadResult:
    relative_path: str
    status: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def load_auth(path: Path) -> tuple[str, str]:
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"Credential file must not be group/world accessible: mode={mode:o}")
    auth = netrc.netrc(str(path)).authenticators("physionet.org")
    if auth is None or not auth[0] or not auth[2]:
        raise ValueError("Credential file has no complete physionet.org entry")
    return auth[0], auth[2]


def wget_environment(credential_file: Path) -> dict[str, str]:
    if credential_file.name != ".netrc":
        raise ValueError("Protected credential must be installed as .netrc in its private directory")
    environment = os.environ.copy()
    environment["HOME"] = str(credential_file.parent)
    return environment


def wget_base_command(retries: int) -> list[str]:
    return [
        "wget",
        "--auth-no-challenge",
        "--tries",
        str(retries),
        "--retry-connrefused",
        "--connect-timeout",
        "30",
        "--read-timeout",
        "180",
        "--waitretry",
        "2",
        "--quiet",
    ]


def list_test_dicoms(environment: dict[str, str], retries: int) -> list[str]:
    listing_url = urljoin(BASE_URL, "test/")
    result = subprocess.run(
        wget_base_command(retries) + ["--output-document", "-", listing_url],
        check=True,
        capture_output=True,
        env=environment,
    )
    parser = DicomLinkParser()
    parser.feed(result.stdout.decode("utf-8"))
    links = sorted({urljoin(listing_url, href) for href in parser.links})
    if len(links) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} DICOM links, found {len(links)}")
    return links


def download_one(
    url: str,
    relative_path: str,
    output_dir: Path,
    environment: dict[str, str],
    retries: int,
) -> DownloadResult:
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return DownloadResult(relative_path, "existing", destination.stat().st_size)

    partial = destination.with_name(destination.name + ".part")
    try:
        subprocess.run(
            wget_base_command(retries)
            + ["--continue", "--output-document", str(partial), url],
            check=True,
            capture_output=True,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"Failed to download {relative_path}: {detail}") from exc
    if not partial.is_file() or partial.stat().st_size == 0:
        raise IOError(f"Downloaded file is missing or empty: {relative_path}")
    final_size = partial.stat().st_size
    os.replace(partial, destination)
    return DownloadResult(relative_path, "downloaded", final_size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, relative_path = parts
        relative_path = relative_path.lstrip("*./")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            checksums[relative_path] = digest.lower()
    return checksums


def expected_checksum(checksums: dict[str, str], relative_path: str) -> str | None:
    if relative_path in checksums:
        return checksums[relative_path]
    matches = [value for key, value in checksums.items() if key.endswith("/" + relative_path)]
    if len(matches) == 1:
        return matches[0]
    return None


def verify_dataset(output_dir: Path, workers: int) -> dict[str, object]:
    dicoms = sorted((output_dir / "test").glob("*.dicom"))
    if len(dicoms) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} DICOM files, found {len(dicoms)}")
    if any(path.stat().st_size == 0 for path in dicoms):
        raise ValueError("One or more DICOM files are empty")

    label_csv = output_dir / "annotations" / "image_labels_test.csv"
    with label_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} label rows, found {len(rows)}")
    if "image_id" not in (rows[0] if rows else {}):
        raise ValueError("image_labels_test.csv has no image_id column")
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != EXPECTED_TEST_IMAGES:
        raise ValueError("image_labels_test.csv image_id values are not unique")
    missing = [image_id for image_id in image_ids if not (output_dir / "test" / f"{image_id}.dicom").is_file()]
    if missing:
        raise ValueError(f"Label/image mismatch: {len(missing)} images are missing")

    checksums = load_checksums(output_dir / "SHA256SUMS.txt")
    selected_paths = [f"test/{path.name}" for path in dicoms] + list(METADATA_PATHS[:-1])
    checksum_targets: list[tuple[str, str]] = []
    missing_checksums: list[str] = []
    for relative_path in selected_paths:
        expected = expected_checksum(checksums, relative_path)
        if expected is None:
            missing_checksums.append(relative_path)
        else:
            checksum_targets.append((relative_path, expected))
    if missing_checksums:
        raise ValueError(f"Missing official checksums for {len(missing_checksums)} selected files")

    checksum_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(sha256_file, output_dir / relative_path): (relative_path, expected)
            for relative_path, expected in checksum_targets
        }
        for future in as_completed(futures):
            relative_path, expected = futures[future]
            if future.result() != expected:
                checksum_failures.append(relative_path)
    if checksum_failures:
        raise ValueError(f"Official SHA-256 mismatch for {len(checksum_failures)} files")

    return {
        "test_images": len(dicoms),
        "label_rows": len(rows),
        "official_checksums_verified": len(checksum_targets),
        "total_bytes": sum(path.stat().st_size for path in dicoms),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        verification = verify_dataset(args.output_dir, args.workers)
        manifest = {
            "dataset": "VinDr-CXR",
            "version": "1.0.0",
            "scope": "consensus-labelled test cohort",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "mode": "verify-only",
            "workers": args.workers,
            "verification": verification,
            "label_csv_sha256": sha256_file(
                args.output_dir / "annotations" / "image_labels_test.csv"
            ),
            "checksum_manifest_sha256": sha256_file(args.output_dir / "SHA256SUMS.txt"),
        }
        manifest_path = args.output_dir / "vindr_test_download_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / ".download_verified").write_text("verified\n", encoding="ascii")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return
    if args.credential_file is None:
        raise ValueError("--credential-file is required unless --verify-only is used")
    load_auth(args.credential_file)
    environment = wget_environment(args.credential_file)
    dicom_urls = list_test_dicoms(environment, args.retries)
    if args.preflight_only:
        print(json.dumps({"authenticated": True, "dicom_links": len(dicom_urls)}, indent=2))
        return

    tasks = [
        (url, f"test/{Path(urlparse(url).path).name}")
        for url in dicom_urls
    ]
    tasks.extend((urljoin(BASE_URL, path), path) for path in METADATA_PATHS)

    progress_lock = threading.Lock()
    completed = 0
    downloaded = 0
    reused = 0
    total = len(tasks)
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                url,
                relative_path,
                args.output_dir,
                environment,
                args.retries,
            ): relative_path
            for url, relative_path in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with progress_lock:
                completed += 1
                downloaded += result.status == "downloaded"
                reused += result.status == "existing"
                if completed == total or completed % 25 == 0:
                    print(
                        f"progress={completed}/{total} downloaded={downloaded} reused={reused}",
                        flush=True,
                    )

    verification = verify_dataset(args.output_dir, args.workers)
    manifest = {
        "dataset": "VinDr-CXR",
        "version": "1.0.0",
        "scope": "consensus-labelled test cohort",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "workers": args.workers,
        "downloaded_files": downloaded,
        "reused_files": reused,
        "verification": verification,
        "label_csv_sha256": sha256_file(args.output_dir / "annotations" / "image_labels_test.csv"),
        "checksum_manifest_sha256": sha256_file(args.output_dir / "SHA256SUMS.txt"),
    }
    manifest_path = args.output_dir / "vindr_test_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / ".download_verified").write_text("verified\n", encoding="ascii")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
