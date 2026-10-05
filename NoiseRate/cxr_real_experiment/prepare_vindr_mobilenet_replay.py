#!/usr/bin/env python3
"""Create an immutable MobileNet replay root from locked VinDr inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from vindr_known_gt_cl_benchmark import atomic_write_text, sha256_file


ROOT_FILES = [
    "scenario_manifest.csv",
    "blind_run_manifest.csv",
    "scenarios.csv",
    "seed_manifest.csv",
    "image_index.csv",
]


def prepare(source: Path, output: Path, protocol: Path, engine: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Replay root already exists: {output}")
    if not (source / ".prepare_complete").is_file() or not (source / ".benchmark_verified").is_file():
        raise FileNotFoundError("Source experiment is not prepared and verified")

    scenario_manifest = pd.read_csv(source / "scenario_manifest.csv")
    blind_manifest = pd.read_csv(source / "blind_run_manifest.csv")
    if len(scenario_manifest) != 60 or len(blind_manifest) != 60:
        raise ValueError("Source does not contain the locked 60-run grid")
    if not scenario_manifest[["scenario_id", "seed"]].equals(
        blind_manifest[["scenario_id", "seed"]]
    ):
        raise ValueError("Private and blind manifests do not have identical run ordering")
    forbidden = [
        column for column in blind_manifest.columns
        if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
    ]
    if forbidden:
        raise ValueError(f"Blind manifest contains outcome columns: {forbidden}")

    staging = output.with_name(f".{output.name}.prepare-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Replay staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        copied_hashes = {}
        for name in ROOT_FILES:
            source_path = source / name
            target_path = staging / name
            shutil.copy2(source_path, target_path)
            source_hash = sha256_file(source_path)
            if sha256_file(target_path) != source_hash:
                raise RuntimeError(f"Replay root copy differs from source: {name}")
            copied_hashes[name] = source_hash

        prepared_links = []
        for row in blind_manifest.itertuples(index=False):
            relative = Path(row.prepared_relpath)
            source_prepared = source / relative
            if not (source_prepared / ".prepare_complete").is_file():
                raise FileNotFoundError(f"Prepared source marker is missing: {source_prepared}")
            target_prepared = staging / relative
            target_prepared.parent.mkdir(parents=True, exist_ok=True)
            target_prepared.symlink_to(source_prepared, target_is_directory=True)
            if target_prepared.resolve() != source_prepared.resolve():
                raise RuntimeError(f"Prepared replay link is invalid: {target_prepared}")
            prepared_links.append(str(relative))

        if len(prepared_links) != 60 or len(set(prepared_links)) != 60:
            raise RuntimeError("Replay prepared-link coverage is not exactly 60 unique paths")
        summary = {
            "protocol": "vindr_mobilenet_direction_sensitivity_v1",
            "source_experiment": str(source),
            "source_prepare_verified": True,
            "blind_runs": 60,
            "prepared_input_links": 60,
            "copied_file_sha256": copied_hashes,
            "protocol_sha256": sha256_file(protocol),
            "mobilenet_engine_sha256": sha256_file(engine),
            "program_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_text(staging / "replay_prepare_summary.json", json.dumps(summary, indent=2))
        atomic_write_text(staging / ".prepare_complete", "complete\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare(args.source_root, args.output_root, args.protocol, args.engine)


if __name__ == "__main__":
    main()
