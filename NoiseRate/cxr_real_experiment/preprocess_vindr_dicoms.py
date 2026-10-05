#!/usr/bin/env python3
"""Audit and deterministically convert verified VinDr-CXR DICOMs to PNG-224."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import pylibjpeg
from PIL import Image, ImageDraw
from pydicom.pixels import apply_modality_lut, apply_voi_lut


EXPECTED_PYDICOM = "3.0.1"
EXPECTED_PYLIBJPEG = "2.1.0"
EXPECTED_IMAGES = 3_000
JPEG2000_LOSSLESS_UID = "1.2.840.10008.1.2.4.90"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def first_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value
    try:
        return value[0]
    except (TypeError, IndexError):
        return value


def optional_float(value: Any) -> float | None:
    value = first_value(value)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    value = first_value(value)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def private_group_hash(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_official_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, relative = line.split(maxsplit=1)
            checksums[relative.lstrip("*./")] = digest
    return checksums


def load_source_table(
    dicom_dir: Path,
    labels_csv: Path,
    checksums_path: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    if "image_id" not in labels.columns:
        raise ValueError("VinDr label CSV is missing image_id")
    if len(labels) != EXPECTED_IMAGES or labels["image_id"].nunique() != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} unique label rows, found "
            f"{len(labels)} rows and {labels['image_id'].nunique()} unique ids"
        )

    checksums = parse_official_checksums(checksums_path)
    records = []
    for image_id in sorted(labels["image_id"].astype(str)):
        source = dicom_dir / f"{image_id}.dicom"
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = f"test/{source.name}"
        if relative not in checksums:
            raise ValueError(f"Official checksum missing for {relative}")
        records.append(
            {
                "image_id": image_id,
                "source_path": str(source),
                "official_source_sha256": checksums[relative],
            }
        )

    disk_ids = {path.stem for path in dicom_dir.glob("*.dicom")}
    label_ids = set(labels["image_id"].astype(str))
    if disk_ids != label_ids:
        raise ValueError(
            f"DICOM/label id mismatch: missing={len(label_ids - disk_ids)}, "
            f"extra={len(disk_ids - label_ids)}"
        )
    return pd.DataFrame(records)


def header_record(record: dict[str, Any]) -> dict[str, Any]:
    source = Path(record["source_path"])
    dataset = pydicom.dcmread(source, stop_before_pixels=True)
    transfer_syntax = str(dataset.file_meta.TransferSyntaxUID)
    return {
        **record,
        "rows": optional_int(getattr(dataset, "Rows", None)),
        "columns": optional_int(getattr(dataset, "Columns", None)),
        "samples_per_pixel": optional_int(getattr(dataset, "SamplesPerPixel", None)),
        "photometric_interpretation": str(
            getattr(dataset, "PhotometricInterpretation", "")
        ),
        "bits_allocated": optional_int(getattr(dataset, "BitsAllocated", None)),
        "bits_stored": optional_int(getattr(dataset, "BitsStored", None)),
        "pixel_representation": optional_int(
            getattr(dataset, "PixelRepresentation", None)
        ),
        "transfer_syntax_uid": transfer_syntax,
        "is_jpeg2000_lossless": int(transfer_syntax == JPEG2000_LOSSLESS_UID),
        "window_center": optional_float(getattr(dataset, "WindowCenter", None)),
        "window_width": optional_float(getattr(dataset, "WindowWidth", None)),
        "has_voi_lut": int(hasattr(dataset, "VOILUTSequence")),
        "rescale_slope": optional_float(getattr(dataset, "RescaleSlope", None)),
        "rescale_intercept": optional_float(
            getattr(dataset, "RescaleIntercept", None)
        ),
        "presentation_lut_shape": str(
            getattr(dataset, "PresentationLUTShape", "")
        ),
        "pixel_padding_value": optional_float(
            getattr(dataset, "PixelPaddingValue", None)
        ),
        "patient_group_hash": private_group_hash(
            getattr(dataset, "PatientID", None)
        ),
        "study_group_hash": private_group_hash(
            getattr(dataset, "StudyInstanceUID", None)
        ),
        "view_position": str(getattr(dataset, "ViewPosition", "")),
    }


def categorical_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = frame[column].fillna("<missing>").astype(str)
    return {str(key): int(value) for key, value in Counter(values).items()}


def run_header_audit(args: argparse.Namespace, sources: pd.DataFrame) -> None:
    records = sources.to_dict(orient="records")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        headers = list(executor.map(header_record, records, chunksize=16))
    frame = pd.DataFrame(headers).sort_values("image_id").reset_index(drop=True)

    if frame[["rows", "columns"]].isna().any().any():
        raise ValueError("At least one DICOM is missing image dimensions")
    if (frame[["rows", "columns"]] <= 0).any().any():
        raise ValueError("At least one DICOM has invalid image dimensions")
    if not frame["samples_per_pixel"].eq(1).all():
        raise ValueError("At least one VinDr DICOM is not single-channel")
    if not frame["photometric_interpretation"].isin(
        ["MONOCHROME1", "MONOCHROME2"]
    ).all():
        raise ValueError("Unsupported photometric interpretation found")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "dicom_header_audit.csv"
    atomic_write_csv(frame, audit_path)
    patient_groups = frame.loc[
        frame["patient_group_hash"].ne(""), "patient_group_hash"
    ]
    summary = {
        "mode": "audit",
        "images": int(len(frame)),
        "pydicom_version": pydicom.__version__,
        "pylibjpeg_version": pylibjpeg.__version__,
        "transfer_syntax_uid": categorical_counts(frame, "transfer_syntax_uid"),
        "photometric_interpretation": categorical_counts(
            frame, "photometric_interpretation"
        ),
        "bits_allocated": categorical_counts(frame, "bits_allocated"),
        "bits_stored": categorical_counts(frame, "bits_stored"),
        "window_present": int(
            (frame["window_center"].notna() & frame["window_width"].notna()).sum()
        ),
        "voi_lut_present": int(frame["has_voi_lut"].sum()),
        "patient_id_present": int(len(patient_groups)),
        "unique_patient_groups": int(patient_groups.nunique()),
        "image_rows_min_max": [int(frame["rows"].min()), int(frame["rows"].max())],
        "image_columns_min_max": [
            int(frame["columns"].min()),
            int(frame["columns"].max()),
        ],
        "header_audit_sha256": sha256_file(audit_path),
    }
    atomic_write_text(output / "header_audit_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".header_audit_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def scale_to_uint8(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.asarray(array, dtype=np.float64)
    if finite.ndim != 2 or not np.isfinite(finite).all():
        raise ValueError("Decoded DICOM is not a finite 2D array")
    lower = float(finite.min())
    upper = float(finite.max())
    if upper <= lower:
        raise ValueError("Decoded DICOM is constant after grayscale processing")
    scaled = np.clip((finite - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), lower, upper


def center_crop_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    height, width = array.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = array[top : top + side, left : left + side]
    image = Image.fromarray(cropped, mode="L")
    image = image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def convert_record(payload: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    record, png_dir_text, image_size = payload
    source = Path(record["source_path"])
    png_dir = Path(png_dir_text)
    output = png_dir / f"{record['image_id']}.png"
    dataset = pydicom.dcmread(source)
    dataset.pixel_array_options(decoding_plugin="pylibjpeg")
    decoded = dataset.pixel_array
    if decoded.ndim != 2:
        raise ValueError(f"Expected one grayscale frame in {source}")

    processed = apply_modality_lut(decoded, dataset)
    has_voi = hasattr(dataset, "VOILUTSequence") or (
        hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth")
    )
    if has_voi:
        processed = apply_voi_lut(processed, dataset, index=0)

    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    presentation_inverse = (
        str(getattr(dataset, "PresentationLUTShape", "")).upper() == "INVERSE"
    )
    should_invert = (photometric == "MONOCHROME1") != presentation_inverse
    processed = np.asarray(processed, dtype=np.float64)
    if should_invert:
        processed = float(processed.max() + processed.min()) - processed

    uint8_full, scale_min, scale_max = scale_to_uint8(processed)
    uint8_resized = center_crop_resize(uint8_full, image_size)
    if np.ptp(uint8_resized) <= 0:
        raise ValueError(f"PNG became constant after resize: {source}")

    png_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    Image.fromarray(uint8_resized, mode="L").save(
        temporary, format="PNG", optimize=False, compress_level=6
    )
    with Image.open(temporary) as check:
        check.load()
        if check.mode != "L" or check.size != (image_size, image_size):
            raise ValueError(f"Saved PNG failed postflight: {temporary}")
    temporary.replace(output)

    return {
        "image_id": record["image_id"],
        "source_path": str(source),
        "official_source_sha256": record["official_source_sha256"],
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "source_rows": int(decoded.shape[0]),
        "source_columns": int(decoded.shape[1]),
        "decoded_dtype": str(decoded.dtype),
        "decoded_min": float(decoded.min()),
        "decoded_max": float(decoded.max()),
        "voi_applied": int(has_voi),
        "inverted": int(should_invert),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "output_min": int(uint8_resized.min()),
        "output_max": int(uint8_resized.max()),
        "output_mean": float(uint8_resized.mean()),
        "output_std": float(uint8_resized.std()),
        "photometric_interpretation": photometric,
        "transfer_syntax_uid": str(dataset.file_meta.TransferSyntaxUID),
        "bits_stored": int(dataset.BitsStored),
        "patient_group_hash": private_group_hash(
            getattr(dataset, "PatientID", None)
        ),
    }


def select_stratified_smoke(
    sources: pd.DataFrame,
    audit_csv: Path,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    required = {
        "image_id",
        "transfer_syntax_uid",
        "photometric_interpretation",
        "bits_stored",
        "window_center",
        "window_width",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"Header audit is missing smoke-stratification columns: {missing}")
    audit = audit[list(required)].copy()
    audit["image_id"] = audit["image_id"].astype(str)
    audit["window_status"] = np.where(
        audit["window_center"].notna() & audit["window_width"].notna(),
        "present",
        "missing",
    )
    candidates = sources.merge(audit, on="image_id", how="left", validate="one_to_one")
    if candidates[list(required - {"image_id"})].isna().all(axis=1).any():
        raise ValueError("Header audit does not cover every source image")

    strata_columns = [
        "transfer_syntax_uid",
        "photometric_interpretation",
        "bits_stored",
        "window_status",
    ]
    tokens_by_index: dict[int, set[tuple[str, str]]] = {}
    uncovered: set[tuple[str, str]] = set()
    for index, row in candidates.iterrows():
        tokens = {(column, str(row[column])) for column in strata_columns}
        tokens_by_index[int(index)] = tokens
        uncovered.update(tokens)

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(candidates)).tolist()
    available = set(range(len(candidates)))
    selected: list[int] = []
    while uncovered and len(selected) < limit:
        scores = {
            index: len(tokens_by_index[index] & uncovered) for index in available
        }
        best_score = max(scores.values())
        if best_score <= 0:
            break
        chosen = next(index for index in tie_order if index in available and scores[index] == best_score)
        selected.append(chosen)
        available.remove(chosen)
        uncovered.difference_update(tokens_by_index[chosen])
    if uncovered:
        raise ValueError(
            f"Smoke limit {limit} cannot cover all audited marginal strata: {sorted(uncovered)}"
        )
    for index in tie_order:
        if len(selected) >= limit:
            break
        if index in available:
            selected.append(index)
            available.remove(index)
    return candidates.iloc[sorted(selected)][sources.columns].reset_index(drop=True)


def build_contact_sheet(manifest: pd.DataFrame, path: Path, image_size: int) -> None:
    examples = manifest.sort_values("image_id").head(20)
    columns = 5
    rows = int(np.ceil(len(examples) / columns))
    label_height = 26
    canvas = Image.new("L", (columns * image_size, rows * (image_size + label_height)), 255)
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(examples.itertuples(index=False)):
        column = index % columns
        grid_row = index // columns
        with Image.open(row.output_path) as image:
            canvas.paste(image.convert("L"), (column * image_size, grid_row * (image_size + label_height)))
        draw.text(
            (column * image_size + 4, grid_row * (image_size + label_height) + image_size + 5),
            str(row.image_id)[:12],
            fill=0,
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(path)


def run_conversion(args: argparse.Namespace, sources: pd.DataFrame) -> None:
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    selected = sources.copy()
    if args.limit:
        if args.limit > len(selected):
            raise ValueError("--limit exceeds available images")
        if args.audit_csv is not None:
            selected = select_stratified_smoke(
                selected,
                audit_csv=args.audit_csv,
                limit=args.limit,
                seed=args.sample_seed,
            )
        else:
            rng = np.random.default_rng(args.sample_seed)
            chosen = rng.choice(len(selected), size=args.limit, replace=False)
            selected = selected.iloc[np.sort(chosen)].reset_index(drop=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    png_dir = output / "images"
    payloads = [
        (record, str(png_dir), int(args.image_size))
        for record in selected.to_dict(orient="records")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(convert_record, payloads, chunksize=1))
    manifest = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)

    expected = len(selected)
    if len(manifest) != expected or manifest["image_id"].nunique() != expected:
        raise RuntimeError("Conversion manifest is incomplete or duplicated")
    if manifest["output_sha256"].str.len().ne(64).any():
        raise RuntimeError("At least one output hash is invalid")
    if manifest["output_std"].le(0).any():
        raise RuntimeError("At least one output PNG is constant")

    manifest_path = output / "conversion_manifest.csv"
    atomic_write_csv(manifest, manifest_path)
    contact_sheet = output / "contact_sheet.png"
    build_contact_sheet(manifest, contact_sheet, int(args.image_size))
    full_run = args.limit == 0 and len(manifest) == EXPECTED_IMAGES
    summary = {
        "mode": "convert",
        "scope": "full" if full_run else "smoke",
        "images": int(len(manifest)),
        "image_size": int(args.image_size),
        "sample_seed": int(args.sample_seed),
        "workers": int(args.workers),
        "voi_applied": int(manifest["voi_applied"].sum()),
        "inverted": int(manifest["inverted"].sum()),
        "output_mean_min_max": [
            float(manifest["output_mean"].min()),
            float(manifest["output_mean"].max()),
        ],
        "output_std_min_max": [
            float(manifest["output_std"].min()),
            float(manifest["output_std"].max()),
        ],
        "conversion_manifest_sha256": sha256_file(manifest_path),
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "conversion_summary.json", json.dumps(summary, indent=2))
    marker = ".conversion_complete" if full_run else ".smoke_complete"
    atomic_write_text(output / marker, "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "convert"])
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--audit-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if pydicom.__version__ != EXPECTED_PYDICOM:
        raise RuntimeError(f"Expected pydicom {EXPECTED_PYDICOM}, found {pydicom.__version__}")
    if pylibjpeg.__version__ != EXPECTED_PYLIBJPEG:
        raise RuntimeError(
            f"Expected pylibjpeg {EXPECTED_PYLIBJPEG}, found {pylibjpeg.__version__}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not args.dicom_dir.is_dir():
        raise FileNotFoundError(args.dicom_dir)
    if not args.labels_csv.is_file() or not args.checksums.is_file():
        raise FileNotFoundError("Labels CSV or checksum manifest is missing")

    sources = load_source_table(args.dicom_dir, args.labels_csv, args.checksums)
    if args.mode == "audit":
        run_header_audit(args, sources)
    else:
        run_conversion(args, sources)


if __name__ == "__main__":
    main()
