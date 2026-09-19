"""Assemble and validate the organizer's labelled train/test benchmark data.

The public prefix contains labelled training data and test videos.  The private
prefix contains only the matching test thermistors and split metadata.  This
module overlays those sources while retaining separate ``train`` and ``test``
directories, since their numeric session indices intentionally overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DEFAULT_S3_ROOT = (
    "s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/"
    "9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a"
)


def download_commands(
    s3_root: str, destination: Path, camera: str, *, dry_run: bool
) -> list[list[str]]:
    sync_filter = ["--exclude", "*"]
    if camera == "both":
        sync_filter = []
    else:
        sync_filter += ["--include", "thermistor_*"]
        sync_filter += ["--include", f"video_{camera}_*"]
    dry = ["--dryrun"] if dry_run else []
    common = ["--no-sign-request", "--only-show-errors"]
    return [
        [
            "aws",
            "s3",
            "sync",
            f"{s3_root}/public/train/",
            str(destination / "train"),
            *common,
            *sync_filter,
            *dry,
        ],
        [
            "aws",
            "s3",
            "sync",
            f"{s3_root}/public/test/",
            str(destination / "test"),
            *common,
            *sync_filter,
            *dry,
        ],
        [
            "aws",
            "s3",
            "sync",
            f"{s3_root}/private/test/",
            str(destination / "test"),
            *common,
            "--exclude",
            "*",
            "--include",
            "thermistor_*",
            *dry,
        ],
        [
            "aws",
            "s3",
            "cp",
            f"{s3_root}/private/split.json",
            str(destination / "split.json"),
            "--no-sign-request",
            *dry,
        ],
    ]


def run_download(s3_root: str, destination: Path, camera: str, dry_run: bool) -> None:
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # --no-sign-request needs no configured profile. A stale workstation-wide
    # profile name would otherwise make even anonymous S3 access fail early.
    environment.pop("AWS_PROFILE", None)
    for command in download_commands(s3_root.rstrip("/"), destination, camera, dry_run=dry_run):
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True, env=environment)


def _indices(data: dict, key: str) -> list[int]:
    return sorted(int(item["video_index"]) for item in data[key])


def validate(destination: Path, camera: str) -> dict:
    split_path = destination / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"missing {split_path}; run the download command first")
    data = json.loads(split_path.read_text())
    train = _indices(data, "train")
    new_animals = _indices(data, "val_new_animals")
    known_animals = _indices(data, "val_held_out")
    if len(train) != 16 or len(new_animals) != 6 or len(known_animals) != 6:
        raise ValueError(
            "unexpected split sizes: expected train=16, new animals=6, "
            f"known animals/new date=6; got {len(train)}, {len(new_animals)}, "
            f"{len(known_animals)}"
        )
    if set(new_animals) & set(known_animals):
        raise ValueError("test strata overlap")

    train_subjects = {str(item["subject_id"]) for item in data["train"]}
    new_subjects = {str(item["subject_id"]) for item in data["val_new_animals"]}
    known_subjects = {str(item["subject_id"]) for item in data["val_held_out"]}
    if train_subjects & new_subjects:
        raise ValueError("new-animal test stratum contains a training subject")
    if not known_subjects <= train_subjects:
        raise ValueError("known-animal/new-date stratum contains an unknown subject")

    cameras = ("face", "side") if camera == "both" else (camera,)
    missing: list[str] = []
    for split, indices in (("train", train), ("test", new_animals + known_animals)):
        for session in indices:
            for part in (1, 2):
                suffix = f"{session}_part_{part}"
                required = [destination / split / f"thermistor_{suffix}.parquet"]
                for selected_camera in cameras:
                    required += [
                        destination / split / f"video_{selected_camera}_{suffix}.mp4",
                        destination / split / f"video_{selected_camera}_{suffix}.parquet",
                    ]
                missing += [str(path) for path in required if not path.exists()]
    if missing:
        preview = "\n".join(missing[:20])
        remainder = f"\n... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"benchmark data is incomplete:\n{preview}{remainder}")

    result = {
        "train_recordings": len(train),
        "train_clips": 2 * len(train),
        "test_recordings": len(new_animals) + len(known_animals),
        "test_clips": 2 * (len(new_animals) + len(known_animals)),
        "new_animal_recordings": len(new_animals),
        "known_animal_new_date_recordings": len(known_animals),
        "camera": camera,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--camera", choices=["face", "side", "both"], default="face")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--s3-root", default=DEFAULT_S3_ROOT)
    download_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "download":
        run_download(args.s3_root, args.dest, args.camera, args.dry_run)
    else:
        validate(args.dest, args.camera)


if __name__ == "__main__":
    main()
