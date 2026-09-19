"""Reproducible factorial benchmark runner for the CNN-TCN.

The harness deliberately keeps orchestration separate from training.  Each
expanded job receives an exact run directory, trains on the complete labelled
training split for a fixed epoch budget, and is evaluated only after training
against the independent packaged test split.

Examples
--------
    python -m breathing_cnn_tcn.benchmark plan
    python -m breathing_cnn_tcn.benchmark run --dry-run
    python -m breathing_cnn_tcn.benchmark run --job gray__signal__seed-17
    python -m breathing_cnn_tcn.benchmark collect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path("baseline-cnn-tcn/artifacts/benchmark.toml")
METRICS = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi")


def _slug(value: str) -> str:
    return value.replace("+", "-").replace("_", "-")


@dataclass(frozen=True)
class Objective:
    name: str
    w_corr: float
    w_onset: float


@dataclass(frozen=True)
class Job:
    representation: str
    objective: Objective
    seed: int

    @property
    def job_id(self) -> str:
        return f"{_slug(self.representation)}__{_slug(self.objective.name)}__seed-{self.seed}"


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    name: str
    packaged_root: Path
    features_dir: Path
    split_manifest: Path
    output_root: Path
    train_split: str
    test_split: str
    camera: str
    epochs: int
    steps_per_epoch: int
    device: str
    amp: str
    deterministic: bool
    representations: tuple[str, ...]
    seeds: tuple[int, ...]
    objectives: tuple[Objective, ...]

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def jobs(self) -> list[Job]:
        return [
            Job(representation, objective, seed)
            for representation in self.representations
            for objective in self.objectives
            for seed in self.seeds
        ]

    def run_dir(self, job: Job) -> Path:
        return self.output_root / "runs" / job.job_id


def load_config(path: Path) -> BenchmarkConfig:
    data = tomllib.loads(path.read_text())
    benchmark = data["benchmark"]
    objectives = tuple(
        Objective(name, float(values["w_corr"]), float(values["w_onset"]))
        for name, values in data["objectives"].items()
    )
    config = BenchmarkConfig(
        path=path,
        name=str(benchmark["name"]),
        packaged_root=Path(benchmark["packaged_root"]),
        features_dir=Path(benchmark["features_dir"]),
        split_manifest=Path(benchmark["split_manifest"]),
        output_root=Path(benchmark["output_root"]),
        train_split=str(benchmark.get("train_split", "train")),
        test_split=str(benchmark.get("test_split", "test")),
        camera=str(benchmark.get("camera", "face")),
        epochs=int(benchmark["epochs"]),
        steps_per_epoch=int(benchmark.get("steps_per_epoch", 200)),
        device=str(benchmark.get("device", "cuda")),
        amp=str(benchmark.get("amp", "bf16")),
        deterministic=bool(benchmark.get("deterministic", True)),
        representations=tuple(str(x) for x in benchmark["representations"]),
        seeds=tuple(int(x) for x in benchmark["seeds"]),
        objectives=objectives,
    )
    _validate_config(config)
    return config


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.representations or not config.objectives or not config.seeds:
        raise ValueError("representations, objectives, and seeds must be non-empty")
    if len(config.jobs()) != len({job.job_id for job in config.jobs()}):
        raise ValueError("benchmark expansion contains duplicate job IDs")
    if config.epochs <= 0 or config.steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if config.amp not in {"bf16", "fp16", "off"}:
        raise ValueError(f"unsupported amp mode: {config.amp}")
    if config.train_split == config.test_split:
        raise ValueError("train_split and test_split must be different")


def train_command(config: BenchmarkConfig, job: Job, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "breathing_cnn_tcn.train",
        "--features-dir",
        str(config.features_dir),
        "--split",
        config.train_split,
        "--camera",
        config.camera,
        "--run-dir",
        str(config.run_dir(job)),
        "--channels",
        job.representation,
        "--w-corr",
        str(job.objective.w_corr),
        "--w-onset",
        str(job.objective.w_onset),
        "--seed",
        str(job.seed),
        "--epochs",
        str(config.epochs),
        "--steps-per-epoch",
        str(config.steps_per_epoch),
        "--val-fraction",
        "0",
        "--no-reserve-test-sessions",
        "--device",
        config.device,
        "--amp",
        config.amp,
    ]
    if config.deterministic:
        command.append("--deterministic")
    if resume:
        command.append("--resume")
    return command


def evaluate_command(config: BenchmarkConfig, job: Job) -> list[str]:
    run_dir = config.run_dir(job)
    return [
        sys.executable,
        "-m",
        "breathing_cnn_tcn.evaluate",
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--features-dir",
        str(config.features_dir),
        "--packaged-root",
        str(config.packaged_root),
        "--split",
        config.test_split,
        "--camera",
        config.camera,
        "--split-manifest",
        str(config.split_manifest),
        "--out",
        str(run_dir / "evaluation.json"),
        "--device",
        config.device,
        "--amp",
        config.amp,
    ]


def _selected_jobs(config: BenchmarkConfig, selected: list[str] | None) -> list[Job]:
    jobs = config.jobs()
    if not selected:
        return jobs
    wanted = set(selected)
    unknown = wanted - {job.job_id for job in jobs}
    if unknown:
        raise SystemExit(f"unknown job ID(s): {', '.join(sorted(unknown))}")
    return [job for job in jobs if job.job_id in wanted]


def _manifest_path(config: BenchmarkConfig, split: str) -> Path:
    return config.features_dir / f"manifest_{split}_{config.camera}.json"


def preflight(config: BenchmarkConfig, *, evaluation: bool) -> None:
    """Fail before launching a job when benchmark preprocessing is incomplete."""
    required = [_manifest_path(config, config.train_split), config.split_manifest]
    if evaluation:
        required.append(_manifest_path(config, config.test_split))
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return
    lines = "\n".join(f"  - {path}" for path in missing)
    raise SystemExit(
        "benchmark preflight failed; required generated files are missing:\n"
        f"{lines}\n\n"
        "Download/validate the benchmark data and preprocess both splits before "
        "training. See 'Fixed train/test factorial benchmark' in "
        "baseline-cnn-tcn/README.md."
    )


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _sha256_if_present(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_provenance() -> dict[str, str | bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def plan(config: BenchmarkConfig) -> None:
    jobs = config.jobs()
    print(
        f"{config.name}: {len(config.representations)} representations x "
        f"{len(config.objectives)} objectives x {len(config.seeds)} seeds "
        f"= {len(jobs)} jobs"
    )
    print(
        f"fixed budget: {config.epochs} epochs x {config.steps_per_epoch} steps; "
        f"train={config.train_split}, test={config.test_split}"
    )
    for job in jobs:
        run_dir = config.run_dir(job)
        status = "complete" if (run_dir / "best.pt").exists() else "pending"
        print(f"{job.job_id:40s} {status:8s} {run_dir}")


def run_jobs(
    config: BenchmarkConfig,
    jobs: list[Job],
    *,
    resume: bool,
    dry_run: bool,
) -> None:
    if not dry_run:
        preflight(config, evaluation=False)
    if not dry_run:
        config.output_root.mkdir(parents=True, exist_ok=True)
        train_manifest = (
            config.features_dir
            / f"manifest_{config.train_split}_{config.camera}.json"
        )
        (config.output_root / "plan.json").write_text(
            json.dumps(
                {
                    "name": config.name,
                    "config": str(config.path),
                    "config_sha256": config.config_sha256,
                    "train_manifest_sha256": _sha256_if_present(train_manifest),
                    "split_manifest_sha256": _sha256_if_present(
                        config.split_manifest
                    ),
                    "jobs": [job.job_id for job in config.jobs()],
                }
                | _git_provenance(),
                indent=2,
            )
        )
    for number, job in enumerate(jobs, start=1):
        run_dir = config.run_dir(job)
        best = run_dir / "best.pt"
        last = run_dir / "last.pt"
        if best.exists():
            print(f"[{number}/{len(jobs)}] skip complete {job.job_id}")
            continue
        job_resume = resume and last.exists()
        if last.exists() and not resume:
            raise SystemExit(
                f"{job.job_id} has last.pt but no best.pt; rerun with --resume"
            )
        command = train_command(config, job, resume=job_resume)
        print(f"[{number}/{len(jobs)}] {_command_text(command)}", flush=True)
        if dry_run:
            continue
        metadata = {
            "job_id": job.job_id,
            "representation": job.representation,
            "objective": asdict(job.objective),
            "seed": job.seed,
            "config_sha256": config.config_sha256,
            "command": command,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
        } | _git_provenance()
        metadata_path = config.output_root / "jobs" / f"{job.job_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        try:
            subprocess.run(command, check=True)
        except BaseException:
            metadata["status"] = "failed"
            metadata["finished_at"] = datetime.now(UTC).isoformat()
            metadata_path.write_text(json.dumps(metadata, indent=2))
            raise
        metadata["status"] = "complete"
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))


def _evaluation_rows(config: BenchmarkConfig, jobs: list[Job]) -> tuple[list[dict], list[dict]]:
    seed_rows: list[dict] = []
    session_rows: list[dict] = []
    for job in jobs:
        result = json.loads((config.run_dir(job) / "evaluation.json").read_text())
        strata = result.get("strata") or {
            "all": {"summary": result["summary_by_session"]}
        }
        for stratum, values in strata.items():
            seed_rows.append(
                {
                    "job_id": job.job_id,
                    "representation": job.representation,
                    "objective": job.objective.name,
                    "seed": job.seed,
                    "stratum": stratum,
                }
                | {metric: values["summary"].get(metric) for metric in METRICS}
            )
        membership = {
            session: name
            for name, values in result.get("strata", {}).items()
            if name != "all"
            for session in values["sessions"]
        }
        for row in result["per_session"]:
            session_rows.append(
                {
                    "job_id": job.job_id,
                    "representation": job.representation,
                    "objective": job.objective.name,
                    "seed": job.seed,
                    "stratum": membership.get(row["session_idx"], "all"),
                }
                | row
            )
    return seed_rows, session_rows


def _across_seed_rows(seed_rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    keys = sorted(
        {
            (row["representation"], row["objective"], row["stratum"])
            for row in seed_rows
        }
    )
    for representation, objective, stratum in keys:
        members = [
            row
            for row in seed_rows
            if (row["representation"], row["objective"], row["stratum"])
            == (representation, objective, stratum)
        ]
        base = {
            "representation": representation,
            "objective": objective,
            "stratum": stratum,
            "n_seeds": len(members),
        }
        for metric in METRICS:
            values = np.array(
                [row[metric] for row in members if row[metric] is not None], float
            )
            values = values[np.isfinite(values)]
            base |= {
                f"{metric}_mean": float(values.mean()) if len(values) else np.nan,
                f"{metric}_sd": (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                ),
                f"{metric}_min": float(values.min()) if len(values) else np.nan,
                f"{metric}_max": float(values.max()) if len(values) else np.nan,
                f"{metric}_median": float(np.median(values)) if len(values) else np.nan,
            }
        output.append(base)
    return output


def _json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def collect(
    config: BenchmarkConfig,
    jobs: list[Job],
    *,
    force: bool,
    dry_run: bool,
) -> None:
    if not dry_run:
        preflight(config, evaluation=True)
    incomplete = [job.job_id for job in jobs if not (config.run_dir(job) / "best.pt").exists()]
    if incomplete:
        raise SystemExit(
            f"cannot collect: {len(incomplete)} job(s) have no best.pt: "
            + ", ".join(incomplete)
        )
    for number, job in enumerate(jobs, start=1):
        result = config.run_dir(job) / "evaluation.json"
        if result.exists() and not force:
            print(f"[{number}/{len(jobs)}] skip evaluated {job.job_id}")
            continue
        command = evaluate_command(config, job)
        print(f"[{number}/{len(jobs)}] {_command_text(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)
    if dry_run:
        return
    seed_rows, session_rows = _evaluation_rows(config, jobs)
    config.output_root.mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark": config.name,
        "config": str(config.path),
        "config_sha256": config.config_sha256,
        "summary_by_seed": seed_rows,
        "metrics_per_session": session_rows,
        "summary_across_seeds": _across_seed_rows(seed_rows),
    }
    (config.output_root / "results.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False)
    )
    print(f"wrote {config.output_root / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the expanded job matrix and status.")
    run_parser = subparsers.add_parser("run", help="Run pending training jobs.")
    run_parser.add_argument("--job", action="append", help="Run only this job ID.")
    run_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume jobs containing last.pt (default: enabled).",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    collect_parser = subparsers.add_parser(
        "collect", help="Evaluate completed jobs and write aggregate tables."
    )
    collect_parser.add_argument("--job", action="append", help="Collect only this job ID.")
    collect_parser.add_argument("--force", action="store_true")
    collect_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "plan":
        plan(config)
    elif args.command == "run":
        run_jobs(
            config,
            _selected_jobs(config, args.job),
            resume=args.resume,
            dry_run=args.dry_run,
        )
    elif args.command == "collect":
        collect(
            config,
            _selected_jobs(config, args.job),
            force=args.force,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
