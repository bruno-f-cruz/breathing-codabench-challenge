import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from breathing_cnn_tcn.benchmark import (
    DEFAULT_CONFIG,
    _across_seed_rows,
    collect,
    evaluate_command,
    load_config,
    preflight,
    run_jobs,
    train_command,
)
from breathing_cnn_tcn.benchmark_data import download_commands, validate


class BenchmarkConfigTests(unittest.TestCase):
    def test_default_config_expands_four_by_two_by_five(self):
        config = load_config(DEFAULT_CONFIG)
        jobs = config.jobs()
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({job.job_id for job in jobs}), 40)
        self.assertEqual(
            set(config.representations), {"gray", "diff", "flow", "gray+flow"}
        )
        self.assertEqual(set(config.seeds), {17, 42, 101, 202, 314})

    def test_train_command_uses_full_training_split_and_fixed_budget(self):
        config = load_config(DEFAULT_CONFIG)
        job = next(
            job
            for job in config.jobs()
            if job.representation == "gray+flow"
            and job.objective.name == "multitask"
            and job.seed == 17
        )
        command = train_command(config, job, resume=False)
        self.assertIn("--val-fraction", command)
        self.assertEqual(command[command.index("--val-fraction") + 1], "0")
        self.assertIn("--no-reserve-test-sessions", command)
        self.assertEqual(command[command.index("--channels") + 1], "gray+flow")
        self.assertEqual(command[command.index("--w-onset") + 1], "0.5")
        self.assertNotIn("--resume", command)

    def test_evaluate_command_uses_independent_test_split(self):
        config = load_config(DEFAULT_CONFIG)
        command = evaluate_command(config, config.jobs()[0])
        self.assertEqual(command[command.index("--split") + 1], "test")
        self.assertIn("--split-manifest", command)

    def test_across_seed_summary_has_requested_descriptives(self):
        rows = []
        for seed, correlation in ((1, 0.2), (2, 0.4), (3, 0.6)):
            rows.append(
                {
                    "representation": "gray",
                    "objective": "signal",
                    "stratum": "all",
                    "seed": seed,
                    "correlation": correlation,
                    "inhale_f1": correlation,
                    "exhale_f1": correlation,
                    "kl_ibi": 1.0 - correlation,
                }
            )
        summary = _across_seed_rows(rows)[0]
        self.assertAlmostEqual(summary["correlation_mean"], 0.4)
        self.assertAlmostEqual(summary["correlation_median"], 0.4)
        self.assertAlmostEqual(summary["correlation_min"], 0.2)
        self.assertAlmostEqual(summary["correlation_max"], 0.6)
        self.assertEqual(summary["n_seeds"], 3)

    def test_runner_does_not_prepopulate_exact_training_directory(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            isolated = replace(config, output_root=Path(temporary) / "benchmark")
            job = isolated.jobs()[0]
            with (
                patch("breathing_cnn_tcn.benchmark.preflight"),
                patch("breathing_cnn_tcn.benchmark._git_provenance", return_value={}),
                patch("breathing_cnn_tcn.benchmark.subprocess.run") as run,
            ):
                run_jobs(isolated, [job], resume=False, dry_run=False)
            run.assert_called_once()
            self.assertFalse(isolated.run_dir(job).exists())
            self.assertTrue(
                (isolated.output_root / "jobs" / f"{job.job_id}.json").exists()
            )

    def test_runner_restarts_a_failure_before_the_first_checkpoint(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            isolated = replace(config, output_root=Path(temporary) / "benchmark")
            job = isolated.jobs()[0]
            run_dir = isolated.run_dir(job)
            run_dir.mkdir(parents=True)
            (run_dir / "args.json").write_text("{}")
            with (
                patch("breathing_cnn_tcn.benchmark.preflight"),
                patch("breathing_cnn_tcn.benchmark._git_provenance", return_value={}),
                patch("breathing_cnn_tcn.benchmark.subprocess.run") as run,
            ):
                run_jobs(isolated, [job], resume=True, dry_run=False)
            command = run.call_args.args[0]
            self.assertIn("--resume", command)

    def test_runner_refuses_unknown_files_without_a_checkpoint(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            isolated = replace(config, output_root=Path(temporary) / "benchmark")
            job = isolated.jobs()[0]
            run_dir = isolated.run_dir(job)
            run_dir.mkdir(parents=True)
            (run_dir / "unrelated.txt").write_text("keep me")
            with patch("breathing_cnn_tcn.benchmark.preflight"):
                with self.assertRaisesRegex(SystemExit, "unexpected files"):
                    run_jobs(isolated, [job], resume=True, dry_run=False)

    def test_collector_writes_one_strict_json_artifact(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            isolated = replace(config, output_root=Path(temporary) / "benchmark")
            job = isolated.jobs()[0]
            run_dir = isolated.run_dir(job)
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").touch()
            metrics = {
                "correlation": 0.5,
                "inhale_f1": 0.4,
                "exhale_f1": 0.3,
                "kl_ibi": 0.2,
            }
            (run_dir / "evaluation.json").write_text(
                json.dumps(
                    {
                        "summary_by_session": metrics,
                        "per_session": [
                            {"session_idx": 1, "n_clips": 2} | metrics
                        ],
                        "strata": {
                            "new_animals": {
                                "sessions": [1],
                                "summary": metrics,
                            },
                            "all": {"sessions": [1], "summary": metrics},
                        },
                    }
                )
            )
            with patch("breathing_cnn_tcn.benchmark.preflight"):
                collect(isolated, [job], force=False, dry_run=False)
            result_path = isolated.output_root / "results.json"
            result = json.loads(result_path.read_text())
            self.assertEqual(len(result["summary_by_seed"]), 2)
            self.assertEqual(len(result["metrics_per_session"]), 1)
            self.assertFalse(list(isolated.output_root.glob("*.csv")))

    def test_preflight_fails_before_training_without_generated_manifest(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            isolated = replace(
                config,
                features_dir=root / "features",
                split_manifest=root / "split.json",
            )
            with self.assertRaisesRegex(SystemExit, "benchmark preflight failed"):
                preflight(isolated, evaluation=False)


class BenchmarkDataTests(unittest.TestCase):
    @staticmethod
    def _record(subject: int, video_index: int) -> dict:
        return {"subject_id": str(subject), "video_index": video_index}

    def test_download_is_an_overlay_not_a_flat_merge(self):
        commands = download_commands("s3://bucket/root", Path("dataset"), "face", dry_run=True)
        self.assertEqual(len(commands), 4)
        self.assertIn(str(Path("dataset") / "train"), commands[0])
        self.assertIn(str(Path("dataset") / "test"), commands[1])
        self.assertIn(str(Path("dataset") / "test"), commands[2])
        self.assertIn("--dryrun", commands[0])
        self.assertIn("video_face_*", commands[0])
        self.assertNotIn("video_side_*", commands[0])

    def test_validate_checks_complete_face_camera_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_subjects = list(range(100, 116))
            split = {
                "train": [
                    self._record(subject, index)
                    for index, subject in enumerate(train_subjects, start=1)
                ],
                "val_new_animals": [
                    self._record(200 + index, index) for index in range(1, 7)
                ],
                "val_held_out": [
                    self._record(train_subjects[index - 7], index)
                    for index in range(7, 13)
                ],
            }
            (root / "split.json").write_text(json.dumps(split))
            for split_name, indices in (
                ("train", range(1, 17)),
                ("test", range(1, 13)),
            ):
                directory = root / split_name
                directory.mkdir()
                for session in indices:
                    for part in (1, 2):
                        suffix = f"{session}_part_{part}"
                        for name in (
                            f"thermistor_{suffix}.parquet",
                            f"video_face_{suffix}.mp4",
                            f"video_face_{suffix}.parquet",
                        ):
                            (directory / name).touch()
            result = validate(root, "face")
            self.assertEqual(result["train_clips"], 32)
            self.assertEqual(result["test_clips"], 24)


if __name__ == "__main__":
    unittest.main()
