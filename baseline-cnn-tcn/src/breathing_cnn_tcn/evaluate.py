"""Score checkpoints against held-out labelled clips, the way the competition would.

Truth comes from the raw signal, not the filtered, z-scored target this package
trains on, and scoring goes through ``scoring.metrics.score_clip`` -- which
resamples both traces onto the canonical grid and detects events itself.
Anything less faithful would report a number the leaderboard will not reproduce.

What this is for
----------------
Two protocols are supported. Legacy development runs score sessions reserved in
``baseline-cnn-tcn/artifacts/holdout_sessions.json``. The factorial benchmark
instead trains on the complete ``train`` split and supplies the organizer's
``split.json`` while scoring the independent ``test`` split. In either protocol
the estimate is consumable: every look influences what gets tried next, so
score models only after the training choices are frozen.

Several ``--checkpoint`` paths are ensembled, each z-scored before averaging --
the models are trained on a correlation loss that leaves output scale free, so
averaging raw outputs would weight by whichever run happened to settle largest.

``--plot`` additionally writes the rate-breakdown and reserved-grid diagnostic
plots (see :mod:`.plot_diagnosis`) from the same predictions computed here --
nothing is re-predicted.  The reserved-grid panel needs one model's own
onset-head probability, so ``--plot`` requires exactly one ``--checkpoint``.

CLI
---
    python -m breathing_cnn_tcn.evaluate \\
        --checkpoint runs/baseline-cnn-tcn/<run>/best.pt --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scoring.metrics import score_clip
from scoring.processing import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

from .channels import ChannelSet
from .clips import PUBLIC_SPLIT
from .dataset import ClipEntry, load_manifest
from .infer import predict_clip
from .model import BreathingNet
from .plot_diagnosis import ClipPrediction, rate_breakdown, reserved_grid


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[BreathingNet, np.ndarray, np.ndarray, dict]:
    """Rebuild a model from a training checkpoint, with its own normalisation.

    ``model`` holds whichever weights the run selected -- the EMA copy when
    averaging was enabled -- so nothing here needs to know how it was trained.

    The channel selection comes from the checkpoint, not from the current
    manifest: it fixes the first convolution's shape, so reading it from
    anywhere else would build a model the weights do not fit.  Checkpoints
    written before ``--channels`` existed have no such field and trained on
    every stored channel, which their own manifest config records.  Returned
    ``mean``/``std`` stay full-width -- :func:`~.infer.predict_clip` slices
    them to the model.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["feature_config"]
    channels = ChannelSet.parse(state.get("channels") or config["channel_names"])
    model = BreathingNet(channels=channels).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["mean"], state["std"], state


def zscore(x: np.ndarray) -> np.ndarray:
    scale = x.std()
    return (x - x.mean()) / (scale if scale > 0 else 1.0)


def predict_entry(
    models: list[tuple[BreathingNet, np.ndarray, np.ndarray]],
    entry: ClipEntry,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    """Ensemble prediction for one clip, z-scored per model then averaged.

    Each model is z-scored before averaging: they are trained on a correlation
    loss that leaves output scale free, so averaging raw outputs would weight by
    whichever run happened to settle on the largest amplitude.
    """
    stack = [
        zscore(
            predict_clip(
                model,
                entry,
                mean,
                std,
                window=window,
                device=device,
                frame_chunk=frame_chunk,
                amp_dtype=amp_dtype,
            )[0]
        )
        for model, mean, std in models
    ]
    return zscore(np.mean(stack, axis=0))


def truth_frame(packaged_root: Path, split: str, session_idx: int, part: int):
    """Raw thermistor trace -- already the columns ``score_clip`` expects."""
    path = packaged_root / split / f"thermistor_{session_idx}_part_{part}.parquet"
    return pd.read_parquet(path)


METRIC_FIELDS = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi")


def summarise_rows(rows: list[dict]) -> dict[str, float]:
    """Mean and spread for metric rows, ignoring undefined values."""
    summary: dict[str, float] = {}
    for field in METRIC_FIELDS:
        values = np.array(
            [r[field] for r in rows if r.get(field) is not None], dtype=float
        )
        values = values[np.isfinite(values)]
        summary[field] = float(values.mean()) if len(values) else float("nan")
        summary[f"{field}_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        )
        summary[f"{field}_n"] = len(values)
    return summary


def aggregate_sessions(rows: list[dict]) -> list[dict]:
    """Average clip metrics within session, the independent sampling unit."""
    sessions: list[dict] = []
    for session_idx in sorted({int(r["session_idx"]) for r in rows}):
        members = [r for r in rows if int(r["session_idx"]) == session_idx]
        sessions.append(
            {"session_idx": session_idx, "n_clips": len(members)}
            | {field: summarise_rows(members)[field] for field in METRIC_FIELDS}
        )
    return sessions


def load_test_strata(path: Path | None) -> dict[str, list[int]]:
    """Map the organizer split manifest to stable benchmark stratum names."""
    if path is None:
        return {}
    data = json.loads(path.read_text())

    def indices(key: str) -> list[int]:
        return sorted(int(item["video_index"]) for item in data.get(key, []))

    strata = {
        "new_animals": indices("val_new_animals"),
        "known_animals_new_date": indices("val_held_out"),
    }
    return {name: values for name, values in strata.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score checkpoints on held-out clips.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--packaged-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default=PUBLIC_SPLIT)
    parser.add_argument("--camera", default="face", choices=["face", "side"])
    parser.add_argument(
        "--holdout-json",
        type=Path,
        default=Path("baseline-cnn-tcn/artifacts/holdout_sessions.json"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Organizer split.json. When --sessions is omitted, score every "
        "test video_index and report new-animal and known-animal/new-date strata.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        help="Sessions to score.  Defaults to the reserved test sessions.",
    )
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument("--out", type=Path, help="Write the full result as JSON.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write the rate-breakdown and reserved-grid diagnostic plots "
        "(see .plot_diagnosis) into --plot-dir.  Needs exactly one --checkpoint: "
        "the reserved-grid panel reads one model's own onset head.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Where the diagnostic plots land. Defaults to a `diagnosis/` "
        "folder next to the checkpoint being scored, so plots from different "
        "runs never collide or need telling apart by filename.",
    )
    parser.add_argument("--grid-window", type=int, default=512)
    parser.add_argument("--corr-window-s", type=float, default=3.0)
    parser.add_argument("--corr-hop-s", type=float, default=1.5)
    parser.add_argument("--corr-min-breaths", type=int, default=3)
    parser.add_argument(
        "--null-shifts",
        type=int,
        default=5,
        help="Circular shifts of the predicted onset train used for the rate-"
        "breakdown plot's chance baseline.",
    )
    args = parser.parse_args()

    if args.plot and len(args.checkpoint) != 1:
        raise SystemExit(
            f"--plot needs exactly one --checkpoint (the reserved-grid panel "
            f"reads one model's own onset head); got {len(args.checkpoint)}"
        )

    device = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    strata = load_test_strata(args.split_manifest)
    sessions = args.sessions
    if sessions is None and strata:
        sessions = sorted({i for values in strata.values() for i in values})
    elif sessions is None:
        sessions = json.loads(args.holdout_json.read_text())["test_sessions"]
    sessions = set(sessions)

    models = []
    trained_on: set[int] = set()
    for path in args.checkpoint:
        model, mean, std, state = load_checkpoint(path, device)
        models.append((model, mean, std))
        reserved = set(state.get("test_sessions") or [])
        train_split = state.get("train_split") or state.get("args", {}).get("split")
        trained_sessions = set(state.get("trained_sessions") or [])
        if train_split is not None and trained_sessions:
            # Numeric session indices restart in each packaged split. Only an
            # overlap within the same split is leakage.
            if args.split == train_split:
                trained_on |= sessions & trained_sessions
        elif train_split is None or args.split == train_split:
            # Backwards compatibility for checkpoints created before split
            # provenance was recorded; those only supported the train split.
            trained_on |= sessions - reserved
        print(
            f"loaded {path}  epoch {state.get('epoch')}  "
            f"reserved {sorted(reserved) or 'none recorded'}"
        )
    if trained_on:
        raise SystemExit(
            f"refusing to score: {sorted(trained_on)} were not held out by every "
            "checkpoint, so this would not be an unbiased estimate."
        )

    _, entries = load_manifest(args.features_dir, args.split, args.camera)
    entries = [e for e in entries if e.session_idx in sessions]
    if not entries:
        raise SystemExit(f"no clips found for sessions {sorted(sessions)}")

    print(
        f"\nscoring {len(entries)} clips / {len(sessions)} sessions "
        f"({', '.join(str(s) for s in sorted(sessions))}) with {len(models)} model(s)\n"
    )
    header = (
        f"{'clip':16s}{'sess':9s}{'corr':>10s}"
        f"{'inh_f1':>8s}{'exh_f1':>8s}{'kl_ibi':>8s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    plot_data: list[ClipPrediction] = []
    for entry in sorted(entries, key=lambda e: (e.session_idx, e.part)):
        signal = predict_entry(
            models,
            entry,
            device,
            window=args.infer_window,
            frame_chunk=args.frame_chunk,
            amp_dtype=amp_dtype,
        )
        times = np.load(entry.times)
        n = min(len(times), len(signal))
        signal, times = signal[:n], times[:n]
        predicted = pd.DataFrame(
            {
                TIME_COLUMN: times.astype(np.float64),
                BREATHING_SIGNAL_COLUMN: signal.astype(np.float64),
            }
        )
        truth = truth_frame(
            args.packaged_root, args.split, entry.session_idx, entry.part
        )
        score = score_clip(truth, predicted)
        row = {
            "clip_id": entry.clip_id,
            "session_idx": entry.session_idx,
            "part": entry.part,
        } | score.to_dict()
        rows.append(row)
        if args.plot:
            plot_data.append(ClipPrediction(entry, signal, times, truth))
        print(
            f"{entry.clip_id:16s}{entry.session_idx:<9d}"
            f"{score.correlation:+10.3f}"
            f"{score.inhale_f1:8.3f}{score.exhale_f1:8.3f}{score.kl_ibi:8.3f}",
            flush=True,
        )

    print("-" * len(header))
    # Keep the historical clip-level summary, and add the academically correct
    # session-level result used by the benchmark harness.
    summary = summarise_rows(rows)
    session_rows = aggregate_sessions(rows)
    summary_by_session = summarise_rows(session_rows)
    stratum_results = {}
    if strata:
        for name, indices in strata.items():
            members = [r for r in session_rows if r["session_idx"] in set(indices)]
            stratum_results[name] = {
                "sessions": indices,
                "summary": summarise_rows(members),
            }
        stratum_results["all"] = {
            "sessions": sorted(sessions),
            "summary": summary_by_session,
        }
    print(
        f"{'mean':25s}{summary['correlation']:+10.3f}"
        f"{summary['inhale_f1']:8.3f}{summary['exhale_f1']:8.3f}{summary['kl_ibi']:8.3f}"
    )

    # The composite proposed in the organiser notes, for orientation only -- the
    # official weighting is still a TODO on the competition page.
    composite = (
        0.50 * summary["inhale_f1"]
        + 0.20 * summary["exhale_f1"]
        + 0.20 * max(0.0, summary["correlation"])
        + 0.10 * float(np.exp(-summary["kl_ibi"]))
    )
    print(f"\nproposed composite (0.5/0.2/0.2/0.1): {composite:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "checkpoints": [str(p) for p in args.checkpoint],
                    "sessions": sorted(sessions),
                    "clips": rows,
                    "summary": summary,
                    "per_session": session_rows,
                    "summary_by_session": summary_by_session,
                    "strata": stratum_results,
                    "composite": composite,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")

    if args.plot:
        # --plot requires exactly one --checkpoint (checked above), so its
        # parent is unambiguously the run directory this checkpoint lives in.
        plot_dir = args.plot_dir or (args.checkpoint[0].parent / "diagnosis")
        plot_dir.mkdir(parents=True, exist_ok=True)
        title_suffix = args.checkpoint[0].parent.name
        rate_breakdown(
            plot_data,
            sorted(sessions),
            plot_dir / "rate_breakdown.png",
            title_suffix,
            n_shifts=args.null_shifts,
            corr_window_s=args.corr_window_s,
            corr_hop_s=args.corr_hop_s,
            corr_min_breaths=args.corr_min_breaths,
        )
        model, mean, std = models[0]
        reserved_grid(
            model,
            mean,
            std,
            device,
            entries,
            sorted(sessions),
            plot_dir / "reserved_grid.png",
            window=args.grid_window,
            infer_window=args.infer_window,
            frame_chunk=args.frame_chunk,
            amp_dtype=amp_dtype,
        )
        print(f"wrote diagnostic plots -> {plot_dir}")


if __name__ == "__main__":
    main()
