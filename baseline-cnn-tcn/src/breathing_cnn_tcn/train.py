"""Train a CNN-TCN, optionally reserving sessions and a validation tail.

Two numbers are reported per epoch:

``val loss`` / ``val corr``
    Cheap, windowed, computed exactly like the training loss.  Use it to watch
    for overfitting between epochs.
``xcorr`` / ``f1``
    The real thing: full-clip prediction, stitched, then scored with the
    competition's own metrics.  Costs a forward pass over every held-out clip,
    so it runs every ``--score-every`` epochs and is what checkpoint selection
    uses.

CLI
---
    python -m breathing_cnn_tcn.train
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scoring.metrics import event_f1, zero_lag_correlation
from scoring.processing import detect_inhalation_events
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from .augment import AugmentConfig
from .channels import CHANNEL_NAMES, ChannelSet
from .dataset import (
    STATS_FILENAME,
    ClipEntry,
    EpochRangeSampler,
    WindowDataset,
    channel_stats,
    load_manifest,
    reserve_test_sessions,
)
from .infer import predict_clip
from .model import BreathingLoss, BreathingNet


def _mean(values: list[dict], key: str) -> float:
    return float(np.mean([v[key] for v in values])) if values else float("nan")


@torch.no_grad()
def validate_windows(
    model: BreathingNet,
    loader: DataLoader,
    criterion: BreathingLoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Windowed validation loss, computed exactly like the training loss."""
    model.eval()
    parts: list[dict] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        t_in = batch["t_in"].to(device, non_blocking=True)
        t_out = batch["t_out"].to(device, non_blocking=True)
        signal = batch["signal"].to(device, non_blocking=True)
        onset = batch["onset"].to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred_signal, pred_onset = model(features, t_in, t_out)
                _, stats = criterion(
                    pred_signal.float(), pred_onset.float(), signal, onset
                )
        else:
            pred_signal, pred_onset = model(features, t_in, t_out)
            _, stats = criterion(pred_signal, pred_onset, signal, onset)
        parts.append(stats)
    return {f"val_{k}": _mean(parts, k) for k in ("loss", "corr", "onset")}


def score_full_clips(
    model: BreathingNet,
    entries: list[ClipEntry],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
    fs: float = 60.0,
    span: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, float], list[dict]]:
    """Predict each clip end to end and score it with the competition metrics.

    *span* restricts which fraction of each clip is *scored*.  The whole clip is
    still predicted -- inference is cheap, and it keeps every scored sample's
    receptive field fully populated with real context rather than zero-padding
    at the span boundary -- but only frames inside the span reach the metrics.
    Under ``--val-fraction`` that is what keeps trained-on frames out of the
    reported number.
    """
    rows: list[dict] = []
    for entry in entries:
        pred, _ = predict_clip(
            model,
            entry,
            mean,
            std,
            window=window,
            device=device,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        truth = pd.read_parquet(entry.target)["signal"].to_numpy(np.float64)
        n = min(len(truth), len(pred))
        truth, pred_n = truth[:n], pred[:n].astype(np.float64)
        if span != (0.0, 1.0):
            lo, hi = int(n * span[0]), int(n * span[1])
            truth, pred_n = truth[lo:hi], pred_n[lo:hi]

        corr = zero_lag_correlation(truth, pred_n)
        truth_on, truth_off = detect_inhalation_events(truth, fs)
        pred_on, pred_off = detect_inhalation_events(pred_n, fs)
        rows.append(
            {
                "clip_id": entry.clip_id,
                "session_idx": entry.session_idx,
                "correlation": corr,
                "inhale_f1": event_f1(truth_on / fs, pred_on / fs),
                "exhale_f1": event_f1(truth_off / fs, pred_off / fs),
                "n_truth_onsets": len(truth_on),
                "n_pred_onsets": len(pred_on),
            }
        )
    summary = {
        "xcorr": _mean(rows, "correlation"),
        "inhale_f1": _mean(rows, "inhale_f1"),
        "exhale_f1": _mean(rows, "exhale_f1"),
    }
    return summary, rows


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 1% of the peak learning rate."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--camera", default="face", choices=["face", "side"])
    parser.add_argument("--out-dir", type=Path, default=Path("runs_all"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Exact directory for this run. Unlike --out-dir, no timestamp or "
        "model-name component is appended. This is the safe option for sweep "
        "jobs and makes --resume unambiguous.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="Hold out the last fraction of every clip in time for early "
        "stopping, so every session still contributes gradients.  0 disables "
        "validation entirely -- then --epochs is a fixed budget with no early "
        "stopping.  Note the signal this gives is within-session, not the same "
        "thing as held-out generalisation.",
    )
    parser.add_argument(
        "--holdout-json",
        type=Path,
        default=Path("baseline-cnn-tcn/artifacts/holdout_sessions.json"),
        help="Where the reserved test sessions live.  Drawn on first use, then "
        "reused verbatim -- delete it only if you mean to invalidate every "
        "result measured against it.",
    )
    parser.add_argument("--n-test-sessions", type=int, default=3)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument(
        "--reserve-test-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve the sessions in --holdout-json. Disable only when an "
        "independent packaged test split is used; then every labelled session "
        "in --split is used for training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt in the run directory if one exists.",
    )

    parser.add_argument(
        "--channels",
        default="gray+diff+flow",
        help="Channels to train on, '+'- or ','-separated.  Names are gray, "
        "diff, flow_x, flow_y, plus the groups flow (both flow planes) and "
        "all.  Preprocessing always stores every channel, so this selects "
        "without reprocessing: 'gray' and 'gray+diff+flow' read byte-identical "
        "crops and share one channel_stats.json.  Order does not matter.",
    )

    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Exponential moving average of the weights, e.g. 0.999.  0 "
        "disables it.  The best epoch is mostly evaluation noise; averaging "
        "removes the lottery of picking one checkpoint.  Buffers are averaged "
        "too, so BatchNorm running statistics stay consistent with the "
        "averaged weights.",
    )

    parser.add_argument(
        "--time-stretch",
        type=float,
        default=3.0,
        help="Bound on the temporal stretch factor.  Without --rate-range, "
        "factors are drawn log-uniformly on [1/f, f]; with it, this clamps the "
        "rate-targeted factor.  1.0 disables stretching.",
    )
    parser.add_argument(
        "--rate-range",
        type=float,
        nargs=2,
        metavar=("LO_HZ", "HI_HZ"),
        default=[1.0, 7.0],
        help="Flatten the target rate distribution: measure each window's own "
        "rate and stretch it to a target drawn uniformly from this band. Needs "
        "--time-stretch to bound the factor.",
    )
    parser.add_argument(
        "--no-scale-motion",
        action="store_true",
        help="Do not rescale diff/flow by the stretch factor.  Off by default "
        "because a stretched window otherwise pairs a slow rhythm with "
        "fast-rhythm motion magnitudes.",
    )
    parser.add_argument(
        "--select-jitter",
        type=float,
        default=0.25,
        help="Max wander of each input frame's position, in selection-grid "
        "units (timestamp moves with it). 0 disables it.",
    )
    parser.add_argument(
        "--motion-noise",
        type=float,
        default=2.0,
        help="Extra noise on the motion channels only, in uint8 code units, "
        "at a level drawn per window. 0 disables it.",
    )
    parser.add_argument("--shift-px", type=int, default=4)
    parser.add_argument("--brightness", type=float, default=0.15)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument(
        "--flip",
        type=float,
        default=0.0,
        help="Horizontal mirror probability.  Negates flow_x to match.  Off by "
        "default when the camera only ever views one side of the subject.",
    )

    parser.add_argument("--w-corr", type=float, default=1.0)
    parser.add_argument("--w-onset", type=float, default=0.5)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="Average-pool factors the correlation loss is computed at, in "
        "output frames.  Scale 1 is the raw signal; larger scales expose "
        "slower structure that a single whole-window correlation averages "
        "away.  See model.multiscale_pearson_loss.",
    )

    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument("--score-every", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop after this many consecutive scored evaluations without a new "
        "best held-out correlation.  0 disables early stopping.  Counts scored "
        "evaluations, not epochs, so the budget is patience x score-every epochs.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=24,
        help="Never stop early before this epoch, regardless of --patience.",
    )
    parser.add_argument("--val-windows", type=int, default=2048)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp",
        default="bf16",
        choices=["bf16", "fp16", "off"],
        help="bf16 needs no loss scaling and is the default on Ada GPUs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms and disable cuDNN "
        "benchmarking. This can reduce throughput and will fail loudly if an "
        "operation has no deterministic implementation.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    config, entries = load_manifest(args.features_dir, args.split, args.camera)
    labelled = [e for e in entries if e.has_target]
    if not labelled:
        raise SystemExit(
            f"no clip in manifest_{args.split}_{args.camera}.json carries a target; "
            "only the public split can be trained on."
        )

    stored = tuple(config["channel_names"])
    if stored != CHANNEL_NAMES:
        raise SystemExit(
            f"{args.features_dir} stores channels {list(stored)} but this "
            f"version expects {list(CHANNEL_NAMES)} -- re-run preprocess."
        )
    try:
        channel_set = ChannelSet.parse(args.channels)
    except ValueError as exc:
        raise SystemExit(f"--channels: {exc}") from exc

    mean, std = channel_stats(labelled, args.features_dir / STATS_FILENAME)

    test_sessions = (
        reserve_test_sessions(
            labelled,
            args.holdout_json,
            n_test=args.n_test_sessions,
            seed=args.test_seed,
        )
        if args.reserve_test_sessions
        else []
    )
    pool = [e for e in labelled if e.session_idx not in set(test_sessions)]
    train_entries, val_entries = pool, pool
    # Every session trains.  Validation, if any, is the time-tail of these
    # same clips -- see --val-fraction.
    train_span = (0.0, 1.0 - args.val_fraction)
    val_span = (1.0 - args.val_fraction, 1.0)
    scoring = args.val_fraction > 0
    if not scoring and not args.epochs:
        raise SystemExit("--val-fraction 0 needs --epochs set explicitly")

    # Cheap, but the failure it guards against is expensive and invisible: a
    # reserved session leaking into training makes every number measured against
    # the holdout meaningless, with no symptom.
    leaked = {e.session_idx for e in [*train_entries, *val_entries]} & set(
        test_sessions
    )
    if leaked:
        raise SystemExit(f"reserved test sessions leaked into training: {leaked}")

    if args.reserve_test_sessions:
        print(
            f"reserved test sessions (never trained or validated on): "
            f"{test_sessions}  <- {args.holdout_json}",
            flush=True,
        )
    else:
        print(
            "no sessions reserved from the training split; an independent "
            "test split is required for unbiased evaluation",
            flush=True,
        )
    tail = (
        f"validating on the last {args.val_fraction:.0%} of each clip"
        if scoring
        else f"NO validation -- fixed {args.epochs} epochs, no early stopping"
    )
    print(
        f"train {len(train_entries)} clips / "
        f"{len({e.session_idx for e in train_entries})} sessions  |  {tail}",
        flush=True,
    )
    selected_mean, selected_std = channel_set.take_stats(mean, std)
    print(
        f"channels {channel_set} ({len(channel_set)} of {len(stored)} stored)  "
        f"mean {np.round(selected_mean, 2)}  std {np.round(selected_std, 2)}",
        flush=True,
    )

    augment = AugmentConfig(
        time_stretch=args.time_stretch,
        rate_range=tuple(args.rate_range) if args.rate_range else None,
        scale_motion=not args.no_scale_motion,
        select_jitter=args.select_jitter,
        motion_noise=args.motion_noise,
        shift_px=args.shift_px,
        brightness=args.brightness,
        contrast=args.contrast,
        noise=args.noise,
        flip=args.flip,
    )
    print(
        f"augmentation: {'on -> ' + str(augment) if augment.enabled else 'off'}",
        flush=True,
    )

    horizon = args.epochs * args.steps_per_epoch * args.batch_size
    grids = dict(
        select_fs=config["select_fs_hz"],
        output_fs=config["output_fs_hz"],
        motion_tau_s=config["motion_tau_s"],
        onset_sigma_s=config["onset_sigma_s"],
    )
    train_set = WindowDataset(
        train_entries,
        window=args.window,
        mean=mean,
        std=std,
        length=horizon,
        seed=args.seed,
        augment=augment,
        span=train_span,
        channels=channel_set,
        **grids,
    )
    # Gridded and non-overlapping, so the windowed number is comparable epoch to
    # epoch; capped because a full grid over every held-out clip is more forward
    # passes than the signal in the number justifies.
    val_set = (
        WindowDataset(
            val_entries,
            window=args.window,
            mean=mean,
            std=std,
            stride=args.window,
            span=val_span,
            channels=channel_set,
            **grids,
        )
        if scoring
        else None
    )
    if val_set is not None and len(val_set) > args.val_windows:
        val_set.index = val_set.index[:: len(val_set) // args.val_windows + 1]

    sampler = EpochRangeSampler(args.steps_per_epoch * args.batch_size)
    common = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, **common
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common)
        if val_set is not None
        else None
    )

    model = BreathingNet(channels=channel_set, dropout=args.dropout).to(device)
    criterion = BreathingLoss(args.w_corr, args.w_onset, tuple(args.scales))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: cosine_schedule(s, total_steps, args.warmup_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    # The averaged weights start life anchored to the random initialisation and
    # relax toward the trained ones with a time constant of 1/(1-decay) steps --
    # 1000 at the 0.999 default.  Selecting on them before that has settled picks
    # a checkpoint that is mostly noise: measured, a 2-epoch (400-step) run scored
    # +0.118 on EMA against +0.716 live.  Three time constants is ~95% relaxed.
    ema_ready_step = int(3.0 / (1.0 - args.ema_decay)) if args.ema_decay > 0 else 0
    global_step = 0

    ema = None
    if args.ema_decay > 0:
        ema = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
            use_buffers=True,
        )

    if ema is not None:
        print(
            f"EMA decay {args.ema_decay}: selection switches to the averaged "
            f"weights after {ema_ready_step} steps "
            f"(~epoch {ema_ready_step // args.steps_per_epoch + 1}); "
            f"until then the log marks them '~'",
            flush=True,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{n_params / 1e6:.2f} M params | receptive field {model.receptive_field} "
        f"frames ({model.receptive_field / 60:.1f} s) | window {args.window} "
        f"x batch {args.batch_size} | amp {args.amp}",
        flush=True,
    )

    runs_root = args.out_dir / "baseline-cnn-tcn"
    # The channel selection is in the directory name so a sweep over channel
    # sets is legible without opening args.json.  The timestamp still leads, so
    # --resume's "most recent" ordering is unchanged.
    fresh_dir = runs_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{channel_set.slug}"
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.resume:
        # Resume the most recently started run that has a checkpoint, rather
        # than a run directory named for this invocation's arguments.
        existing = sorted(p for p in runs_root.glob("*") if (p / "last.pt").exists())
        run_dir = existing[-1] if existing else fresh_dir
    else:
        run_dir = fresh_dir
    if run_dir.exists() and not args.resume and any(run_dir.iterdir()):
        raise SystemExit(
            f"{run_dir} already exists and is not empty; pass --resume to "
            "continue that exact run or choose another --run-dir"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    def save(
        path: Path,
        epoch: int,
        metrics: dict,
        per_clip: list[dict],
        weights: torch.nn.Module | None = None,
    ) -> None:
        """Write a checkpoint.

        ``model`` holds whichever weights were selected -- the EMA copy when
        averaging is on -- so downstream loaders need no special case.  The live
        weights and the averaging state are stored alongside so that ``last.pt``
        resumes exactly: optimiser, scheduler, and a cosine schedule restarted
        mid-run would otherwise not be the same schedule.
        """
        state = {
                "model": (weights or model).state_dict(),
                "model_live": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "ema_decay": args.ema_decay,
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "since_best": since_best,
                "history": history,
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "test_sessions": test_sessions,
                "train_split": args.split,
                "trained_sessions": sorted({e.session_idx for e in train_entries}),
                # Not part of state_dict, but the weights are unusable without
                # it: it fixes the first conv's input width and which planes of
                # the stored array to feed it.
                "channels": list(channel_set.names),
                # Full-width, over every stored channel, so checkpoints trained
                # on different selections stay comparable.  Consumers slice with
                # ChannelSet.take_stats.
                "mean": mean,
                "std": std,
                "args": vars(args)
                | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
                "feature_config": config,
                "metrics": metrics,
                "per_clip": per_clip,
            }
        # Keep the previous checkpoint intact until the replacement is fully
        # serialized. A process or machine failure during torch.save then costs
        # at most one epoch instead of corrupting the only resumable file.
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    best = -np.inf
    since_best = 0
    history: list[dict] = []
    start_epoch = 0

    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["test_sessions"] != test_sessions:
            raise SystemExit(
                f"{last_path} reserved {state['test_sessions']} but this run "
                f"reserves {test_sessions} -- refusing to mix holdouts."
            )
        resumed_channels = tuple(
            state.get("channels") or state["feature_config"]["channel_names"]
        )
        if resumed_channels != channel_set.names:
            raise SystemExit(
                f"{last_path} was trained on channels {list(resumed_channels)} "
                f"but this run asks for {list(channel_set.names)} -- the first "
                "convolution has a different shape, so it cannot be resumed."
            )
        model.load_state_dict(state.get("model_live") or state["model"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best = state["best"]
        since_best = state.get("since_best", 0)
        history = state["history"]
        start_epoch = state["epoch"] + 1
        global_step = state.get("global_step", start_epoch * args.steps_per_epoch)
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
        print(
            f"resumed from {last_path} at epoch {start_epoch} (best xcorr {best:+.4f})"
        )
    elif args.resume:
        print(f"--resume given but {last_path} does not exist; starting fresh")

    for epoch in range(start_epoch, args.epochs):
        sampler.epoch = epoch
        model.train()
        started = time.perf_counter()
        parts: list[dict] = []
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            t_in = batch["t_in"].to(device, non_blocking=True)
            t_out = batch["t_out"].to(device, non_blocking=True)
            signal = batch["signal"].to(device, non_blocking=True)
            onset = batch["onset"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred_signal, pred_onset = model(features, t_in, t_out)
                # Losses in fp32: the correlation term normalises by a sum of
                # squares over 512 samples, which is exactly the kind of
                # reduction that loses precision in half.
                loss, stats = criterion(
                    pred_signal.float(), pred_onset.float(), signal, onset
                )
            else:
                pred_signal, pred_onset = model(features, t_in, t_out)
                loss, stats = criterion(pred_signal, pred_onset, signal, onset)

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update_parameters(model)
            parts.append(stats)

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row |= {f"train_{k}": _mean(parts, k) for k in ("loss", "corr", "onset")}
        if val_loader is not None:
            row |= validate_windows(model, val_loader, criterion, device, amp_dtype)

        scored = scoring and (
            (epoch + 1) % args.score_every == 0 or epoch == args.epochs - 1
        )
        if scored:
            summary, per_clip = score_full_clips(
                model,
                val_entries,
                mean,
                std,
                device,
                window=args.infer_window,
                frame_chunk=args.frame_chunk,
                amp_dtype=amp_dtype,
                span=val_span,
            )
            row |= summary
            # Scoring 6-8 clips costs ~8 s against a ~115 s epoch, so evaluating
            # the averaged weights as well is close to free -- and it is the only
            # way to know whether EMA actually helped.
            selected, selected_clips = model, per_clip
            if ema is not None and global_step < ema_ready_step:
                row["ema_warming"] = True
            if ema is not None:
                ema_summary, ema_clips = score_full_clips(
                    ema.module,
                    val_entries,
                    mean,
                    std,
                    device,
                    window=args.infer_window,
                    frame_chunk=args.frame_chunk,
                    amp_dtype=amp_dtype,
                    span=val_span,
                )
                row |= {f"ema_{k}": v for k, v in ema_summary.items()}
                # Select on the averaged weights -- they are what would ship --
                # but only once they have relaxed away from the initialisation.
                if global_step >= ema_ready_step:
                    summary, per_clip = ema_summary, ema_clips
                    selected, selected_clips = ema.module, ema_clips

            if summary["xcorr"] > best:
                best = summary["xcorr"]
                since_best = 0
                save(
                    run_dir / "best.pt",
                    epoch,
                    summary,
                    selected_clips,
                    weights=selected,
                )
                (run_dir / "best_per_clip.json").write_text(
                    json.dumps(selected_clips, indent=2)
                )
            else:
                since_best += 1
            row["since_best"] = since_best

        row["seconds"] = round(time.perf_counter() - started, 1)
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Written every epoch, scored or not, so a crash costs one epoch.
        save(last_path, epoch, row, [])

        message = f"[{epoch + 1:3d}/{args.epochs}] train corr {row['train_corr']:+.3f}"
        if "val_corr" in row:
            message += f"  val corr {row['val_corr']:+.3f}  loss {row['val_loss']:.4f}"
        else:
            message += f"  loss {row['train_loss']:.4f}"
        if scored:
            marker = " *" if row.get("ema_xcorr", row["xcorr"]) >= best else ""
            message += f"  |  xcorr {row['xcorr']:+.3f}"
            if "ema_xcorr" in row:
                message += f"  ema {row['ema_xcorr']:+.3f}"
                if row.get("ema_warming"):
                    message += "~"
            message += f"  inhale_f1 {row['inhale_f1']:.3f}{marker}"
        print(f"{message}  ({row['seconds']:.0f}s)", flush=True)

        if (
            args.patience
            and scored
            and since_best >= args.patience
            and epoch + 1 >= args.min_epochs
        ):
            print(
                f"early stop: {since_best} scored evaluations "
                f"({since_best * args.score_every} epochs) without improving on "
                f"{best:+.4f}",
                flush=True,
            )
            break

    if not scoring:
        # Nothing was ever scored, so there is no "best" to choose -- the final
        # weights are the deliverable.  Written as best.pt too so downstream
        # loaders need no special case.
        save(run_dir / "best.pt", epoch, {}, [], weights=ema.module if ema else None)
        print(f"final weights (no validation) -> {run_dir / 'best.pt'}")
    else:
        print(f"best held-out correlation {best:+.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
