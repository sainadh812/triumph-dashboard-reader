"""Colab-only uploaded-video inference; no private storage or credentials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import zipfile

from .inference import DashboardPredictor, _colab_guard, DEFAULT_THRESHOLDS
from .video import process_video, write_json
from .weights import download_weights
from .release import RELEASE_TAG, WEIGHTS_SHA256


def run_video(
    source,
    output_dir,
    *,
    device="cuda",
    batch_size=None,
    cpu_threads=2,
    last_seconds=None,
    weights_cache="/content/triumph_weights",
):
    """Return output paths and measured timings; run every selected input frame."""
    _colab_guard()
    import torch

    if device not in ("cuda", "cpu"):
        raise ValueError("Choose cuda or cpu explicitly")
    if device == "cpu" and torch.cuda.is_available():
        raise RuntimeError("Select CPU hardware first to avoid holding an unused GPU")
    if device == "cpu":
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 32:
            raise ValueError("cpu_threads must be1..32")
        torch.set_num_threads(cpu_threads)
    if batch_size is None:
        batch_size = 16 if device == "cuda" else 4
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = Path(source)
    output_dir = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(
            "Choose a new results folder; existing outputs are preserved"
        )
    run_start = time.perf_counter()
    before = time.perf_counter()
    paths, manifest = download_weights(weights_cache)
    download_seconds = time.perf_counter() - before
    before = time.perf_counter()
    predictor = DashboardPredictor(paths, batch_size=batch_size, device=device)
    model_load_seconds = time.perf_counter() - before
    report = process_video(
        source,
        output_dir,
        predictor,
        batch_size=batch_size,
        last_seconds=last_seconds,
        label=source.name,
    )
    report.update(
        release_tag=RELEASE_TAG,
        release_archive_sha256=WEIGHTS_SHA256,
        selected_epochs=manifest["selected_epochs"],
        model_identity=predictor.metadata,
        thresholds=dict(DEFAULT_THRESHOLDS),
        model_load_seconds=model_load_seconds,
        release_download_and_verification_seconds=download_seconds,
        wall_seconds_through_verified_video=time.perf_counter() - run_start,
        browser_upload_download_included_in_timing=False,
        reported_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "report.json", report)
    archive = output_dir / "results.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as z:
        for name in (
            "labeled.mp4",
            "report.json",
            "predictions.csv",
            "predictions.jsonl",
        ):
            z.write(output_dir / name, name)
    print(
        f'Complete: {report["frames_processed"]} frames; source {report["source_fps"]:.2f} FPS; '
        f'prediction pipeline {report["prediction_pipeline_fps"]:.2f} FPS; '
        f'full processing {report["end_to_end_processing_fps"]:.2f} FPS.'
    )
    print("Actual model execution coverage:", report["actual_task_inference_coverage"])
    print("Unavailable reading reasons:", report["unavailable_reasons"])
    print(
        "Results are local to this Colab runtime. Download before disconnecting or changing runtime."
    )
    return {
        "video": str(output_dir / "labeled.mp4"),
        "report": str(output_dir / "report.json"),
        "archive": str(archive),
        "metrics": report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--last-seconds",
        type=float,
        default=None,
        help="Default processes the full video",
    )
    parser.add_argument("--weights-cache", default="/content/triumph_weights")
    args = parser.parse_args(argv)
    result = run_video(
        args.video,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        last_seconds=args.last_seconds,
        weights_cache=args.weights_cache,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
