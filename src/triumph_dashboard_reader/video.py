"""Video IO, full-frame overlays and honest conditional-pipeline timing.

Adapted from dashboard_joint_v1 revision 7a3fd991; no Drive/auth dependencies.
Importing this module does not construct or execute neural models.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


def probe_video(path):
    """Read timestamps for every source frame, without neural inference."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_streams",
        "-show_entries",
        "frame=best_effort_timestamp_time:stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    document = json.loads(result.stdout)
    if len(document.get("streams", [])) != 1:
        raise ValueError("Expected a readable primary video stream")
    stream = document["streams"][0]

    def fraction(value):
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator) if float(denominator) else 0.0

    fps = fraction(stream.get("avg_frame_rate", "0/0"))
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("Video frame rate is unavailable or unsupported")
    frames = document.get("frames", [])
    pts = [float(row["best_effort_timestamp_time"]) for row in frames]
    if (
        not pts
        or any(not math.isfinite(value) for value in pts)
        or any(b < a for a, b in zip(pts, pts[1:]))
    ):
        raise ValueError("Video has missing or non-monotonic frame timestamps")
    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    period = sorted(deltas)[len(deltas) // 2] if deltas else 1 / fps
    duration = pts[-1] - pts[0] + period
    variable = bool(deltas and max(deltas) - min(deltas) > max(0.001, 0.02 / fps))
    return {
        "fps": fps,
        "timestamps": pts,
        "duration_seconds": duration,
        "last_frame_period_seconds": period,
        "variable_frame_rate": variable,
        "stream_width": int(stream["width"]),
        "stream_height": int(stream["height"]),
        "reported_frame_count": stream.get("nb_frames"),
        "decoded_timestamp_count": len(pts),
    }


def clip_indices(metadata, seconds):
    pts = metadata["timestamps"]
    end = pts[-1] + metadata["last_frame_period_seconds"]
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("last_seconds must be positive, or None for the full video")
    start = pts[0] if seconds is None else max(pts[0], end - seconds)
    indices = [
        index for index, timestamp in enumerate(pts) if timestamp >= start - 1e-9
    ]
    if not indices:
        raise ValueError("Requested video window contains no frames")
    return indices[0], indices[-1] + 1, start, end


def _point(point):
    if (
        point is None
        or len(point) != 2
        or any(not math.isfinite(float(v)) for v in point)
    ):
        return None
    if any(abs(float(v)) >= 2**30 for v in point):
        return None
    return tuple(int(round(float(v))) for v in point)


def draw_prediction(frame, prediction, source_time, source_name):
    """Full input frame with basic geometry and readings; no reader-input crops."""
    import cv2
    import numpy as np

    output = frame.copy()
    height, width = output.shape[:2]
    scale = max(0.45, min(width, height) / 700.0)
    thickness = max(1, round(scale * 2))

    def label(text, xy, color, size=1):
        font = scale * size
        cv2.putText(
            output,
            str(text),
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            str(text),
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def polygon(points, color, close=True):
        values = [_point(p) for p in points or []]
        if len(values) >= 2 and all(v is not None for v in values):
            cv2.polylines(
                output,
                [np.asarray(values, np.int32)],
                close,
                color,
                thickness,
                cv2.LINE_AA,
            )

    box = prediction.get("dashboard_box")
    if (
        box
        and len(box) == 4
        and _point(box[:2]) is not None
        and _point(box[2:]) is not None
    ):
        cv2.rectangle(
            output,
            (round(box[0]), round(box[1])),
            (round(box[2]), round(box[3])),
            (0, 220, 255),
            thickness,
        )
    polygon(prediction.get("lcd_polygon"), (255, 230, 0))
    speed = prediction.get("speed", {})
    points = {"pivot": speed.get("pivot"), "tip": speed.get("tip")}
    points.update(
        {"tick_" + str(name): point for name, point in speed.get("ticks", {}).items()}
    )
    pivot, tip = _point(points.get("pivot")), _point(points.get("tip"))
    if pivot is not None and tip is not None:
        cv2.line(output, pivot, tip, (0, 80, 255), thickness + 1, cv2.LINE_AA)
    for name, point in points.items():
        xy = _point(point)
        if xy is None:
            continue
        cv2.circle(output, xy, max(2, round(scale * 3)), (50, 255, 80), -1, cv2.LINE_AA)
        if name.startswith("tick_"):
            label(name[5:], (xy[0] + 3, xy[1] - 3), (50, 255, 80), 0.6)
    rpm = prediction.get("rpm", {})
    polygon(prediction.get("rpm_roi_corners"), (255, 140, 0))
    for name, point in rpm.get("anchors", {}).items():
        xy = _point(point)
        if xy is not None:
            cv2.circle(
                output, xy, max(2, round(scale * 3)), (255, 200, 0), -1, cv2.LINE_AA
            )
            label("R" + str(name), (xy[0] + 3, xy[1] - 3), (255, 200, 0), 0.6)
    end = _point(rpm.get("bar_end"))
    if end is not None:
        cv2.drawMarker(
            output,
            end,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            max(8, round(scale * 12)),
            thickness + 1,
            cv2.LINE_AA,
        )
    value = speed.get("value_kmh")
    speed_text = (
        f"{value:.1f} km/h"
        if isinstance(value, (int, float)) and math.isfinite(value)
        else "unavailable"
    )
    value = rpm.get("value")
    rpm_text = (
        f"{value:.0f}"
        if isinstance(value, (int, float)) and math.isfinite(value)
        else "unavailable"
    )
    line = max(24, round(31 * scale))
    label("Speed: " + speed_text, (12, line), (255, 255, 255))
    label("RPM: " + rpm_text, (12, line * 2 + 5), (0, 255, 255), 1.45)
    lcd_status = prediction.get("localizer", {}).get(
        "lcd_status",
        prediction.get(
            "lcd_status", "detected" if prediction.get("lcd_polygon") else "unavailable"
        ),
    )
    label(
        f"LCD: {lcd_status} | t={source_time:.2f}s",
        (12, height - 12),
        (255, 255, 255),
        0.75,
    )
    return output


def _encode_h264(source, target):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    if not Path(target).is_file() or Path(target).stat().st_size <= 0:
        raise ValueError("H264 output was not created")


def pad_for_encoder(frame):
    """Pad only the right/bottom edges; preserve original pixels and coordinates."""
    import cv2

    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(
        frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def _task_inference_flags(prediction):
    """Executed readers retain raw landmark visibility even if geometry rejects.

    An empty visibility dictionary is the predictor's explicit skipped-reader
    representation; unavailable numeric readings alone do not mean a skip.
    """
    return {
        "localizer": True,
        "speed": bool(prediction.get("speed", {}).get("point_visibility")),
        "rpm": bool(prediction.get("rpm", {}).get("point_visibility")),
    }


def process_video(
    source, output_dir, predictor, batch_size=16, last_seconds=None, label="video"
):
    """Decode each selected frame once; batch prediction keeps original order."""
    import cv2

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(
        f"{label}: scanning source frame timestamps; prediction has not started.",
        flush=True,
    )
    before = time.perf_counter()
    metadata = probe_video(source)
    initial_probe_seconds = time.perf_counter() - before
    first, stop, requested_start, clip_end = clip_indices(metadata, last_seconds)
    selected_count = stop - first
    # CFR output contains every selected frame. For VFR sources original PTS
    # stays in CSV/JSONL; playback cadence is explicitly documented as CFR.
    playback_duration = clip_end - metadata["timestamps"][first]
    output_fps = selected_count / playback_duration
    print(
        f"{label}: {selected_count} selected frames, source indices {first}..{stop-1}; "
        f"timestamp scan {initial_probe_seconds:.2f}s. Decoding source prefix before selected window.",
        flush=True,
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Could not open source video")
    raw = output_dir / "labeled_intermediate.mp4"
    final = output_dir / "labeled.mp4"
    csv_path, jsonl_path = (
        output_dir / "predictions.csv",
        output_dir / "predictions.jsonl",
    )
    writer = None
    timers = {
        "initial_probe_seconds": initial_probe_seconds,
        "decode_seconds": 0.0,
        "model_pipeline_seconds": 0.0,
        "draw_seconds": 0.0,
        "intermediate_encode_seconds": 0.0,
        "prediction_log_seconds": 0.0,
    }
    decode_breakdown = {"prefix_decode_seconds": 0.0, "selected_decode_seconds": 0.0}
    processed, frame_index, batch, batch_indices = 0, 0, [], []
    batch_timings = []
    statuses = {"speed_available": 0, "rpm_available": 0}
    rejection_reasons = {"speed": {}, "rpm": {}}
    task_frames = {"localizer": 0, "speed": 0, "rpm": 0, "all_three": 0}
    output_shape = None
    with (
        csv_path.open("w", newline="", encoding="utf8") as csv_file,
        jsonl_path.open("w", encoding="utf8") as log_file,
    ):
        columns = [
            "frame_index",
            "clip_frame_index",
            "source_time_seconds",
            "speed_kmh",
            "speed_status",
            "rpm",
            "rpm_status",
            "lcd_status",
        ]
        table = csv.DictWriter(csv_file, fieldnames=columns)
        table.writeheader()

        def flush():
            nonlocal processed, batch, batch_indices, writer, output_shape
            if not batch:
                return
            before = time.perf_counter()
            predictions = predictor.predict_frames(batch)
            batch_wall = time.perf_counter() - before
            timers["model_pipeline_seconds"] += batch_wall
            batch_timings.append(
                {
                    "frames": len(batch),
                    "wall_seconds": batch_wall,
                    "predictor_timing": getattr(predictor, "last_batch_timing", None),
                }
            )
            if len(predictions) != len(batch):
                raise ValueError("Predictor returned a different frame count")
            for frame, index, prediction in zip(batch, batch_indices, predictions):
                before = time.perf_counter()
                annotated = draw_prediction(
                    frame, prediction, metadata["timestamps"][index], label
                )
                timers["draw_seconds"] += time.perf_counter() - before
                if output_shape is None:
                    output_shape = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(raw),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (
                            output_shape[1] + output_shape[1] % 2,
                            output_shape[0] + output_shape[0] % 2,
                        ),
                    )
                    if not writer.isOpened():
                        raise ValueError("Could not create full-frame output video")
                if annotated.shape[:2] != output_shape:
                    raise ValueError(
                        "Variable decoded dimensions cannot preserve full-frame layout"
                    )
                before = time.perf_counter()
                writer.write(pad_for_encoder(annotated))
                timers["intermediate_encode_seconds"] += time.perf_counter() - before
                before = time.perf_counter()
                speed, rpm = prediction.get("speed", {}), prediction.get("rpm", {})
                executed = _task_inference_flags(prediction)
                for task, ran in executed.items():
                    task_frames[task] += int(ran)
                task_frames["all_three"] += int(all(executed.values()))
                record = {
                    "frame_index": index,
                    "clip_frame_index": processed,
                    "source_time_seconds": metadata["timestamps"][index],
                    "speed_kmh": speed.get("value_kmh"),
                    "speed_status": speed.get("reason")
                    or (
                        "available"
                        if speed.get("value_kmh") is not None
                        else "unavailable"
                    ),
                    "rpm": rpm.get("value"),
                    "rpm_status": rpm.get("reason")
                    or ("available" if rpm.get("value") is not None else "unavailable"),
                    "lcd_status": prediction.get("localizer", {}).get(
                        "lcd_status",
                        prediction.get(
                            "lcd_status",
                            (
                                "detected"
                                if prediction.get("lcd_polygon")
                                else "unavailable"
                            ),
                        ),
                    ),
                }
                for task, status_key in (
                    ("speed", "speed_status"),
                    ("rpm", "rpm_status"),
                ):
                    if record[status_key] != "available":
                        key = record[status_key]
                        rejection_reasons[task][key] = (
                            rejection_reasons[task].get(key, 0) + 1
                        )
                table.writerow(record)
                log_file.write(
                    json.dumps(dict(record, prediction=prediction), allow_nan=False)
                    + "\n"
                )
                statuses["speed_available"] += speed.get("value_kmh") is not None
                statuses["rpm_available"] += rpm.get("value") is not None
                processed += 1
                timers["prediction_log_seconds"] += time.perf_counter() - before
            csv_file.flush()
            log_file.flush()
            batch, batch_indices = [], []
            if processed % (batch_size * 10) == 0 or processed == selected_count:
                elapsed = time.perf_counter() - started
                pipeline_fps = processed / max(timers["model_pipeline_seconds"], 1e-12)
                print(
                    f"{label}: {processed}/{selected_count} predicted frames; "
                    f"prediction_pipelineFPS={pipeline_fps:.2f} (conditional task coverage); "
                    f"elapsed={elapsed:.2f}s; export pending.",
                    flush=True,
                )

        try:
            while frame_index < stop:
                before = time.perf_counter()
                if frame_index < first:
                    ok, frame = capture.grab(), None
                else:
                    ok, frame = capture.read()
                decode_elapsed = time.perf_counter() - before
                timers["decode_seconds"] += decode_elapsed
                decode_breakdown[
                    (
                        "prefix_decode_seconds"
                        if frame_index < first
                        else "selected_decode_seconds"
                    )
                ] += decode_elapsed
                if not ok:
                    raise ValueError(
                        "Source decode ended before its verified timestamp sequence"
                    )
                if frame_index >= first:
                    batch.append(frame)
                    batch_indices.append(frame_index)
                    if len(batch) == batch_size:
                        flush()
                frame_index += 1
            flush()
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    if processed != selected_count:
        raise ValueError("Not all frames in the requested final window were processed")
    print(
        f"{label}: predictions complete. Encoding full-frame H264 output now.",
        flush=True,
    )
    before = time.perf_counter()
    _encode_h264(raw, final)
    timers["h264_encode_seconds"] = time.perf_counter() - before
    print(
        f'{label}: H264 encoding complete ({timers["h264_encode_seconds"]:.2f}s). Verifying output frame count.',
        flush=True,
    )
    before = time.perf_counter()
    encoded = probe_video(final)
    if encoded["decoded_timestamp_count"] != processed:
        raise ValueError("Encoded output does not contain every predicted frame")
    expected_shape = [size + size % 2 for size in output_shape]
    if [encoded["stream_height"], encoded["stream_width"]] != expected_shape:
        raise ValueError("Encoded output dimensions do not preserve the original frame")
    timers["output_verification_seconds"] = time.perf_counter() - before
    timers["total_processing_seconds"] = time.perf_counter() - started
    report = {
        "status": "complete",
        "source_label": label,
        "frames_processed": processed,
        "first_source_frame_index": first,
        "stop_source_frame_index_exclusive": stop,
        "requested_start_seconds": requested_start,
        "first_source_timestamp_seconds": metadata["timestamps"][first],
        "clip_end_seconds": clip_end,
        "clip_duration_seconds": playback_duration,
        "requested_last_seconds": last_seconds,
        "source_fps": metadata["fps"],
        "source_variable_frame_rate": metadata["variable_frame_rate"],
        "source_frame_count": len(metadata["timestamps"]),
        "output_playback_fps": output_fps,
        "output_frame_count": encoded["decoded_timestamp_count"],
        "batch_size": batch_size,
        "original_decoded_height_width": list(output_shape),
        "encoded_height_width": [encoded["stream_height"], encoded["stream_width"]],
        "timing": timers,
        "model_pipeline_fps": processed / max(timers["model_pipeline_seconds"], 1e-12),
        "prediction_pipeline_fps": processed
        / max(timers["model_pipeline_seconds"], 1e-12),
        "decode_timing_breakdown": decode_breakdown,
        "decode_timing_note": "Breakdown sums to timing.decode_seconds; it is not additional work and must not be added a second time.",
        "end_to_end_processing_fps": processed
        / max(timers["total_processing_seconds"], 1e-12),
        "availability": statuses,
        "unavailable_reasons": rejection_reasons,
        "every_selected_frame_processed": True,
        "actual_task_inference_frames": task_frames,
        "actual_task_inference_coverage": {
            task: count / processed for task, count in task_frames.items()
        },
        "pipeline_work_scope": "Conditional work: localizer runs every frame; speed requires a localized dashboard, RPM requires a valid RPM region. Pipeline FPS is not a benchmark with all three readers forced on every frame. Rejected numeric readings can still have executed their reader.",
        "batch_timings": batch_timings,
        "warmup_policy": "No frames are discarded for warmup. First-batch startup costs are included. Models remain loaded across source videos.",
        "execution": getattr(predictor, "execution_metadata", None),
        "timing_scope": "Model pipeline includes preprocessing, prediction and geometry; CUDA execution is synchronized before predict_frames returns, and CPU execution is synchronous. Model construction is excluded from per-video processing and reported separately. Release download and browser upload/download are excluded.",
        "full_frame_preserved": True,
        "audio_included": False,
        "playback_note": "CFR playback preserves every selected frame; original timestamps are in CSV/JSONL. Playback FPS is not inference throughput.",
        "metric_scope": "Unlabeled video inspection; no ground-truth speed/RPM accuracy is claimed.",
    }
    write_json(output_dir / "report.json", report)
    # Intermediate is disposable only after frame-count verification succeeds.
    raw.unlink()
    return report
