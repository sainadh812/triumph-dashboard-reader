# Release v0.1.0

Three inference checkpoints for dashboard/LCD localization, analog speed and LCD RPM reading. Selected epochs: **localizer 5, speed 0, RPM 14**. Speed epoch 0 retains the unchanged starting weights because fine-tuning did not improve the validation selection score.

This is a prototype. Numeric-reading availability is **not accuracy**. No 98% accuracy, physical speed/RPM accuracy or unseen-recording performance is established. Validation changes are mixed: daylight localization worsened slightly, daylight RPM endpoint error stayed unchanged, and night RPM endpoint error improved. Motion blur, glare, partial framing and the empty RPM bar at engine-off remain failure cases.

The selected weights and original pipeline were tested on the same two 120-second clips in Colab GPU and CPU modes. These are separate hardware measurements, not a before/after training comparison. The public notebook wrapper has not yet completed its own full Colab video run.

| Execution | Prediction FPS, day / night | Full-output FPS, day / night |
|---|---:|---:|
| T4 GPU, float16 autocast, batch 16 | 52.90 / 53.19 | 15.34 / 8.84 |
| CPU, float32, batch 4, two threads | 4.17 / 4.23 | 3.30 / 2.74 |

GPU run on **19 October 2026**: daylight **3,602 frames in 234.80 seconds**; night **3,599 frames in 407.05 seconds**. Each clip is the last two minutes of its recording. Prediction-pipeline time alone was **68.09 seconds day / 67.67 seconds night**. These figures include the first batch; no warmup frames were discarded.

Full-output timing includes decoding, overlays and encoding; it excludes model setup and Drive transfer. A reader may be skipped when its region is unusable. Different precision and batch settings can change predictions near decision thresholds; the hardware runs are not an accuracy comparison. The clear empty-bar example still produces a false RPM reading.

## Download integrity

- Archive: `triumph-dashboard-weights-v0.1.0.zip`
- Bytes: **17,717,987**
- SHA256: `2ae6c5539fa097154fd57eec984bdeaffecd92d0f67fe18e6e9747c86d585be5`

The archive contains only `weights_manifest.json` and the three original inference checkpoint files below. Checkpoint tensors and file bytes are unchanged. Inference metadata retains selection and initialization statistics; no optimizer, scheduler, RNG state, credentials, raw video, Drive links or private filesystem paths are included. Load through the repository's verified release loader.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `localizer/best_model.pt` | 6,415,603 | `e3d6cac90715685f0e7075ac6e4e729d0fc37423c7e8df0785e3f3d4643f591e` |
| `speed/best_model.pt` | 6,420,287 | `af2bc4b1736c71cb74cd0368d654da7856a2d057b7328162380adc99a90611be` |
| `rpm/best_model.pt` | 6,409,407 | `55e585494d7feba30f5f60e4765db711b1836ed4074215ce92d2df4f13176012` |

The architecture and point order are recorded in `weights_manifest.json`; the model's dense state keys are unchanged. See [summary.json](https://github.com/sainadh812/triumph-dashboard-reader/blob/main/results/summary.json) for metric definitions, validation sample counts and hardware timing details.
