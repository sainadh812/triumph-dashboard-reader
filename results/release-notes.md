# Release v0.1.0

Three inference checkpoints for dashboard/LCD localization, analog speed and LCD RPM reading. Selected epochs: **localizer 5, speed 0, RPM 14**. Speed epoch 0 retains the unchanged starting weights because fine-tuning did not improve the validation selection score.

This is a prototype. Numeric-reading availability is **not accuracy**. No 98% accuracy, physical speed/RPM accuracy or unseen-recording performance is established. Validation changes are mixed: daylight localization worsened slightly, daylight RPM endpoint error stayed unchanged, and night RPM endpoint error improved. Motion blur, glare, partial framing and the empty RPM bar at engine-off remain failure cases.

The selected weights and original pipeline were tested on two 120-second clips in Colab CPU mode, float32, two threads and batch 4. The new public notebook wrapper has not yet completed a full Colab video run. Prediction-pipeline throughput was 4.17 FPS day and 4.23 FPS night; full processing throughput was 3.30 and 2.74 FPS respectively. Work is conditional: a reader may be skipped when its region is invalid. GPU training does not establish GPU inference throughput for these selected weights.

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

The architecture and point order are recorded in `weights_manifest.json`; the model's dense state keys are unchanged. See [summary.json](https://github.com/sainadh812/triumph-dashboard-reader/blob/main/results/summary.json) for metric definitions, validation sample counts and CPU timing details.
