# Triumph dashboard reader

Reading the analog speedometer and LCD RPM bar on a Triumph Scrambler 400 XC. Built from a synthetic dial, then improved with reviewed daylight and night video frames.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sainadh812/triumph-dashboard-reader/blob/v0.1.0/notebooks/Test_Your_Video.ipynb)
[**Download selected weights**](https://github.com/sainadh812/triumph-dashboard-reader/releases/tag/v0.1.0) · [Experiment archive](https://github.com/sainadh812/triumph-speedometer-dataset)

| Daylight | Night |
|---|---|
| ![Saved daylight predictions](assets/demo-day.gif) | ![Saved night predictions](assets/demo-night.gif) |

Actual saved predictions. Readings are estimates; glare and empty RPM bars still cause errors. [Inspect still frames](assets/clear-day-night.png).

## Try a video

Open Colab → select GPU (or CPU) → run setup → upload your video → download the labeled MP4 and reports. **No Drive sign-in required.** The full frame stays visible, with needle, ticks, LCD outline and larger RPM text. Save downloads before disconnecting the runtime.

## How it works

![Actual input regions and model outputs](assets/pipeline.png)

| Model | Input | Predicts |
|---|---|---|
| Localizer | Whole frame, 512 × 512 | Dashboard, LCD mask/corners, RPM region |
| Speed | Dashboard, 512 × 512; LCD filled gray | Needle pivot/tip and named ticks |
| RPM | Rectified scale, 256 wide × 384 high | 0–12 scale anchors and bar endpoint |

Three MobileNetV3-Small models; grayscale inputs. Geometry converts landmarks into readings. A reader runs only when its region is usable. Unavailable readings keep their rejection reason in the report. Gear and odometer are not implemented.

## What changed along the way

| Problem | Decision | What we learned |
|---|---|---|
| Clean synthetic images looked easier than real video | Add reviewed real day/night frames | Synthetic success did not establish real-camera accuracy. |
| Needle direction alone missed camera pose | Predict named ticks as well | Geometry needs reliable anchors. |
| Fixed LCD masks covered the needle | Learn dashboard/LCD regions | Keep the speed mask precise; give RPM its own crop. |
| One camera view displaced another | Mix views and lighting; retain old validation | Rotation and mirrors do not create a new physical viewpoint. |
| Augmentation erased labeled detail | Review crops; reject destructive appearance changes | More variation is useful only while labels remain credible. |
| Fine-tuning sometimes got worse | Include starting weights in checkpoint selection | Latest speed training kept **epoch 0**. |

![Rejected augmentation and preserved details](assets/augmentation-guard.png)

Latest dataset: **94 reviewed source frames → 8,864 training inputs** across three tasks. Daylight gets more sampling weight; night remains included. Labels follow geometry, and every source's variants stay in one split. Bulk synthesis uses scripts/OpenCV; early source cleanup used ImageGen. Full recordings and the current training dataset are not bundled. [Earlier experiments and generators](EXPERIMENTS.md).

## Results, including failures

![Selected validation changes](assets/validation-changes.png)

Latest selected epochs: **localizer 5 · speed 0 · RPM 14**. Night RPM endpoint error improved; daylight speed weights stayed unchanged. These are small validation groups from the same recordings, not an independent accuracy benchmark.

![Glare before and after fine-tuning](assets/glare-before-after.png)

Daylight RPM availability rose **51.1% → 82.0%** on matched frames. That means more displayed readings, not 82% accuracy: glare can still produce confident wrong landmarks.

![False RPM on a clearly empty bar](assets/empty-bar-failure.png)

The latest CPU test processed all **7,201 frames** from two two-minute windows: **3.30 FPS day / 2.74 FPS night**, including rendering/encoding. Earlier GPU timings used a different pipeline execution mix; they are reported separately. **30–60 FPS and 98% real-world accuracy remain unmet goals.** [Metrics and provenance](results/summary.json).

<details>
<summary>Timing context: CPU versus earlier GPU runs</summary>

![Execution context for the reported timings](assets/results-context.png)

</details>
