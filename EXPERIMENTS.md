# Experiment notes

The question throughout: **what must the reader see to estimate speed and RPM reliably?** These experiments used different data, labels and pipelines. Their numbers are local evidence, not one continuous accuracy curve.

## Speed: from needle cues to a localized dial

| Stage | Change and reason | Observed result / limit |
|---|---|---|
| **1 · One dial, 500 images** | Scripted needle rotation, known speed and pivot/tip labels. MobileNetV3-Small predicted speed directly plus auxiliary needle heatmaps. | One ImageGen-reconstructed grayscale source did not cover real glass, lighting and framing, even with augmentation. |
| **2 · Camera roll** | Rotate images and labels together through the full circle, preserving speed. | Synthetic main-validation MAE **46.62 → 2.54 km/h**; upright MAE **2.16 → 2.34**. Roll gains did not improve every group. |
| **3 · Arc + named ticks** | Add visible arc, six semantic ticks and visibility; later a dedicated tick decoder with a separate frozen needle branch. Compute speed from geometry. | Synthetic test: **81/100 within ±2 km/h** using geometry, **28/100** using direct regression. Real startup video: **0/541 readings accepted**. Several things changed together. |
| **4 · Real night + gray LCD, V3** | Build from repaired real frames, mask the LCD and fine-tune the needle. Use predicted needle with **image-registered manual reference ticks**. | Eight-frame diagnostic from the development video: **0/8 → 7/8 within ±2 km/h**. The startup demo's amber ticks are calibration, not learned detections. |
| **5 · Real day + learned ticks, V4** | Night-view mask covered the day-view hub. Add actual day plates, correct the mask and learn ten ticks plus pivot/tip. | Old masked and unmasked runs each accepted **0/600** day frames; the revised pipeline accepted all **600 distinct source frames**. Coverage is not correctness. |
| **6 · Mixed views, V5** | Combine straight/night and tilted/day data, balance sources, add left/straight/right roll. Registration projects only the LCD mask; it does not rotate the reader's frame. | **7,500 images**. Retaining old views mattered. Mirrors and in-plane rotation do not generate a true opposite-side camera view. |
| **7 · Learned LCD mask** | Segment the LCD in the current frame instead of matching a reference for speed masking. Speed weights remain frozen. | A mask can follow framing, but a bad border can still erase the needle. Training localization alone does not train the readers. |
| **8 · Dashboard + two readers** | Learn the head box, LCD mask/corners and separate RPM region. Feed the masked dashboard to speed; predict pivot/tip and named ticks. | Current startup retest exposes landmark and crop failures. Latest daylight tuning retained **speed epoch 0**, rather than a worse fine-tuned checkpoint. |

Archive: [pilot](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/training_pilot) · [rotation evidence](https://github.com/sainadh812/triumph-speedometer-dataset/blob/main/results/rotation_v2_pilot/rotation_before_after.png) · [arc/tick evidence](https://github.com/sainadh812/triumph-speedometer-dataset/blob/main/results/arc_ticks_augmented_v2/synthetic_comparison.png) · [night V3](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/real_adaptation_v3) · [day V4](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/real_adaptation_v4) · [mixed V5](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/mixed_view_v5).

## RPM: from a fixed strip to scale landmarks

| Stage | What the reader received | Why we changed it |
|---|---|---|
| **Reference prototype** | Match static features, align to a fixed reference, crop/enlarge the curved scale. Manual 0, 2, …, 12 anchors convert the continuous fill edge to RPM. | No learned LCD detector yet. Of 20 reviewed frames, nine had tentative readings and eleven abstained. Thin segments, reflections and startup animation were ambiguous. |
| **First RPM CNN** | **2,000 scripted images**; masked scale, aspect-preserving resize/letterbox to 192 wide × 256 high. Predict fill fraction and readability; suppress unrelated LCD text. | Still depended on correct reference alignment and an external LCD-state check. Best synthetic validation MAE: **135.97 RPM against draft targets**, not engine measurements. |
| **Learned LCD segmentation** | Predict the screen mask. Combined experiments still used CPU feature matching for the old RPM crop. | Good screen overlap did not ensure correct scale boundaries. Batching CNN calls could not remove per-frame CPU matching. |
| **Detected LCD → direct warp** | Derive a quadrilateral from the mask and warp into the old RPM reference. | Corners ordered by image axes were not consistent semantic corners. Saved crops clipped the lower scale and admitted the display border. One ungated diagnostic also fed unmasked input to the masked-trained speed model. |
| **Semantic corners + scale anchors** | Learn LCD corners and separate wider RPM-region corners; rectify to **256 wide × 384 high**. Predict 0, 2, …, 12 anchors and bar endpoint. | Preserve the relevant source detail and geometry. Upscaling alone cannot recover missing detail. Current failures show the crop still needs work. |

The learned-LCD dataset had **4,800 train / 614 validation / 212 test** variants. Its test IoU was **0.9179**, but the test variants came from only **12 original frames**, with no LCD-absent scenes. That result cannot establish generalization to arbitrary scenes or speed/RPM accuracy.

Archive: [first RPM reader](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/rpm_training_v1) · [learned LCD](https://github.com/sainadh812/triumph-speedometer-dataset/tree/main/learned_lcd_v1) · [pinned dashboard workflow](https://github.com/sainadh812/triumph-speedometer-dataset/tree/7a3fd991d6c0e85e050e593c76fd895d59d516ed/dashboard_joint_v1).

## Data and training decisions

![Real source frame, reviewed geometry and transformed inputs for the three tasks](assets/dataset.png)

| Finding | Response |
|---|---|
| One cleaned source did not cover real glass and lighting | Use reviewed real daylight/night frames as source plates; synthesize with scripts/OpenCV. |
| Transforming only pixels breaks labels | Transform coordinates, regions and visibility with each geometric augmentation. Keep every source's variants within one split. |
| Daylight remained weaker | Add **22 reviewed daylight training sources**; sample daylight at **70%** for speed/RPM while retaining night. |
| Crop errors change what RPM sees | Add constrained corner jitter and preserve scale context; keep the speed mask separate and precise. |
| Blur/glare augmentation sometimes destroyed supervised detail | Reject destructive appearance variants; do not treat unknown labels as known invisible points. |
| More epochs sometimes made validation worse | Evaluate starting weights too. Select localizer **5**, speed **0**, RPM **14**. |

![Augmentation rejected for destroying detail beside a retained example](assets/augmentation-guard.png)

Current data: **94 reviewed sources → 8,864 inputs across the three tasks**. Source-level splitting prevents sibling augmentations crossing splits; it does not make frames from the same recording an independent recording-level test.

Training runs three tasks sequentially in one workflow, not one end-to-end network. All current branches use grayscale. RGB remains an untested hypothesis. Gear and total odometer reading remain unimplemented.

## What remains unresolved

The current startup retest processed **541 frames**: LCD detected **528**, speed displayed **272**, RPM displayed **54**. **457** frames had invalid RPM-region geometry. Detecting the screen is therefore insufficient to guarantee an RPM input. See the [matched-frame failure in the README](README.md#a-later-version-regressed-on-the-startup-video).

That comparison changes both models and processing, so it does not isolate training as the cause. The startup sweep also represents instrument animation, not road speed or engine-RPM ground truth.

Before promoting another checkpoint, retain a fixed regression suite covering startup, straight/tilted views, day/night, empty bars and glare. Inspect final crops and readings, score each scenario separately, and then test on independently recorded sessions. Old recordings remain useful regression checks, not unseen tests.

[Current day/night results](results/summary.json) · [Startup retest](results/startup-regression.json) · [Image provenance](assets/research-provenance.json)
