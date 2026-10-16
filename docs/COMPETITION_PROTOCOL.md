# Competition-grade evaluation protocol

This repository is a reproducible local-analysis prototype.  It must not be
described as competition-grade, nationally award-winning, clinically valid, or
production-ready until the following protocol has been completed on a frozen,
independent test set.

## 1. Data governance

- Use only self-recorded or explicitly licensed video for the default product.
- Store consent, sport, camera position, frame rate, resolution and intended
  athlete/track for every clip.
- Freeze `train`, `validation` and `test` clip IDs before tuning thresholds.
  Never tune using the final test split.
- Keep public research datasets in a separately documented, research-only
  benchmark configuration; do not silently bundle non-commercial data or
  weights into the application.

## 2. Annotation contract

Annotate each test clip independently twice where possible:

| Capability | Required ground truth | Primary metric |
|---|---|---|
| Pose | visible joint coordinates + visibility | OKS/PCK and joint error in calibrated scenes |
| Person tracking | box/keypoint tracks with stable identities | HOTA, DetA, AssA, IDF1, MOTA |
| Court calibration | image-to-court point pairs | held-out reprojection RMSE (pixels) |
| Event rules | start/end frame and event class | event F1 with a declared time tolerance |
| Motion metrics | reference system or manual protocol | MAE, bias and limits of agreement |

Record inter-annotator disagreement before using a metric as a claim.

## 3. Quality gates

The application already records `run-manifest.json`, `artifact-manifest.json`,
frame metrics and rule events.  A release candidate needs all of the following:

1. 100% of test runs have a valid artifact manifest and reproducible input/model
   hashes.
2. No metric is reported in metres or metres/second without a persisted,
   accepted calibration record.
3. Track IDs, occlusions and association failures are evaluated rather than
   treating the largest visible person as the same athlete forever.
4. Rule outputs are labelled as heuristic candidates unless calibrated against
   held-out event annotations.
5. A failure-case appendix covers occlusion, camera movement, motion blur,
   left/right-handed athletes, multiple players and non-standard viewpoints.

## 4. Report template

For each final experiment include the commit hash **and source-tree dirty
state**, operating system, Python and dependency versions, model hashes,
configuration, dataset split IDs, random seed, speed (FPS), metrics with
confidence intervals, and representative success/failure clips. Compare at
least: raw landmarks, temporal smoothing, and the final rule configuration.

## 5. Safety boundary

The local-rules engine reports image-space pose measurements and rule-match
scores.  It does not diagnose injury, determine athletic potential, make
selection decisions, or replace a qualified coach/clinician.
