# Benchmark annotation starter kit

This directory contains schemas and examples only; it deliberately contains no
competition footage, athlete identity data, research-only dataset samples or
model weights. Keep the real annotations in access-controlled storage and
record their immutable revision alongside each experiment.

## 1. Freeze the split first

Copy `templates/split-manifest.example.json`, replace the anonymous clip IDs,
and validate it before tuning rules or models:

```bash
python -m backend.evaluation validate-splits \
  --manifest benchmarks/split-manifest.json \
  --output benchmark-results/split-integrity.json
```

The validator rejects repeated clips within a split and train/validation/test
overlap. It cannot infer athlete, match or camera leakage: group clips by those
factors yourself and document the grouping policy in the experiment report.

## 2. Event annotations

`templates/events-ground-truth.example.json` uses a point frame for each
event. `start_frame`/`end_frame` may also be recorded for audit, but current
F1 matches one point (`frame`, or `start_frame` when no peak frame exists) per
event. Freeze the frame tolerance before evaluating the held-out test split:

```bash
python -m backend.evaluation events \
  --predicted outputs/<job-id>/events.json \
  --ground-truth benchmarks/events-test.json \
  --tolerance-frames 3 \
  --output benchmark-results/events-test.json
```

The local-rules engine emits candidates and heuristic scores, not calibrated
event probabilities. F1 is evidence only after the annotation protocol,
classes and tolerance are declared in advance.

## 3. Pose annotations

For each ground-truth frame, provide a fixed `normalization_scale_px` and only
mark genuinely annotated joints visible. A missing prediction is counted as a
failure, so coverage cannot silently inflate PCK:

```bash
python -m backend.evaluation pose-pck \
  --predicted benchmarks/pose-predictions-test.json \
  --ground-truth benchmarks/pose-ground-truth-test.json \
  --threshold 0.05 --threshold 0.10 \
  --output benchmark-results/pose-test.json
```

Use the same scale definition throughout a split (for example, an independently
annotated bounding-box diagonal). This is 2-D image-space PCK, not a 3-D or
clinical measurement.

## 4. Calibration and artifact evidence

Fit a court homography on one set of correspondences, then put only unused
points into the paired projected/held-out JSON files and evaluate their pixel
error. Verify every generated job before publishing a result:

```bash
python -m backend.evaluation calibration \
  --projected benchmarks/calibration-projected-test.json \
  --held-out benchmarks/calibration-held-out-test.json

python -m backend.evaluation verify-artifacts --output-dir outputs/<job-id>
```

For tracking HOTA, DetA, AssA, IDF1 and MOTA, format the data for the official
[TrackEval](https://github.com/JonathonLuiten/TrackEval) pipeline. The local
`TrackManager` is an auditable association primitive, not an official benchmark
implementation.

See [`docs/COMPETITION_PROTOCOL.md`](../docs/COMPETITION_PROTOCOL.md) for the
full evaluation gates and reporting requirements.

## 5. Immutable experiment evidence bundle

Once every individual receipt has been generated, copy
`templates/experiment-manifest.example.json`, fill in the run ID and the
metric reports, then assemble one reviewable report. The command verifies the
split, every artifact hash, the run ID binding, the checked-in runtime/code
provenance and each declared metric type; it does not manufacture an accuracy
claim.

```bash
python -m backend.evaluation experiment-report \
  --manifest benchmarks/experiment-manifest.json \
  --output benchmark-results/racket-rules-baseline-001.json
```

Archive the manifest, returned report and original receipts together. A
non-zero exit code means an evidence link failed and the result should not be
used in a submission table.
