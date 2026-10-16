# Multi-person pose and tracking contract

`backend.multi_person.MultiPersonPoseAnalyzer` is an opt-in, local developer
API for multi-athlete footage. It requests up to a configured number of
MediaPipe poses, converts each valid pose into a pixel-space candidate, and
uses `TrackManager` to give short-term geometric associations stable IDs.

It is intentionally separate from the default single-athlete streaming UI:
the default pipeline remains a reproducible single-person rule-analysis path,
while multi-person score/technique analysis must first prove which athlete is
selected and whether track identity survives the scene.

```python
import cv2
from backend.multi_person import MultiPersonPoseAnalyzer

frame_bgr = cv2.imread("frame.png")
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
analyzer = MultiPersonPoseAnalyzer(num_poses=4)

# First frame: inspect stable candidate/track IDs, but no athlete is selected.
first = analyzer.process_frame(frame_rgb, timestamp_ms=0, frame_index=0)
print(first["tracks"])

# The caller records an explicit choice; it is never inferred from size/order.
selected = analyzer.process_frame(
    frame_rgb,
    timestamp_ms=33,
    frame_index=1,
    primary_track_id=1,
)
print(selected["primary"])
analyzer.close()
```

Each result includes:

- `candidates`: raw visible detector candidates, their application `track_id`,
  pelvis/bounding-box centre and scale evidence;
- `tracks`: visible and short-term occluded track state, missed count,
  association cost and solver used;
- `primary`: only the track explicitly requested by `primary_track_id`, or
  `null` when selection has not been made; and
- `capabilities`: configured pose limit and observed per-frame counts. The
  configured limit is not a guarantee of detections.

## Non-claims and evaluation gate

IDs use position, scale, optional landmark consistency and Hungarian/greedy
assignment. They are not appearance-based re-identification. Crossings, long
occlusions, similar uniforms and camera cuts can therefore require a dedicated
detector/re-ID model and athlete-level annotations.

Before publishing tracking claims, produce TrackEval-compatible annotations and
report HOTA, DetA, AssA, IDF1 and MOTA on a frozen test split. The local unit
tests prove interface behavior (ordering, short occlusion, explicit selection),
not sports-benchmark accuracy. See
[the competition protocol](COMPETITION_PROTOCOL.md) and the
[benchmark starter kit](../benchmarks/README.md).
