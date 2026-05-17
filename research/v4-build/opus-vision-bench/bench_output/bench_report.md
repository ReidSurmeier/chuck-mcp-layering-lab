# Opus 4.7 vision cell-ID benchmark — report

_generated: 2026-05-17T02:52:49.972992+00:00_
_dry-run: False_

## Headline

- images scored: 9 / 10
- images skipped: 1
- **overall mean Jaccard: 0.027**
- overall median Jaccard: 0.000
- overall min Jaccard: 0.000
- overall mean F1: 0.043
- cost spent: $5.5076 over 9 Opus calls
- latency (ms): median=52372  min=45862  max=78104

## Routing decision

**Global route: `MEDIAPIPE`** (threshold 0.95)

> MEDIAPIPE — overall mean Jaccard 0.027 < global threshold 0.95. Opus is not yet trusted to write cell IDs. Ship MediaPipe; revisit after a model bump or after Opus prompt iteration.

Per-region routes (floor 0.85):

- `background        ` -> `mediapipe`  mean Jaccard 0.124  (n=9)
- `chin              ` -> `mediapipe`  mean Jaccard 0.000  (n=7)
- `face              ` -> `mediapipe`  mean Jaccard 0.082  (n=9)
- `forehead          ` -> `mediapipe`  mean Jaccard 0.052  (n=9)
- `hair              ` -> `mediapipe`  mean Jaccard 0.079  (n=9)
- `left_cheek        ` -> `mediapipe`  mean Jaccard 0.030  (n=8)
- `left_eye          ` -> `mediapipe`  mean Jaccard 0.000  (n=5)
- `left_eyebrow      ` -> `mediapipe`  mean Jaccard 0.000  (n=5)
- `left_jaw          ` -> `mediapipe`  mean Jaccard 0.000  (n=6)
- `left_temple       ` -> `mediapipe`  mean Jaccard 0.000  (n=6)
- `lips              ` -> `mediapipe`  mean Jaccard 0.000  (n=8)
- `lower_lip         ` -> `mediapipe`  mean Jaccard 0.000  (n=5)
- `nose              ` -> `mediapipe`  mean Jaccard 0.027  (n=9)
- `right_cheek       ` -> `mediapipe`  mean Jaccard 0.007  (n=8)
- `right_eye         ` -> `mediapipe`  mean Jaccard 0.000  (n=5)
- `right_eyebrow     ` -> `mediapipe`  mean Jaccard 0.000  (n=5)
- `right_jaw         ` -> `mediapipe`  mean Jaccard 0.000  (n=6)
- `right_temple      ` -> `mediapipe`  mean Jaccard 0.024  (n=6)
- `upper_lip         ` -> `mediapipe`  mean Jaccard 0.000  (n=3)

## Per-image headline

| image_id | mean Jaccard | min | F1 | cost USD | duration ms |
|---|---|---|---|---|---|
| close_emma_2002 | 0.041 | 0.000 | 0.069 | 0.7113 | 52372 |
| reid_mike_portrait | 0.000 | 0.000 | 0.000 | 0.5596 | 49433 |
| reid_untitled_01 | 0.106 | 0.000 | 0.166 | 0.7721 | 78104 |
| reid_untitled_02 | 0.000 | 0.000 | 0.000 | 0.5661 | 45862 |
| toy_print_face_masks | 0.000 | 0.000 | 0.000 | 0.5748 | 50509 |
| synth_face_01 | 0.000 | 0.000 | 0.000 | 0.5615 | 48154 |
| synth_face_02 | 0.053 | 0.000 | 0.084 | 0.5482 | 54809 |
| synth_face_03 | 0.000 | 0.000 | 0.000 | 0.6029 | 52954 |
| synth_face_04 | 0.040 | 0.000 | 0.066 | 0.6111 | 52633 |

### Skipped
- synth_face_00: claude -p failed after 83.2s: ClaudePInvocationError: claude -p exited 1; stderr=''

## Per-region detail

| region | mean Jaccard | median | min | max | n |
|---|---|---|---|---|---|
| background | 0.124 | 0.000 | 0.000 | 0.461 | 9 |
| chin | 0.000 | 0.000 | 0.000 | 0.000 | 7 |
| face | 0.082 | 0.000 | 0.000 | 0.300 | 9 |
| forehead | 0.052 | 0.000 | 0.000 | 0.256 | 9 |
| hair | 0.079 | 0.000 | 0.000 | 0.341 | 9 |
| left_cheek | 0.030 | 0.000 | 0.000 | 0.111 | 8 |
| left_eye | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| left_eyebrow | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| left_jaw | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| left_temple | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| lips | 0.000 | 0.000 | 0.000 | 0.000 | 8 |
| lower_lip | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| nose | 0.027 | 0.000 | 0.000 | 0.107 | 9 |
| right_cheek | 0.007 | 0.000 | 0.000 | 0.056 | 8 |
| right_eye | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| right_eyebrow | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| right_jaw | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| right_temple | 0.024 | 0.000 | 0.000 | 0.143 | 6 |
| upper_lip | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
