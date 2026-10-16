# Temporal video benchmark summary

Primary temporal analysis of the RAVDESS subset. Clip-level emotion labels are not treated as ground truth for each individual frame.

| Metric | EmoAR | ViT |
|---|---:|---:|
| Frames | 1620 | 1620 |
| Class switches | 124 | 116 |
| Change / flickering rate | 0.0772 | 0.0722 |
| Consecutive agreement | 0.9228 | 0.9278 |
| Prediction runs | 138 | 130 |
| Mean run length (frames) | 11.74 | 12.46 |
| Median run length (frames) | 4.00 | 5.00 |
| Maximum run length (frames) | 106 | 111 |
| Mean confidence | 0.6771 | 0.8544 |
| Confidence SD | 0.2023 | 0.1909 |
| Mean |Δ confidence| | 0.0431 | 0.0489 |
| P95 |Δ confidence| | 0.1601 | 0.2470 |
| Mean latency (ms) | 50.49 | 116.82 |
| Median latency (ms) | 49.37 | 115.59 |
| P95 latency (ms) | 59.69 | 132.87 |
| Effective classifier FPS | 19.81 | 8.56 |

Change rate is computed only between consecutive frames belonging to the same video.

Latency represents model-specific preprocessing plus classification; it does not represent end-to-end AR application latency.
