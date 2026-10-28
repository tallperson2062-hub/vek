# VEK280 YOLOv7 Profiling Results

## 1. Overview

This document summarizes the recorded profiling and performance measurements for YOLOv7 running on the AMD/Xilinx VEK280 (VE2802) board.

The main objective was to evaluate **NPU-only inference**, measure end-to-end (E2E) throughput, identify CPU-side bottlenecks, and assess AI Engine utilization and power consumption.

> **Important:** The figures below come from different experiments and timing scopes. Do not treat them as one directly comparable benchmark set. In particular, isolated NPU inference time is not the same as full-application E2E latency.

## 2. Platform and Model Configuration

- **Board:** AMD/Xilinx VEK280 / VE2802
- **NPU IP:** `VE2802_NPU_IP_O00_A304_M3`
- **NPU architecture:** AI Engine-ML, 304 AIE tiles
- **Model:** YOLOv7
- **Model input:** `640 × 640`
- **Input tensor:** INT8, NHWC4, `(1, 640, 640, 4)`
- **Output tensors:**
  - `(1, 80, 80, 256)`
  - `(1, 40, 40, 256)`
  - `(1, 20, 20, 256)`
- **Detection settings:** confidence threshold `0.25`; NMS IoU threshold `0.45`
- **Runtime mode:** VART NPU-only mode, with CPU subgraphs skipped

The recorded runtime message confirmed that CPU subgraphs were skipped, including the message:

```text
NPU only mode set. Skipping node wrp_network_embd_export_2_cpu_subgraph_call.
```

## 3. End-to-End Performance by Source Resolution

The model input remains fixed at `640 × 640`; the source image resolution changes.

| Source resolution | E2E latency | E2E FPS | NPU inference time | CPU utilization |
|---|---:|---:|---:|---:|
| 320 × 240 | 37.02 ms | 27.01 | 7.18 ms | 72% |
| 640 × 480 | 39.57 ms | 25.27 | 7.14 ms | 72% |
| 1280 × 720 | 56.32 ms | 17.76 | 7.12 ms | 80% |
| 1920 × 1080 | 74.99 ms | 13.33 | 7.12 ms | 95% |

### Observations

- NPU inference remained approximately **7.1 ms** as source resolution increased.
- E2E latency increased with source resolution because image reading and CPU-side processing required more time.
- At 1920 × 1080, the recorded E2E throughput was **13.33 FPS**, despite the NPU execution itself taking only about 7.12 ms.
- These values are clean, unprofiled E2E measurements; instrumented profiling can introduce substantial overhead.

## 4. CPU-Enabled Versus NPU-Only Execution

| Execution mode | Latency | Throughput | Notes |
|---|---:|---:|---|
| VART with CPU subgraphs enabled | 175.44 ms | 5.70 FPS | Earlier baseline |
| VART NPU-only mode | 73.75 ms | 13.56 FPS | CPU subgraphs skipped |
| Isolated NPU execution | Approximately 7.12 ms | Approximately 140 FPS equivalent | Accelerator timing only; not E2E FPS |

Switching from the CPU-subgraph-enabled baseline to NPU-only mode improved the recorded throughput from **5.70 FPS to 13.56 FPS**.

The approximately 140 FPS equivalent is calculated from isolated NPU latency. It must not be presented as full application throughput because it excludes image I/O, preprocessing, output decoding, NMS, and drawing.

## 5. Optimized Multi-Producer Pipeline

The recorded true E2E pipeline comparison was:

| Pipeline configuration | Measured E2E throughput |
|---|---:|
| 1 producer | 54.44 FPS |
| 2 producers | **59.69 FPS** |
| 3 producers | 57.62 FPS |

The **two-producer configuration at 59.69 FPS** was the best trusted baseline in this comparison. The three-producer configuration did not improve throughput further.

### Fused preprocessing experiment

A separate preprocessing experiment reduced the preparation stage from approximately **11.6–11.8 ms** to approximately **3.0 ms**. The recorded test reported byte-identical prepared inputs.

This result belongs to a separate experiment and should not be substituted for the 59.69 FPS two-producer baseline unless the same full E2E test is rerun and confirms the result.

## 6. Detailed 1080p E2E Breakdown

Recorded timings for a `1920 × 1080` source image:

| Pipeline stage | Latency |
|---|---:|
| Image read | 44.31 ms |
| Letterboxing | 7.97 ms |
| Input preparation | 12.22 ms |
| NPU inference | 7.12 ms |
| Output decoding | 2.42 ms |
| Non-maximum suppression (NMS) | 0.24 ms |
| Drawing | 0.69 ms |
| **Total E2E latency** | **74.99 ms** |
| **Total E2E throughput** | **13.33 FPS** |

Stage timings are rounded, so their sum may differ slightly from the reported total.

### Main bottlenecks

1. **Image reading:** 44.31 ms, the largest recorded stage.
2. **Input preparation:** 12.22 ms.
3. **Letterboxing:** 7.97 ms.
4. **NPU inference:** 7.12 ms.

At 1080p, image reading alone accounted for the largest share of the pipeline time. This indicates that optimizing the accelerator alone will not remove the main E2E bottleneck.

## 7. AI Engine Utilization and Power

| Metric | Recorded value |
|---|---:|
| Average AIE utilization | Approximately 52.48% |
| Recorded AIE utilization range | Approximately 52.21–52.77% |
| Average measured board power | 3.55 W |
| Recorded peak power | 3.97 W |
| Energy per frame at approximately 27 FPS | Approximately 0.131 J |
| Energy per frame at approximately 13.3 FPS | Approximately 0.267 J |

Additional notes:

- Another recorded AIE profiling run reported approximately 52.15% average utilization, with a range of 51.68–52.82%.
- The instrumented 1080p profiling run dropped to approximately 7.3 FPS (about 136 ms/frame). This is a profiling-overhead result and must not replace the clean 13.33 FPS E2E measurement.
- Power values depend on the particular test and measurement interval; do not assume every power figure was captured simultaneously with every FPS result.

## 8. Additional Recorded Experiments

| Test or experiment | Recorded result | Interpretation |
|---|---|---|
| Original 100-image YOLOv7 test | 5.74 detections/frame | Detection-count baseline, not FPS |
| Reduced-decode experiment | 4.57 detections/frame | Fewer detections than the original test |
| 100-frame YOLO-family test | 80.03 ms average; approximately 12.50 FPS | Separate workload; not confirmed as the final YOLOv7 NPU-only E2E result |
| Earlier 5000-image YOLOv7 benchmark | 105.69 ms average; 9.46 FPS | Earlier VART benchmark with a different timing scope |
| Separate 100-inference NPU-only benchmark | 42.051 ms average; 23.781 FPS | Different experiment; keep separate from isolated NPU latency |
| Webcam/video tests | Tests were performed | Exact final per-test FPS is not available in the currently retrieved records |

These results should not be combined into a single ranking because the workloads, timing boundaries, and configurations differ.

## 9. Key Conclusions

1. **Best trusted optimized E2E baseline:** 59.69 FPS using the two-producer pipeline.
2. **1080p E2E baseline:** 74.99 ms, or 13.33 FPS.
3. **Isolated NPU execution:** approximately 7.12 ms.
4. **Average AIE utilization:** approximately 52.48%.
5. **Recorded power:** 3.55 W average and 3.97 W peak.
6. The major 1080p bottleneck was CPU-side image handling, especially image reading, rather than NPU inference.
7. The next optimization work should target image I/O, preprocessing, and pipeline overlap, while preserving NPU-only mode and validating changes using the same E2E benchmark.

## 10. Benchmarking and Reporting Notes

- Clearly label every result as **E2E**, **isolated NPU**, or **instrumented profiling**.
- Do not report isolated NPU throughput as application FPS.
- Use the same input set, resolution, warm-up procedure, runtime mode, and timing boundaries when comparing optimizations.
- Record both FPS and latency.
- Track detection quality as well as speed when changing preprocessing or decoding.
- Mark results as measured or estimated; do not mix estimates with measured values.
