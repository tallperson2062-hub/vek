# VEK280 YOLOv7 Optimization and Profiling Report

## 1. Objective

The objective of this work was to optimize and profile the YOLOv7 inference pipeline on the AMD/Xilinx Versal VEK280 board and identify the major contributors to end-to-end latency.

The work focused on:

* Improving YOLOv7 inference execution on the VEK280 NPU.
* Separating NPU inference time from CPU-side processing time.
* Profiling individual stages of the complete inference pipeline.
* Evaluating the effect of different input image source resolutions.
* Measuring AIE/NPU activity.
* Examining power consumption.
* Identifying the main bottleneck limiting end-to-end FPS.

The YOLOv7 model itself continues to use a fixed **640 × 640 model input**, regardless of the original source-image resolution.

---

# 2. Initial YOLOv7 Performance

The initial YOLOv7 execution used the standard VART execution path.

The measured inference time was approximately:

* Average latency: **175.4 ms/frame**
* Approximate throughput: **5.7 FPS**

The execution trace showed that the normal execution path included a CPU-side ONNX subgraph:

`wrp_network_embd_export_2_cpu_subgraph_call`

Therefore, the measured latency was not representative of NPU-only execution.

---

# 3. NPU-Only Execution

To determine how much of the latency was caused by the CPU-side execution path, YOLOv7 was executed using the NPU-only mode:

```bash
python3 /usr/bin/vart_ml_runner.py \
    --snapshot "$SNAPSHOT" \
    --npu_only
```

The CPU subgraph was skipped:

`NPU only mode set. Skipping node wrp_network_embd_export_2_cpu_subgraph_call.`

Measured results:

* Average inference latency: **≈73.75 ms**
* Throughput: **≈13.56 FPS**

This reduced the inference latency from approximately **175.4 ms to 73.75 ms**.

The improvement demonstrated that the CPU subgraph was a major contributor to the original latency.

However, 73.75 ms should not be interpreted as pure NPU hardware latency because the NPU-only runner still includes CPU-side runner and input-handling overhead.

---

# 4. Native NPU Input Optimization

The VEK280 NPU uses an optimized native tensor representation for the YOLOv7 input:

```text
(1, 640, 640, 4)
INT8
NHWC4
```

The normal application input is:

```text
(1, 3, 640, 640)
float32
NCHW
```

The optimized profiling path was therefore used to reduce unnecessary conversion and processing overhead before NPU execution.

The YOLOv7 NPU output consists of three feature maps:

```text
[1, 80, 80, 256]   NHWC8
[1, 40, 40, 256]   NHWC8
[1, 20, 20, 256]   NHWC8
```

The model remains fixed at 640 × 640 internally.

---

# 5. Stage-Level Profiling

A custom profiling application was used to divide the complete YOLOv7 pipeline into individual stages:

1. Image read/decode
2. Letterbox resize
3. Input preparation
4. NPU inference
5. Output decode
6. Non-Maximum Suppression (NMS)
7. Drawing
8. Complete frame latency

The profiling was performed using **101 images per resolution**.

The following source-image resolutions were tested:

* 320 × 240
* 640 × 480
* 1280 × 720
* 1920 × 1080

The YOLOv7 model input remained **640 × 640 for every test**.

---

# 6. Resolution Profiling Results

## 6.1 320 × 240

| Stage           | Average Time |
| --------------- | -----------: |
| Read            |      9.52 ms |
| Letterbox       |      4.58 ms |
| Preparation     |     12.78 ms |
| NPU             |      7.18 ms |
| Decode          |      2.39 ms |
| NMS             |      0.24 ms |
| Draw            |      0.31 ms |
| **Frame Total** | **37.02 ms** |

Result:

**27.01 FPS**

CPU busy time was approximately **72%**.

---

## 6.2 640 × 480

| Stage           | Average Time |
| --------------- | -----------: |
| Read            |     16.07 ms |
| Letterbox       |      0.69 ms |
| Preparation     |     12.41 ms |
| NPU             |      7.14 ms |
| Decode          |      2.55 ms |
| NMS             |      0.25 ms |
| Draw            |      0.44 ms |
| **Frame Total** | **39.57 ms** |

Result:

**25.27 FPS**

CPU busy time was approximately **72%**.

---

## 6.3 1280 × 720

| Stage           | Average Time |
| --------------- | -----------: |
| Read            |     30.50 ms |
| Letterbox       |      3.17 ms |
| Preparation     |     12.25 ms |
| NPU             |      7.12 ms |
| Decode          |      2.44 ms |
| NMS             |      0.24 ms |
| Draw            |      0.57 ms |
| **Frame Total** | **56.32 ms** |

Result:

**17.76 FPS**

CPU busy time was approximately **80%**.

---

## 6.4 1920 × 1080

| Stage           | Average Time |
| --------------- | -----------: |
| Read            |     44.31 ms |
| Letterbox       |      7.97 ms |
| Preparation     |     12.22 ms |
| NPU             |      7.12 ms |
| Decode          |      2.42 ms |
| NMS             |      0.24 ms |
| Draw            |      0.69 ms |
| **Frame Total** | **74.99 ms** |

Result:

**13.33 FPS**

CPU busy time was approximately **95%**.

---

# 7. Resolution Comparison

| Source Resolution | E2E Latency | E2E FPS |     Read | Letterbox |     Prep |     NPU | CPU Busy |
| ----------------- | ----------: | ------: | -------: | --------: | -------: | ------: | -------: |
| 320×240           |    37.02 ms |   27.01 |  9.52 ms |   4.58 ms | 12.78 ms | 7.18 ms |      72% |
| 640×480           |    39.57 ms |   25.27 | 16.07 ms |   0.69 ms | 12.41 ms | 7.14 ms |      72% |
| 1280×720          |    56.32 ms |   17.76 | 30.50 ms |   3.17 ms | 12.25 ms | 7.12 ms |      80% |
| 1920×1080         |    74.99 ms |   13.33 | 44.31 ms |   7.97 ms | 12.22 ms | 7.12 ms |      95% |

---

# 8. Main Observation from Resolution Testing

The most important result is that **NPU latency remains almost constant across all source resolutions**.

The measured NPU latency was:

```text
320×240    → 7.18 ms
640×480    → 7.14 ms
1280×720   → 7.12 ms
1920×1080  → 7.12 ms
```

This happens because the YOLOv7 network always receives a **640 × 640 input**.

Therefore, increasing the source image resolution does not increase the neural-network computation performed by the NPU.

Instead, the additional latency is primarily introduced by the CPU-side image-processing pipeline.

The strongest example is the image-read stage:

```text
320×240     →  9.52 ms
640×480     → 16.07 ms
1280×720    → 30.50 ms
1920×1080   → 44.31 ms
```

At 1920 × 1080, image reading alone accounts for approximately **59% of the measured frame latency**.

The CPU also becomes increasingly busy as source resolution increases:

```text
320×240     → 72%
640×480     → 72%
1280×720    → 80%
1920×1080   → 95%
```

This indicates that the end-to-end performance limitation is increasingly dominated by the CPU-side input pipeline rather than the NPU.

---

# 9. NPU/AIE Profiling

AIE profiling was enabled using the XRT profiling configuration.

The VEK280 uses the VE2802 NPU configuration with:

* **304 AIE tiles**
* Arrangement: **38 × 8**
* NPU frequency: **500 MHz**
* Backend: **AIEML_V1C**
* NPU kernel: `npu_aieml_O0`

The profiling configuration enabled AIE core activity and tile-level metrics.

The profile contained measurements for all **304 AIE tiles**.

The latest profile produced the following activity results:

```text
AIE tiles measured: 304

Average core activity: 52.48%
Minimum:                52.21%
Maximum:                52.77%
```

This activity value was calculated from the AIE event counters corresponding to:

* `CORE_ACTIVE`
* `CORE_GROUP_STALL`

The result should therefore be described as **measured AIE core activity**, rather than claiming that only 52% of the NPU is being used.

All 304 tiles were included in the measurement.

---

# 10. Important AIE Profiling Observation

AIE profiling introduces significant measurement overhead.

A clean 1920 × 1080 run produced approximately:

```text
NPU stage:       7.12 ms
Frame latency:  74.99 ms
FPS:             13.33
```

When AIE profiling was enabled, the same type of run showed approximately:

```text
NPU stage:       ~10 ms
Frame latency:  ~136 ms
FPS:             ~7.3
```

Therefore, AIE profiling should **not** be used for performance benchmarking.

It is used to obtain utilization/activity information.

After utilization profiling, the XRT configuration was restored to the clean configuration so that subsequent performance measurements were not affected by profiling overhead.

---

# 11. Power Measurement

The VEK280 board power rails were monitored using the INA226-based power-monitoring interface.

A representative power reading was:

```text
MONITORED RAIL POWER: 3.378 W
```

The monitored rails included:

* VCCINT
* VCC_SOC
* VCC_PMC
* VCC_RAM
* VCC_PSLP_CPM5
* VCC_PSFP
* VCCO_HDIO_3V3
* VCCAUX_PMC
* VCCAUX
* MGTAVCC
* VCC1V5
* VCCO_MIO
* MGTAVTT
* VCCO_502
* MGTVCCAUX
* VCC1V1_LP4
* VADJ_FMC
* LPDMGTYAVCC
* LPDMGTYAVTT
* LPDMGTYVCCAUX

The measured 3.378 W value is a **single monitored-rail reading** and should not be treated as the average active power for every tested resolution.

For a valid power-versus-resolution comparison, power must be sampled continuously while each resolution test is executing and then averaged over the active inference period.

---

# 12. Profiling Parameters Captured

The profiling methodology provides several useful performance parameters:

### Timing parameters

* Image read time
* Letterbox/resize time
* Input preparation time
* NPU inference time
* Output decoding time
* NMS time
* Drawing time
* Total frame latency
* End-to-end FPS

### CPU parameters

* CPU busy percentage
* Peak memory/RSS usage

### NPU/AIE parameters

* AIE tile activity
* Core active cycles
* Core stall activity
* Tile-level metrics
* Number of measured AIE tiles
* NPU execution time

### Power parameters

* Rail voltage
* Rail current
* Individual rail power
* Total monitored-rail power

These parameters allow the system to be analyzed at three levels:

```text
Application
     ↓
CPU preprocessing/postprocessing
     ↓
NPU/AIE execution
     ↓
Hardware power consumption
```

---

# 13. Overall Optimization Progress

The optimization can be summarized as follows:

### Initial implementation

```text
~175.4 ms/frame
~5.7 FPS
```

The standard execution path included CPU-side model execution.

### NPU-only execution

```text
~73.75 ms/frame
~13.56 FPS
```

The CPU ONNX subgraph was skipped.

### Optimized native-input profiling path

For the tested source resolutions:

```text
320×240     → 27.01 FPS
640×480     → 25.27 FPS
1280×720    → 17.76 FPS
1920×1080   → 13.33 FPS
```

The NPU stage remained approximately:

```text
~7.1 ms
```

across all resolutions.

---

# 14. Final Findings

The profiling work establishes several important conclusions.

**1. The NPU is not the main resolution-dependent bottleneck.**

The NPU requires approximately 7.1 ms regardless of whether the original image is 320 × 240 or 1920 × 1080.

**2. Source-image resolution primarily affects CPU-side processing.**

Higher-resolution JPEG images require more time to read/decode and process before they reach the NPU.

**3. Image reading is the dominant bottleneck at high resolution.**

At 1920 × 1080, image reading takes approximately 44.31 ms compared with only 7.12 ms for NPU execution.

**4. CPU utilization increases with source resolution.**

The CPU reaches approximately 95% busy time for the 1920 × 1080 workload.

**5. NPU/AIE activity is approximately 52.48% in the latest measured profile.**

This is an activity measurement across all 304 AIE tiles, not a statement that only 52% of the NPU tiles are active.

**6. AIE profiling affects performance.**

Consequently, utilization profiling and performance benchmarking must be treated as separate measurements.

**7. The optimization direction should now focus on the CPU input pipeline.**

Further NPU-level optimization is unlikely to provide a large improvement in the 1920 × 1080 end-to-end case unless the CPU-side read, decode, resize, and preparation stages are also optimized.

---

# 15. Recommended Next Optimization Target

Based on the measured data, the next optimization target should be:

```text
JPEG read/decode
        ↓
Image preprocessing
        ↓
Input preparation
        ↓
NPU
```

rather than attempting to reduce the already stable ~7.1 ms NPU execution time.

For the 1920 × 1080 case, reducing the **44.31 ms image-read/decode cost** provides substantially more potential benefit than attempting to optimize the ~7.12 ms NPU stage.

The profiling therefore provides a clear separation between **NPU capability** and **system-level end-to-end performance**.

---

## Conclusion

The YOLOv7 implementation on the VEK280 was moved from a CPU-involved execution path to an NPU-focused execution path and subsequently analyzed using detailed stage-level profiling.

The key result is that the VEK280 NPU provides stable inference performance of approximately **7.1 ms per 640 × 640 inference**, while end-to-end performance varies significantly with the original image resolution because of CPU-side processing.

For the highest tested resolution of 1920 × 1080, the system achieved approximately **13.33 FPS**, with **44.31 ms spent reading the source image**, **12.22 ms preparing the input**, and **7.12 ms executing the NPU inference**.

The profiling therefore identifies the **CPU-side input pipeline, particularly image read/decode, as the primary current bottleneck**, while the NPU itself remains comparatively stable across source resolutions.

