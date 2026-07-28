# Pre-upstream verification

Date: 2026-07-28 UTC

## Source under test

- Branch: `andy/update-miles`
- FP8 frozen-base commit: `81ca261b38475c87bf77d206a80f0b5b4af0e5ea`
- Consolidated chunked + fused TP log-prob commit:
  `df0cc01a2b5464eb2232e04ce546f753fcaa2492`

The log-prob commit includes the Qwen3.5 nested-output-layer compatibility fix
found during this verification and three focused regression tests.

## Environment

- Namespace/pod: `trainers/andy-miles-dev-node`
- Priority class: `skypilot-high-priority`
- Node: `gpu-dp-xfbc8-9q5bq`
- Allocation: one node, 8 NVIDIA H200 GPUs
- Image:
  `docker.io/radixark/miles@sha256:6c1a7d471b5410f1c6f1841d305f401716e3aba7f62bdcd2acad586cfbd05fed`
- Driver/CUDA: 575.57.08 / CUDA 13.0
- PyTorch/Triton/Transformer Engine: 2.11.0+cu130 / 3.6.0 / 2.12.0

An initially selected newer image had an argument-parser mismatch, and the
historical fallback tag no longer existed. All reported runs used the pinned
digest above.

## Results

| Run | Topology | Result | Peak GPU memory | Key training result |
| --- | --- | --- | ---: | --- |
| Qwen3-4B full parameter | TP2, PP1, CP1, EP1, DP4 | Passed, exit 0 | 64,763 MiB | step 0, loss -1.632e-5, grad norm 1.789e-3 |
| Qwen3.5-35B-A3B LoRA + frozen-base FP8 | TP2, PP1, CP1, EP8, DP1 | Passed, exit 0 | 70,008 MiB | step 0, loss -1.078e-3, grad norm 5.346e-4 |

Both runs:

- installed the chunked TP log-prob bypass;
- enabled the fused TP log-prob kernel;
- completed rollout generation, log-prob calculation, backward, optimizer step,
  and post-step weight synchronization;
- produced finite metrics with no traceback or CUDA OOM.

### Full-parameter metrics

- Trainer/rollout log-prob absolute difference: `0.00805684`
- Trainer/rollout KL: `0.000319791`
- Actor train throughput: `187.93 tokens/s`
- Fused log-prob time: `0.1025 s`
- Actor train time: `5.18 s`
- End-to-end measured step time: `17.69 s`

### LoRA + FP8 metrics

- Frozen tensors quantized: `2,896`
- Reported storage freed: approximately `4.70 GB/rank`
- Sampled FP8 round-trip relative error: `0.0356`
- Actor train throughput: `9.99 tokens/s`
- Fused log-prob time: `70.05 s`, including first-use compilation
- Actor train time: `98.73 s`
- End-to-end measured step time: `201.22 s`

The LoRA run has a material numerical caveat: the FP8 trainer versus BF16
rollout log-prob absolute difference was `2.9381`, with KL `2.3287`. The
execution path is operational and trainable, but this divergence should be
treated as an explicit quality tradeoff and rechecked after the upstream port.

## Defect found and fixed

The first Qwen3.5 LoRA attempt failed before quantization because the bridge
model stores its LM head at
`module.module.language_model.output_layer`. The chunked-log-prob installer
only followed `.module` wrappers and expected a top-level `output_layer`.

The consolidated log-prob commit was amended to discover a unique nested
output layer, reject ambiguous nested heads, and retain the direct-head path.
The focused regression suite passed `3/3`, after which both real H200 runs
passed on the amended commit.

## Artifacts

- `metrics/pre-upstream/full-parameter/`
- `metrics/pre-upstream/lora-fp8/`

Each directory contains the launch script and exact launch command, profile,
runtime versions, raw training log, 2-second GPU telemetry, exit status, and
parsed `metrics.json`.

## Verdict

Pre-upstream execution verification passed for full-parameter and LoRA
training on one 8xH200 node. The full-parameter numerical agreement is close.
The FP8 LoRA run is operational but carries the recorded trainer/rollout
log-prob divergence caveat.
