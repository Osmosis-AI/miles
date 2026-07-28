# Post-upstream verification

Date: 2026-07-28 UTC

## Source under test

- Branch: `andy/update-miles`
- Latest upstream baseline:
  `423160507a83d350dddffd27fa7a0325fd739f79`
- Code-under-test commit:
  `35a5d0e490480008b01b11adcb8d829b5bcbe6e6`
- FP8 frozen-base feature commit:
  `eb12aa83`
- Consolidated chunked + fused TP log-prob feature commit:
  `39828428`

The branch was rebuilt on the latest fetched upstream baseline and was
`0` commits behind upstream at verification time. Subsequent report and
artifact commits do not change the production code under test.

## Environment

- Namespace/pod: `trainers/andy-miles-dev-node`
- Priority class: `skypilot-high-priority`
- Node: `gpu-dp-xfbc8-9q5bq`
- Allocation: one node, 8 NVIDIA H200 GPUs
- Image:
  `docker.io/radixark/miles@sha256:c6e09b6b20cad09aefe312bc4311bf6d4978b9ff2679d41b2f80fa0e4d19d12a`
- Driver/CUDA: 575.57.08 / CUDA 13.0
- PyTorch/Triton/Transformer Engine:
  2.11.0+cu130 / 3.6.0 / 2.17.0

No second node or additional GPU workload was used.

## Upstream adaptations

The upstream port required four material adjustments:

1. The frozen-base FP8 path was moved onto the current native Qwen3.5
   Megatron-Bridge LoRA flow. Dequantization now requests the model output
   dtype directly, avoiding an unnecessary transient FP32 tensor, and the
   per-layer release lifecycle is validated.
2. The chunked/fused log-prob path was updated for current parallel-state
   APIs, `bshd` activations, direct or uniquely nested output heads, and
   stricter incompatibility checks.
3. The Triton fused kernel was corrected to match BF16 projection semantics
   in both forward and backward instead of comparing FP32-accumulated logits
   against BF16-rounded reference logits.
4. The Bridge LoRA provider now receives `args.attention_backend` before
   `provider.finalize()`. This allows an explicit Transformer Engine
   FlashAttention path instead of silently retaining `auto`.

The verification matrix was also made topology-aware so TP1/DP8 uses a global
batch of 8, while TP2/DP4 uses a global batch of 4.

## H200 results

All four final runs completed rollout, fused log-prob calculation, backward,
an optimizer step, and post-step weight synchronization with exit status 0.
All metrics were finite, and no run contained a traceback or CUDA OOM.

| Run | Topology | Peak GPU | Loss | Grad norm | Log-prob | Step | Actor tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B full parameter | TP1, DP8 | 64,510 MiB | -1.4993e-5 | 6.9692e-4 | 0.0861 s | 15.0097 s | 287.14 |
| Qwen3-4B full parameter | TP2, DP4 | 64,819 MiB | -1.4433e-5 | 1.2638e-3 | 0.2103 s | 15.2136 s | 396.34 |
| Qwen3.5-35B-A3B LoRA + FP8 | TP1, EP8, DP8 | 72,024 MiB | -1.07838e-3 | 2.3580e-3 | 50.8627 s | 182.7126 s | 18.75 |
| Qwen3.5-35B-A3B LoRA + FP8 | TP2, EP8, DP4 | 66,962 MiB | -1.07850e-3 | 7.4195e-3 | 11.5279 s | 53.5863 s | 93.39 |

The LoRA TP1 run uses 8 rollout samples and the TP2 run uses 4 so each global
batch is divisible by its data-parallel size. Their throughput and step-time
figures are therefore execution evidence, not a controlled scaling benchmark.
Warm kernel and model caches also affect these single-step smoke measurements.

### Numerical agreement

| Run | Trainer/rollout abs diff | Trainer/rollout KL |
| --- | ---: | ---: |
| Full parameter TP1 | 0.008920 | 0.000624 |
| Full parameter TP2 | 0.010154 | 0.000720 |
| LoRA + FP8 TP1 | 2.961108 | 2.338918 |
| LoRA + FP8 TP2 | 3.163439 | 2.434124 |

Full-parameter trainer and rollout log-probs remain close. The FP8 trainer
versus BF16 rollout divergence remains material after the upstream port. The
path is operational and trainable, but the divergence is still a quality
tradeoff rather than numerical parity.

### FP8 storage evidence

Both LoRA variants quantized `2,896` frozen tensors.

| Topology | Storage freed | Sampled round-trip relative error |
| --- | ---: | ---: |
| TP1, EP8, DP8 | 5.57 GB/rank | 0.0369 |
| TP2, EP8, DP4 | 4.70 GB/rank | 0.0356 |

## Final targeted tests

The exact code-under-test commit passed `141/141` targeted tests in `27.33 s`:

- 130 LoRA, Bridge, checkpoint, weight-sync, slicing, and Qwen3.5 mapping
  tests;
- 3 FP8 frozen-base H200 tests;
- 5 chunked TP log-prob validation tests;
- 3 fused Triton tests, including two-rank NCCL forward/backward parity.

## Adaptation attempts preserved

The failed intermediate runs are retained because they identify concrete
upstream/runtime compatibility boundaries:

| Attempt | Failure | Resolution |
| --- | --- | --- |
| TE `auto` / fused attention | cuDNN `CUDNN_STATUS_BAD_PARAM` in backward | Select the TE FlashAttention backend |
| Forced FAv2 backward | TE 2.17 called flash-attn 2.7.4 with an incompatible `varlen_bwd` signature | Use the supported full FlashAttention provider path |
| Flash env before Bridge fix | Provider remained `AttnBackend.auto` and rejected the environment | Propagate `args.attention_backend` before `finalize()` |
| Initial TP1 LoRA matrix | Global batch 4 was not divisible by DP8 | Use topology-aware rollout/global batches |

## Artifacts

Successful run artifacts:

- `metrics/post-upstream/full-parameter-tp1/`
- `metrics/post-upstream/full-parameter-tp2/`
- `metrics/post-upstream/lora-fp8-tp1-ep8/`
- `metrics/post-upstream/lora-fp8-tp2-ep8/`
- `metrics/post-upstream/validation-tests/`

Preserved failed attempts:

- `metrics/post-upstream/attempts/`

Each successful training directory contains the canonical launch script,
actual launch command, profile, runtime versions and commit, raw training log,
2-second GPU telemetry, exit status, and parsed `metrics.json`.

## Verdict

Post-upstream execution verification passed for full-parameter and
frozen-base FP8 LoRA training across TP1 and TP2 on one 8xH200 node. The
feature commits remain distinct, the branch is based on the latest fetched
upstream commit, and targeted tests are green. The remaining caveat is the
documented FP8-trainer/BF16-rollout log-prob divergence.
