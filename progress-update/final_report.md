# FP8 and fused log-prob upstream integration: final report

Date: 2026-07-28 UTC

## Outcome

The FP8 frozen-base and chunked/fused TP log-prob work is consolidated into
distinct feature commits, ported onto the latest fetched upstream baseline,
and verified with real full-parameter and LoRA training on one 8xH200 node.

- Upstream baseline:
  `423160507a83d350dddffd27fa7a0325fd739f79`
- Code-under-test:
  `35a5d0e490480008b01b11adcb8d829b5bcbe6e6`
- Branch: `andy/update-miles`
- Code-under-test relation at verification: `0` behind, `6` ahead. The final
  report/artifact commit follows without production-code changes.

## Commit structure

| Commit | Purpose |
| --- | --- |
| `eb12aa83` | FP8 blockwise frozen-base storage for LoRA |
| `39828428` | Consolidated chunked and fused TP log-prob path |
| `ae1a4a4f` | Pre-upstream H200 evidence |
| `f044165b` | Post-upstream H200 verification matrix |
| `94113525` | Bridge LoRA attention-backend propagation |
| `35a5d0e4` | Topology-aware LoRA verification batches |

The FP8 and log-prob implementations are not mixed into one commit.

## Verification summary

### Before the upstream port

| Run | Topology | Result | Peak GPU | Step |
| --- | --- | --- | ---: | ---: |
| Qwen3-4B full parameter | TP2, DP4 | Passed | 64,763 MiB | 17.6877 s |
| Qwen3.5-35B-A3B LoRA + FP8 | TP2, EP8 | Passed | 70,008 MiB | 201.2225 s |

The pre-upstream report is in
`progress-update/pre-upstream/report.md`.

### After the upstream port

All final runs used code-under-test commit `35a5d0e4`.

| Run | Topology | Result | Peak GPU | Loss | Grad norm | Step |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B full parameter | TP1, DP8 | Passed | 64,510 MiB | -1.4993e-5 | 6.9692e-4 | 15.0097 s |
| Qwen3-4B full parameter | TP2, DP4 | Passed | 64,819 MiB | -1.4433e-5 | 1.2638e-3 | 15.2136 s |
| Qwen3.5-35B-A3B LoRA + FP8 | TP1, EP8, DP8 | Passed | 72,024 MiB | -1.07838e-3 | 2.3580e-3 | 182.7126 s |
| Qwen3.5-35B-A3B LoRA + FP8 | TP2, EP8, DP4 | Passed | 66,962 MiB | -1.07850e-3 | 7.4195e-3 | 53.5863 s |

All four runs completed an optimizer step with finite metrics, no traceback,
and no CUDA OOM. The exact post-upstream details are in
`progress-update/post-upstream/report.md`.

## Functional evidence

- FP8 LoRA quantized `2,896` frozen tensors in both TP layouts.
- Reported frozen-base storage reduction was `5.57 GB/rank` at TP1 and
  `4.70 GB/rank` at TP2.
- The chunked output-layer bypass was installed in every real training run.
- The fused Triton selected-logprob path completed forward and backward.
- Standard Bridge LoRA weights loaded into SGLang before and after the
  optimizer step.
- Explicit TE FlashAttention completed the Qwen3.5 backward path.
- `141/141` targeted tests passed, including two-rank NCCL parity.

## Issues found while adapting upstream

1. Current Bridge LoRA did not receive the requested attention backend.
   Propagating `args.attention_backend` before provider finalization fixed the
   real distributed path and is covered by a regression test.
2. TE 2.17 fused-attention backward failed in cuDNN for this shape, while the
   forced FAv2 compatibility path exposed a flash-attn API mismatch. The
   supported TE FlashAttention backend passed direct microtests and both real
   LoRA variants.
3. The fused Triton kernel needed model-dtype rounding in forward and backward
   to match the ordinary BF16 projection path.
4. TP1/DP8 required a global batch divisible by 8; the matrix now carries
   topology-specific rollout and global batch values.

The raw failed attempts remain under
`metrics/post-upstream/attempts/`.

## Numerical caveat

Full-parameter trainer/rollout agreement is close:

- TP1 absolute difference `0.008920`, KL `0.000624`
- TP2 absolute difference `0.010154`, KL `0.000720`

The FP8 trainer versus BF16 rollout gap remains material:

- TP1 absolute difference `2.961108`, KL `2.338918`
- TP2 absolute difference `3.163439`, KL `2.434124`

This does not block execution or optimization, but it should be treated as an
explicit model-quality tradeoff and monitored before broader rollout.

## Artifact map

- Pre-upstream report: `progress-update/pre-upstream/report.md`
- Post-upstream report: `progress-update/post-upstream/report.md`
- Pre-upstream metrics: `metrics/pre-upstream/`
- Post-upstream metrics: `metrics/post-upstream/`
- Kubernetes workload manifest:
  `progress-update/k8s/andy-miles-dev-node.yaml`

The post-upstream training directories contain launch/profile/metrics plus
runtime commit/version evidence, raw logs, and GPU telemetry.

## Final verdict

Ready for upstream review from an execution and integration perspective.
Both feature paths work on the latest fetched upstream code across the
requested full-parameter, LoRA, TP1, TP2, and EP8 variants. The only material
open concern is the recorded FP8 trainer/BF16 rollout log-prob divergence.
