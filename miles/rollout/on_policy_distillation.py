import asyncio
import json
import logging
import math
import os
from argparse import Namespace
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any

import aiohttp
import numpy as np
import torch
from tokenizers.decoders import DecodeStream

from miles.rollout.token_aligner import TokenAligner
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_DECODE_KWARGS = {
    "skip_special_tokens": False,
    "clean_up_tokenization_spaces": False,
}

_CROSS_VOCAB_ALIGN_WORKER: dict[str, Any] = {}
_CROSS_VOCAB_ALIGN_POOL: ProcessPoolExecutor | None = None

try:
    # Scoring responses carry O(response_len * top_k) logprob entries; stdlib json
    # decode dominates the reward path at that scale (~5-10x slower than orjson).
    import orjson

    _json_loads = orjson.loads
except ImportError:  # orjson is an sglang dependency, not a miles one
    _json_loads = json.loads

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, str]:
    """Parse ``NAME=URL`` entries from ``--opd-teacher-urls`` into a routing map.

    Splits on the first ``=`` only, so URLs containing ``=`` (e.g. query
    strings) survive intact. Raises on malformed entries and duplicate names
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, str] = {}
    for value in values or []:
        name, sep, url = value.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL.")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        url_map[name] = url
    return url_map


def _teacher_url_for_sample(args: Namespace, sample: Sample) -> str:
    """Resolve the teacher scoring endpoint for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--rm-url`` (the
    original single-teacher path, unchanged). With it, the sample is routed by
    the teacher name in ``sample.metadata[--opd-teacher-key]``; samples whose
    name is missing or unknown fall back to the reserved ``default`` entry,
    and raise if no default is configured — silently distilling from the
    wrong teacher is worse than failing the rollout.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return args.rm_url

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        url = url_map.get(str(name))
        if url is not None:
            return url
        if DEFAULT_TEACHER_NAME in url_map:
            return url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


def _get_opd_top_k(args: Namespace) -> int:
    return max(0, int(getattr(args, "opd_log_prob_top_k", 0) or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    mode = getattr(args, "opd_reward_weight_mode", "student_p")
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _score_payload(
    input_ids: list[int],
    top_k: int = 0,
    token_ids: list[int] | None = None,
    token_ids_positions: list[list[int]] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids_positions is not None:
        # Per-position scoring (patched sglang): one id-list per input position, so the
        # teacher returns each position's own ids (sparse) instead of the global union
        # broadcast to every position (dense O(R^2)). Aligned to logprob_start_len=0.
        payload["token_ids_logprob_positions"] = token_ids_positions
    elif token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _per_position_ids(top_logprobs: TopLogprobs, prompt_len: int) -> list[list[int]]:
    """Build one token-id list per scored input position for ``token_ids_logprob_positions``.

    ``top_logprobs`` is per response position (length == response_length). Prompt
    positions are padded with empty id-lists so the layout aligns with
    ``logprob_start_len=0`` and the existing ``_trim_input_field`` extraction
    (``values[1:][-response_length:]``) — i.e. response position r lands at index
    ``prompt_len + r``.
    """
    per_pos: list[list[int]] = [[] for _ in range(prompt_len)]
    for entries in top_logprobs:
        per_pos.append([_top_entry_token_id(e) for e in (entries or []) if e is not None])
    return per_pos


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _get_teacher_semaphore(args: Namespace) -> asyncio.Semaphore | None:
    concurrency = int(getattr(args, "opd_teacher_concurrency", 0))
    if concurrency <= 0:
        return None

    loop_key = id(asyncio.get_running_loop())
    semaphores = getattr(args, "_opd_teacher_semaphores", None)
    if semaphores is None:
        semaphores = {}
        args._opd_teacher_semaphores = semaphores
    if loop_key not in semaphores:
        semaphores[loop_key] = asyncio.Semaphore(concurrency)
    return semaphores[loop_key]


async def _post_json_once(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with session.post(url, json=payload) as resp:
        if resp.status >= 400:
            body = (await resp.text())[:500]
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=f"{resp.reason}; body={body}",
                headers=resp.headers,
            )
        # Decode from raw bytes: skips aiohttp's charset detection + str copy,
        # and lets orjson (when present) parse the large logprob payloads.
        result = _json_loads(await resp.read())
        if not isinstance(result, dict):
            raise ValueError(f"OPD scoring endpoint returned {type(result).__name__}, expected a JSON object")
        return result


async def _post_json(
    args: Namespace,
    url: str,
    payload: dict[str, Any],
    sample: Sample,
    warning_prefix: str,
) -> dict[str, Any]:
    retries = max(0, int(getattr(args, "opd_teacher_retries", 2)))
    timeout = aiohttp.ClientTimeout(total=float(getattr(args, "opd_teacher_timeout", 300.0)))
    semaphore = _get_teacher_semaphore(args)
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if semaphore is None:
                    return await _post_json_once(session, url, payload)
                async with semaphore:
                    return await _post_json_once(session, url, payload)
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(min(0.2 * (2**attempt), 2.0))

    payload_size = (
        f"input_tokens={len(payload['input_ids'])}"
        if "input_ids" in payload
        else f"text_chars={len(payload.get('text', ''))}"
    )
    reason = type(last_exc).__name__
    logger.warning(
        "%s failed after %s attempt(s); disabling OPD for sample index=%s (%s): %s: %s",
        warning_prefix,
        retries + 1,
        sample.index,
        payload_size,
        reason,
        last_exc,
    )
    if getattr(args, "opd_teacher_strict", False):
        raise RuntimeError(
            f"{warning_prefix} failed after {retries + 1} attempt(s) for sample index={sample.index}"
        ) from last_exc
    return {"_opd_teacher_fallback": True, "_opd_teacher_fallback_reason": reason}


def _is_teacher_fallback(reward: Any) -> bool:
    return isinstance(reward, dict) and bool(reward.get("_opd_teacher_fallback"))


def _sample_reward(args: Namespace, sample: Sample) -> Any:
    if _is_teacher_fallback(sample.reward):
        return sample.reward
    return sample.get_reward_value(args)


def _mark_teacher_fallback(sample: Sample, reason: str) -> None:
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["opd_teacher_fallback"] = True
    sample.metadata["opd_teacher_fallback_reason"] = reason


def _use_student_log_probs_for_teacher(sample: Sample, reason: str) -> None:
    """Make same-tokenizer OPD a strict no-op after a scoring failure."""
    response_length = sample.response_length
    if response_length == 0:
        sample.teacher_log_probs = torch.empty(0, dtype=torch.float32)
        sample.opd_loss_mask = torch.empty(0, dtype=torch.float32)
        _mark_teacher_fallback(sample, reason)
        return

    rollout_values = [] if sample.rollout_log_probs is None else sample.rollout_log_probs
    rollout_log_probs = torch.as_tensor(rollout_values, dtype=torch.float32).flatten()
    if rollout_log_probs.numel() >= response_length:
        rollout_log_probs = rollout_log_probs[-response_length:]
    else:
        rollout_log_probs = torch.cat(
            [torch.zeros(response_length - rollout_log_probs.numel(), dtype=torch.float32), rollout_log_probs]
        )
    sample.teacher_log_probs = rollout_log_probs
    sample.opd_loss_mask = torch.zeros(response_length, dtype=torch.float32)
    _mark_teacher_fallback(sample, reason)


def _use_zero_precomputed_kl(sample: Sample, reason: str) -> None:
    """Make a precomputed OPD path a strict no-op for this response."""
    sample.opd_reverse_kl = torch.zeros(sample.response_length, dtype=torch.float32)
    _mark_teacher_fallback(sample, reason)


def _top_entry_token_id(entry: list[Any]) -> int:
    return int(entry[1])


def _top_entry_logprob(entry: list[Any]) -> float:
    return float(entry[0])


def _top_entries_to_map(entries: Iterable[list[Any]] | None) -> dict[int, float]:
    if not entries:
        return {}
    return {_top_entry_token_id(entry): _top_entry_logprob(entry) for entry in entries if entry is not None}


def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:
    values = meta_info.get(field)
    if values is None:
        raise ValueError(f"Teacher response is missing meta_info.{field}.")
    # SGLang's first input logprob/top-logprob position is a placeholder.
    return values[1:][-response_length:] if response_length > 0 else []


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], response_length: int) -> torch.Tensor:
    input_token_logprobs = _trim_input_field(response["meta_info"], "input_token_logprobs", response_length)
    if len(input_token_logprobs) != response_length:
        raise ValueError(
            f"Teacher returned {len(input_token_logprobs)} response log-probs, expected {response_length}."
        )
    if any(item is None or item[0] is None for item in input_token_logprobs):
        raise ValueError("Teacher returned a missing sampled-token log-probability.")
    return torch.tensor([item[0] for item in input_token_logprobs], dtype=torch.float32)


def _load_student_tokenizer(args: Namespace):
    if not hasattr(args, "_opd_student_tok"):
        from transformers import AutoTokenizer

        args._opd_student_tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return args._opd_student_tok


def _get_opd_mask_token_sequences(args: Namespace) -> list[list[int]]:
    token_strings = getattr(args, "opd_mask_teacher_logprob_tokens", None)
    if not token_strings:
        return []
    if hasattr(args, "_opd_mask_teacher_logprob_token_ids"):
        return args._opd_mask_teacher_logprob_token_ids

    tokenizer = _load_student_tokenizer(args)
    sequences = []
    for token_string in token_strings:
        token_ids = tokenizer.encode(token_string, add_special_tokens=False)
        if token_ids:
            sequences.append(token_ids)
        else:
            logger.warning("Skipping empty OPD teacher-logprob mask token string: %r", token_string)
    args._opd_mask_teacher_logprob_token_ids = sequences
    return sequences


def _matching_token_positions(tokens: list[int], sequences: list[list[int]]) -> set[int]:
    positions = set()
    for sequence in sequences:
        sequence_length = len(sequence)
        for start in range(len(tokens) - sequence_length + 1):
            if tokens[start : start + sequence_length] == sequence:
                positions.update(range(start, start + sequence_length))
    return positions


def _mask_teacher_logprobs_with_student(
    args: Namespace,
    sample: Sample,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    sequences = _get_opd_mask_token_sequences(args)
    if not sequences or sample.response_length == 0 or sample.rollout_log_probs is None:
        return teacher_log_probs

    response_tokens = sample.tokens[-sample.response_length :]
    student_log_probs = list(sample.rollout_log_probs)[-sample.response_length :]
    masked = torch.as_tensor(teacher_log_probs, dtype=torch.float32).clone()
    for position in _matching_token_positions(response_tokens, sequences):
        if position < masked.numel() and position < len(student_log_probs):
            masked[position] = float(student_log_probs[position])
    return masked


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    top_logprobs = sample.metadata.get("opd_student_top_logprobs")
    if top_logprobs is None:
        raise ValueError(
            "Top-k OPD requires student output_top_logprobs. "
            "Ensure --opd-log-prob-top-k is set before rollout generation starts."
        )
    top_logprobs = top_logprobs[-response_length:] if response_length > 0 else []
    if len(top_logprobs) != response_length:
        raise ValueError(
            f"Student top-k logprob length mismatch: got {len(top_logprobs)}, expected {response_length}."
        )
    return top_logprobs


def _unique_ids(top_logprobs: Iterable[Iterable[list[Any]]]) -> list[int]:
    ids = set()
    for entries in top_logprobs:
        for entry in entries or []:
            if entry is not None:
                ids.add(_top_entry_token_id(entry))
    return sorted(ids)


def _ordered_unique(ids: Iterable[int]) -> list[int]:
    seen = set()
    ordered = []
    for token_id in ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token_id)
    return ordered


def _selected_token_ids(strategy: str, student_ids: list[int], teacher_ids: list[int]) -> list[int]:
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)
    if strategy == "only-student":
        return student_ids
    if strategy == "only-teacher":
        return teacher_ids
    if strategy == "intersection":
        return [token_id for token_id in student_ids if token_id in teacher_set]
    if strategy == "union":
        return _ordered_unique([*student_ids, *teacher_ids])
    if strategy == "xor":
        return [
            token_id
            for token_id in [*student_ids, *teacher_ids]
            if (token_id in student_set) != (token_id in teacher_set)
        ]
    raise ValueError(f"Unknown OPD top-k strategy: {strategy}")


def _lookup_logprob(
    token_id: int,
    primary: dict[int, float],
    fallback: dict[int, float] | None,
    *,
    source: str,
) -> float:
    if token_id in primary:
        return primary[token_id]
    if fallback is not None and token_id in fallback:
        return fallback[token_id]
    raise ValueError(f"Missing {source} logprob for token id {token_id}.")


def _reward_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    mode: str,
    *,
    normalize: bool,
) -> list[float]:
    if not student_logps:
        return []
    if mode == "student_p":
        logps = student_logps
    elif mode == "teacher_p":
        logps = teacher_logps
    elif mode == "none":
        logps = [0.0] * len(student_logps)
    else:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")

    if not normalize:
        return [math.exp(logp) for logp in logps]

    max_logp = max(logps)
    exp_vals = [math.exp(logp - max_logp) for logp in logps]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [0.0] * len(logps)
    return [v / denom for v in exp_vals]


def _fast_topk_reverse_kl(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> torch.Tensor | None:
    """Vectorized only-student path over position-aligned arrays.

    With per-position teacher scoring, ``input_token_ids_logprobs[i]`` carries
    the logprobs of exactly the student's top-k ids at position ``i``, in
    request order -- so student and teacher collapse to two aligned ``[N, k]``
    arrays and the weighted reverse-KL needs no per-entry dict building
    (which dominates the reward path at trajectory scale: tens of millions of
    Python-level map inserts per batch).

    Returns ``None`` whenever the inputs don't match that shape (flat-union
    responses, ragged widths, id mismatches, malformed entries); the caller
    then falls back to the generic map-based path, so this is a pure
    optimization with identical semantics.
    """
    response_length = sample.response_length
    student_top = _student_top_logprobs(sample, response_length)
    teacher_rows = _trim_input_field(
        reward_payload["teacher"]["meta_info"], "input_token_ids_logprobs", response_length
    )
    if len(teacher_rows) != response_length:
        return None

    # Blanked positions (e.g. loss-masked observation tokens) carry no ids on
    # either side and contribute zero KL.
    nonempty = [i for i, row in enumerate(student_top) if row]
    if not nonempty:
        return torch.zeros(response_length, dtype=torch.float32)

    width = len(student_top[nonempty[0]])
    s_pairs, t_pairs = [], []
    for i in nonempty:
        s_row = student_top[i]
        t_row = teacher_rows[i] or []
        if len(s_row) != width or len(t_row) != width:
            return None
        # Entries are [logprob, token_id] or [logprob, token_id, text]; keep
        # the numeric pair so the ndarray conversion stays C-speed.
        s_pairs.append([entry[:2] for entry in s_row])
        t_pairs.append([entry[:2] for entry in t_row])

    try:
        s = np.asarray(s_pairs, dtype=np.float64)
        t = np.asarray(t_pairs, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if s.ndim != 3 or s.shape != t.shape or s.shape[2] != 2:
        return None
    # Alignment contract: same ids, same order, at every position. Token ids
    # are exact in float64 (< 2^53).
    if not np.array_equal(s[:, :, 1], t[:, :, 1]):
        return None

    s_logps, t_logps = s[:, :, 0], t[:, :, 0]
    weight_mode = _get_reward_weight_mode(args)
    if weight_mode == "student_p":
        ref = s_logps
    elif weight_mode == "teacher_p":
        ref = t_logps
    else:
        ref = np.zeros_like(s_logps)
    # Mirrors _reward_weights(..., normalize=True): softmax over the k ids.
    shifted = np.exp(ref - ref.max(axis=1, keepdims=True))
    denom = shifted.sum(axis=1, keepdims=True)
    weights = np.divide(shifted, denom, out=np.zeros_like(shifted), where=denom != 0)
    kl_nonempty = (weights * (s_logps - t_logps)).sum(axis=1)

    reverse_kls = np.zeros(response_length, dtype=np.float32)
    reverse_kls[nonempty] = kl_nonempty.astype(np.float32)
    return torch.from_numpy(reverse_kls)


def _compute_topk_reverse_kl(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> torch.Tensor:
    response_length = sample.response_length
    if response_length == 0:
        return torch.zeros((0,), dtype=torch.float32)

    strategy = _get_top_k_strategy(args)
    weight_mode = _get_reward_weight_mode(args)

    if strategy == "only-student" and "student_on_teacher" not in reward_payload:
        fast = _fast_topk_reverse_kl(args, sample, reward_payload)
        if fast is not None:
            return fast

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_response = reward_payload["teacher"]
    teacher_top_maps = (
        _input_logprob_maps(teacher_response, "input_top_logprobs", response_length)
        if strategy in TEACHER_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    teacher_on_student_maps = (
        _input_logprob_maps(teacher_response, "input_token_ids_logprobs", response_length)
        if strategy in TEACHER_ON_STUDENT_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    student_on_teacher_maps = (
        _input_logprob_maps(reward_payload["student_on_teacher"], "input_token_ids_logprobs", response_length)
        if strategy in STUDENT_ON_TEACHER_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    reverse_kls = []
    normalize_weights = strategy != "xor"
    for i in range(response_length):
        student_ids = list(student_top_maps[i].keys())
        teacher_ids = list(teacher_top_maps[i].keys())
        selected_ids = _selected_token_ids(strategy, student_ids, teacher_ids)

        student_logps = []
        teacher_logps = []
        for token_id in selected_ids:
            student_logps.append(
                _lookup_logprob(
                    token_id,
                    student_top_maps[i],
                    student_on_teacher_maps[i],
                    source="student",
                )
            )
            teacher_logps.append(
                _lookup_logprob(
                    token_id,
                    teacher_top_maps[i],
                    teacher_on_student_maps[i],
                    source="teacher",
                )
            )

        weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
        reverse_kl = sum(
            w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
        )
        reverse_kls.append(reverse_kl)

    return torch.tensor(reverse_kls, dtype=torch.float32)


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    if kwargs.get("evaluation"):
        from miles.rollout.rm_hub import default_async_rm

        return await default_async_rm(args, sample, **kwargs)

    top_k = _get_opd_top_k(args)
    # Multi-teacher routing: pick this sample's teacher endpoint (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_url = _teacher_url_for_sample(args, sample)
    if top_k == 0:
        payload = _score_payload(sample.tokens)
        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            payload["image_data"] = [
                encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
            ]
        return await _post_json(args, teacher_url, payload, sample, "OPD teacher request")

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = getattr(args, "opd_topk_per_position", False)
    prompt_len = len(sample.tokens) - sample.response_length

    teacher_top_k = top_k if strategy in TEACHER_TOP_STRATEGIES else 0
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)
    else:
        student_top = None
        teacher_token_ids = None

    if student_top is not None and per_position:
        teacher_payload = _score_payload(
            sample.tokens, top_k=teacher_top_k, token_ids_positions=_per_position_ids(student_top, prompt_len)
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k, token_ids=teacher_token_ids)
    else:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k)
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        teacher_payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]
    teacher_response = await _post_json(args, teacher_url, teacher_payload, sample, "OPD teacher request")
    if _is_teacher_fallback(teacher_response):
        return teacher_response

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        if per_position:
            student_payload = _score_payload(
                sample.tokens, token_ids_positions=_per_position_ids(teacher_top, prompt_len)
            )
        else:
            student_payload = _score_payload(sample.tokens, token_ids=_unique_ids(teacher_top))
        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            student_payload["image_data"] = [
                encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
            ]
        student_response = await _post_json(
            args,
            _student_score_url(args),
            student_payload,
            sample,
            "OPD student scoring request",
        )
        if _is_teacher_fallback(student_response):
            return student_response
        reward_payload["student_on_teacher"] = student_response

    return reward_payload


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Extract OPD signals from teacher responses.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing a precomputed weighted reverse-KL estimate.
    """
    raw_rewards = [_sample_reward(args, sample) for sample in samples]

    if _get_opd_top_k(args) > 0:
        for sample, reward in zip(samples, raw_rewards, strict=True):
            if _is_teacher_fallback(reward):
                _use_zero_precomputed_kl(
                    sample,
                    reward.get("_opd_teacher_fallback_reason", "teacher_failed"),
                )
                continue
            try:
                sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Invalid top-k OPD scoring response for sample index=%s; disabling OPD: %s",
                    sample.index,
                    exc,
                )
                _use_zero_precomputed_kl(sample, "invalid_teacher_response")
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    for sample, reward in zip(samples, raw_rewards, strict=True):
        if _is_teacher_fallback(reward):
            _use_student_log_probs_for_teacher(
                sample,
                reward.get("_opd_teacher_fallback_reason", "teacher_failed"),
            )
            continue
        try:
            teacher_log_probs = _teacher_sampled_log_probs(reward, sample.response_length)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Invalid sampled-token OPD scoring response for sample index=%s; disabling OPD: %s",
                sample.index,
                exc,
            )
            _use_student_log_probs_for_teacher(sample, "invalid_teacher_response")
            continue
        sample.teacher_log_probs = _mask_teacher_logprobs_with_student(args, sample, teacher_log_probs)
        sample.opd_loss_mask = torch.ones(sample.response_length, dtype=torch.float32)

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


def _load_cross_vocab_components(args: Namespace):
    if not hasattr(args, "_cross_vocab_student_tok") or not hasattr(args, "_cross_vocab_teacher_tok"):
        teacher_path = getattr(args, "teacher_tokenizer_path", None)
        if not teacher_path:
            raise ValueError("--teacher-tokenizer-path is required for xToken-aligned cross-tokenizer OPD.")

        from transformers import AutoTokenizer

        args._cross_vocab_student_tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        args._cross_vocab_teacher_tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)

    if not hasattr(args, "_opd_student_tok"):
        args._opd_student_tok = args._cross_vocab_student_tok
    if not hasattr(args, "_cross_vocab_aligner"):
        args._cross_vocab_aligner = TokenAligner(
            args._cross_vocab_student_tok,
            args._cross_vocab_teacher_tok,
        )
    return args._cross_vocab_student_tok, args._cross_vocab_teacher_tok, args._cross_vocab_aligner


def _render_teacher_prompt(args: Namespace, sample: Sample, teacher_tokenizer) -> tuple[str, list[int]]:
    """Render preserved chat messages with the teacher's chat template."""
    messages_key = getattr(args, "opd_prompt_messages_key", None)
    prompt_messages = (sample.metadata or {}).get(messages_key) if messages_key else None
    if prompt_messages is None and isinstance(sample.prompt, list):
        prompt_messages = sample.prompt

    if prompt_messages is None:
        if not isinstance(sample.prompt, str):
            raise TypeError(
                "Cross-tokenizer OPD requires a string prompt or raw chat messages; "
                f"got {type(sample.prompt).__name__}."
            )
        return sample.prompt, teacher_tokenizer.encode(sample.prompt, add_special_tokens=False)

    if isinstance(prompt_messages, str):
        return prompt_messages, teacher_tokenizer.encode(prompt_messages, add_special_tokens=False)
    if not isinstance(prompt_messages, list):
        raise TypeError(
            f"OPD prompt messages under metadata key {messages_key!r} must be a list or string, "
            f"got {type(prompt_messages).__name__}."
        )

    template_kwargs = {"add_generation_prompt": True}
    tools = (sample.metadata or {}).get("tools")
    if tools is not None:
        template_kwargs["tools"] = tools
    try:
        prompt_text = teacher_tokenizer.apply_chat_template(prompt_messages, tokenize=False, **template_kwargs)
        prompt_ids = teacher_tokenizer.apply_chat_template(prompt_messages, tokenize=True, **template_kwargs)
    except TypeError:
        # Some chat templates do not accept tools even when a dataset includes them.
        template_kwargs.pop("tools", None)
        prompt_text = teacher_tokenizer.apply_chat_template(prompt_messages, tokenize=False, **template_kwargs)
        prompt_ids = teacher_tokenizer.apply_chat_template(prompt_messages, tokenize=True, **template_kwargs)
    if isinstance(prompt_ids, Mapping):
        prompt_ids = prompt_ids["input_ids"]
    return prompt_text, list(prompt_ids)


async def reward_func_cross_vocab(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    """Score the student's decoded response using teacher-tokenizer IDs."""
    if kwargs.get("evaluation"):
        from miles.rollout.rm_hub import default_async_rm

        return await default_async_rm(args, sample, **kwargs)

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        raise NotImplementedError("xToken-aligned cross-tokenizer OPD currently supports text-only samples.")

    student_tokenizer, teacher_tokenizer, _ = _load_cross_vocab_components(args)
    _, teacher_prompt_ids = _render_teacher_prompt(args, sample, teacher_tokenizer)
    student_response_ids = sample.tokens[-sample.response_length :] if sample.response_length else []
    response_text = student_tokenizer.decode(
        student_response_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    teacher_response_ids = teacher_tokenizer.encode(response_text, add_special_tokens=False)

    payload = _score_payload(list(teacher_prompt_ids) + list(teacher_response_ids))
    result = await _post_json(
        args,
        args.rm_url,
        payload,
        sample,
        "Cross-tokenizer OPD teacher request",
    )
    if _is_teacher_fallback(result):
        return result

    result["_cross_vocab_meta"] = {
        "teacher_prompt_len": len(teacher_prompt_ids),
        "teacher_response_ids": list(teacher_response_ids),
    }
    return result


def _backend_tokenizer(tokenizer) -> Any | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend
    return getattr(tokenizer, "_tokenizer", None)


def _build_decode_offsets_stream(
    tokenizer,
    token_ids: list[int],
    boundary_indices: set[int],
) -> tuple[str, dict[int, int]] | None:
    """Decode via :class:`DecodeStream`, stepping token-id lists between chunk boundaries."""
    backend_tokenizer = _backend_tokenizer(tokenizer)
    if backend_tokenizer is None:
        return None

    n = len(token_ids)
    bounds = sorted({0, n, *boundary_indices})
    stream = DecodeStream(skip_special_tokens=_DECODE_KWARGS["skip_special_tokens"])
    text_parts: list[str] = []
    text_len = 0
    offsets: dict[int, int] = {0: 0}
    prev = 0
    for bound in bounds[1:]:
        if bound < prev:
            offsets[bound] = text_len
            continue
        segment = token_ids[prev:bound]
        if segment:
            piece = stream.step(backend_tokenizer, [int(token_id) for token_id in segment])
            if piece:
                text_parts.append(piece)
                text_len += len(piece)
        offsets[bound] = text_len
        prev = bound
    offsets.setdefault(n, text_len)
    return "".join(text_parts), offsets


def _span_text_from_offsets(full_text: str, offsets: dict[int, int], positions: list[int]) -> str | None:
    start = positions[0]
    end = positions[-1] + 1
    if positions != list(range(start, end)):
        return None
    try:
        return full_text[offsets[start] : offsets[end]]
    except KeyError:
        return None


def _decode_prefix_text(
    tokenizer,
    token_ids: list[int],
    end: int,
    prefix_cache: dict[int, str],
) -> str:
    cached = prefix_cache.get(end)
    if cached is not None:
        return cached
    text = tokenizer.decode(token_ids[:end], **_DECODE_KWARGS)
    prefix_cache[end] = text
    return text


def _decode_token_span(
    tokenizer,
    token_ids: list[int],
    positions: list[int],
    prefix_cache: dict[int, str] | None = None,
) -> str | None:
    """Decode a contiguous token span with its left context preserved."""
    start = positions[0]
    end = positions[-1] + 1
    if positions != list(range(start, end)):
        return None

    if prefix_cache is None:
        prefix_cache = {}
    prefix_text = _decode_prefix_text(tokenizer, token_ids, start, prefix_cache)
    through_span_text = _decode_prefix_text(tokenizer, token_ids, end, prefix_cache)
    if not through_span_text.startswith(prefix_text):
        return None
    return through_span_text[len(prefix_text) :]


def _chunk_positions_by_id(chunk_ids: torch.Tensor, num_chunks: int) -> list[list[int]]:
    """Map chunk index -> token positions in a single pass over ``chunk_ids``."""
    positions: list[list[int]] = [[] for _ in range(num_chunks)]
    for idx, chunk_id in enumerate(chunk_ids.tolist()):
        if chunk_id >= 0:
            positions[chunk_id].append(idx)
    return positions


def _init_cross_vocab_align_worker(student_path: str, teacher_path: str) -> None:
    from transformers import AutoTokenizer

    student_tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    _CROSS_VOCAB_ALIGN_WORKER["aligner"] = TokenAligner(student_tok, teacher_tok)


def _cross_vocab_align_sample_task(task: dict[str, Any]) -> dict[str, Any]:
    """ProcessPool worker: align one sample and return serializable deltas."""
    aligner = _CROSS_VOCAB_ALIGN_WORKER["aligner"]
    try:
        deltas, aligned_token_count, aligned_chunk_count = _aligned_chunk_logprob_deltas(
            aligner,
            task["student_ids"],
            task["teacher_ids"],
            task["student_log_probs"],
            task["teacher_log_probs"],
            protected_student_positions=set(task["protected_positions"]),
        )
        return {
            "sample_index": task["sample_index"],
            "deltas": deltas.tolist(),
            "aligned_token_count": aligned_token_count,
            "aligned_chunk_count": aligned_chunk_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_index": task["sample_index"],
            "deltas": None,
            "aligned_token_count": 0,
            "aligned_chunk_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cross_vocab_align_workers(args: Namespace, num_tasks: int) -> int:
    configured = int(getattr(args, "opd_cross_vocab_align_workers", 0) or 0)
    if configured > 0:
        return min(configured, num_tasks)
    return min(num_tasks, os.cpu_count() or 1)


def _get_cross_vocab_align_pool(args: Namespace, workers: int) -> ProcessPoolExecutor:
    global _CROSS_VOCAB_ALIGN_POOL
    if _CROSS_VOCAB_ALIGN_POOL is None:
        student_path = args.hf_checkpoint
        teacher_path = getattr(args, "teacher_tokenizer_path", None)
        if not teacher_path:
            raise ValueError("--teacher-tokenizer-path is required for xToken-aligned cross-tokenizer OPD.")
        _CROSS_VOCAB_ALIGN_POOL = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cross_vocab_align_worker,
            initargs=(student_path, teacher_path),
        )
    return _CROSS_VOCAB_ALIGN_POOL


def _aligned_chunk_logprob_deltas(
    aligner: TokenAligner,
    student_ids: list[int],
    teacher_ids: list[int],
    student_log_probs: list[float],
    teacher_log_probs: list[float],
    *,
    protected_student_positions: set[int] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Return equal per-student-token shares of each aligned chunk log-ratio."""
    if len(student_ids) != len(student_log_probs):
        raise ValueError("student token/log-prob length mismatch")
    if len(teacher_ids) != len(teacher_log_probs):
        raise ValueError("teacher token/log-prob length mismatch")

    deltas = torch.zeros(len(student_ids), dtype=torch.float32)
    if not student_ids or not teacher_ids:
        return deltas, 0, 0

    alignment = aligner.align(
        torch.tensor(student_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(teacher_ids, dtype=torch.long).unsqueeze(0),
    )
    student_chunk_ids = alignment.student_chunk_id[0]
    teacher_chunk_ids = alignment.teacher_chunk_id[0]
    protected_student_positions = protected_student_positions or set()
    aligned_token_count = 0
    aligned_chunk_count = 0

    num_chunks = int(alignment.num_chunks[0])
    student_positions_by_chunk = _chunk_positions_by_id(student_chunk_ids, num_chunks)
    teacher_positions_by_chunk = _chunk_positions_by_id(teacher_chunk_ids, num_chunks)

    student_boundaries: set[int] = set()
    teacher_boundaries: set[int] = set()
    for chunk_id in range(num_chunks):
        if not alignment.pair_valid[0, chunk_id] or not alignment.pair_is_correct[0, chunk_id]:
            continue
        student_positions = student_positions_by_chunk[chunk_id]
        teacher_positions = teacher_positions_by_chunk[chunk_id]
        if student_positions:
            student_boundaries.add(student_positions[0])
            student_boundaries.add(student_positions[-1] + 1)
        if teacher_positions:
            teacher_boundaries.add(teacher_positions[0])
            teacher_boundaries.add(teacher_positions[-1] + 1)

    student_stream = _build_decode_offsets_stream(
        aligner.student_tokenizer, student_ids, student_boundaries
    )
    teacher_stream = _build_decode_offsets_stream(
        aligner.teacher_tokenizer, teacher_ids, teacher_boundaries
    )
    use_decode_stream = student_stream is not None and teacher_stream is not None
    if use_decode_stream:
        student_text, student_offsets = student_stream
        teacher_text, teacher_offsets = teacher_stream
    else:
        student_prefix_cache = {0: ""}
        teacher_prefix_cache = {0: ""}

    decode_executor_ctx = nullcontext() if use_decode_stream else ThreadPoolExecutor(max_workers=2)
    with decode_executor_ctx:
        for chunk_id in range(num_chunks):
            if not alignment.pair_valid[0, chunk_id] or not alignment.pair_is_correct[0, chunk_id]:
                continue

            student_positions = student_positions_by_chunk[chunk_id]
            teacher_positions = teacher_positions_by_chunk[chunk_id]
            if not student_positions or not teacher_positions:
                continue
            if protected_student_positions.intersection(student_positions):
                continue

            if use_decode_stream:
                student_chunk_text = _span_text_from_offsets(student_text, student_offsets, student_positions)
                teacher_chunk_text = _span_text_from_offsets(teacher_text, teacher_offsets, teacher_positions)
            else:
                student_future = decode_executor.submit(
                    _decode_token_span,
                    aligner.student_tokenizer,
                    student_ids,
                    student_positions,
                    student_prefix_cache,
                )
                teacher_future = decode_executor.submit(
                    _decode_token_span,
                    aligner.teacher_tokenizer,
                    teacher_ids,
                    teacher_positions,
                    teacher_prefix_cache,
                )
                student_chunk_text = student_future.result()
                teacher_chunk_text = teacher_future.result()
            if student_chunk_text is None or teacher_chunk_text is None or student_chunk_text != teacher_chunk_text:
                continue

            student_chunk_log_prob = sum(float(student_log_probs[position]) for position in student_positions)
            teacher_chunk_log_prob = sum(float(teacher_log_probs[position]) for position in teacher_positions)
            if not math.isfinite(student_chunk_log_prob) or not math.isfinite(teacher_chunk_log_prob):
                continue

            per_token_delta = (student_chunk_log_prob - teacher_chunk_log_prob) / len(student_positions)
            deltas[student_positions] = per_token_delta
            aligned_token_count += len(student_positions)
            aligned_chunk_count += 1

    return deltas, aligned_token_count, aligned_chunk_count


def _extract_teacher_response(reward: dict[str, Any]) -> tuple[list[int], list[float]]:
    metadata = reward["_cross_vocab_meta"]
    teacher_prompt_len = int(metadata["teacher_prompt_len"])
    teacher_response_ids = [int(token_id) for token_id in metadata["teacher_response_ids"]]
    all_logprob_info = reward["meta_info"]["input_token_logprobs"]
    response_info = all_logprob_info[teacher_prompt_len : teacher_prompt_len + len(teacher_response_ids)]
    if len(response_info) != len(teacher_response_ids):
        raise ValueError(
            f"Teacher returned {len(response_info)} response log-probs for {len(teacher_response_ids)} tokens."
        )

    returned_ids = [int(item[1]) for item in response_info]
    if returned_ids != teacher_response_ids:
        raise ValueError("Teacher response token IDs do not match the locally tokenized teacher response.")
    if any(item[0] is None for item in response_info):
        raise ValueError("Teacher returned a missing response log-probability.")
    return teacher_response_ids, [float(item[0]) for item in response_info]


def post_process_rewards_cross_vocab(
    args: Namespace,
    samples: list[Sample],
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Build sampled reverse-KL deltas over xToken-aligned response chunks."""
    has_teacher_response = any(not _is_teacher_fallback(sample.reward) for sample in samples)
    if has_teacher_response:
        _, _, aligner = _load_cross_vocab_components(args)
    else:
        aligner = None

    align_tasks: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        if sample.response_length == 0:
            sample.opd_reverse_kl = torch.empty(0, dtype=torch.float32)
            sample.reward = 0.0
            continue

        reward = _sample_reward(args, sample)
        if _is_teacher_fallback(reward):
            _use_zero_precomputed_kl(
                sample,
                reward.get("_opd_teacher_fallback_reason", "teacher_failed"),
            )
            sample.reward = 0.0
            continue
        if sample.rollout_log_probs is None or len(sample.rollout_log_probs) < sample.response_length:
            _use_zero_precomputed_kl(sample, "missing_student_rollout_logprobs")
            sample.reward = 0.0
            continue

        try:
            teacher_ids, teacher_log_probs = _extract_teacher_response(reward)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Invalid cross-tokenizer teacher response for sample index=%s; disabling OPD: %s",
                sample.index,
                exc,
            )
            _use_zero_precomputed_kl(sample, "invalid_teacher_response")
            sample.reward = 0.0
            continue

        student_ids = sample.tokens[-sample.response_length :]
        student_log_probs = list(sample.rollout_log_probs)[-sample.response_length :]
        protected_positions = _matching_token_positions(
            student_ids,
            _get_opd_mask_token_sequences(args),
        )
        align_tasks.append(
            {
                "sample_index": sample_index,
                "student_ids": student_ids,
                "teacher_ids": teacher_ids,
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "protected_positions": sorted(protected_positions),
            }
        )

    align_results: dict[int, dict[str, Any]] = {}
    if align_tasks:
        workers = _cross_vocab_align_workers(args, len(align_tasks))
        if workers > 1:
            pool = _get_cross_vocab_align_pool(args, workers)
            futures = [pool.submit(_cross_vocab_align_sample_task, task) for task in align_tasks]
            for future in as_completed(futures):
                result = future.result()
                align_results[result["sample_index"]] = result
        else:
            assert aligner is not None
            for task in align_tasks:
                try:
                    deltas, aligned_token_count, aligned_chunk_count = _aligned_chunk_logprob_deltas(
                        aligner,
                        task["student_ids"],
                        task["teacher_ids"],
                        task["student_log_probs"],
                        task["teacher_log_probs"],
                        protected_student_positions=set(task["protected_positions"]),
                    )
                    align_results[task["sample_index"]] = {
                        "sample_index": task["sample_index"],
                        "deltas": deltas.tolist(),
                        "aligned_token_count": aligned_token_count,
                        "aligned_chunk_count": aligned_chunk_count,
                        "error": None,
                    }
                except Exception as exc:
                    align_results[task["sample_index"]] = {
                        "sample_index": task["sample_index"],
                        "deltas": None,
                        "aligned_token_count": 0,
                        "aligned_chunk_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

    overlaps = []
    for task in align_tasks:
        sample = samples[task["sample_index"]]
        result = align_results[task["sample_index"]]
        if result["error"] is not None:
            logger.warning(
                "Cross-vocab alignment failed for sample index=%s; disabling OPD: %s",
                sample.index,
                result["error"],
            )
            _use_zero_precomputed_kl(sample, "invalid_teacher_response")
            sample.reward = 0.0
            continue

        aligned_token_count = int(result["aligned_token_count"])
        aligned_chunk_count = int(result["aligned_chunk_count"])
        if aligned_token_count == 0:
            _use_zero_precomputed_kl(sample, "no_alignment")
            sample.reward = 0.0
            continue

        sample.opd_reverse_kl = torch.tensor(result["deltas"], dtype=torch.float32)
        overlap = aligned_token_count / sample.response_length
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["cross_vocab_token_overlap"] = overlap
        sample.metadata["cross_vocab_aligned_chunks"] = aligned_chunk_count
        sample.reward = 0.0
        overlaps.append(overlap)

    fallback_count = sum(bool((sample.metadata or {}).get("opd_teacher_fallback")) for sample in samples)
    if overlaps:
        logger.info(
            "Cross-tokenizer OPD alignment coverage: mean=%.4f min=%.4f max=%.4f samples=%s fallbacks=%s",
            sum(overlaps) / len(overlaps),
            min(overlaps),
            max(overlaps),
            len(overlaps),
            fallback_count,
        )
    elif samples:
        logger.warning(
            "Cross-tokenizer OPD produced no alignment coverage for %s sample(s); fallbacks=%s",
            len(samples),
            fallback_count,
        )

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
