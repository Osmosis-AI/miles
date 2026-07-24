import asyncio
import math
import os
from argparse import Namespace
from collections import UserDict
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

import miles.rollout.on_policy_distillation as opd
from miles.rollout.on_policy_distillation import _compute_topk_reverse_kl
from miles.rollout.token_aligner import TokenAligner
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


class FakeTokenizer:
    def __init__(self, tokens=None, encodings=None):
        self.tokens = tokens or {}
        self.encodings = encodings or {}
        self.rendered_messages = None

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[token_id] for token_id in token_ids]

    def decode(self, token_ids, **kwargs):
        return "".join(self.tokens[token_id] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return list(self.encodings.get(text, []))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.rendered_messages = messages
        rendered = " " + "|".join(message["content"] for message in messages) + " "
        return [100, 101] if tokenize else rendered


class MetaspaceTokenizer(FakeTokenizer):
    def decode(self, token_ids, **kwargs):
        text = "".join(self.tokens[token_id] for token_id in token_ids).replace("▁", " ")
        return text.lstrip(" ")


class DictChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return UserDict({"input_ids": rendered, "attention_mask": [1] * len(rendered)}) if tokenize else rendered


def test_teacher_prompt_prefers_preserved_messages():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "solve"}]
    sample = Sample(prompt="student-rendered", metadata={"opd_messages": messages})

    prompt_text, prompt_ids = opd._render_teacher_prompt(
        SimpleNamespace(opd_prompt_messages_key="opd_messages"),
        sample,
        tokenizer,
    )

    assert prompt_text == " solve "
    assert prompt_ids == [100, 101]
    assert tokenizer.rendered_messages == messages


def test_teacher_prompt_extracts_ids_from_batch_encoding():
    tokenizer = DictChatTemplateTokenizer()
    sample = Sample(
        prompt="student-rendered",
        metadata={"opd_messages": [{"role": "user", "content": "solve"}]},
    )

    _, prompt_ids = opd._render_teacher_prompt(
        SimpleNamespace(opd_prompt_messages_key="opd_messages"),
        sample,
        tokenizer,
    )

    assert prompt_ids == [100, 101]


def test_cross_vocab_reward_sends_teacher_owned_token_ids(monkeypatch):
    student_tokenizer = FakeTokenizer(tokens={1: "A", 2: "B"})
    teacher_tokenizer = FakeTokenizer(encodings={"AB": [10]}, tokens={10: "AB"})
    args = SimpleNamespace(
        rm_url="http://teacher/generate",
        opd_prompt_messages_key="opd_messages",
        _cross_vocab_student_tok=student_tokenizer,
        _cross_vocab_teacher_tok=teacher_tokenizer,
    )
    sample = Sample(
        prompt="student-rendered",
        tokens=[99, 1, 2],
        response_length=2,
        metadata={"opd_messages": [{"role": "user", "content": "solve"}]},
    )
    captured = {}

    async def fake_post(_args, _url, payload, _sample, _warning_prefix):
        captured.update(payload)
        return {"meta_info": {"input_token_logprobs": []}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    result = asyncio.run(opd.reward_func_cross_vocab(args, sample))

    assert captured["input_ids"] == [100, 101, 10]
    assert 99 not in captured["input_ids"]
    assert result["_cross_vocab_meta"] == {
        "teacher_prompt_len": 2,
        "teacher_response_ids": [10],
    }


def test_cross_vocab_reward_uses_default_rm_during_evaluation(monkeypatch):
    import miles.rollout.rm_hub as rm_hub

    async def fake_default_rm(_args, _sample, **_kwargs):
        return 1.0

    monkeypatch.setattr(rm_hub, "default_async_rm", fake_default_rm)
    result = asyncio.run(
        opd.reward_func_cross_vocab(
            SimpleNamespace(),
            Sample(response="4"),
            evaluation=True,
        )
    )

    assert result == 1.0


def test_non_json_teacher_response_falls_back(monkeypatch):
    async def fake_post(*_args, **_kwargs):
        raise ValueError("invalid JSON")

    monkeypatch.setattr(opd, "_post_json_once", fake_post)
    args = SimpleNamespace(
        opd_teacher_retries=0,
        opd_teacher_timeout=1,
        opd_teacher_concurrency=0,
    )

    result = asyncio.run(
        opd._post_json(
            args,
            "http://teacher/generate",
            {"input_ids": [1]},
            Sample(index=3),
            "teacher request",
        )
    )

    assert result["_opd_teacher_fallback"] is True
    assert result["_opd_teacher_fallback_reason"] == "ValueError"


def test_strict_teacher_request_raises_after_retries(monkeypatch):
    async def fake_post(*_args, **_kwargs):
        raise ValueError("invalid JSON")

    monkeypatch.setattr(opd, "_post_json_once", fake_post)
    args = SimpleNamespace(
        opd_teacher_retries=0,
        opd_teacher_timeout=1,
        opd_teacher_concurrency=0,
        opd_teacher_strict=True,
    )

    with pytest.raises(RuntimeError, match="failed after 1 attempt"):
        asyncio.run(
            opd._post_json(
                args,
                "http://teacher/generate",
                {"input_ids": [1]},
                Sample(index=3),
                "teacher request",
            )
        )


def test_sampled_token_fallback_builds_strict_zero_mask():
    sample = Sample(
        response_length=2,
        rollout_log_probs=[-0.2, -0.3],
        reward={
            "_opd_teacher_fallback": True,
            "_opd_teacher_fallback_reason": "TimeoutError",
        },
    )

    opd.post_process_rewards(
        SimpleNamespace(reward_key=None, opd_log_prob_top_k=0),
        [sample],
    )

    assert torch.equal(sample.teacher_log_probs, torch.tensor([-0.2, -0.3]))
    assert torch.equal(sample.opd_loss_mask, torch.zeros(2))
    assert sample.metadata["opd_teacher_fallback_reason"] == "TimeoutError"


def test_topk_fallback_builds_zero_precomputed_reverse_kl():
    sample = Sample(
        response_length=2,
        reward={
            "_opd_teacher_fallback": True,
            "_opd_teacher_fallback_reason": "TimeoutError",
        },
    )

    opd.post_process_rewards(
        SimpleNamespace(reward_key=None, opd_log_prob_top_k=16),
        [sample],
    )

    assert torch.equal(sample.opd_reverse_kl, torch.zeros(2))
    assert sample.metadata["opd_teacher_fallback_reason"] == "TimeoutError"


def test_topk_student_scoring_preserves_multimodal_images(monkeypatch):
    args = SimpleNamespace(
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-teacher",
        rm_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=30000,
    )
    sample = Sample(
        tokens=[10, 11],
        response_length=1,
        multimodal_inputs={"images": ["raw-image"]},
    )
    calls = []

    async def fake_post(_args, url, payload, _sample, warning_prefix):
        calls.append((url, payload, warning_prefix))
        if url == args.rm_url:
            return {
                "meta_info": {
                    "input_top_logprobs": [
                        None,
                        [[-0.1, 42]],
                    ]
                }
            }
        return {"meta_info": {"input_token_ids_logprobs": [None, [[-0.2, 42]]]}}

    monkeypatch.setattr(opd, "_post_json", fake_post)
    monkeypatch.setattr(opd, "encode_image_for_rollout_engine", lambda image: f"encoded:{image}")

    result = asyncio.run(opd.reward_func(args, sample))

    assert result["student_on_teacher"]["meta_info"]["input_token_ids_logprobs"]
    assert len(calls) == 2
    assert calls[0][1]["image_data"] == ["encoded:raw-image"]
    assert calls[1][0] == "http://student:30000/generate"
    assert calls[1][1]["image_data"] == ["encoded:raw-image"]


def test_many_student_tokens_share_one_chunk_delta():
    aligner = TokenAligner(
        FakeTokenizer(tokens={1: "A", 2: "B"}),
        FakeTokenizer(tokens={10: "AB"}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1, 2],
        teacher_ids=[10],
        student_log_probs=[-0.2, -0.3],
        teacher_log_probs=[-1.0],
    )

    assert torch.allclose(deltas, torch.tensor([0.25, 0.25]))
    assert deltas.sum().item() == pytest.approx((-0.2 - 0.3) - (-1.0))
    assert aligned_tokens == 2
    assert aligned_chunks == 1


def test_many_teacher_tokens_conserve_chunk_delta():
    aligner = TokenAligner(
        FakeTokenizer(tokens={1: "AB"}),
        FakeTokenizer(tokens={10: "A", 11: "B"}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1],
        teacher_ids=[10, 11],
        student_log_probs=[-0.5],
        teacher_log_probs=[-0.4, -0.6],
    )

    assert deltas.item() == pytest.approx(0.5)
    assert aligned_tokens == 1
    assert aligned_chunks == 1


def test_protected_position_disables_the_whole_chunk():
    aligner = TokenAligner(
        FakeTokenizer(tokens={1: "A", 2: "B"}),
        FakeTokenizer(tokens={10: "AB"}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1, 2],
        teacher_ids=[10],
        student_log_probs=[-0.2, -0.3],
        teacher_log_probs=[-1.0],
        protected_student_positions={1},
    )

    assert torch.equal(deltas, torch.zeros(2))
    assert aligned_tokens == 0
    assert aligned_chunks == 0


def test_mismatched_text_has_zero_delta():
    aligner = TokenAligner(
        FakeTokenizer(tokens={1: "A"}),
        FakeTokenizer(tokens={10: "Z"}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1],
        teacher_ids=[10],
        student_log_probs=[-0.2],
        teacher_log_probs=[-1.0],
    )

    assert torch.equal(deltas, torch.zeros(1))
    assert aligned_tokens == 0
    assert aligned_chunks == 0


def test_canonical_match_still_requires_exact_decoded_chunk_text():
    aligner = TokenAligner(
        FakeTokenizer(tokens={1: "Ġ,"}),
        FakeTokenizer(tokens={10: ","}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1],
        teacher_ids=[10],
        student_log_probs=[-0.2],
        teacher_log_probs=[-1.0],
    )

    assert torch.equal(deltas, torch.zeros(1))
    assert aligned_tokens == 0
    assert aligned_chunks == 0


def test_exact_chunk_check_preserves_left_context_for_metaspace_decoder():
    aligner = TokenAligner(
        MetaspaceTokenizer(tokens={1: "A", 2: "▁,"}),
        FakeTokenizer(tokens={10: "A", 11: ","}),
    )

    deltas, aligned_tokens, aligned_chunks = opd._aligned_chunk_logprob_deltas(
        aligner,
        student_ids=[1, 2],
        teacher_ids=[10, 11],
        student_log_probs=[-0.2, -0.3],
        teacher_log_probs=[-0.2, -1.0],
    )

    assert torch.equal(deltas, torch.zeros(2))
    assert aligned_tokens == 1
    assert aligned_chunks == 1


def _cross_vocab_args(student_tokenizer, teacher_tokenizer):
    return SimpleNamespace(
        reward_key=None,
        opd_mask_teacher_logprob_tokens=None,
        _cross_vocab_student_tok=student_tokenizer,
        _cross_vocab_teacher_tok=teacher_tokenizer,
        _cross_vocab_aligner=TokenAligner(student_tokenizer, teacher_tokenizer),
    )


def test_cross_vocab_post_process_builds_chunk_deltas():
    student_tokenizer = FakeTokenizer(tokens={1: "A", 2: "B"})
    teacher_tokenizer = FakeTokenizer(tokens={10: "AB"})
    args = _cross_vocab_args(student_tokenizer, teacher_tokenizer)
    sample = Sample(
        tokens=[99, 1, 2],
        response_length=2,
        rollout_log_probs=[-0.2, -0.3],
        reward={
            "_cross_vocab_meta": {
                "teacher_prompt_len": 2,
                "teacher_response_ids": [10],
            },
            "meta_info": {
                "input_token_logprobs": [
                    [None, 100],
                    [-0.1, 101],
                    [-1.0, 10],
                ]
            },
        },
    )

    rewards, raw_rewards = opd.post_process_rewards_cross_vocab(args, [sample])

    assert rewards == raw_rewards == [0.0]
    assert torch.allclose(sample.opd_reverse_kl, torch.tensor([0.25, 0.25]))
    assert sample.metadata["cross_vocab_token_overlap"] == 1.0
    assert sample.metadata["cross_vocab_aligned_chunks"] == 1


def test_cross_vocab_fallback_is_strict_zero_without_loading_tokenizers():
    sample = Sample(
        response_length=2,
        rollout_log_probs=[-0.2, -0.3],
        reward={
            "_opd_teacher_fallback": True,
            "_opd_teacher_fallback_reason": "TimeoutError",
        },
    )

    opd.post_process_rewards_cross_vocab(SimpleNamespace(reward_key=None), [sample])

    assert torch.equal(sample.opd_reverse_kl, torch.zeros(2))
    assert sample.metadata["opd_teacher_fallback_reason"] == "TimeoutError"
    assert sample.reward == 0.0


def test_malformed_teacher_response_is_strict_zero():
    student_tokenizer = FakeTokenizer(tokens={1: "A"})
    teacher_tokenizer = FakeTokenizer(tokens={10: "A", 11: "B"})
    args = _cross_vocab_args(student_tokenizer, teacher_tokenizer)
    sample = Sample(
        tokens=[99, 1],
        response_length=1,
        rollout_log_probs=[-0.2],
        reward={
            "_cross_vocab_meta": {
                "teacher_prompt_len": 1,
                "teacher_response_ids": [10],
            },
            "meta_info": {"input_token_logprobs": [[None, 100], [-1.0, 11]]},
        },
    )

    opd.post_process_rewards_cross_vocab(args, [sample])

    assert torch.equal(sample.opd_reverse_kl, torch.zeros(1))
    assert sample.metadata["opd_teacher_fallback_reason"] == "invalid_teacher_response"


def test_zero_cross_vocab_alignment_is_reported_as_fallback(caplog):
    student_tokenizer = FakeTokenizer(tokens={1: "A"})
    teacher_tokenizer = FakeTokenizer(tokens={10: "Z"})
    args = _cross_vocab_args(student_tokenizer, teacher_tokenizer)
    sample = Sample(
        tokens=[99, 1],
        response_length=1,
        rollout_log_probs=[-0.2],
        reward={
            "_cross_vocab_meta": {
                "teacher_prompt_len": 1,
                "teacher_response_ids": [10],
            },
            "meta_info": {"input_token_logprobs": [[None, 100], [-1.0, 10]]},
        },
    )

    with caplog.at_level("WARNING"):
        opd.post_process_rewards_cross_vocab(args, [sample])

    assert torch.equal(sample.opd_reverse_kl, torch.zeros(1))
    assert sample.metadata["opd_teacher_fallback_reason"] == "no_alignment"
    assert "produced no alignment coverage" in caplog.text
    assert "fallbacks=1" in caplog.text


@pytest.mark.skipif(
    not (
        os.getenv("MILES_OPD_TEACHER_RM_URL")
        and os.getenv("MILES_OPD_STUDENT_TOKENIZER_PATH")
        and os.getenv("MILES_OPD_TEACHER_TOKENIZER_PATH")
    ),
    reason="set the MILES_OPD_* tokenizer paths and teacher URL to run the live SGLang check",
)
def test_cross_vocab_reward_and_post_process_with_live_teacher():
    from transformers import AutoTokenizer

    student_path = os.environ["MILES_OPD_STUDENT_TOKENIZER_PATH"]
    teacher_path = os.environ["MILES_OPD_TEACHER_TOKENIZER_PATH"]
    student_tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    messages = [{"role": "user", "content": os.getenv("MILES_OPD_LIVE_PROMPT", "What is 2+2?")}]
    response = os.getenv("MILES_OPD_LIVE_RESPONSE", "4")
    prompt_ids = student_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    response_ids = student_tokenizer.encode(response, add_special_tokens=False)
    sample = Sample(
        prompt=student_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        tokens=prompt_ids + response_ids,
        response=response,
        response_length=len(response_ids),
        rollout_log_probs=[-1.0] * len(response_ids),
        metadata={"opd_messages": messages},
    )
    args = SimpleNamespace(
        hf_checkpoint=student_path,
        teacher_tokenizer_path=teacher_path,
        rm_url=os.environ["MILES_OPD_TEACHER_RM_URL"],
        reward_key=None,
        opd_prompt_messages_key="opd_messages",
        opd_mask_teacher_logprob_tokens=None,
        opd_teacher_timeout=float(os.getenv("MILES_OPD_TEACHER_TIMEOUT", "300")),
        opd_teacher_retries=0,
        opd_teacher_concurrency=0,
    )

    sample.reward = asyncio.run(opd.reward_func_cross_vocab(args, sample))
    assert not sample.reward.get("_opd_teacher_fallback")
    opd.post_process_rewards_cross_vocab(args, [sample])

    assert len(sample.opd_reverse_kl) == sample.response_length
    assert torch.isfinite(sample.opd_reverse_kl).all()
    assert 0.0 <= sample.metadata["cross_vocab_token_overlap"] <= 1.0
