"""CPU coverage for xToken-style span alignment."""

import torch
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.token_aligner import TokenAligner, canonical_token

register_cpu_ci(est_time=60, suite="stage-a-cpu")


class TokenMap:
    def __init__(self, tokens):
        self.tokens = tokens

    def convert_ids_to_tokens(self, token_ids):
        return [self.tokens[token_id] for token_id in token_ids]


def _align(student_tokens, teacher_tokens, student_ids=None, teacher_ids=None, **kwargs):
    student_ids = student_ids or list(range(len(student_tokens)))
    teacher_ids = teacher_ids or list(range(len(teacher_tokens)))
    aligner = TokenAligner(TokenMap(student_tokens), TokenMap(teacher_tokens))
    return aligner.align(
        torch.tensor([student_ids], dtype=torch.long),
        torch.tensor([teacher_ids], dtype=torch.long),
        **kwargs,
    )


def test_one_to_one_tokens_form_correct_chunks():
    alignment = _align(["A", "B"], ["A", "B"])

    assert alignment.pair_is_correct[0, :2].tolist() == [True, True]
    assert alignment.student_chunk_id[0].tolist() == [0, 1]
    assert alignment.teacher_chunk_id[0].tolist() == [0, 1]
    assert alignment.student_exact_partition_mask[0].tolist() == [True, True]


def test_one_student_token_aligns_to_multiple_teacher_tokens():
    alignment = _align(["AB"], ["A", "B"])

    student_chunk = int(alignment.student_chunk_id[0, 0])
    assert student_chunk >= 0
    assert alignment.pair_is_correct[0, student_chunk]
    assert alignment.teacher_chunk_id[0].tolist() == [student_chunk, student_chunk]
    assert not alignment.student_exact_partition_mask[0, 0]


def test_multiple_student_tokens_align_to_one_teacher_token():
    alignment = _align(["A", "B"], ["AB"])

    teacher_chunk = int(alignment.teacher_chunk_id[0, 0])
    assert teacher_chunk >= 0
    assert alignment.pair_is_correct[0, teacher_chunk]
    assert alignment.student_chunk_id[0].tolist() == [teacher_chunk, teacher_chunk]
    assert not alignment.teacher_exact_partition_mask[0, 0]


def test_mismatched_text_is_not_marked_correct():
    alignment = _align(["A"], ["Z"])

    assert not alignment.pair_is_correct.any()
    assert not alignment.student_exact_partition_mask.any()
    assert not alignment.teacher_exact_partition_mask.any()


def test_padding_masks_clear_chunk_membership():
    alignment = _align(
        ["A", "<pad>", "<pad>"],
        ["A", "<pad>", "<pad>"],
        student_attention_mask=torch.tensor([[1, 0, 0]]),
        teacher_attention_mask=torch.tensor([[1, 0, 0]]),
    )

    assert alignment.student_chunk_id[0, 1:].tolist() == [-1, -1]
    assert alignment.teacher_chunk_id[0, 1:].tolist() == [-1, -1]
    assert not alignment.student_exact_partition_mask[0, 1:].any()
    assert not alignment.teacher_exact_partition_mask[0, 1:].any()


def test_multitoken_unicode_artifact_maps_back_to_original_positions():
    alignment = _align(["ä¸", "Ń"], ["中"])

    teacher_chunk = int(alignment.teacher_chunk_id[0, 0])
    assert alignment.pair_is_correct[0, teacher_chunk]
    assert alignment.student_chunk_id[0].tolist() == [teacher_chunk, teacher_chunk]
    assert canonical_token("<0x41>") == "A"
