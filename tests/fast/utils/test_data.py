import json

from miles.utils import chat_template_utils
from miles.utils.data import Dataset


def test_dataset_preserves_deep_copy_of_messages_before_student_template(monkeypatch, tmp_path):
    messages = [{"role": "user", "content": {"text": "teacher prompt"}}]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps({"text": messages, "metadata": {"source": "unit-test"}}) + "\n",
        encoding="utf-8",
    )

    def render_student_prompt(prompt, **_kwargs):
        prompt[0]["content"]["text"] = "mutated by student template"
        return "student-rendered-prompt"

    monkeypatch.setattr(chat_template_utils, "apply_chat_template", render_student_prompt)

    dataset = Dataset(
        str(prompt_path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        apply_chat_template=True,
        prompt_messages_key="opd_messages",
    )

    sample = dataset[0]
    assert sample.prompt == "student-rendered-prompt"
    assert sample.metadata["source"] == "unit-test"
    assert sample.metadata["opd_messages"] == messages
    assert sample.metadata["opd_messages"][0]["content"]["text"] == "teacher prompt"
