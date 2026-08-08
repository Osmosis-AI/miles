#!/usr/bin/env python3
"""Build the RuneBench training JSONL (single-prompt POC: one skill task).

The agent's real prompt is the task dir's instruction.md — Harbor passes it to
the agent; miles never sends the JSONL prompt to the model on this path. The
JSONL prompt mirrors instruction.md anyway so dataset-side bookkeeping
(lengths, dumps) reflects what the agent actually saw.

Usage:
    python make_prompt_data.py \
        --task-dir /path/to/RuneBench/tasks/fishing-xp-15m \
        --output /data/runescape/prompts/fishing.jsonl \
        [--agent-name rune_mini_swe_agent:RuneMiniSweAgent]

The task dir must come from a RUNEBENCH_CLI_ONLY=1 generation (bash-only
instructions, mean-rate reward) and must be present in the agent server's
HARBOR_TASKS_DIR under the same name.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--agent-name",
        default="rune_mini_swe_agent:RuneMiniSweAgent",
        help="Harbor agent import path (module:Class) or registered name",
    )
    args = ap.parse_args()

    instruction = (args.task_dir / "instruction.md").read_text()
    instance_id = args.task_dir.name

    record = {
        # Message-list form: the checkpoint is a VL class so miles loads a
        # processor, and Dataset then requires list prompts (str is rejected).
        # The agent never sees this prompt on the Harbor path — instruction.md
        # in the task dir is the real prompt; this mirrors it for bookkeeping.
        "prompt": [{"role": "user", "content": instruction}],
        "metadata": {
            "instance_id": instance_id,
            "agent_name": args.agent_name,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps(record) + "\n")
    print(f"wrote 1 record ({instance_id}, agent={args.agent_name}) -> {args.output}")


if __name__ == "__main__":
    main()
