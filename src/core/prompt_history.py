"""Helpers for compact prompt history construction."""

import html
import json
import re
from typing import Any, Dict, List

_MAX_PRIOR_USER_TURNS = 5
_CODE_TAG_PATTERN = re.compile(r"<code>(.*?)</code>", re.IGNORECASE | re.DOTALL)


def _extract_function_calls(content: str) -> List[str]:
    """Extract clean function-call strings from frontend-rendered assistant content."""
    matches = _CODE_TAG_PATTERN.findall(content)
    if matches:
        return [html.unescape(match.strip()) for match in matches if match.strip()]

    stripped = content.strip()
    return [stripped] if stripped else []


def build_compact_conversation_history(
    conversation: List[Dict[str, Any]],
    max_prior_user_turns: int = _MAX_PRIOR_USER_TURNS,
) -> str:
    """Serialize recent prior user turns and their function calls for prompt context."""
    turns: List[Dict[str, Any]] = []
    current_turn: Dict[str, Any] | None = None

    for message in conversation:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "user_input": str(content or ""),
                "function_calls": [],
            }
            continue

        if (
            role == "assistant"
            and current_turn is not None
            and message.get("is_function_call")
            and isinstance(content, str)
        ):
            current_turn["function_calls"].extend(_extract_function_calls(content))

    if current_turn is not None:
        turns.append(current_turn)

    if conversation and conversation[-1].get("role") == "user" and turns:
        turns = turns[:-1]

    turns = turns[-max_prior_user_turns:]
    serialized_turns = [
        {
            "turn": index + 1,
            "user_input": turn["user_input"],
            "function_calls": turn["function_calls"],
        }
        for index, turn in enumerate(turns)
    ]

    return json.dumps({"conversation_history": serialized_turns}, indent=2)
