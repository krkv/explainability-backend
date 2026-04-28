"""Unit tests for compact prompt history construction."""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.core.prompt_history import build_compact_conversation_history
from src.domain.interfaces.llm_provider import AgentRole
from src.usecases.energy.energy_config import EnergyConfig
from src.usecases.energy.energy_usecase import EnergyUseCase
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_usecase import HeartUseCase


def test_build_compact_conversation_history_keeps_only_recent_prior_user_turns():
    conversation = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<p>plain assistant text</p>"},
        {"role": "assistant", "content": "<code>show_one(id=1)</code>", "is_function_call": True},
        {"role": "assistant", "content": "<p>Function result</p>"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<code>show_ids()</code>, <code>count_all()</code>", "is_function_call": True},
        {"role": "user", "content": "u3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "<code>predict(id=4)</code>", "is_function_call": True},
        {"role": "user", "content": "u5"},
        {"role": "assistant", "content": "<code>show_one(id=5)</code>", "is_function_call": True},
        {"role": "assistant", "content": "<p>Displayed patient 5</p>"},
        {"role": "user", "content": "u6"},
        {"role": "assistant", "content": "<code>show_one(id=6)</code>", "is_function_call": True},
        {"role": "user", "content": "u7"},
    ]

    history = json.loads(build_compact_conversation_history(conversation))

    assert history == {
        "conversation_history": [
            {"turn": 1, "user_input": "u2", "function_calls": ["show_ids()", "count_all()"]},
            {"turn": 2, "user_input": "u3", "function_calls": []},
            {"turn": 3, "user_input": "u4", "function_calls": ["predict(id=4)"]},
            {"turn": 4, "user_input": "u5", "function_calls": ["show_one(id=5)"]},
            {"turn": 5, "user_input": "u6", "function_calls": ["show_one(id=6)"]},
        ]
    }


def test_build_compact_conversation_history_excludes_latest_user_message():
    conversation = [
        {"role": "user", "content": "show ids"},
        {"role": "assistant", "content": "<code>show_ids()</code>", "is_function_call": True},
        {"role": "user", "content": "what about the first one"},
    ]

    history = json.loads(build_compact_conversation_history(conversation))

    assert history == {
        "conversation_history": [
            {"turn": 1, "user_input": "show ids", "function_calls": ["show_ids()"]},
        ]
    }


def test_build_compact_conversation_history_handles_messages_without_code_tags():
    conversation = [
        {"role": "user", "content": "count all"},
        {"role": "assistant", "content": "count_all()", "is_function_call": True},
        {"role": "user", "content": "and record 5"},
    ]

    history = json.loads(build_compact_conversation_history(conversation))

    assert history["conversation_history"][0]["function_calls"] == ["count_all()"]


def test_energy_usecase_prompt_uses_compact_history(tmp_path):
    functions_path = tmp_path / "functions.json"
    functions_path.write_text("[]", encoding="utf-8")

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    data_loader.load_dataset.return_value = pd.DataFrame({"feature": [1], "y": [0]})

    usecase = EnergyUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=EnergyConfig(
            dataset_path=Path("unused.csv"),
            functions_json_path=functions_path,
        ),
    )

    prompt = usecase.get_system_prompt(
        [
            {"role": "user", "content": "show ids"},
            {"role": "assistant", "content": "<code>show_ids()</code>", "is_function_call": True},
            {"role": "assistant", "content": "<p>Here are the ids</p>"},
            {"role": "user", "content": "what about the first one"},
        ]
    )

    assert "recent prior conversation turns with invoked function calls" in prompt
    assert '"conversation_history": [' in prompt
    assert '"user_input": "show ids"' in prompt
    assert '"function_calls": [' in prompt
    assert "show_ids()" in prompt
    assert "Here are the ids" not in prompt
    assert "what about the first one" not in prompt
    assert "full history of user's messages" not in prompt


def test_heart_usecase_assistant_prompt_uses_compact_history_but_suggester_keeps_full_history(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text('{"age": {"display_name": "Age", "aliases": []}}', encoding="utf-8")
    functions_path = tmp_path / "functions.json"
    functions_path.write_text("[]", encoding="utf-8")

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    data_loader.load_dataset.return_value = pd.DataFrame({"age": [54], "num": [1]}, index=[10])

    usecase = HeartUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=HeartConfig(
            dataset_path=Path("unused.csv"),
            feature_metadata_path=metadata_path,
            functions_json_path=functions_path,
            shap_cache_path=tmp_path / "shap_cache.pkl",
            cf_cache_path=tmp_path / "cf_cache.pkl",
            global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
        ),
    )

    conversation = [
        {"role": "user", "content": "show patient 10"},
        {"role": "assistant", "content": "<code>show_one(id=10)</code>", "is_function_call": True},
        {"role": "assistant", "content": "<p>Displayed patient 10</p>"},
        {"role": "user", "content": "why this patient"},
    ]

    assistant_prompt = usecase.get_system_prompt(conversation)
    suggester_prompt = usecase.get_generation_config(
        conversation=conversation,
        agent_role=AgentRole.SUGGESTER,
        context={"latest_assistant_response": "Displayed patient 10"},
    ).system_prompt

    assert "recent prior conversation turns with invoked function calls" in assistant_prompt
    assert "show_one(id=10)" in assistant_prompt
    assert "Displayed patient 10" not in assistant_prompt
    assert "why this patient" not in assistant_prompt
    assert "Here is the full conversation history" in suggester_prompt
    assert "why this patient" in suggester_prompt
