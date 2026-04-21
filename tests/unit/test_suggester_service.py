"""Unit tests for the heart follow-up suggester service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.constants import Model, UseCase
from src.domain.entities.assistant_response import AssistantResponse
from src.services.assistant.follow_up_defaults import HEART_DEFAULT_FOLLOW_UPS
from src.services.assistant.suggester_service import SuggesterService


class TestSuggesterService:
    """Suggester service behavior and fallback coverage."""

    @pytest.fixture
    def service(self):
        return SuggesterService(usecase_registry=Mock())

    @pytest.fixture
    def assistant_response(self):
        return AssistantResponse(
            function_calls=[],
            freeform_response="Patient 2 appears to be high risk.",
            parse="",
        )

    @pytest.fixture
    def conversation(self):
        return [{"role": "user", "content": "What should I ask next?"}]

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_none_for_non_heart_usecase(
        self, service, assistant_response, conversation
    ):
        result = await service.generate_follow_ups(
            conversation=conversation,
            assistant_response=assistant_response,
            usecase=UseCase.ENERGY,
            model=Model.GEMINI_2_0_FLASH,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_valid_suggestions(
        self, service, assistant_response, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(
            return_value=(
                '{"suggestions": ['
                '"Why did the model flag this patient as high risk?", '
                '"What features mattered most for this patient?", '
                '"How would the prediction change if blood pressure were lower?"'
                " ]}"
            )
        )

        with patch("src.services.assistant.suggester_service.get_llm_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                assistant_response=assistant_response,
                usecase=UseCase.HEART,
                model=Model.GEMINI_2_5_FLASH,
            )

        assert result == [
            "Why did the model flag this patient as high risk?",
            "What features mattered most for this patient?",
            "How would the prediction change if blood pressure were lower?",
        ]

    @pytest.mark.asyncio
    async def test_generate_follow_ups_falls_back_on_malformed_json(
        self, service, assistant_response, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(return_value="not-json")

        with patch("src.services.assistant.suggester_service.get_llm_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                assistant_response=assistant_response,
                usecase=UseCase.HEART,
                model=Model.GEMINI_2_5_FLASH,
            )

        assert result == HEART_DEFAULT_FOLLOW_UPS

    @pytest.mark.asyncio
    async def test_generate_follow_ups_dedupes_and_then_falls_back_if_too_short(
        self, service, assistant_response, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(
            return_value=(
                '{"suggestions": ['
                '"What features mattered most?", '
                '"What features mattered most?", '
                '"predict(patient_id=2)"'
                "]}"
            )
        )

        with patch("src.services.assistant.suggester_service.get_llm_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                assistant_response=assistant_response,
                usecase=UseCase.HEART,
                model=Model.GEMINI_2_0_FLASH,
            )

        assert result == HEART_DEFAULT_FOLLOW_UPS

    @pytest.mark.asyncio
    async def test_generate_follow_ups_falls_back_when_provider_raises(
        self, service, assistant_response, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("src.services.assistant.suggester_service.get_llm_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                assistant_response=assistant_response,
                usecase=UseCase.HEART,
                model=Model.GEMINI_2_0_FLASH,
            )

        assert result == HEART_DEFAULT_FOLLOW_UPS
