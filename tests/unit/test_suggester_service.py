"""Unit tests for the heart follow-up suggester service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.constants import UseCase
from src.services.assistant.suggester_service import SuggesterService


class TestSuggesterService:
    """Suggester service behavior and fallback coverage."""

    @pytest.fixture
    def service(self):
        return SuggesterService(usecase_registry=Mock())

    @pytest.fixture
    def conversation(self):
        return [{"role": "user", "content": "What should I ask next?"}]

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_none_for_non_heart_usecase(
        self, service, conversation
    ):
        result = await service.generate_follow_ups(
            conversation=conversation,
            usecase=UseCase.ENERGY,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_valid_suggestions(
        self, service, conversation
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

        with patch("src.services.assistant.suggester_service.get_google_gemini_provider", return_value=llm_provider) as mock_get_provider:
            result = await service.generate_follow_ups(
                conversation=conversation,
                usecase=UseCase.HEART,
            )

        mock_get_provider.assert_called_once_with(
            "gemini-3-flash-preview",
            location="global",
        )
        assert result == [
            "Why did the model flag this patient as high risk?",
            "What features mattered most for this patient?",
            "How would the prediction change if blood pressure were lower?",
        ]

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_none_on_malformed_json(
        self, service, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(return_value="not-json")

        with patch("src.services.assistant.suggester_service.get_google_gemini_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                usecase=UseCase.HEART,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_follow_ups_dedupes_and_then_returns_none_if_too_short(
        self, service, conversation
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

        with patch("src.services.assistant.suggester_service.get_google_gemini_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                usecase=UseCase.HEART,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_follow_ups_returns_none_when_provider_raises(
        self, service, conversation
    ):
        llm_provider = Mock()
        llm_provider.generate_response = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("src.services.assistant.suggester_service.get_google_gemini_provider", return_value=llm_provider):
            result = await service.generate_follow_ups(
                conversation=conversation,
                usecase=UseCase.HEART,
            )

        assert result is None
