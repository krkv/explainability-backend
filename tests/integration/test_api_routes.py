"""Integration tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import ANY, Mock, AsyncMock, patch
from src.main import app
from src.core.constants import UseCase, Model
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_assistant_service():
    """Create a mock assistant service."""
    from src.domain.entities.assistant_response import AssistantResponse
    
    service = Mock()
    service.process_message = AsyncMock(
        return_value=AssistantResponse(
            function_calls=[],
            freeform_response="Test response",
            parse=""
        )
    )
    return service


@pytest.fixture
def mock_suggester_service():
    """Create a mock suggester service."""
    service = Mock()
    service.generate_follow_ups = AsyncMock(return_value=None)
    return service


class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def test_ready_endpoint(self, client):
        """Test the /ready endpoint returns OK."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"


class TestAssistantResponseEndpoint:
    """Test cases for assistant response endpoint."""
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_success(
        self, mock_get_assistant_service, client, mock_assistant_service
    ):
        """Test successful assistant response generation."""
        mock_get_assistant_service.return_value = mock_assistant_service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "What is the dataset size?"}
            ],
            "model": "gemini-3.1-flash-lite-preview",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "assistantResponse" in data
        assert data["assistantResponse"]["freeform_response"] == "Test response"
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_with_function_calls(
        self, mock_get_assistant_service, client
    ):
        """Test assistant response with function calls."""
        from src.domain.entities.assistant_response import AssistantResponse
        
        service = Mock()
        service.process_message = AsyncMock(
            return_value=AssistantResponse(
                function_calls=["count_all()"],
                freeform_response="Here are the results",
                parse="Total: 100"
            )
        )
        mock_get_assistant_service.return_value = service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Count all records"}
            ],
            "model": "gemini-3.1-pro-preview",
            "usecase": "Heart Disease"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["assistantResponse"]["function_calls"]) == 1
        assert data["assistantResponse"]["parse"] == "Total: 100"
    
    def test_get_assistant_response_empty_conversation(self, client):
        """Test assistant response with empty conversation."""
        request_data = {
            "conversation": [],
            "model": "gemini-3.1-flash-lite-preview",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"].lower()
    
    def test_get_assistant_response_invalid_model(self, client):
        """Test assistant response with invalid model."""
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "InvalidModel",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 400
        assert "model" in response.json()["detail"].lower()
    
    def test_get_assistant_response_invalid_usecase(self, client):
        """Test assistant response with invalid usecase."""
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "gemini-3.1-flash-lite-preview",
            "usecase": "InvalidUsecase"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 400
        assert "usecase" in response.json()["detail"].lower()
    
    def test_get_assistant_response_missing_fields(self, client):
        """Test assistant response with missing required fields."""
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ]
            # Missing model and usecase
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_service_error(
        self, mock_get_assistant_service, client
    ):
        """Test assistant response when service raises an error."""
        service = Mock()
        service.process_message = AsyncMock(
            side_effect=LLMProviderException("Service error")
        )
        mock_get_assistant_service.return_value = service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "gemini-3.1-flash-lite-preview",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 502
        assert "error" in response.json()["detail"].lower()

    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_upstream_rate_limit(
        self, mock_get_assistant_service, client
    ):
        """Test upstream rate limit errors are returned as temporary unavailability."""
        service = Mock()
        service.process_message = AsyncMock(
            side_effect=UpstreamRateLimitException("Rate limited")
        )
        mock_get_assistant_service.return_value = service

        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "gemini-3.1-pro-preview",
            "usecase": "Energy Consumption"
        }

        response = client.post("/getAssistantResponse", json=request_data)

        assert response.status_code == 503
        assert response.json()["detail"] == "Rate limited"
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_case_insensitive_usecase(
        self, mock_get_assistant_service, client, mock_assistant_service
    ):
        """Test that usecase accepts case-insensitive values."""
        mock_get_assistant_service.return_value = mock_assistant_service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "gemini-3.1-flash-lite-preview",
            "usecase": "energy consumption"  # Lowercase
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 200


class TestSuggestedFollowUpsEndpoint:
    """Test cases for suggested follow-ups endpoint."""

    @patch('src.api.dependencies.get_suggester_service')
    def test_get_suggested_follow_ups_success(
        self, mock_get_suggester_service, client, mock_suggester_service
    ):
        """Heart suggestion requests should return suggestions when available."""
        mock_suggester_service.generate_follow_ups = AsyncMock(
            return_value=[
                "Why was this patient classified as high risk?",
                "What would happen if their cholesterol were lower?",
                "Which features matter most overall?",
            ]
        )
        mock_get_suggester_service.return_value = mock_suggester_service

        response = client.post(
            "/getSuggestedFollowUps",
            json={
                "conversation": [{"role": "user", "content": "Assess patient 2"}],
                "usecase": "Heart Disease",
            },
        )

        assert response.status_code == 200
        assert response.json()["suggested_follow_ups"] == [
            "Why was this patient classified as high risk?",
            "What would happen if their cholesterol were lower?",
            "Which features matter most overall?",
        ]
        mock_suggester_service.generate_follow_ups.assert_awaited_once_with(
            conversation=[{"role": "user", "content": "Assess patient 2"}],
            usecase=UseCase.HEART,
            limit=None,
            exclude_suggestions=None,
            trace_context=ANY,
        )

    @patch('src.api.dependencies.get_suggester_service')
    def test_get_suggested_follow_ups_for_single_replacement(
        self, mock_get_suggester_service, client, mock_suggester_service
    ):
        """The endpoint should pass replacement options through to the suggester."""
        mock_suggester_service.generate_follow_ups = AsyncMock(
            return_value=["Ask about the strongest risk factor for this patient."]
        )
        mock_get_suggester_service.return_value = mock_suggester_service

        response = client.post(
            "/getSuggestedFollowUps",
            json={
                "conversation": [{"role": "user", "content": "Assess patient 2"}],
                "usecase": "Heart Disease",
                "limit": 1,
                "exclude_suggestions": [
                    "Why was this patient classified as high risk?",
                    "What would happen if their cholesterol were lower?",
                ],
            },
        )

        assert response.status_code == 200
        assert response.json()["suggested_follow_ups"] == [
            "Ask about the strongest risk factor for this patient."
        ]
        mock_suggester_service.generate_follow_ups.assert_awaited_once_with(
            conversation=[{"role": "user", "content": "Assess patient 2"}],
            usecase=UseCase.HEART,
            limit=1,
            exclude_suggestions=[
                "Why was this patient classified as high risk?",
                "What would happen if their cholesterol were lower?",
            ],
            trace_context=ANY,
        )

    @patch('src.api.dependencies.get_suggester_service')
    def test_get_suggested_follow_ups_returns_none_on_failure(
        self, mock_get_suggester_service, client, mock_suggester_service
    ):
        """Suggestion endpoint should degrade gracefully to no suggestions."""
        mock_suggester_service.generate_follow_ups = AsyncMock(side_effect=RuntimeError("boom"))
        mock_get_suggester_service.return_value = mock_suggester_service

        response = client.post(
            "/getSuggestedFollowUps",
            json={
                "conversation": [{"role": "user", "content": "Assess patient 2"}],
                "usecase": "Heart Disease",
            },
        )

        assert response.status_code == 200
        assert response.json()["suggested_follow_ups"] is None

    @patch('src.api.dependencies.get_suggester_service')
    def test_get_suggested_follow_ups_requires_conversation(
        self, mock_get_suggester_service, client, mock_suggester_service
    ):
        """Empty conversations should still be rejected."""
        mock_get_suggester_service.return_value = mock_suggester_service

        response = client.post(
            "/getSuggestedFollowUps",
            json={
                "conversation": [],
                "usecase": "Heart Disease",
            },
        )

        assert response.status_code == 400
