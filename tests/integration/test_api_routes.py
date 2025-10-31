"""Integration tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from src.main import app
from src.core.constants import UseCase, Model


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
        self, mock_get_service, client, mock_assistant_service
    ):
        """Test successful assistant response generation."""
        mock_get_service.return_value = mock_assistant_service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "What is the dataset size?"}
            ],
            "model": "Llama-3.3-70B-Instruct",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "assistantResponse" in data
        assert data["assistantResponse"]["freeform_response"] == "Test response"
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_with_function_calls(
        self, mock_get_service, client
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
        mock_get_service.return_value = service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Count all records"}
            ],
            "model": "Gemini-2.0-Flash",
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
            "model": "Llama-3.3-70B-Instruct",
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
            "model": "Llama-3.3-70B-Instruct",
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
        self, mock_get_service, client
    ):
        """Test assistant response when service raises an error."""
        from src.core.exceptions import LLMProviderException
        
        service = Mock()
        service.process_message = AsyncMock(
            side_effect=LLMProviderException("Service error")
        )
        mock_get_service.return_value = service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "Llama-3.3-70B-Instruct",
            "usecase": "Energy Consumption"
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
    
    @patch('src.api.dependencies.get_assistant_service')
    def test_get_assistant_response_case_insensitive_usecase(
        self, mock_get_service, client, mock_assistant_service
    ):
        """Test that usecase accepts case-insensitive values."""
        mock_get_service.return_value = mock_assistant_service
        
        request_data = {
            "conversation": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "Llama-3.3-70B-Instruct",
            "usecase": "energy consumption"  # Lowercase
        }
        
        response = client.post("/getAssistantResponse", json=request_data)
        
        assert response.status_code == 200

