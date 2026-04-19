"""Google Gemini LLM provider implementation."""

import asyncio
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from src.domain.interfaces.llm_provider import LLMProvider
from src.core.constants import UseCase
from src.core.exceptions import LLMProviderException, UpstreamRateLimitException
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class GoogleGeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation using exact patterns from googlecloud.py."""

    _MAX_RETRIES = 3
    _BASE_RETRY_DELAY_SECONDS = 1.0
    _MAX_RETRY_DELAY_SECONDS = 8.0
    
    def __init__(self, model_name: str, project_id: str, location: str, api_key: Optional[str] = None):
        """
        Initialize the Google Gemini provider.
        
        Args:
            model_name: Name of the Gemini model
            project_id: Google Cloud project ID
            location: Google Cloud location
            api_key: Google API key (optional)
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.api_key = api_key
        self._client = None
        self._is_available = False
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Google Gemini client exactly as in googlecloud.py."""
        try:
            # Initialize the client exactly as in googlecloud.py
            self._client = genai.Client(
                http_options=types.HttpOptions(api_version="v1"),
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            self._is_available = True
            logger.info(f"Google Gemini provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini provider: {e}")
            self._is_available = False
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase
    ) -> str:
        """
        Generate a response from the Google Gemini model using exact pattern from googlecloud.py.
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content' keys
            usecase: The use case context (energy or heart)
            
        Returns:
            Raw response string from the LLM (structured JSON)
            
        Raises:
            LLMProviderException: If the LLM call fails
        """
        if not self._is_available:
            raise LLMProviderException("Google Gemini provider is not available")
        
        try:
            # Retry transient quota/rate-limit errors from Vertex AI.
            response = await self._generate_with_retry(conversation, usecase.value)

            if not response or not response.strip():
                raise LLMProviderException("Empty response content from Google Gemini API")
            
            logger.debug(f"Generated response from Google Gemini: {len(response)} characters")
            return response
            
        except LLMProviderException:
            raise
        except Exception as e:
            logger.error(f"Google Gemini generation failed: {e}")
            raise LLMProviderException(f"Failed to generate response: {e}")

    async def _generate_with_retry(self, conversation: List[Dict[str, str]], usecase: str) -> str:
        """Retry transient upstream rate-limit failures with exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                return await self._generate_google_cloud_response(conversation, usecase)
            except Exception as e:
                last_error = e

                if not self._is_resource_exhausted_error(e):
                    raise

                if attempt == self._MAX_RETRIES - 1:
                    break

                delay = min(
                    self._BASE_RETRY_DELAY_SECONDS * (2 ** attempt),
                    self._MAX_RETRY_DELAY_SECONDS,
                )
                logger.warning(
                    "Google Gemini rate limited by Vertex AI; retrying in %.1f seconds "
                    "(attempt %s/%s)",
                    delay,
                    attempt + 1,
                    self._MAX_RETRIES,
                )
                await asyncio.sleep(delay)

        raise UpstreamRateLimitException(
            "Google Gemini rate limit exceeded. Vertex AI returned RESOURCE_EXHAUSTED (429). "
            "Please retry shortly."
        ) from last_error
    
    async def _generate_google_cloud_response(self, conversation: List[Dict[str, str]], usecase: str) -> str:
        """
        Generate response using the exact same pattern as googlecloud.py
        
        Args:
            conversation: List of message dictionaries
            usecase: The use case context as string
            
        Returns:
            Generated response (structured JSON)
        """
        from pydantic import BaseModel
        
        # Define Response model exactly as in googlecloud.py
        class Response(BaseModel):
            function_calls: list[str]
            freeform_response: str
        
        # Get system prompt from use case registry
        from src.services.service_factory import get_usecase_registry
        from src.core.constants import UseCase
        
        # Convert string usecase to enum (handles both frontend and backend formats)
        usecase_enum = UseCase.from_string(usecase)
        registry = get_usecase_registry()
        system_prompt = registry.get_system_prompt(usecase_enum, conversation)
        
        # Get user input from the last message
        user_input = conversation[len(conversation) - 1]['content']
        
        # Configure generation exactly as in googlecloud.py
        generate_content_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type='application/json',
            response_schema=Response,
        )
        
        # Generate response using async wrapper
        # Use asyncio.to_thread() instead of deprecated asyncio.get_event_loop()
        response = await asyncio.to_thread(
            self._generate_sync, 
            user_input,
            generate_content_config
        )
        return response
    
    def _generate_sync(self, user_input: str, config: types.GenerateContentConfig) -> str:
        """
        Synchronous generation method.
        
        Args:
            user_input: The user input
            config: Generation configuration
            
        Returns:
            Generated response
        """
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_input,
            config=config
        )
        return response.text

    def _is_resource_exhausted_error(self, error: Exception) -> bool:
        """Best-effort detection for Vertex AI quota and rate-limit failures."""
        error_text = str(error).upper()

        if "RESOURCE_EXHAUSTED" in error_text:
            return True
        if "429" in error_text:
            return True

        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        code = getattr(error, "code", None)
        if code == 429:
            return True
        if isinstance(code, str) and code.upper() == "RESOURCE_EXHAUSTED":
            return True

        return False
    
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name string
        """
        return self.model_name
    
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and configured.
        
        Returns:
            True if available, False otherwise
        """
        return self._is_available
