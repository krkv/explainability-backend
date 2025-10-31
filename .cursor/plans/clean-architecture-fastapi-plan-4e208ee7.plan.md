<!-- 4e208ee7-ebee-4ffd-b67b-b4ae6c591366 00ed395b-d58e-4cc5-a366-105d48e4c2e4 -->
# Clean Architecture & FastAPI Migration Plan

## Executive Summary

This plan outlines the refactoring of the XAI LLM Chat Backend from a Flask-based monolithic application into a clean, maintainable Python architecture using FastAPI. The plan addresses critical security issues (eval() usage), introduces proper separation of concerns, dependency injection, and modern Python best practices.

**Current Issues:**

- Security vulnerability: `eval()` used to execute LLM-generated function calls
- Global state: models and datasets loaded at module import time
- Tight coupling: direct imports throughout, no dependency injection
- Monolithic structure: mixed concerns, difficult to test
- No type safety: minimal type hints, manual JSON parsing
- Code duplication: similar patterns in energy/heart instances
- Flask limitations: no automatic API docs, limited validation

**Target Architecture:**

- Clean architecture with clear layer boundaries
- FastAPI with automatic validation and documentation
- Dependency injection throughout
- Type-safe with comprehensive type hints
- Testable with dependency injection
- Secure function execution without eval()
- Lazy loading of heavy resources

---

## Implementation Status

**Last Updated**: 2025-01-27 - Conversation history management removed from backend

## Changelog

### 2025-01-27 - Conversation History Management Removed

**Summary**: Removed all backend conversation state management. Frontend now manages and sends entire conversation history with each request.

**Changes**:
- **AssistantService** (`src/services/assistant/assistant_service.py`):
  - ✅ Removed `_conversations` dictionary and all conversation tracking methods
  - ✅ Removed `_get_or_create_conversation()`, `get_conversation()`, `get_all_conversations()`, `get_conversation_stats()` methods
  - ✅ Removed `_prepare_conversation_for_llm()` method (no longer needed)
  - ✅ Simplified `process_message()` to directly accept `conversation: List[Dict[str, str]]` from frontend
  - ✅ Conversation is now passed directly to LLM providers without backend state management
- **API Schemas** (`src/api/schemas.py`):
  - ✅ Removed `conversation_id` field from `AssistantRequest`
  - ✅ `AssistantRequest` now only includes `conversation`, `model`, and `usecase`
- **API Routes** (`src/api/routes.py`):
  - ✅ Updated to pass full conversation directly from request to `assistant_service.process_message()`
  - ✅ Removed `conversation_id` extraction logic
- **Service Factory** (`src/services/service_factory.py`):
  - ✅ Removed `get_conversation_stats()` call from service initialization

**Architecture Impact**:
- Backend is now stateless regarding conversation history (matches legacy implementation)
- Frontend is responsible for maintaining and sending conversation history with each request
- Conversation history is embedded in system prompts as JSON (as per legacy implementation)
- Reduced backend complexity and memory usage (no conversation storage)

**Status**: ✅ Complete

### Phase Completion Status

| Phase | Status | Completion | Notes |

|-------|--------|------------|-------|

| **Phase 1: Security Fix** | ✅ **COMPLETE** | 100% | eval() replaced with AST parser, both instances updated |

| **Phase 2: Core Infrastructure** | ✅ **COMPLETE** | 100% | Config, exceptions, constants, logging all implemented |

| **Phase 3: Domain Layer** | ✅ **COMPLETE** | 100% | Interfaces/protocols and entities created |

| **Phase 4: Infrastructure Layer** | ✅ **COMPLETE** | 100% | Lazy loaders, caching, factories implemented |

| **Phase 5: Service Layer** | ✅ **COMPLETE** | 100% | Assistant service, LLM providers, function executor all implemented |

| **Phase 6: UseCase Refactoring** | ✅ **COMPLETE** | 100% | All usecase classes created, functions refactored, registry updated |

| **Phase 7: FastAPI Migration** | ✅ **COMPLETE** | 100% | FastAPI routes, schemas, dependencies, and main app implemented |

| **Phase 8: Testing Infrastructure** | ❌ **NOT STARTED** | 0% | Test directories exist but empty |

| **Phase 9: Configuration Files** | ✅ **COMPLETE** | 100% | requirements.txt updated with FastAPI, Dockerfile updated for uvicorn |

| **Phase 10: Cleanup Legacy Code** | ❌ **NOT STARTED** | 0% | All legacy files still present |

### Key Achievements ✅

1. **Security Fixed**: `eval()` completely removed, replaced with safe AST-based function parser
2. **Clean Architecture Foundation**: Core, domain, infrastructure, and service layers fully implemented
3. **Dependency Injection**: Services use dependency injection via factories
4. **Type Safety**: Enums, type hints, and interfaces throughout
5. **Lazy Loading**: Infrastructure for lazy loading models and data created
6. **Frontend Compatibility**: UseCase enum now properly handles frontend values ("Energy Consumption", "Heart Disease")
7. **UseCase Refactoring**: Complete - functions moved to `src/usecases/` with lazy loading and dependency injection
8. **FastAPI Migration**: Complete - FastAPI application with routes, schemas, dependencies, and backward-compatible API responses

### Critical Remaining Work ❌

1. **Testing**: No tests written yet
2. **Legacy Code Cleanup**: Legacy `instances/` and `app.py` files still present (can be removed in Phase 10)

### Next Steps (Priority Order)

1. **Phase 8**: Write tests for all components
2. **Phase 10**: Remove legacy code after verification (legacy `instances/` files and `app.py` can be removed once FastAPI is verified)

---

## Current Architecture Analysis

### Code Structure

```
explainability-backend/
├── app.py                          # Flask app (36 lines) - LEGACY, to be replaced
├── assistant.py                    # Orchestration (48 lines) - LEGACY, to be replaced
├── huggingface.py                  # HF provider (32 lines) - LEGACY, replaced
├── googlecloud.py                  # Google provider (35 lines) - LEGACY, replaced
├── instances/                       # LEGACY - functions migrated to src/usecases/, files kept for reference (remove in Phase 10)
│   ├── energy/
│   │   ├── executive.py           # ⚠️ LEGACY - functions migrated to src/usecases/energy/energy_functions.py
│   │   ├── parser.py               # ⚠️ LEGACY - functionality in src/services/parser/function_parser.py
│   │   ├── prompt.py               # ⚠️ LEGACY - functionality in src/usecases/energy/energy_usecase.py
│   │   └── functions.json          # ✅ Still used by usecase for function definitions
│   └── heart/
│       ├── executive.py           # ⚠️ LEGACY - functions migrated to src/usecases/heart/heart_functions.py
│       ├── parser.py               # ⚠️ LEGACY - functionality in src/services/parser/function_parser.py
│       ├── prompt.py               # ⚠️ LEGACY - functionality in src/usecases/heart/heart_usecase.py
│       └── functions.json          # ✅ Still used by usecase for function definitions
└── src/                            # ✅ NEW CLEAN ARCHITECTURE
    ├── api/                        # ✅ COMPLETE - FastAPI routes, schemas, dependencies
    ├── main.py                     # ✅ COMPLETE - FastAPI application entry point
    ├── core/                       # ✅ COMPLETE - config, exceptions, constants, logging
    ├── domain/                     # ✅ COMPLETE - entities, interfaces/protocols
    ├── infrastructure/             # ✅ COMPLETE - loaders, caching, factories
    ├── services/                   # ✅ COMPLETE - assistant, LLM, parser, function executor
    └── usecases/                   # ✅ COMPLETE - base, energy, heart usecases implemented
└── requirements.txt                # ✅ COMPLETE - FastAPI and uvicorn dependencies added
```

### Critical Issues Identified

1. **Security: eval() Usage** ✅ **FIXED**

   - ✅ `instances/energy/parser.py`: Now uses `FunctionParser` with AST-based parsing
   - ✅ `instances/heart/parser.py`: Now uses `FunctionParser` with AST-based parsing
   - ✅ No `eval()` usage remaining in codebase
   - ✅ Safe function execution implemented via `src/services/parser/function_parser.py`

2. **Global State** ✅ **ADDRESSED IN NEW CODE** ⚠️ **LEGACY FILES REMAIN**

   - ✅ New usecases in `src/usecases/` use lazy loading (models/datasets load on demand)
   - ✅ Lazy loading infrastructure created (`src/infrastructure/loaders/`)
   - ✅ Use cases refactored to use lazy loading (`src/usecases/energy/energy_usecase.py`, `src/usecases/heart/heart_usecase.py`)
   - ⚠️ Legacy `instances/energy/executive.py` and `instances/heart/executive.py` still have global state (will be removed in Phase 10)

3. **Tight Coupling** ✅ **ADDRESSED**

   - ✅ Dependency injection implemented via services and factories
   - ✅ Abstraction layers created (domain interfaces)
   - ⚠️ Legacy `app.py` still uses direct imports (needs FastAPI migration)

4. **Type Safety Issues** ✅ **MOSTLY ADDRESSED**

   - ✅ Enums created for models and use cases (`src/core/config.py`, `src/core/constants.py`)
   - ✅ Type hints throughout services and domain layer
   - ⚠️ FastAPI schemas not yet created (no request/response validation)

5. **Code Duplication** ✅ **ADDRESSED IN NEW CODE** ⚠️ **LEGACY FILES REMAIN**

   - ✅ Use case refactoring complete - functions moved to `src/usecases/` with shared base class
   - ✅ Common infrastructure extracted (loaders, parsers, services)
   - ✅ Energy and heart functions use same base class pattern
   - ⚠️ Legacy `instances/` files still exist but are no longer used (will be removed in Phase 10)

---

## Proposed Architecture

### Layer Structure (Clean Architecture)

```
┌─────────────────────────────────────────────┐
│  API Layer (FastAPI)                        │  Presentation
│  - Routes, Schemas, Dependencies           │
├─────────────────────────────────────────────┤
│  Application Layer                          │  Use Cases
│  - Assistant Service, Orchestration        │
├─────────────────────────────────────────────┤
│  Domain Layer                               │  Business Logic
│  - Entities, Interfaces/Protocols          │
├─────────────────────────────────────────────┤
│  Infrastructure Layer                       │  Technical Details
│  - Model/Data Loaders, LLM Providers       │
└─────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── main.py                           # FastAPI app entry point
├── api/
│   ├── __init__.py
│   ├── routes.py                     # API endpoints
│   ├── schemas.py                    # Pydantic request/response models
│   ├── dependencies.py               # Dependency injection
│   └── errors.py                     # Error handlers
├── core/
│   ├── __init__.py
│   ├── config.py                     # Settings (pydantic-settings)
│   ├── exceptions.py                 # Custom exception hierarchy
│   ├── constants.py                  # Enums, constants
│   └── logging_config.py            # Logging setup
├── domain/
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── message.py                # Conversation message
│   │   └── function_call.py          # Function call entity
│   └── interfaces/
│       ├── __init__.py
│       ├── llm_provider.py           # LLM provider protocol
│       ├── function_executor.py     # Function execution protocol
│       └── usecase_registry.py      # UseCase registry protocol
├── services/
│   ├── __init__.py
│   ├── assistant/
│   │   ├── __init__.py
│   │   └── assistant_service.py      # Main orchestration
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_provider.py          # Base LLM provider
│   │   ├── huggingface_provider.py
│   │   └── google_provider.py
│   └── parser/
│       ├── __init__.py
│       ├── function_parser.py        # Safe function parser
│       ├── ast_parser.py             # AST-based parsing
│       └── function_registry.py      # Function registry
├── infrastructure/
│   ├── __init__.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── model_loader.py           # Lazy model loading
│   │   ├── data_loader.py            # Lazy data loading
│   │   └── explainer_loader.py       # Lazy explainer loading
│   └── caching/
│       ├── __init__.py
│       └── cache_manager.py          # Cache management
└── usecases/
    ├── __init__.py
    ├── base/
    │   ├── __init__.py
    │   └── base_usecase.py           # ✅ Abstract base class with lazy loading
    ├── energy/
    │   ├── __init__.py
    │   ├── energy_usecase.py         # ✅ Energy usecase class
    │   ├── energy_functions.py       # ✅ Refactored energy functions
    │   └── energy_config.py          # ✅ Energy-specific config
    └── heart/
        ├── __init__.py
        ├── heart_usecase.py          # ✅ Heart usecase class
        ├── heart_functions.py         # ✅ Refactored heart functions
        └── heart_config.py           # ✅ Heart-specific config

tests/
├── __init__.py
├── conftest.py                       # Pytest fixtures
├── unit/
│   ├── test_function_parser.py
│   ├── test_llm_providers.py
│   ├── test_assistant_service.py
│   └── test_usecases.py
├── integration/
│   ├── test_api_routes.py
│   └── test_end_to_end.py
└── fixtures/
    └── sample_data.py
```

---

## Implementation Phases

### Phase 1: Security Fix (CRITICAL)

**Priority: HIGH - Must be done first**

#### 1.1 Safe Function Parser Implementation

**File**: `src/services/parser/function_parser.py`

Replace `eval()` with AST-based safe parsing:

```python
import ast
from typing import Callable, Dict, Any, List, Optional
from src.core.exceptions import FunctionExecutionException

class FunctionParser:
    """Safely parses and executes function calls without eval()."""
    
    def __init__(self, function_registry: Dict[str, Callable]):
        self.function_registry = function_registry
    
    def parse_and_execute(self, function_call_str: str) -> Any:
        """Parse function call string and execute safely."""
        # Parse using AST, validate structure, extract function name and args
        # Execute only registered functions
        pass
    
    def parse_calls(self, function_calls: List[str]) -> str:
        """Parse multiple function calls and return concatenated results."""
        results = []
        for call in function_calls:
            result = self.parse_and_execute(call)
            # Handle both string and dict return types
            if isinstance(result, dict) and "text" in result:
                results.append(result["text"])
            elif isinstance(result, str):
                results.append(result)
            else:
                results.append(str(result))
        return '\n'.join(results)
```

**File**: `src/services/parser/ast_parser.py`

AST parsing utilities:

```python
import ast
from typing import Dict, Any

class ASTParser:
    """AST-based function call parser."""
    
    @staticmethod
    def parse_function_call(function_call_str: str) -> tuple[str, Dict[str, Any]]:
        """Extract function name and arguments from string."""
        # Parse AST, validate it's a Call node
        # Extract function name and keyword arguments
        # Use ast.literal_eval() for safe value extraction
        pass
```

**File**: `src/services/parser/function_registry.py`

Function registry system:

```python
from typing import Dict, Callable, Optional

class FunctionRegistry:
    """Registry of available functions per usecase."""
    
    def __init__(self):
        self._registries: Dict[str, Dict[str, Callable]] = {}
    
    def register_usecase(
        self, 
        usecase: str, 
        functions: Dict[str, Callable]
    ) -> None:
        """Register functions for a usecase."""
        self._registries[usecase] = functions
    
    def get_registry(self, usecase: str) -> Dict[str, Callable]:
        """Get function registry for usecase."""
        if usecase not in self._registries:
            raise ValueError(f"Unknown usecase: {usecase}")
        return self._registries[usecase]
```

**Action Items:**

- [x] Implement AST-based function parser (`src/services/parser/function_parser.py`)
- [x] Create function registry (`src/services/parser/function_registry.py`)
- [x] Create AST parser utilities (`src/services/parser/ast_parser.py`)
- [x] Update energy parser to use safe parser (`instances/energy/parser.py`)
- [x] Update heart parser to use safe parser (`instances/heart/parser.py`)
- [x] Remove all `eval()` usage
- [ ] Add unit tests for parser (deferred to Phase 8)

---

### Phase 2: Core Infrastructure

#### 2.1 Configuration Management

**File**: `src/core/config.py`

```python
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class Model(str, Enum):
    LLAMA_3_3_70B = "Llama-3.3-70B-Instruct"
    GEMINI_2_0_FLASH = "Gemini-2.0-Flash"
    
    @classmethod
    def from_string(cls, value: str) -> "Model":
        """Map legacy string format to enum."""
        mapping = {
            "Llama 3.3 70B Instruct": cls.LLAMA_3_3_70B,
            "Gemini 2.0 Flash": cls.GEMINI_2_0_FLASH,
        }
        return mapping.get(value, cls.LLAMA_3_3_70B)

class UseCase(str, Enum):
    ENERGY = "energy"
    HEART = "heart"
    
    @classmethod
    def from_string(cls, value: str) -> "UseCase":
        """Map legacy string format to enum."""
        mapping = {
            "Heart Disease": cls.HEART,
            "Energy Consumption": cls.ENERGY,
        }
        return mapping.get(value, cls(value.lower()))

class Settings(BaseSettings):
    # LLM Configuration
    hf_token: Optional[str] = None
    google_project: str = "explainability-app"
    google_location: str = "europe-north1"
    
    # Paths
    instances_path: str = "instances"
    energy_instance_path: str = "instances/energy"
    heart_instance_path: str = "instances/heart"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

#### 2.2 Exception Hierarchy

**File**: `src/core/exceptions.py`

```python
class ExplainabilityException(Exception):
    """Base exception for all application errors."""
    pass

class InvalidModelException(ExplainabilityException):
    """Raised when invalid model is specified."""
    pass

class InvalidUseCaseException(ExplainabilityException):
    """Raised when invalid usecase is specified."""
    pass

class FunctionExecutionException(ExplainabilityException):
    """Raised when function execution fails."""
    pass

class LLMProviderException(ExplainabilityException):
    """Raised when LLM provider fails."""
    pass

class ModelLoadException(ExplainabilityException):
    """Raised when model fails to load."""
    pass

class DataLoadException(ExplainabilityException):
    """Raised when dataset fails to load."""
    pass
```

**Action Items:**

- [x] Create directory structure
- [x] Implement configuration management (`src/core/config.py`)
- [x] Define exception hierarchy (`src/core/exceptions.py`)
- [x] Create constants/enums (`src/core/constants.py`)
- [x] Setup logging configuration (`src/core/logging_config.py`)
- [x] ✅ **FIXED**: Added `from_string()` method to `UseCase` enum in `constants.py` to handle frontend values ("Energy Consumption", "Heart Disease")
- [x] ✅ **FIXED**: Updated LLM providers (`huggingface_provider.py`, `google_gemini_provider.py`) to use `UseCase.from_string()` instead of manual conversion
- [ ] Consolidate duplicate `UseCase` enums (currently in both `config.py` and `constants.py`) - **LOW PRIORITY** - Both work correctly now

---

### Phase 3: Domain Layer

#### 3.1 Interfaces (Protocols)

**File**: `src/domain/interfaces/llm_provider.py`

```python
from typing import Protocol, List, Dict

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        """Generate LLM response with function calls."""
        ...
```

**File**: `src/domain/interfaces/function_executor.py`

```python
from typing import Protocol, List

class FunctionExecutor(Protocol):
    """Protocol for function execution."""
    
    def execute_calls(self, function_calls: List[str]) -> str:
        """Execute function calls and return results."""
        ...
```

#### 3.2 Entities

**File**: `src/domain/entities/message.py`

```python
from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
```

**Action Items:**

- [x] Define domain interfaces (Protocols) (`src/domain/interfaces/`)
  - [x] `llm_provider.py`
  - [x] `function_executor.py`
  - [x] `usecase_registry.py`
  - [x] `model_loader.py`
  - [x] `data_loader.py`
- [x] Create entity models (`src/domain/entities/`)
  - [x] `message.py` (includes `Conversation` class)
  - [x] `function_call.py`
  - [x] `assistant_response.py`
- [x] Establish type contracts

---

### Phase 4: Infrastructure Layer

#### 4.1 Lazy Loading Infrastructure

**File**: `src/infrastructure/loaders/model_loader.py`

```python
from functools import lru_cache
import pickle
import joblib
from pathlib import Path
from typing import Any
from src.core.exceptions import ModelLoadException

class ModelLoader:
    """Lazy loading and caching of ML models."""
    
    @lru_cache(maxsize=10)
    def load_model(self, model_path: str) -> Any:
        """Load model with caching."""
        # Validate path, load model, handle errors
        pass
```

**File**: `src/infrastructure/loaders/data_loader.py`

```python
import pandas as pd
from functools import lru_cache
from pathlib import Path
from src.core.exceptions import DataLoadException

class DataLoader:
    """Lazy loading and caching of datasets."""
    
    @lru_cache(maxsize=10)
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset with caching."""
        # Validate path, load CSV, handle errors
        pass
```

**Action Items:**

- [x] Implement lazy model loading (`src/infrastructure/loaders/model_loader.py`)
- [x] Implement lazy data loading (`src/infrastructure/loaders/data_loader.py`)
- [x] Add caching mechanism (`src/infrastructure/caching/cache_manager.py`)
- [x] Create explainer loader (`src/infrastructure/loaders/explainer_loader.py`)
- [x] Create infrastructure factory (`src/infrastructure/factory.py`)

---

### Phase 5: Service Layer

#### 5.1 LLM Providers

**File**: `src/services/llm/base_provider.py`

```python
from src.domain.interfaces.llm_provider import LLMProvider
from typing import List, Dict
from abc import ABC

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        """Generate response - to be implemented by subclasses."""
        raise NotImplementedError
```

**File**: `src/services/llm/huggingface_provider.py`

Refactor `huggingface.py` into class:

```python
from huggingface_hub import InferenceClient
from src.services.llm.base_provider import BaseLLMProvider
from src.core.config import settings
from src.core.exceptions import LLMProviderException
from typing import List, Dict

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider."""
    
    def __init__(self, client: InferenceClient = None):
        self.client = client or InferenceClient(
            "meta-llama/Llama-3.3-70B-Instruct",
            token=settings.hf_token,
        )
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        # Implementation from huggingface.py, refactored
        pass
```

**File**: `src/services/llm/google_provider.py`

Refactor `googlecloud.py` into class:

```python
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from pydantic import BaseModel
from src.services.llm.base_provider import BaseLLMProvider
from src.core.config import settings
from typing import List, Dict

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, client=None):
        self.client = client or genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=True,
            project=settings.google_project,
            location=settings.google_location,
        )
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        # Implementation from googlecloud.py, refactored
        pass
```

#### 5.2 Assistant Service

**File**: `src/services/assistant/assistant_service.py`

Refactor `assistant.py` into service:

```python
from src.domain.interfaces.llm_provider import LLMProvider
from src.services.parser.function_parser import FunctionParser
from src.core.exceptions import FunctionExecutionException
import json
from typing import List, Dict

class AssistantService:
    """Orchestrates LLM calls and function execution.
    
    Note: Conversation history is managed by the frontend and passed with each request.
    The backend is stateless and does not track conversations.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        function_parser: FunctionParser,
    ):
        self.llm_provider = llm_provider
        self.function_parser = function_parser
    
    async def process_message(
        self,
        conversation: List[Dict[str, str]],
        usecase: UseCase,
        model: Model
    ) -> AssistantResponse:
        """Process a conversation and generate an assistant response.
        
        Args:
            conversation: Full conversation history from frontend
            usecase: The use case context
            model: The LLM model to use
        """
        # Get LLM response (conversation history is embedded in system prompt)
        llm_response = await self.llm_provider.generate_response(
            conversation, usecase
        )
        
        # Parse and validate JSON response
        response_data = json.loads(llm_response)
        function_calls = response_data.get("function_calls", [])
        freeform_response = response_data.get("freeform_response", "")
        
        # Execute functions if present
        parse_result = None
        if function_calls:
            try:
                parse_result = self.function_parser.parse_calls(function_calls)
            except FunctionExecutionException as e:
                # Log and handle gracefully
                parse_result = f"Error executing functions: {e}"
        
        return AssistantResponse(
            function_calls=function_calls,
            freeform_response=freeform_response.strip(),
            parse=parse_result,
        )
```

**Action Items:**

- [x] Refactor LLM providers into classes
  - [x] `src/services/llm/huggingface_provider.py` (Note: Named `HuggingFaceProvider`, uses `google_gemini_provider.py` naming pattern)
  - [x] `src/services/llm/google_gemini_provider.py` (Note: Named `GoogleGeminiProvider`)
  - [x] `src/services/llm/llm_factory.py`
- [x] Create assistant service (`src/services/assistant/assistant_service.py`)
- [x] Create function executor service (`src/services/function/function_executor_service.py`)
- [x] Create use case registry service (`src/services/usecase/usecase_registry_service.py`)
- [x] Create service factory (`src/services/service_factory.py`)
- [x] Implement dependency injection via factories
- [x] Add error handling throughout

---

### Phase 6: UseCase Refactoring

#### 6.1 Base UseCase

**File**: `src/usecases/base/base_usecase.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Callable, List
from src.infrastructure.loaders.model_loader import ModelLoader
from src.infrastructure.loaders.data_loader import DataLoader

class BaseUseCase(ABC):
    """Base class for all usecases."""
    
    def __init__(
        self,
        model_loader: ModelLoader,
        data_loader: DataLoader,
    ):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self._model = None
        self._dataset = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @property
    def dataset(self):
        """Lazy load dataset."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset
    
    @abstractmethod
    def _load_model(self):
        """Load model for this usecase."""
        pass
    
    @abstractmethod
    def _load_dataset(self):
        """Load dataset for this usecase."""
        pass
    
    @abstractmethod
    def get_functions(self) -> Dict[str, Callable]:
        """Get function registry for this usecase."""
        pass
    
    @abstractmethod
    def get_system_prompt(self, conversation: List[Dict]) -> str:
        """Generate system prompt for this usecase."""
        pass
```

#### 6.2 Energy UseCase

**File**: `src/usecases/energy/energy_usecase.py`

```python
from src.usecases.base.base_usecase import BaseUseCase
from src.usecases.energy.energy_functions import EnergyFunctions
from typing import Dict, Callable, List

class EnergyUseCase(BaseUseCase):
    """Energy consumption usecase."""
    
    def __init__(self, model_loader, data_loader, config):
        super().__init__(model_loader, data_loader)
        self.config = config
        self._functions = None
    
    def _load_model(self):
        return self.model_loader.load_model(
            self.config.model_path
        )
    
    def _load_dataset(self):
        return self.data_loader.load_dataset(
            self.config.dataset_path
        )
    
    def get_functions(self) -> Dict[str, Callable]:
        """Get all energy functions."""
        if self._functions is None:
            energy_funcs = EnergyFunctions(
                model=self.model,
                dataset=self.dataset,
                config=self.config
            )
            self._functions = {
                'count_all': energy_funcs.count_all,
                'show_one': energy_funcs.show_one,
                # ... register all functions
            }
        return self._functions
    
    def get_system_prompt(self, conversation: List[Dict]) -> str:
        """Generate energy system prompt."""
        # Move logic from instances/energy/prompt.py
        pass
```

**File**: `src/usecases/energy/energy_functions.py`

Refactor `instances/energy/executive.py`:

- Extract functions into class
- Remove global state
- Accept dependencies via constructor
- Consistent return types

**File**: `src/usecases/energy/energy_config.py`

```python
from pydantic import BaseModel
from pathlib import Path

class EnergyConfig(BaseModel):
    instance_path: Path
    model_path: Path
    dataset_path: Path
    functions_json_path: Path
```

**Action Items:**

- [x] Create base usecase class (`src/usecases/base/base_usecase.py`) ✅ **COMPLETE**
- [x] Create energy config class (`src/usecases/energy/energy_config.py`) ✅ **COMPLETE**
- [x] Refactor energy functions into class (`src/usecases/energy/energy_functions.py`) ✅ **COMPLETE**
- [x] Create energy usecase class (`src/usecases/energy/energy_usecase.py`) ✅ **COMPLETE**
- [x] Create heart config class (`src/usecases/heart/heart_config.py`) ✅ **COMPLETE**
- [x] Refactor heart functions into class (`src/usecases/heart/heart_functions.py`) ✅ **COMPLETE**
- [x] Create heart usecase class (`src/usecases/heart/heart_usecase.py`) ✅ **COMPLETE**
- [x] Move prompt generation into usecases (`get_system_prompt()` in each usecase class) ✅ **COMPLETE**
- [x] Update usecase registry to use new usecase classes (`src/services/usecase/usecase_registry_service.py`) ✅ **COMPLETE**
- [ ] Remove global state ⚠️ **PARTIAL** - New usecases use lazy loading, but legacy `instances/` files still have global state (can be removed in Phase 10)

---

### Phase 7: FastAPI Migration

**Status**: ✅ **COMPLETE** - FastAPI application fully implemented and configured

**Note**: The service layer has been implemented differently than the plan:

- `AssistantService` uses `process_message()` method instead of `generate_response()`
- Services are created via factories (`service_factory.py`) rather than direct dependency injection in routes
- `FunctionExecutorService` and `UseCaseRegistryService` are separate services wired into FastAPI dependencies via service factory

#### 7.1 API Schemas

**File**: `src/api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from src.core.config import Model, UseCase

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class AssistantRequest(BaseModel):
    conversation: List[Message]  # Full conversation history from frontend
    model: str  # Will be validated and converted to Model enum
    usecase: str  # Will be validated and converted to UseCase enum
    # Note: conversation_id removed - backend is stateless, frontend manages conversation

class AssistantResponse(BaseModel):
    function_calls: List[str]
    freeform_response: str
    parse: Optional[str] = None
```

#### 7.2 API Dependencies

**File**: `src/api/dependencies.py`

```python
from functools import lru_cache
from fastapi import Depends
from src.core.config import settings, Model, UseCase
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_provider import GoogleProvider
from src.services.parser.function_parser import FunctionParser
from src.services.parser.function_registry import FunctionRegistry
from src.services.assistant.assistant_service import AssistantService
from src.domain.interfaces.llm_provider import LLMProvider
from src.api.schemas import AssistantRequest

def get_llm_provider(model_str: str) -> LLMProvider:
    """Factory for LLM providers."""
    model = Model.from_string(model_str)
    if model == Model.LLAMA_3_3_70B:
        return HuggingFaceProvider()
    elif model == Model.GEMINI_2_0_FLASH:
        return GoogleProvider()
    else:
        raise ValueError(f"Unknown model: {model_str}")

def get_usecase_instance(usecase_str: str):
    """Get usecase instance - to be implemented."""
    # Factory pattern to get energy/heart usecase
    pass

def get_function_parser(usecase_str: str) -> FunctionParser:
    """Factory for function parser."""
    usecase = UseCase.from_string(usecase_str)
    usecase_instance = get_usecase_instance(usecase_str)
    functions = usecase_instance.get_functions()
    
    registry = FunctionRegistry()
    registry.register_usecase(usecase.value, functions)
    
    return FunctionParser(registry.get_registry(usecase.value))

def get_assistant_service(request: AssistantRequest) -> AssistantService:
    """Factory for assistant service."""
    llm_provider = get_llm_provider(request.model)
    function_parser = get_function_parser(request.usecase)
    
    return AssistantService(
        llm_provider=llm_provider,
        function_parser=function_parser,
    )
```

#### 7.3 API Routes

**File**: `src/api/routes.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import AssistantRequest, AssistantResponse
from src.api.dependencies import get_assistant_service
from src.services.assistant.assistant_service import AssistantService

router = APIRouter()

@router.get("/ready")
async def ready():
    """Health check endpoint."""
    return {"status": "OK"}

@router.post("/getAssistantResponse", response_model=AssistantResponse)
async def get_assistant_response(
    request: AssistantRequest,
    assistant_service: AssistantService = Depends(get_assistant_service),
):
    """Generate assistant response with function execution."""
    try:
        response = await assistant_service.generate_response(
            conversation=[msg.dict() for msg in request.conversation],
            usecase=request.usecase,
        )
        return AssistantResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 7.4 Main Application

**File**: `src/main.py`

```python
from fastapi import FastAPI
from src.api.routes import router
from src.core.logging_config import setup_logging

app = FastAPI(
    title="XAI LLM Chat Backend",
    description="LLM-powered assistant for ML model explanations",
    version="2.0.0",
)

app.include_router(router)
setup_logging()

@app.on_event("startup")
async def startup():
    """Startup tasks."""
    pass

@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks."""
    pass
```

**Action Items:**

- [x] Create Pydantic schemas (`src/api/schemas.py`) ✅ **COMPLETE**
- [x] Implement dependency injection (`src/api/dependencies.py`) ✅ **COMPLETE**
- [x] Create FastAPI routes (`src/api/routes.py`) ✅ **COMPLETE**
- [x] Setup main application (`src/main.py`) ✅ **COMPLETE**
- [x] Add error handlers and CORS middleware ✅ **COMPLETE**
- [x] Update requirements.txt with FastAPI dependencies ✅ **COMPLETE**
- [x] Update Dockerfile for FastAPI/uvicorn ✅ **COMPLETE**
- [x] Remove backend conversation history management ✅ **COMPLETE** - Backend is now stateless

**Implementation Details:**

- **Schemas** (`src/api/schemas.py`): Created request/response models including `AssistantRequest` (with `conversation`, `model`, `usecase` - no `conversation_id`), `AssistantResponseWrapper`, and `HealthResponse`
- **Dependencies** (`src/api/dependencies.py`): Implemented validation functions for model and usecase, dependency injection for `AssistantService` via service factory
- **Routes** (`src/api/routes.py`): Implemented `/ready` health check endpoint and `/getAssistantResponse` POST endpoint with full error handling
  - Passes full conversation history from frontend directly to `AssistantService.process_message()`
  - Backend is stateless - no conversation tracking or extraction of user messages
- **Main App** (`src/main.py`): FastAPI application with CORS middleware, logging setup, startup/shutdown events, and automatic API documentation at `/docs`
- **Backward Compatibility**: Response format matches legacy Flask API (`{"assistantResponse": {...}}`) for seamless frontend integration
- **Configuration**: Updated `requirements.txt` with `fastapi>=0.104.0` and `uvicorn[standard]>=0.24.0`, updated `Dockerfile` to use uvicorn instead of gunicorn

---

### Phase 8: Testing Infrastructure

#### 8.1 Test Structure

**File**: `tests/conftest.py`

```python
import pytest
from unittest.mock import Mock
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.parser.function_parser import FunctionParser

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock(spec=HuggingFaceProvider)
    provider.generate_response.return_value = '{"function_calls": [], "freeform_response": "test"}'
    return provider

@pytest.fixture
def function_registry():
    """Test function registry."""
    registry = {
        'test_function': lambda x: f"result: {x}",
    }
    return FunctionParser(registry)
```

#### 8.2 Unit Tests

**File**: `tests/unit/test_function_parser.py`

Test safe function parsing, error handling, registry validation.

**File**: `tests/unit/test_llm_providers.py`

Test LLM provider implementations with mocks.

**Action Items:**

- [ ] Create test directory structure
- [ ] Write unit tests for parsers
- [ ] Write unit tests for services
- [ ] Write integration tests for API
- [ ] Achieve >80% code coverage

---

### Phase 9: Update Configuration Files

#### 9.1 Requirements

**File**: `requirements.txt`

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
# ... existing dependencies ...
# Remove Flask
```

#### 9.2 Dockerfile

**File**: `Dockerfile`

```dockerfile
FROM python:3.11
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Action Items:**

- [x] Update requirements.txt ✅ **COMPLETE** - FastAPI and uvicorn added, Flask kept for backward compatibility
- [x] Update Dockerfile ✅ **COMPLETE** - Changed to use uvicorn instead of gunicorn
- [ ] Update README with FastAPI instructions ⚠️ **OPTIONAL** - Can be done later if needed

---

### Phase 10: Cleanup Legacy Code

**Action Items:**

- [ ] Remove `app.py` (replaced by `src/main.py`)
- [ ] Remove `assistant.py` (replaced by `src/services/assistant/`)
- [ ] Remove `huggingface.py` (replaced by `src/services/llm/huggingface_provider.py`)
- [ ] Remove `googlecloud.py` (replaced by `src/services/llm/google_provider.py`)
- [ ] Keep `instances/` temporarily for backward compatibility
- [ ] Gradually migrate instance code to `src/usecases/`

---

## Implementation Order

1. **Phase 1 (Security)**: Critical - Must be done first
2. **Phase 2 (Core)**: Foundation for everything else
3. **Phase 3 (Domain)**: Define contracts
4. **Phase 4 (Infrastructure)**: Technical implementations
5. **Phase 5 (Services)**: Business logic
6. **Phase 6 (UseCases)**: Domain-specific logic
7. **Phase 7 (FastAPI)**: API migration
8. **Phase 8 (Testing)**: Quality assurance
9. **Phase 9 (Config)**: Deployment readiness
10. **Phase 10 (Cleanup)**: Remove legacy code

---

## Success Criteria

### Must Have

- [x] No `eval()` usage in codebase ✅ **ACHIEVED**
- [ ] All tests pass (>80% coverage) ⚠️ **PENDING** - No tests written yet
- [x] API documentation available at `/docs` ✅ **ACHIEVED** - FastAPI automatic docs at `/docs` and `/redoc`
- [x] Request validation working ✅ **ACHIEVED** - Pydantic schemas validate all requests
- [x] Backward compatible API responses ✅ **ACHIEVED** - Response format matches legacy Flask API
- [ ] No performance regression ⚠️ **PENDING** - Not yet measured

### Should Have

- [x] Type hints throughout (>90% coverage) ✅ **ACHIEVED** - Comprehensive type hints in services/domain
- [x] Comprehensive error handling ✅ **ACHIEVED** - Exception hierarchy and error handling implemented
- [x] Structured logging ✅ **ACHIEVED** - Logging configuration implemented
- [ ] Code duplication reduced by >50% ⚠️ **PENDING** - Use case refactoring needed
- [x] Clean architecture principles followed ✅ **ACHIEVED** - Layers properly separated

### Nice to Have

- [x] Async/await where beneficial ✅ **ACHIEVED** - LLM providers use async
- [ ] Metrics/monitoring ❌ **NOT STARTED**
- [ ] Performance improvements ❌ **NOT STARTED**
- [ ] Comprehensive documentation ❌ **NOT STARTED**

---

## Migration Strategy

### Parallel Development

1. Create new structure alongside old code
2. Implement incrementally
3. Test thoroughly before switching
4. Switch API when ready
5. Remove old code after verification

### Testing Strategy

1. Unit tests for each component
2. Integration tests for API endpoints
3. End-to-end tests for critical paths
4. Mock external dependencies (LLM providers)

---

## Risk Mitigation

### High Risks

1. **Security during migration**

   - Mitigation: Phase 1 must be completed first, before any other work

2. **Breaking changes**

   - Mitigation: Maintain API compatibility, gradual rollout

3. **Performance impact**

   - Mitigation: Use lazy loading, caching, benchmarking

### Medium Risks

4. **Learning curve**

   - Mitigation: Documentation, code reviews

5. **Time overrun**

   - Mitigation: Phased approach, prioritize security and core features

---

## Notes

- All file paths use `src/` prefix to avoid import conflicts
- Use Protocol for interfaces (Python 3.8+ feature)
- Pydantic v2 for validation and settings
- Type hints required for all public APIs
- Async/await can be added incrementally (start with sync, migrate later)
- Maintain backward compatibility during transition

### Implementation Notes

**Important**: The actual implementation differs slightly from the plan:

- LLM providers are implemented as `HuggingFaceProvider` and `GoogleGeminiProvider` (not `GoogleProvider`)
- Service layer includes additional services: `FunctionExecutorService`, `UseCaseRegistryService`
- Infrastructure includes factories for dependency injection
- Use case registry pattern implemented via `UseCaseRegistryService` instead of direct registry

**✅ RESOLVED - UseCase Enum Issue**:

- **Issue Fixed**: Added `from_string()` method to `UseCase` enum in `constants.py`
  - ✅ Method correctly maps "Energy Consumption" → `ENERGY` and "Heart Disease" → `HEART`
  - ✅ Also handles backend format ("energy", "heart") for backward compatibility
  - ✅ Includes case-insensitive matching for robustness
- **Code Updated**:
  - ✅ Updated `src/services/llm/huggingface_provider.py` to use `UseCase.from_string(usecase)`
  - ✅ Updated `src/services/llm/google_gemini_provider.py` to use `UseCase.from_string(usecase)`
- **Note**: Legacy `app.py` still exists but is no longer needed (will be removed in Phase 10)
- **Note**: Two `UseCase` enums still exist (in `config.py` and `constants.py`), but both work correctly now. Consolidation is low priority and can be done later if desired.

**✅ COMPLETE - Phase 6 UseCase Refactoring**:

- **All usecase classes created**:
  - ✅ `src/usecases/base/base_usecase.py` - Abstract base with lazy loading
  - ✅ `src/usecases/energy/energy_usecase.py` - Energy usecase implementation
  - ✅ `src/usecases/heart/heart_usecase.py` - Heart usecase implementation
- **Functions refactored**:
  - ✅ `src/usecases/energy/energy_functions.py` - All energy functions moved from `instances/energy/executive.py`
  - ✅ `src/usecases/heart/heart_functions.py` - All heart functions moved from `instances/heart/executive.py`
- **Configuration**:
  - ✅ `src/usecases/energy/energy_config.py` - Energy-specific configuration
  - ✅ `src/usecases/heart/heart_config.py` - Heart-specific configuration
- **Registry updated**:
  - ✅ `src/services/usecase/usecase_registry_service.py` - Now initializes and uses new usecase classes
  - ✅ System prompts moved into usecase classes
- **Lazy loading**: All models, datasets, and explainers load on demand via properties
- **Dependency injection**: Functions receive dependencies via constructor, no global state
- **Legacy files**: `instances/` directory still exists but is no longer used by new code (will be removed in Phase 10)

**✅ COMPLETE - Phase 7 FastAPI Migration**:

- **API Schemas** (`src/api/schemas.py`):
  - ✅ Created `AssistantRequest` with conversation, model, usecase (conversation_id removed - backend is stateless)
  - ✅ Created `AssistantResponseWrapper` to match legacy Flask API format (`{"assistantResponse": {...}}`)
  - ✅ Created `HealthResponse` for health check endpoint
  - ✅ Used existing `AssistantResponse` entity from domain layer
- **API Dependencies** (`src/api/dependencies.py`):
  - ✅ Implemented `validate_model()` function to convert model strings to Model enum
  - ✅ Implemented `validate_usecase()` function to convert usecase strings to UseCase enum
  - ✅ Created `get_assistant_service_dependency()` using service factory for dependency injection
  - ✅ Proper error handling with HTTPException for invalid inputs
- **API Routes** (`src/api/routes.py`):
  - ✅ Implemented `/ready` GET endpoint for health checks
  - ✅ Implemented `/getAssistantResponse` POST endpoint with full error handling
  - ✅ Passes full conversation history from frontend to `AssistantService.process_message()`
  - ✅ Validates and converts model/usecase strings to enums
  - ✅ Backend is stateless - no conversation tracking
  - ✅ Returns response in legacy format for backward compatibility
- **Main Application** (`src/main.py`):
  - ✅ FastAPI application with title, description, version
  - ✅ CORS middleware configured (currently allows all origins - should be configured for production)
  - ✅ Automatic API documentation at `/docs` and `/redoc`
  - ✅ Logging setup on startup
  - ✅ Startup and shutdown event handlers
  - ✅ Root endpoint with API information
- **Configuration Updates**:
  - ✅ `requirements.txt`: Added `fastapi>=0.104.0` and `uvicorn[standard]>=0.24.0`
  - ✅ `Dockerfile`: Changed CMD to use `uvicorn src.main:app` instead of gunicorn
  - ✅ Flask dependencies kept temporarily for backward compatibility
- **Conversation Management**:
  - ✅ Backend does not track conversation history (stateless architecture)
  - ✅ Frontend sends entire conversation history with each request
  - ✅ Conversation history embedded in system prompts as JSON (matching legacy behavior)
- **Backward Compatibility**: All responses match legacy Flask API format for seamless frontend integration

### To-dos

- [x] Phase 1: Implement safe function parser using AST to replace eval() - critical security fix ✅ **COMPLETE**
- [x] Phase 2: Set up core infrastructure (config, exceptions, constants, directory structure) ✅ **COMPLETE**
- [x] Phase 3: Create domain layer (interfaces/protocols, entities) ✅ **COMPLETE**
- [x] Phase 4: Implement infrastructure layer (lazy loaders for models, data, explainers) ✅ **COMPLETE**
- [x] Phase 5: Implement service layer (LLM providers, assistant service, function parser) ✅ **COMPLETE**
- [x] Phase 6: Refactor usecases (base class, energy/heart implementations, move functions) ✅ **COMPLETE**
- [x] Phase 7: Migrate to FastAPI (schemas, routes, dependencies, main app) ✅ **COMPLETE**
- [ ] Phase 8: Create testing infrastructure (unit tests, integration tests, fixtures) ❌ **NOT STARTED**
- [x] Phase 9: Update configuration files (requirements.txt, Dockerfile) ✅ **COMPLETE** - FastAPI dependencies added, Dockerfile updated
- [ ] Phase 10: Remove legacy code (app.py, assistant.py, old instance files) ❌ **NOT STARTED**