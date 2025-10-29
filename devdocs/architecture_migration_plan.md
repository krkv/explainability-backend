# Clean Architecture & FastAPI Migration Plan

## Executive Summary

This document provides a comprehensive analysis of the current XAI LLM Chat Backend application and outlines a detailed migration plan to:
1. Implement clean architecture principles
2. Migrate from Flask to FastAPI
3. Address critical security vulnerabilities
4. Improve code maintainability, testability, and scalability

**Estimated Migration Time**: 4-6 weeks (depending on team size)
**Priority**: High (security vulnerabilities present)

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Architectural Issues & Technical Debt](#architectural-issues--technical-debt)
3. [Clean Architecture Principles](#clean-architecture-principles)
4. [FastAPI Migration Benefits](#fastapi-migration-benefits)
5. [Proposed Architecture](#proposed-architecture)
6. [Implementation Plan](#implementation-plan)
7. [Risk Assessment & Mitigation](#risk-assessment--mitigation)
8. [Success Criteria](#success-criteria)

---

## Current Architecture Analysis

### Application Overview

The application is a Flask-based REST API that provides an LLM-powered assistant for explaining machine learning models. It supports:
- Multiple ML model instances (heart disease classification, energy consumption regression)
- Multiple LLM providers (HuggingFace, Google Gemini)
- Function-calling pattern where LLM generates function call strings that are executed

### Current Structure

```
explainability-backend/
├── app.py                    # Flask application entry point
├── assistant.py              # Core assistant logic (orchestration)
├── huggingface.py            # HuggingFace LLM integration
├── googlecloud.py            # Google Gemini LLM integration
├── instances/
│   ├── energy/
│   │   ├── executive.py      # 433 lines - all energy functions
│   │   ├── parser.py         # Uses eval() - SECURITY RISK
│   │   ├── prompt.py         # Prompt generation
│   │   └── functions.json    # Function definitions
│   └── heart/
│       ├── executive.py      # 464 lines - all heart functions
│       ├── parser.py         # Uses eval() - SECURITY RISK
│       ├── prompt.py         # Prompt generation
│       └── functions.json    # Function definitions
└── requirements.txt
```

### Data Flow

1. **Request** → Flask route (`/getAssistantResponse`)
2. **Parse** → Extract conversation, model, usecase from JSON
3. **LLM Call** → Generate response with function calls
4. **Parse Response** → Extract function_calls and freeform_response
5. **Execute Functions** → Use `eval()` to execute function calls
6. **Return** → Combine parsed results with freeform response

---

## Architectural Issues & Technical Debt

### 🔴 Critical Issues

#### 1. **Security Vulnerability: eval() Usage**
- **Location**: `instances/energy/parser.py`, `instances/heart/parser.py`
- **Risk**: Arbitrary code execution vulnerability
- **Impact**: Attackers could execute any Python code
- **Current Code**:
  ```python
  def parse_calls(calls):
      for call in calls:
          result = eval(call)  # DANGEROUS!
  ```

#### 2. **Global State & Module-Level Initialization**
- **Location**: `instances/*/executive.py`
- **Issue**: Datasets, models, explainers loaded at import time
- **Impact**: 
  - High memory usage even when not needed
  - Cannot test without loading large datasets
  - Circular import risks
  - Difficult to mock in tests

#### 3. **No Request Validation**
- **Location**: `app.py`
- **Issue**: Manual JSON parsing, no schema validation
- **Impact**: Runtime errors, no type safety

### 🟡 High Priority Issues

#### 4. **Tight Coupling**
- All components directly import each other
- No dependency injection
- Hard to swap implementations
- Difficult to test in isolation

#### 5. **Code Duplication**
- Similar logic in energy and heart instances
- Duplicated filtering logic (group functions)
- Similar prompt generation patterns
- Repeated error handling

#### 6. **Magic Strings**
- Model names: `"Llama 3.3 70B Instruct"`, `"Gemini 2.0 Flash"`
- Usecase names: `"Heart Disease"` vs `"heart"`
- String-based switching in multiple places

#### 7. **Monolithic Functions**
- `executive.py` files are 400+ lines
- Functions do too much (data loading, computation, formatting)
- Mixed concerns (business logic + HTML formatting)

#### 8. **No Error Handling Strategy**
- Inconsistent error handling
- No custom exceptions
- Generic error messages
- No error logging structure

#### 9. **Mixed Return Types**
- Some functions return strings (energy)
- Some return dicts with "text" key (heart)
- Inconsistent formatting patterns

#### 10. **No Type Hints**
- Limited use of type annotations
- Difficult to understand interfaces
- No static type checking possible

### 🟢 Medium Priority Issues

#### 11. **Hard-coded Paths**
- `INSTANCE_PATH = 'instances/energy/'` in multiple places
- No configuration management
- Difficult to test with different paths

#### 12. **Limited Logging**
- Basic print statements
- No structured logging
- No request/response logging

#### 13. **No Testing Infrastructure**
- No visible test files
- Cannot verify changes safely

#### 14. **Documentation Gaps**
- Minimal API documentation
- No architecture documentation
- Missing type information

---

## Clean Architecture Principles

### Why Clean Architecture?

Clean Architecture provides:
- **Independence**: Framework, UI, database, external services
- **Testability**: Business logic testable without dependencies
- **Flexibility**: Easy to swap implementations
- **Maintainability**: Clear boundaries and responsibilities

### Layers

```
┌─────────────────────────────────────────┐
│         Presentation Layer (API)        │  FastAPI routes, schemas
├─────────────────────────────────────────┤
│         Application Layer (Services)    │  Use cases, orchestration
├─────────────────────────────────────────┤
│         Domain Layer (Business Logic)   │  Entities, interfaces
├─────────────────────────────────────────┤
│      Infrastructure Layer (Details)     │  DB, file I/O, external APIs
└─────────────────────────────────────────┘
```

### Dependency Rule

**Dependencies point inward**: Outer layers depend on inner layers, not vice versa.

---

## FastAPI Migration Benefits

### Advantages Over Flask

1. **Automatic API Documentation**
   - OpenAPI/Swagger at `/docs`
   - Redoc at `/redoc`
   - Always up-to-date

2. **Type Safety & Validation**
   - Pydantic models for request/response validation
   - Automatic JSON schema generation
   - Runtime type checking

3. **Modern Python Features**
   - Async/await support
   - Better dependency injection
   - Type hints throughout

4. **Performance**
   - Built on Starlette (ASGI)
   - Better performance for concurrent requests
   - Native async support

5. **Developer Experience**
   - Better IDE support
   - Clearer error messages
   - Less boilerplate

---

## Proposed Architecture

### Directory Structure

```
src/
├── main.py                          # FastAPI app entry point
├── api/
│   ├── __init__.py
│   ├── routes.py                    # API endpoints
│   ├── schemas.py                   # Pydantic models
│   ├── dependencies.py              # Dependency injection
│   └── middleware.py                # Request/response middleware
├── core/
│   ├── __init__.py
│   ├── config.py                    # Settings (pydantic-settings)
│   ├── exceptions.py                # Custom exceptions
│   ├── constants.py                 # Constants & enums
│   └── logging.py                   # Logging configuration
├── domain/
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── message.py               # Conversation message entity
│   │   └── function_call.py         # Function call entity
│   └── interfaces/
│       ├── __init__.py
│       ├── llm_provider.py          # LLM provider interface
│       ├── function_executor.py     # Function execution interface
│       └── usecase_registry.py      # UseCase registry interface
├── services/
│   ├── __init__.py
│   ├── assistant/
│   │   ├── __init__.py
│   │   └── assistant_service.py     # Main orchestration service
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_provider.py         # Base LLM provider
│   │   ├── huggingface_provider.py
│   │   └── google_provider.py
│   └── parser/
│       ├── __init__.py
│       ├── function_parser.py       # Safe function parser
│       └── ast_parser.py            # AST-based parsing
├── infrastructure/
│   ├── __init__.py
│   ├── model_loader.py              # Model loading & caching
│   ├── data_loader.py               # Dataset loading & caching
│   └── explainer_factory.py         # SHAP/DiCE explainer creation
└── usecases/
    ├── __init__.py
    ├── base/
    │   ├── __init__.py
    │   ├── base_usecase.py          # Base usecase class
    │   └── base_functions.py        # Common function implementations
    ├── energy/
    │   ├── __init__.py
    │   ├── energy_usecase.py        # Energy-specific logic
    │   ├── energy_functions.py      # Energy functions (refactored)
    │   └── energy_config.py         # Energy configuration
    └── heart/
        ├── __init__.py
        ├── heart_usecase.py
        ├── heart_functions.py
        └── heart_config.py

instances/                          # Legacy - to be removed after migration
tests/
├── __init__.py
├── conftest.py                      # Pytest fixtures
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

### Component Responsibilities

#### API Layer (`src/api/`)
- **Routes**: Define HTTP endpoints
- **Schemas**: Request/response validation (Pydantic)
- **Dependencies**: Dependency injection setup
- **Middleware**: Logging, error handling, CORS

#### Services Layer (`src/services/`)
- **Assistant Service**: Orchestrates LLM calls and function execution
- **LLM Providers**: Abstract LLM interactions
- **Function Parser**: Safely parses and executes function calls

#### Domain Layer (`src/domain/`)
- **Entities**: Core business objects
- **Interfaces**: Abstract contracts (protocols/ABCs)

#### Infrastructure Layer (`src/infrastructure/`)
- **Model Loader**: Lazy loading, caching of ML models
- **Data Loader**: Dataset loading with caching
- **Explainer Factory**: Creates SHAP/DiCE explainers

#### UseCase Layer (`src/usecases/`)
- **Base UseCase**: Common functionality
- **Instance-specific**: Energy, Heart implementations

---

## Implementation Plan

### Phase 1: Security Fix (CRITICAL - Week 1)

**Priority**: 🔴 **MUST DO FIRST**

#### 1.1 Create Safe Function Parser

**File**: `src/services/parser/function_parser.py`

Replace `eval()` with safe AST-based parsing:

```python
import ast
from typing import Callable, Dict, Any, List
from src.core.exceptions import FunctionExecutionException

class FunctionParser:
    """Safely parses and executes function calls without eval()."""
    
    def __init__(self, function_registry: Dict[str, Callable]):
        self.function_registry = function_registry
    
    def parse_and_execute(self, function_call_str: str) -> Any:
        """Parse function call string and execute safely."""
        try:
            # Parse into AST
            parsed = ast.parse(function_call_str, mode='eval')
            
            # Validate it's a function call
            if not isinstance(parsed.body, ast.Call):
                raise FunctionExecutionException(
                    f"Invalid function call format: {function_call_str}"
                )
            
            func_name = parsed.body.func.id
            if func_name not in self.function_registry:
                raise FunctionExecutionException(
                    f"Unknown function: {func_name}"
                )
            
            # Extract arguments
            args = self._extract_args(parsed.body)
            
            # Execute function
            func = self.function_registry[func_name]
            return func(**args)
            
        except SyntaxError as e:
            raise FunctionExecutionException(f"Invalid syntax: {e}")
        except Exception as e:
            raise FunctionExecutionException(f"Execution error: {e}")
    
    def _extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract keyword arguments from AST call node."""
        args = {}
        for keyword in call_node.keywords:
            key = keyword.arg
            value = ast.literal_eval(keyword.value)  # Safe literal evaluation
            args[key] = value
        return args
```

#### 1.2 Create Function Registry System

**File**: `src/services/parser/function_registry.py`

```python
from typing import Dict, Callable, Any
from functools import lru_cache

class FunctionRegistry:
    """Registry of available functions per usecase."""
    
    def __init__(self):
        self._registries: Dict[str, Dict[str, Callable]] = {}
    
    def register_usecase(self, usecase: str, functions: Dict[str, Callable]):
        """Register functions for a usecase."""
        self._registries[usecase] = functions
    
    def get_registry(self, usecase: str) -> Dict[str, Callable]:
        """Get function registry for usecase."""
        if usecase not in self._registries:
            raise ValueError(f"Unknown usecase: {usecase}")
        return self._registries[usecase]
```

#### 1.3 Update Parsers

**File**: `instances/energy/parser.py` (temporary - will be replaced)

```python
from src.services.parser.function_parser import FunctionParser
from src.services.parser.function_registry import FunctionRegistry

# Create registry with energy functions
registry = FunctionRegistry()
registry.register_usecase("energy", {
    'count_all': count_all,
    'show_one': show_one,
    # ... all energy functions
})

def parse_calls(calls, usecase="energy"):
    parser = FunctionParser(registry.get_registry(usecase))
    results = []
    for call in calls:
        result = parser.parse_and_execute(call)
        results.append(result)
    return '\n'.join(results)
```

**Deliverables**:
- ✅ Safe function parser implementation
- ✅ No `eval()` usage in codebase
- ✅ Function registry system
- ✅ Updated parsers use safe parser

---

### Phase 2: Core Infrastructure (Week 1-2)

#### 2.1 Create Directory Structure

```bash
mkdir -p src/{api,core,domain/{entities,interfaces},services/{assistant,llm,parser},infrastructure,usecases/{base,energy,heart}}
mkdir -p tests/{unit,integration,fixtures}
```

#### 2.2 Core Configuration

**File**: `src/core/config.py`

```python
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class Model(str, Enum):
    LLAMA_3_3_70B = "Llama-3.3-70B-Instruct"
    GEMINI_2_0_FLASH = "Gemini-2.0-Flash"
    
    @classmethod
    def from_string(cls, value: str):
        """Map legacy strings to enum."""
        mapping = {
            "Llama 3.3 70B Instruct": cls.LLAMA_3_3_70B,
            "Gemini 2.0 Flash": cls.GEMINI_2_0_FLASH,
        }
        return mapping.get(value, cls.LLAMA_3_3_70B)

class UseCase(str, Enum):
    ENERGY = "energy"
    HEART = "heart"
    
    @classmethod
    def from_string(cls, value: str):
        """Map legacy strings to enum."""
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
    
    # Model Paths
    energy_model_path: str = "instances/energy/model/custom_gp_model.pkl"
    heart_model_path: str = "instances/heart/model/best_model_3_DecisionTreeClassifier.pkl"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

#### 2.3 Custom Exceptions

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

#### 2.4 Constants

**File**: `src/core/constants.py`

```python
from enum import Enum

class Model(str, Enum):
    LLAMA_3_3_70B = "Llama-3.3-70B-Instruct"
    GEMINI_2_0_FLASH = "Gemini-2.0-Flash"

class UseCase(str, Enum):
    ENERGY = "energy"
    HEART = "heart"
```

**Deliverables**:
- ✅ Directory structure created
- ✅ Configuration management
- ✅ Exception hierarchy
- ✅ Constants/enums defined

---

### Phase 3: Domain Layer (Week 2)

#### 3.1 Interfaces

**File**: `src/domain/interfaces/llm_provider.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        """Generate LLM response with function calls."""
        pass
```

**File**: `src/domain/interfaces/function_executor.py`

```python
from abc import ABC, abstractmethod
from typing import List, Any

class FunctionExecutor(ABC):
    """Interface for function execution."""
    
    @abstractmethod
    def execute_calls(self, function_calls: List[str]) -> str:
        """Execute function calls and return results."""
        pass
```

**File**: `src/domain/entities/message.py`

```python
from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
```

**Deliverables**:
- ✅ Domain interfaces defined
- ✅ Entity models created
- ✅ Type safety foundation

---

### Phase 4: Service Layer (Week 2-3)

#### 4.1 LLM Providers

**File**: `src/services/llm/base_provider.py`

```python
from src.domain.interfaces.llm_provider import LLMProvider
from typing import List, Dict
from abc import ABC

class BaseLLMProvider(LLMProvider, ABC):
    """Base class for LLM providers with common functionality."""
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for LLM."""
        # Common formatting logic
        pass
```

**File**: `src/services/llm/huggingface_provider.py`

```python
from huggingface_hub import InferenceClient
from src.services.llm.base_provider import BaseLLMProvider
from src.core.config import settings
from src.core.exceptions import LLMProviderException
from typing import List, Dict

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider implementation."""
    
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
        """Generate response using HuggingFace."""
        try:
            # Get system prompt based on usecase
            system_prompt = self._get_system_prompt(conversation, usecase)
            user_input = conversation[-1]['content']
            
            llama_prompt = self._format_prompt(system_prompt, user_input)
            response = self.client.text_generation(llama_prompt).strip()
            return response
        except Exception as e:
            raise LLMProviderException(f"HuggingFace error: {e}")
```

**File**: `src/services/llm/google_provider.py`

```python
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from src.services.llm.base_provider import BaseLLMProvider
from src.core.config import settings
from typing import List, Dict

class Response(BaseModel):
    function_calls: list[str]
    freeform_response: str

class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self, client=None):
        self.client = client or genai.Client(
            http_options=genai.types.HttpOptions(api_version="v1"),
            vertexai=True,
            project=settings.google_project,
            location=settings.google_location,
        )
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> str:
        """Generate response using Google Gemini."""
        try:
            system_prompt = self._get_system_prompt(conversation, usecase)
            user_input = conversation[-1]['content']
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=user_input,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type='application/json',
                    response_schema=Response
                )
            )
            return response.text
        except Exception as e:
            raise LLMProviderException(f"Google error: {e}")
```

#### 4.2 Function Parser (Enhanced from Phase 1)

**File**: `src/services/parser/function_parser.py`

- Enhanced version with better error handling
- Support for both string and dict return types
- Logging integration

#### 4.3 Assistant Service

**File**: `src/services/assistant/assistant_service.py`

```python
from src.domain.interfaces.llm_provider import LLMProvider
from src.services.parser.function_parser import FunctionParser
from src.core.exceptions import FunctionExecutionException
import json
from typing import List, Dict

class AssistantService:
    """Orchestrates LLM calls and function execution."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        function_parser: FunctionParser,
    ):
        self.llm_provider = llm_provider
        self.function_parser = function_parser
    
    async def generate_response(
        self,
        conversation: List[Dict[str, str]],
        usecase: str
    ) -> Dict:
        """Generate assistant response with function execution."""
        # Get LLM response
        llm_response = await self.llm_provider.generate_response(
            conversation, usecase
        )
        
        # Parse LLM response
        try:
            response_data = json.loads(llm_response)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from LLM")
        
        function_calls = response_data.get("function_calls", [])
        freeform_response = response_data.get("freeform_response", "")
        
        # Execute functions if present
        parse_result = ""
        if function_calls:
            try:
                parse_result = self.function_parser.parse_calls(function_calls)
            except FunctionExecutionException as e:
                # Log error but continue with freeform response
                parse_result = f"Error executing functions: {e}"
        
        return {
            "function_calls": function_calls,
            "freeform_response": freeform_response,
            "parse": parse_result if parse_result else None,
        }
```

**Deliverables**:
- ✅ LLM provider implementations
- ✅ Enhanced function parser
- ✅ Assistant service orchestration
- ✅ Dependency injection ready

---

### Phase 5: Infrastructure Layer (Week 3)

#### 5.1 Model Loader

**File**: `src/infrastructure/model_loader.py`

```python
from functools import lru_cache
import pickle
import joblib
from pathlib import Path
from src.core.config import settings
from src.core.exceptions import ModelLoadException
from typing import Any

class ModelLoader:
    """Lazy loading and caching of ML models."""
    
    @lru_cache(maxsize=10)
    def load_model(self, model_path: str) -> Any:
        """Load model with caching."""
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadException(f"Model not found: {model_path}")
        
        try:
            if path.suffix == '.pkl':
                # Try pickle first
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Try joblib
                return joblib.load(path)
        except Exception as e:
            raise ModelLoadException(f"Failed to load model: {e}")
```

#### 5.2 Data Loader

**File**: `src/infrastructure/data_loader.py`

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
        path = Path(dataset_path)
        if not path.exists():
            raise DataLoadException(f"Dataset not found: {dataset_path}")
        
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise DataLoadException(f"Failed to load dataset: {e}")
```

**Deliverables**:
- ✅ Lazy loading infrastructure
- ✅ Caching mechanism
- ✅ Error handling

---

### Phase 6: UseCase Refactoring (Week 3-4)

#### 6.1 Base UseCase

**File**: `src/usecases/base/base_usecase.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Callable, List
from src.infrastructure.model_loader import ModelLoader
from src.infrastructure.data_loader import DataLoader

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
    
    def __init__(self, model_loader, data_loader):
        super().__init__(model_loader, data_loader)
        self.energy_functions = EnergyFunctions(
            model=self.model,
            dataset=self.dataset
        )
    
    def _load_model(self):
        return self.model_loader.load_model(self.model_path)
    
    def _load_dataset(self):
        return self.data_loader.load_dataset(self.dataset_path)
    
    def get_functions(self) -> Dict[str, Callable]:
        """Get all energy functions."""
        return {
            'count_all': self.energy_functions.count_all,
            'show_one': self.energy_functions.show_one,
            # ... all energy functions
        }
    
    def get_system_prompt(self, conversation: List[Dict]) -> str:
        """Generate energy system prompt."""
        # Move prompt generation logic here
        pass
```

**File**: `src/usecases/energy/energy_functions.py`

- Refactor `executive.py` functions into a class
- Remove global state
- Accept model/dataset as constructor parameters
- Return consistent types (strings or dicts with "text" key)

**Deliverables**:
- ✅ Base usecase class
- ✅ Energy usecase implementation
- ✅ Heart usecase implementation
- ✅ Functions refactored

---

### Phase 7: FastAPI Migration (Week 4)

#### 7.1 API Schemas

**File**: `src/api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from src.core.constants import Model, UseCase

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class AssistantRequest(BaseModel):
    conversation: List[Message]
    model: Model
    usecase: UseCase

class AssistantResponse(BaseModel):
    function_calls: List[str]
    freeform_response: str
    parse: Optional[str] = None
```

#### 7.2 API Routes

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
            usecase=request.usecase.value,
        )
        return AssistantResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 7.3 Dependencies

**File**: `src/api/dependencies.py`

```python
from functools import lru_cache
from src.core.config import settings, Model, UseCase
from src.services.llm.huggingface_provider import HuggingFaceProvider
from src.services.llm.google_provider import GoogleProvider
from src.services.parser.function_parser import FunctionParser
from src.services.assistant.assistant_service import AssistantService
from src.domain.interfaces.llm_provider import LLMProvider

def get_llm_provider(model: Model) -> LLMProvider:
    """Factory for LLM providers."""
    if model == Model.LLAMA_3_3_70B:
        return HuggingFaceProvider()
    elif model == Model.GEMINI_2_0_FLASH:
        return GoogleProvider()
    else:
        raise ValueError(f"Unknown model: {model}")

def get_function_parser(usecase: UseCase):
    """Factory for function parser."""
    # Get usecase instance and register functions
    usecase_instance = get_usecase_instance(usecase)
    functions = usecase_instance.get_functions()
    
    from src.services.parser.function_registry import FunctionRegistry
    registry = FunctionRegistry()
    registry.register_usecase(usecase.value, functions)
    
    parser = FunctionParser(registry.get_registry(usecase.value))
    return parser

def get_assistant_service(
    request: AssistantRequest,
    llm_provider: LLMProvider = Depends(get_llm_provider),
    function_parser = Depends(get_function_parser),
) -> AssistantService:
    """Factory for assistant service."""
    return AssistantService(
        llm_provider=llm_provider,
        function_parser=function_parser,
    )
```

#### 7.4 Main Application

**File**: `src/main.py`

```python
from fastapi import FastAPI
from src.api.routes import router
from src.core.logging import setup_logging

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

**Deliverables**:
- ✅ FastAPI application
- ✅ Request/response validation
- ✅ Automatic API docs
- ✅ Dependency injection

---

### Phase 8: Testing Infrastructure (Week 4-5)

#### 8.1 Test Structure

Create comprehensive test suite:
- Unit tests for parsers, services
- Integration tests for API endpoints
- Mock LLM providers
- Fixtures for test data

**Deliverables**:
- ✅ Test infrastructure
- ✅ Unit tests (>80% coverage)
- ✅ Integration tests
- ✅ CI/CD ready

---

### Phase 9: Cleanup & Migration (Week 5-6)

#### 9.1 Update Requirements

**File**: `requirements.txt`

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
# ... existing dependencies
# Remove Flask
```

#### 9.2 Update Dockerfile

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

#### 9.3 Remove Legacy Code

- Remove `app.py`
- Remove `assistant.py`
- Remove `huggingface.py`, `googlecloud.py`
- Keep `instances/` temporarily for backward compatibility
- Create migration script if needed

#### 9.4 Update Documentation

- Update README with FastAPI instructions
- Document new architecture
- API documentation (auto-generated)

**Deliverables**:
- ✅ Requirements updated
- ✅ Dockerfile updated
- ✅ Legacy code removed
- ✅ Documentation updated

---

## Risk Assessment & Mitigation

### High Risks

1. **Breaking Changes During Migration**
   - **Risk**: Existing clients break
   - **Mitigation**: Maintain API compatibility, version endpoints, gradual rollout

2. **Performance Regression**
   - **Risk**: Refactoring introduces performance issues
   - **Mitigation**: Benchmark before/after, use caching, lazy loading

3. **Security Issues During Transition**
   - **Risk**: `eval()` still accessible during migration
   - **Mitigation**: Phase 1 (security fix) must be completed first, before any other work

### Medium Risks

4. **Team Learning Curve**
   - **Risk**: Team unfamiliar with new architecture
   - **Mitigation**: Documentation, code reviews, pair programming

5. **Migration Time Overrun**
   - **Risk**: Takes longer than estimated
   - **Mitigation**: Phased approach, parallel development

---

## Success Criteria

### Must Have ✅

- [ ] No `eval()` usage in codebase
- [ ] All tests pass (>80% coverage)
- [ ] API documentation at `/docs`
- [ ] Request validation working
- [ ] Backward compatible API (during transition)
- [ ] No performance regression

### Should Have ✨

- [ ] Type hints throughout
- [ ] Comprehensive error handling
- [ ] Structured logging
- [ ] Code duplication reduced by >50%
- [ ] Clean architecture principles followed

### Nice to Have 🌟

- [ ] Async/await implemented
- [ ] Metrics/monitoring added
- [ ] Performance improvements
- [ ] Comprehensive documentation

---

## Migration Timeline

```
Week 1: Phase 1 (Security) + Phase 2 (Core Infrastructure)
Week 2: Phase 3 (Domain) + Phase 4 (Services)
Week 3: Phase 5 (Infrastructure) + Phase 6 (UseCases)
Week 4: Phase 7 (FastAPI) + Start Phase 8 (Testing)
Week 5: Complete Phase 8 (Testing) + Phase 9 (Cleanup)
Week 6: Final testing, documentation, deployment
```

---

## Next Steps

1. **Review this plan** with team
2. **Approve Phase 1** (security fix) for immediate execution
3. **Set up development branch** for migration work
4. **Create project board** with phases as epics
5. **Begin Phase 1** implementation

---

## Appendix: Code Quality Metrics

### Current State
- Cyclomatic Complexity: High (large functions)
- Code Duplication: ~40% (estimate)
- Test Coverage: ~0%
- Type Coverage: ~10%

### Target State
- Cyclomatic Complexity: <10 per function
- Code Duplication: <10%
- Test Coverage: >80%
- Type Coverage: >90%

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: Architecture Review  
**Status**: Ready for Review

