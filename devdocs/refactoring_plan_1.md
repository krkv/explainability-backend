# FastAPI Migration and Architecture Refactoring Plan

## Overview

This plan addresses critical security vulnerabilities, architectural issues, and migrates the application from Flask to FastAPI while introducing clean architecture principles.

## Critical Issues Identified

1. **Security**: Use of `eval()` in parsers allows arbitrary code execution
2. **Architecture**: Tight coupling, no dependency injection, global state
3. **API Layer**: Manual JSON parsing, no request validation
4. **Code Organization**: Duplication, missing abstractions, magic strings

## Phase 1: Security Fix - Replace eval() (HIGH PRIORITY)

### 1.1 Create Safe Function Parser

**File**: `src/services/parser/function_parser.py`

- Create a safe function execution system using a function registry
- Parse function call strings like `"count_all()"` or `"show_one(id=5)"`
- Validate function names against registry
- Extract and validate parameters
- Execute functions safely without eval()

**Implementation Strategy**:

- Use `ast.parse()` and `ast.literal_eval()` to safely parse function calls
- Create function registries per usecase instance
- Validate function exists in registry before execution
- Type-check and validate parameters

### 1.2 Update Energy Parser

**File**: `instances/energy/parser.py`

- Replace `eval(call)` with safe parser
- Import and use `FunctionParser` from new location
- Maintain backward compatibility with existing function signatures

### 1.3 Update Heart Parser

**File**: `instances/heart/parser.py`

- Replace `eval(call)` with safe parser (same approach as energy)
- Handle different return format (heart returns dict with "text" key)

## Phase 2: Architecture Foundation

### 2.1 Create Directory Structure

Create new directory structure:

```
src/
├── api/
│   ├── __init__.py
│   ├── routes.py
│   ├── schemas.py
│   └── dependencies.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   └── constants.py
├── domain/
│   ├── __init__.py
│   └── interfaces/
│       ├── __init__.py
│       ├── llm_provider.py
│       ├── usecase.py
│       └── parser.py
├── services/
│   ├── __init__.py
│   ├── assistant_service.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base_provider.py
│   │   ├── huggingface_provider.py
│   │   └── google_provider.py
│   └── parser/
│       ├── __init__.py
│       ├── function_parser.py
│       └── ast_parser.py
└── infrastructure/
    ├── __init__.py
    ├── model_loader.py
    └── data_loader.py
```

### 2.2 Core Configuration

**File**: `src/core/config.py`

- Use `pydantic-settings` for configuration
- Load environment variables
- Define model enum, usecase enum
- Centralize all configuration

### 2.3 Custom Exceptions

**File**: `src/core/exceptions.py`

- Define custom exception hierarchy:
  - `ExplainabilityException` (base)
  - `InvalidModelException`
  - `FunctionExecutionException`
  - `InvalidRequestException`
  - `UseCaseNotFoundException`

### 2.4 Constants

**File**: `src/core/constants.py`

- Replace magic strings with constants
- Model names enum
- Usecase names enum
- Path constants

## Phase 3: Domain Interfaces

### 3.1 LLM Provider Interface

**File**: `src/domain/interfaces/llm_provider.py`

- Abstract base class defining `generate_response(conversation, usecase)` method
- Type hints for conversation format

### 3.2 UseCase Interface

**File**: `src/domain/interfaces/usecase.py`

- Abstract base class for usecases
- Methods: `get_system_prompt()`, `get_parser()`, `get_functions()`
- Abstract away instance-specific logic

### 3.3 Parser Interface

**File**: `src/domain/interfaces/parser.py`

- Abstract base class for function parsers
- `parse_calls(function_calls: List[str]) -> str` method

## Phase 4: Service Layer Implementation

### 4.1 Base LLM Provider

**File**: `src/services/llm/base_provider.py`

- Abstract base class implementing LLM provider interface
- Common prompt formatting logic

### 4.2 HuggingFace Provider

**File**: `src/services/llm/huggingface_provider.py`

- Refactor `huggingface.py` into a class
- Inherit from `BaseLLMProvider`
- Dependency injection for client
- Load token from config

### 4.3 Google Provider

**File**: `src/services/llm/google_provider.py`

- Refactor `googlecloud.py` into a class
- Inherit from `BaseLLMProvider`
- Use existing Pydantic Response model
- Dependency injection for client

### 4.4 Function Parser

**File**: `src/services/parser/function_parser.py`

- Safe function execution engine
- Function registry per usecase
- Parameter validation
- Error handling with custom exceptions

### 4.5 AST Parser

**File**: `src/services/parser/ast_parser.py`

- Use Python AST module to parse function call strings
- Extract function name and arguments
- Validate syntax before execution

### 4.6 Assistant Service

**File**: `src/services/assistant_service.py`

- Orchestrates LLM calls and function parsing
- Replace logic from `assistant.py`
- Dependency injection for LLM provider, parser, usecase
- Better error handling

## Phase 5: Infrastructure Layer

### 5.1 Model Loader

**File**: `src/infrastructure/model_loader.py`

- Lazy loading with caching for models
- Load SHAP explainers on demand
- Context managers for resource cleanup

### 5.2 Data Loader

**File**: `src/infrastructure/data_loader.py`

- Lazy loading for datasets
- Caching mechanism
- Path resolution

## Phase 6: UseCase Refactoring

### 6.1 Base UseCase Class

**File**: `instances/base/base_usecase.py` (new)

- Abstract base class implementing UseCase interface
- Common functionality (dataset loading, model loading)
- Template methods for prompt generation

### 6.2 Energy UseCase

**File**: `instances/energy/usecase.py` (new)

- Create class wrapping executive functions
- Load model/dataset lazily or via factory
- Register functions in parser registry
- Implement UseCase interface

### 6.3 Heart UseCase

**File**: `instances/heart/usecase.py` (new)

- Same approach as Energy
- Handle different return format for parser

### 6.4 UseCase Factory

**File**: `src/core/usecase_factory.py` (new)

- Factory pattern to create usecase instances
- Registry of usecases
- Lazy initialization

## Phase 7: FastAPI Migration

### 7.1 Pydantic Schemas

**File**: `src/api/schemas.py`

- `Message` model (role, content)
- `AssistantRequest` model (conversation, model, usecase)
- `FunctionCall` model
- `AssistantResponse` model
- `AssistantResponseWrapper` model
- Use Literal types for models and usecases

### 7.2 API Routes

**File**: `src/api/routes.py`

- FastAPI app initialization
- `/ready` endpoint (GET)
- `/getAssistantResponse` endpoint (POST)
- Dependency injection for services
- Error handlers for custom exceptions

### 7.3 API Dependencies

**File**: `src/api/dependencies.py`

- Dependency injection setup
- Factory functions for LLM providers
- UseCase factory dependency
- Assistant service factory

### 7.4 Main Application Entry

**File**: `src/main.py` (new, replaces app.py)

- FastAPI app creation
- Include routers
- Configuration loading
- Startup/shutdown events

## Phase 8: Update Requirements and Configuration

### 8.1 Requirements Update

**File**: `requirements.txt`

- Add `fastapi>=0.104.0`
- Add `uvicorn[standard]>=0.24.0`
- Add `pydantic-settings>=2.0.0`
- Keep existing dependencies
- Update Flask removal note

### 8.2 Environment Configuration

**File**: `.env.example` (update if exists)

- Document all required environment variables
- Add FastAPI-specific configs if needed

### 8.3 Dockerfile Update

**File**: `Dockerfile`

- Update to use uvicorn instead of gunicorn
- Update entrypoint command

## Phase 9: Refactor Existing Files

### 9.1 Remove/Deprecate Old Files

- `app.py` → Remove after migration
- `assistant.py` → Functionality moved to services
- `huggingface.py` → Functionality moved to provider class
- `googlecloud.py` → Functionality moved to provider class

### 9.2 Update Instance Modules

- Refactor `instances/energy/executive.py`:
  - Extract to class if beneficial
  - Keep functions but make them part of UseCase class

- Refactor `instances/heart/executive.py`:
  - Same approach

- Update `instances/energy/prompt.py`:
  - Integrate into UseCase class
  - Remove global state

- Update `instances/heart/prompt.py`:
  - Same approach

- Update parsers:
  - Use new safe parser

## Phase 10: Testing Structure

### 10.1 Test Directory

Create `tests/` directory structure:

```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── test_function_parser.py
│   ├── test_llm_providers.py
│   └── test_assistant_service.py
├── integration/
│   ├── test_api_routes.py
│   └── test_usecases.py
└── fixtures/
    └── sample_data.py
```

### 10.2 Test Configuration

- Use pytest
- Mock LLM providers for fast tests
- Fixtures for usecases
- Test data fixtures

## Phase 11: Documentation

### 11.1 Update README

**File**: `readme.md`

- Update setup instructions for FastAPI
- Update run commands (uvicorn instead of flask)
- Document new architecture
- API documentation link

### 11.2 API Documentation

- FastAPI auto-generates at `/docs` and `/redoc`
- Document in README

## Implementation Order

1. **Phase 1** (Security): Must be done first
2. **Phase 2** (Foundation): Core infrastructure
3. **Phase 3-4** (Interfaces & Services): Business logic
4. **Phase 5-6** (Infrastructure & UseCases): Domain layer
5. **Phase 7** (FastAPI): API migration
6. **Phase 8-9** (Cleanup): Remove old code
7. **Phase 10-11** (Testing & Docs): Quality assurance

## Migration Strategy

### Parallel Development Approach

1. Create new structure alongside old code
2. Implement new services incrementally
3. Switch API endpoint when ready
4. Remove old code after verification

### Backward Compatibility

- Maintain same API response format
- Keep function signatures compatible where possible
- Gradual migration of internal code

## Risks and Mitigation

### Risks

1. Breaking existing functionality during migration

   - **Mitigation**: Comprehensive testing, gradual migration

2. Performance impact from refactoring

   - **Mitigation**: Use caching, lazy loading

3. Learning curve for team

   - **Mitigation**: Document architecture decisions, code reviews

### Success Criteria

- All tests pass
- No eval() usage in codebase
- API documentation available at /docs
- Request validation working
- Error handling improved
- Code duplication reduced
- Global state eliminated