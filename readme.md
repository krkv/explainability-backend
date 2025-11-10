# Explainability Assistant Backend

A FastAPI-based backend service that provides an API for LLM-powered assistant requests with explainable AI capabilities. The service integrates with Large Language Models (LLMs) to provide intelligent explanations and analysis of machine learning models through natural language conversations.

## 🚀 Features

- **LLM Integration**: Support for multiple LLM providers (Hugging Face, Google Gemini)
- **Explainable AI**: Provides explanations for ML model predictions using SHAP, DICE, and other XAI techniques
- **Multiple Use Cases**: 
  - **Energy Consumption**: Analyze and explain energy consumption predictions
  - **Heart Disease**: Analyze and explain heart disease prediction models
- **Function Calling**: LLM can execute specialized functions for model analysis, predictions, and explanations
- **RESTful API**: Clean, well-documented FastAPI endpoints
- **Production Ready**: Optimized Docker container for Google Cloud Run deployment
- **Comprehensive Logging**: Structured logging for debugging and monitoring

## 🛠️ Tech Stack

- **Framework**: FastAPI 0.104+
- **ASGI Server**: Uvicorn with standard extensions
- **Python**: 3.12
- **ML Libraries**: 
  - scikit-learn
  - SHAP (SHapley Additive exPlanations)
  - DICE-ML (Diverse Counterfactual Explanations)
  - gplearn
- **LLM Providers**:
  - Hugging Face Hub
  - Google Generative AI (Gemini)
- **Data Processing**: pandas, numpy
- **Validation**: Pydantic 2.0+
- **Testing**: pytest, pytest-asyncio, pytest-cov

## 📋 Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- (Optional) Docker for containerized deployment
- (Optional) Google Cloud SDK for Cloud Run deployment

## 🔧 Installation

### Local Development Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd explainability-backend
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create environment file**:
   Create a `.env` file in the root directory:
   ```bash
   # LLM Configuration
   HF_TOKEN="your_huggingface_token_here"  # Required for Hugging Face models
   
   # Google Cloud Configuration (for Gemini)
   GOOGLE_PROJECT="explainability-app"
   GOOGLE_LOCATION="europe-north1"
   
   # Logging (optional)
   LOG_LEVEL="INFO"
   ```

5. **Verify installation**:
   ```bash
   python -c "from src.main import app; print('Installation successful!')"
   ```

## ⚙️ Configuration

The application uses environment variables for configuration. Key settings:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `HF_TOKEN` | Hugging Face API token | Yes (for HF models) | - |
| `GOOGLE_PROJECT` | Google Cloud project ID | No | `explainability-app` |
| `GOOGLE_LOCATION` | Google Cloud region | No | `europe-north1` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `PORT` | Server port (Cloud Run sets this) | No | `8080` |

Configuration is managed through `src/core/config.py` using Pydantic Settings.

## 🏃 Running the Application

### Development Mode

Start the development server with auto-reload:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at:
- **API**: http://localhost:8080
- **Interactive Docs (Swagger)**: http://localhost:8080/docs
- **Alternative Docs (ReDoc)**: http://localhost:8080/redoc

### Production Mode

For production, use uvicorn with workers:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers 4
```

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t explainability-backend:latest .
```

### Run Locally with Docker

```bash
docker run -p 8080:8080 \
  -e HF_TOKEN="your_token_here" \
  -e GOOGLE_PROJECT="your-project" \
  explainability-backend:latest
```

### Google Cloud Run Deployment

1. **Create an Artifact Registry repository** (if it doesn't exist):
   ```bash
   gcloud artifacts repositories create explainability-backend \
     --repository-format=docker \
     --location=europe-north1 \
     --description="Docker repository for Explainability Assistant Backend"
   ```

2. **Configure Docker to use gcloud as a credential helper**:
   ```bash
   gcloud auth configure-docker europe-north1-docker.pkg.dev
   ```

3. **Build and push to Artifact Registry**:
   ```bash
   # Set your project ID and location
   export PROJECT_ID="your-project-id"
   export LOCATION="europe-north1"
   export REPOSITORY="explainability-backend"
   export IMAGE_NAME="explainability-backend"
   
   # Build and push using Cloud Build
   gcloud builds submit --tag ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest
   
   # Or build locally and push
   docker build -t ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest .
   docker push ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest
   ```

4. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy explainability-backend \
     --image ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest \
     --platform managed \
     --region europe-north1 \
     --allow-unauthenticated \
     --set-env-vars HF_TOKEN="your_token_here" \
     --port 8080
   ```

The Dockerfile is optimized for Cloud Run with:
- Multi-stage build for smaller image size
- Non-root user for security
- Health check endpoint
- Dynamic PORT environment variable support

## 📡 API Documentation

### Base URL

- **Local**: `http://localhost:8080`
- **Production**: Your Cloud Run service URL

### Endpoints

#### Health Check

```http
GET /ready
```

Returns the health status of the service.

**Response**:
```json
{
  "status": "OK"
}
```

#### Root Endpoint

```http
GET /
```

Returns API information.

**Response**:
```json
{
  "name": "Explainability Assistant Backend",
  "version": "2.0.0",
  "docs": "/docs",
  "health": "/ready"
}
```

#### Get Assistant Response

```http
POST /getAssistantResponse
```

Generate an LLM assistant response with function execution capabilities.

**Request Body**:
```json
{
  "conversation": [
    {
      "role": "user",
      "content": "What is the prediction for this patient?"
    }
  ],
  "model": "Llama-3.3-70B-Instruct",
  "usecase": "Heart Disease"
}
```

**Response**:
```json
{
  "assistantResponse": {
    "functionCalls": [...],
    "freeformResponse": "...",
    "parse": "..."
  }
}
```

**Supported Models**:
- `Llama-3.3-70B-Instruct` (Hugging Face)
- `Gemini-2.0-Flash` (Google)

**Supported Use Cases**:
- `Energy Consumption` or `energy`
- `Heart Disease` or `heart`

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

These provide interactive documentation where you can test endpoints directly.

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_specific.py

# Run with verbose output
pytest -v
```

### Test Structure

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for API endpoints
- `tests/fixtures/` - Test fixtures and mock data

## 📁 Project Structure

```
explainability-backend/
├── src/
│   ├── api/              # API routes, schemas, dependencies
│   ├── core/             # Configuration, constants, logging, exceptions
│   ├── domain/           # Domain entities and interfaces
│   ├── infrastructure/  # Infrastructure layer (loaders, caching)
│   ├── services/         # Business logic services
│   ├── usecases/         # Use case implementations (energy, heart)
│   └── main.py           # FastAPI application entry point
├── instances/            # ML models, datasets, and configurations
│   ├── energy/          # Energy consumption use case data
│   └── heart/           # Heart disease use case data
├── tests/               # Test suite
├── Dockerfile           # Docker configuration for deployment
├── .dockerignore        # Files to exclude from Docker build
├── requirements.txt     # Python dependencies
└── readme.md           # This file
```

## 🔍 Architecture

The application follows a clean architecture pattern:

- **API Layer**: FastAPI routes and request/response schemas
- **Service Layer**: Business logic and orchestration
- **Domain Layer**: Entities and interfaces
- **Infrastructure Layer**: External integrations (model loading, data loading, caching)
- **Use Cases**: Domain-specific implementations

Key design patterns:
- Dependency Injection
- Factory Pattern
- Lazy Loading
- Caching

## 🔐 Security Considerations

- The application runs as a non-root user in Docker
- CORS is configured (adjust `allow_origins` for production)
- Environment variables for sensitive data (tokens, API keys)
- Input validation using Pydantic schemas

## 📝 Logging

Logging is configured through `src/core/logging_config.py`. Logs include:
- Request/response information
- Model loading events
- Function execution details
- Error traces

Log level can be configured via `LOG_LEVEL` environment variable.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure virtual environment is activated
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **Model Loading Errors**:
   - Verify `instances/` directory contains required model files
   - Check file paths in configuration

3. **LLM API Errors**:
   - Verify API tokens are set correctly in `.env`
   - Check network connectivity
   - Verify API quotas/limits

4. **Port Already in Use**:
   - Change the port: `uvicorn src.main:app --port 8081`
   - Or kill the process using port 8080

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG uvicorn src.main:app --reload
```

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Update documentation as needed
6. Submit a pull request

## 📄 License

MIT

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- SHAP and DICE-ML for explainable AI capabilities
- Hugging Face and Google for LLM APIs

---

For more information, visit the interactive API documentation at `/docs` when the server is running.
