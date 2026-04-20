#!/bin/bash

set -e

# Configuration (matching build-and-push.sh)
PROJECT_ID="explainability-app"
LOCATION="europe-north1"
REPOSITORY="explainability-backend"
IMAGE_NAME="explainability-backend"
IMAGE_TAG="latest"

# Full image name with latest tag
FULL_IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop automatically exporting
fi

LANGFUSE_TRACING_ENVIRONMENT=${LANGFUSE_TRACING_ENVIRONMENT:-production}

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please set it in your .env file or export it."
    exit 1
fi

# Check if Langfuse variables are set
if [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
    echo "Error: LANGFUSE_PUBLIC_KEY is not set. Please set it in your .env file or export it."
    exit 1
fi

if [ -z "$LANGFUSE_SECRET_KEY" ]; then
    echo "Error: LANGFUSE_SECRET_KEY is not set. Please set it in your .env file or export it."
    exit 1
fi

if [ -z "$LANGFUSE_BASE_URL" ]; then
    echo "Error: LANGFUSE_BASE_URL is not set. Please set it in your .env file or export it."
    exit 1
fi

echo "🚀 Deploying ${IMAGE_NAME} with image: ${FULL_IMAGE_NAME}"

gcloud run deploy ${IMAGE_NAME} \
--image=${FULL_IMAGE_NAME} \
--min-instances=0 \
--allow-unauthenticated \
--set-env-vars=HF_TOKEN="${HF_TOKEN}",LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY}",LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY}",LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL}",LANGFUSE_TRACING_ENVIRONMENT="${LANGFUSE_TRACING_ENVIRONMENT}" \
--region=${LOCATION} \
--project=${PROJECT_ID}
