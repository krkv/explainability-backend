#!/usr/bin/env bash

# Build and push script for explainability-backend Docker image
# Usage: ./build-and-push.sh [tag]
# If no tag is provided, uses 'latest'
# If a version tag is provided (e.g., v1.0.0), it will also be tagged as 'latest'

set -euo pipefail

PROJECT_ID="explainability-assistant"
LOCATION="us-central1"
REPOSITORY="explainability-backend"
IMAGE_NAME="explainability-backend"

# Get tag from argument or default to 'latest'
TAG=${1:-latest}

# Full image name
FULL_IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Configure Docker to use gcloud as credential helper (if not already done)
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev --quiet

# Build the Docker image
echo "Building image..."
docker build --platform linux/amd64 -t ${FULL_IMAGE_NAME} .

# If a version tag was provided (not 'latest'), also tag as 'latest'
if [ "${TAG}" != "latest" ]; then
    LATEST_IMAGE_NAME="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
    echo "Also tagging as latest: ${LATEST_IMAGE_NAME}"
    docker tag ${FULL_IMAGE_NAME} ${LATEST_IMAGE_NAME}
fi

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push ${FULL_IMAGE_NAME}

# Push latest tag if it was created
if [ "${TAG}" != "latest" ]; then
    echo "Pushing latest tag..."
    docker push ${LATEST_IMAGE_NAME}
fi

echo "Successfully built and pushed: ${FULL_IMAGE_NAME}"
if [ "${TAG}" != "latest" ]; then
    echo "Also tagged and pushed as: ${LATEST_IMAGE_NAME}"
fi
echo ""
echo "To deploy to Cloud Run, run:"
echo "  gcloud run deploy ${IMAGE_NAME} \\"
echo "    --image ${FULL_IMAGE_NAME} \\"
echo "    --region ${LOCATION} \\"
echo "    --allow-unauthenticated \\"
echo "    --set-env-vars HF_TOKEN=\"your_token_here\",OPENAI_API_KEY=\"your_openai_key_here\",LANGFUSE_PUBLIC_KEY=\"...\",LANGFUSE_SECRET_KEY=\"...\",LANGFUSE_BASE_URL=\"...\",LANGFUSE_TRACING_ENVIRONMENT=\"production\" \\"
echo "    --port 8080"
