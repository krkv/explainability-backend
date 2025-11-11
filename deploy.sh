#!/bin/bash

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

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please set it in your .env file or export it."
    exit 1
fi

echo "🚀 Deploying ${IMAGE_NAME} with image: ${FULL_IMAGE_NAME}"

gcloud run deploy ${IMAGE_NAME} \
--image=${FULL_IMAGE_NAME} \
--min-instances=0 \
--set-env-vars=HF_TOKEN="${HF_TOKEN}" \
--region=us-central1 \
--project=${PROJECT_ID} \
&& gcloud run services update-traffic ${IMAGE_NAME} --to-latest