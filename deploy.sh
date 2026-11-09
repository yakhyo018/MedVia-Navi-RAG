#!/bin/bash
# PRODUCTION: pull main, rebuild the image if anything changed, recreate what changed.
# Qdrant's data and the downloaded models live in volumes and survive this.
set -e
cd "$(dirname "$0")"
git checkout main
git pull origin main
docker compose -f docker-compose.prod.yml up -d --build
