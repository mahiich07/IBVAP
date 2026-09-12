#!/usr/bin/env bash
# ==============================================================================
# IBVAP Single-Command Edge Orchestration Launcher
# Detects host architecture (Jetson vs. Standard), builds, and launches all containers.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "  IBVAP — Launching Edge Surveillance Node"
echo "======================================================================"

# 1. Check or Generate .env Configuration
if [[ ! -f .env ]]; then
    echo "[+] Creating .env from .env.example..."
    cp .env.example .env
fi

# 2. Detect NVIDIA Jetson Edge Hardware
IS_JETSON=false
if [[ -f /etc/nv_tegra_release ]] || uname -r | grep -iqE "(tegra|nvidia)"; then
    IS_JETSON=true
    echo "[✓] Hardware Detection: NVIDIA Jetson Edge Processor Identified."
else
    echo "[i] Hardware Detection: Standard Host Architecture."
fi

# 3. Choose Compose Profile
if [[ "$IS_JETSON" == "true" ]]; then
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.jetson.yml"
    echo "[+] Launching with Jetson hardware acceleration override..."
else
    COMPOSE_CMD="docker compose -f docker-compose.yml"
    echo "[+] Launching with standard edge container profile..."
fi

# 4. Build and Start Multi-Service Containers
echo "[+] Orchestrating services: [Frontend, API Gateway, Edge Vision Worker, DB Volume]..."
$COMPOSE_CMD up -d --build

# 5. Wait for Backend Health Probe
echo "[+] Waiting for backend service readiness..."
RETRIES=15
READY=false

for ((i=1;i<=RETRIES;i++)); do
    if curl -s -f http://localhost:8000/health &> /dev/null; then
        READY=true
        break
    fi
    echo "    - Waiting for API healthcheck ($i/$RETRIES)..."
    sleep 2
done

if [[ "$READY" == "true" ]]; then
    echo "======================================================================"
    echo "  [✓] IBVAP Edge Services Operational!"
    echo "  ------------------------------------------------------------------"
    echo "  Command Center UI:        http://localhost:3000"
    echo "  REST API Documentation:   http://localhost:8000/docs"
    echo "  Real-Time WebSocket:      ws://localhost:8000/ws/stream"
    echo "======================================================================"
else
    echo "[!] Warning: Services started, but healthcheck did not respond in time."
    echo "    Run 'docker compose logs' to inspect service telemetry."
fi
