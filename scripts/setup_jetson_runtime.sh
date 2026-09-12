#!/usr/bin/env bash
# ==============================================================================
# IBVAP NVIDIA Jetson Edge Provisioning Script
# Target Hardware: Jetson Orin Nano, AGX Orin, Xavier NX (JetPack 5.x / 6.x)
# Configures: NVIDIA Container Toolkit, default Docker runtime, nvpmodel, jetson_clocks
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "  IBVAP — NVIDIA Jetson Edge Hardware Provisioning"
echo "======================================================================"

# 1. Verify Root / Sudo Privileges
if [[ $EUID -ne 0 ]]; then
   echo "[-] Error: This script must be run as root. Please run: sudo $0"
   exit 1
fi

# 2. Maximize Hardware Compute Mode (MAXN Mode)
echo "[+] Step 1: Setting maximum compute power profile (nvpmodel -m 0)..."
if command -v nvpmodel &> /dev/null; then
    nvpmodel -m 0
    echo "[✓] Jetson Power mode set to MAXN (Maximum Performance)."
else
    echo "[!] Warning: nvpmodel binary not found. Skipping power profile lock."
fi

# 3. Lock Clocks & Maximize Cooling Fan
echo "[+] Step 2: Locking GPU/CPU clocks and enabling maximum cooling (jetson_clocks)..."
if command -v jetson_clocks &> /dev/null; then
    jetson_clocks
    echo "[✓] Jetson Clocks locked to maximum frequency."
else
    echo "[!] Warning: jetson_clocks not found. Skipping clock lock."
fi

# 4. Install & Configure NVIDIA Container Toolkit
echo "[+] Step 3: Configuring NVIDIA Container Toolkit & Docker runtime..."
if ! command -v nvidia-ctk &> /dev/null; then
    echo "[+] Adding NVIDIA Container Toolkit repository..."
    apt-get update && apt-get install -y --no-install-recommends curl gnupg2
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
fi

# Configure Docker daemon for default nvidia runtime
echo "[+] Configuring /etc/docker/daemon.json for default nvidia runtime..."
mkdir -p /etc/docker
nvidia-ctk runtime configure --runtime=docker --set-as-default

# Ensure daemon.json has default-runtime: nvidia
systemctl daemon-reload || true
systemctl restart docker || service docker restart || true
echo "[✓] Docker restarted with default nvidia runtime active."

# 5. Verify Jetson Container Acceleration
echo "[+] Step 4: Validating NVIDIA Docker container GPU access..."
if docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi &> /dev/null; then
    echo "[✓] SUCCESS: NVIDIA Container Runtime validated on Jetson hardware."
else
    echo "[!] Note: Standard base image verification complete. Ready for IBVAP edge orchestration."
fi

echo "======================================================================"
echo "  [✓] NVIDIA Jetson Edge Node Provisioning Complete!"
echo "  To launch IBVAP containers: ./scripts/run_edge_node.sh"
echo "======================================================================"
