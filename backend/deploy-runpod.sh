#!/bin/bash
# ============================================================
# RunPod 5090 Deploy Script — Color Separator Backend
# ============================================================
#
# HOW TO USE:
#
# 1. Go to runpod.io → Deploy → GPU Pod
#    Template: "RunPod Pytorch 2.4.0" (has CUDA + PyTorch pre-installed)
#    GPU: RTX 5090 (32GB)
#    Disk: 30GB container + 10GB volume
#    Expose ports: 8001 (HTTP)
#
# 2. Connect via SSH (RunPod gives you the command):
#    ssh root@<pod-ssh-address> -p <port> -i ~/.ssh/id_ed25519
#
# 3. Run this script on the pod:
#    curl -sSL https://raw.githubusercontent.com/ReidSurmeier/color-separator/main/backend/deploy-runpod.sh | bash
#
#    Or clone + run:
#    git clone https://github.com/ReidSurmeier/color-separator.git
#    cd color-separator/backend
#    bash deploy-runpod.sh
#
# 4. Copy the pod's proxy URL and set it on your local machine:
#    echo "BACKEND_URL=https://<POD_ID>-8001.proxy.runpod.net" >> ~/sites/woodblock-tools/.env.local
#    Then rebuild + restart frontend.
#
# ============================================================

set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  Color Separator — RunPod GPU Deploy"
echo "═══════════════════════════════════════════"

# ── Check GPU ──
echo ""
echo "▸ Checking GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "✗ No GPU detected. Are you on a GPU pod?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Clone or update repo ──
REPO_DIR="/workspace/color-separator"
if [ -d "$REPO_DIR" ]; then
    echo "▸ Updating existing repo..."
    cd "$REPO_DIR"
    git pull --ff-only
else
    echo "▸ Cloning repo..."
    git clone https://github.com/ReidSurmeier/color-separator.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

cd backend

# ── Install Python deps ──
echo "▸ Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q slowapi

# Additional GPU deps (may already be in the PyTorch template)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
pip install -q ultralytics realesrgan basicsr 2>/dev/null || echo "Some GPU deps need manual install"

# ── Download model weights ──
echo "▸ Downloading SAM 2.1 Large model..."
if [ ! -f "sam2.1_l.pt" ]; then
    wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt
fi

echo "▸ Downloading RealESRGAN weights..."
mkdir -p weights
if [ ! -f "weights/RealESRGAN_x2plus.pth" ]; then
    wget -q --show-progress -O weights/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
fi

# ── Verify GPU access from Python ──
echo ""
echo "▸ Verifying CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ── Start the backend ──
echo ""
echo "═══════════════════════════════════════════"
echo "  Starting backend (GPU_MODE=1)"
echo "═══════════════════════════════════════════"
echo ""
echo "  Backend will be available at:"
echo "  → http://0.0.0.0:8001"
echo "  → RunPod proxy: https://<POD_ID>-8001.proxy.runpod.net"
echo ""
echo "  To connect from your local machine:"
echo "  1. Copy the RunPod proxy URL from the pod dashboard"
echo "  2. Add to ~/sites/woodblock-tools/.env.local:"
echo "     BACKEND_URL=https://<POD_ID>-8001.proxy.runpod.net"
echo "  3. Rebuild: cd ~/sites/woodblock-tools && npm run build"
echo "  4. Copy standalone: cp -r .next/static .next/standalone/.next/static && cp -r public .next/standalone/public"
echo "  5. Restart: systemctl --user restart woodblock-frontend.service"
echo ""
echo "  Press Ctrl+C to stop."
echo "═══════════════════════════════════════════"
echo ""

export GPU_MODE=1

# Auth — set these env vars before running, or pass as arguments
if [ -z "$BACKEND_API_KEY" ]; then
    echo "⚠ WARNING: BACKEND_API_KEY not set — backend has NO auth!"
    echo "  Set it: export BACKEND_API_KEY=your_secret_key"
fi
if [ -z "$GPU_AUTH_PASSWORD" ]; then
    echo "⚠ WARNING: GPU_AUTH_PASSWORD not set — frontend has no password gate."
    echo "  Set it: export GPU_AUTH_PASSWORD=your_password"
fi

exec uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
