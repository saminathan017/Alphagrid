#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  AlphaGrid Cloud Bootstrap — EC2 g4dn.xlarge (NVIDIA T4)
#  Run this once after SSH-ing into a fresh EC2 instance.
#
#  USAGE:
#    1. Fill in S3_BUCKET and GITHUB_REPO below
#    2. SSH into your EC2 instance
#    3. curl -s <this_file_url> | bash
#        OR: scp this file to EC2 and run: bash cloud_bootstrap.sh
#
#  After it runs, training starts in a screen session.
#  Detach with Ctrl+A D. Reattach with: screen -r alphagrid
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── CONFIG — edit these ───────────────────────────────────────────
S3_BUCKET="your-s3-bucket-name"           # e.g. "alphagrid-models-2024"
GITHUB_REPO="https://github.com/YOUR_USERNAME/alphagrid-final.git"
LOOKBACK=3650                             # 10 years
# SYMBOLS="AAPL,MSFT,NVDA"               # uncomment to train specific symbols only
# ─────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════"
echo "  AlphaGrid Cloud Setup — $(date)"
echo "══════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo yum update -y -q 2>/dev/null || sudo apt-get update -qq
sudo yum install -y git screen htop 2>/dev/null || sudo apt-get install -y git screen htop -qq

# ── 2. Clone repository ───────────────────────────────────────────
echo "[2/6] Cloning repository..."
if [ -d "alphagrid-final" ]; then
    cd alphagrid-final && git pull
else
    git clone "$GITHUB_REPO" alphagrid-final
    cd alphagrid-final
fi

# ── 3. Python environment ─────────────────────────────────────────
echo "[3/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# PyTorch with CUDA 12.1 (matches Deep Learning AMI)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install --quiet \
    lightgbm scikit-learn yfinance pandas numpy loguru scipy \
    pandas-ta boto3 imbalanced-learn fastapi uvicorn

echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU:             $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

# ── 4. Create directories ─────────────────────────────────────────
echo "[4/6] Creating output directories..."
mkdir -p logs models

# ── 5. Verify S3 access ───────────────────────────────────────────
echo "[5/6] Verifying S3 access..."
if aws s3 ls "s3://$S3_BUCKET" > /dev/null 2>&1; then
    echo "  S3 bucket accessible: s3://$S3_BUCKET"
else
    echo "  WARNING: Cannot access s3://$S3_BUCKET"
    echo "  Either create the bucket or configure AWS credentials:"
    echo "    aws configure"
    echo "  Then re-run this script."
fi

# ── 6. Launch training in screen session ──────────────────────────
echo "[6/6] Launching training..."

TRAIN_CMD="source venv/bin/activate && python scripts/train_models.py --lookback $LOOKBACK --s3-bucket $S3_BUCKET 2>&1 | tee logs/cloud_training.log"

# Add symbol filter if set
if [ -n "${SYMBOLS:-}" ]; then
    TRAIN_CMD="source venv/bin/activate && python scripts/train_models.py --symbols $SYMBOLS --lookback $LOOKBACK --s3-bucket $S3_BUCKET 2>&1 | tee logs/cloud_training.log"
fi

screen -dmS alphagrid bash -c "
    cd $(pwd)
    $TRAIN_CMD
    echo '━━━ Training complete — syncing logs to S3 ━━━'
    aws s3 sync logs/ s3://$S3_BUCKET/alphagrid/logs/ --quiet
    echo 'Done. Check S3 bucket: s3://$S3_BUCKET/alphagrid/'
"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Training started in background screen session."
echo ""
echo "  Monitor:   screen -r alphagrid"
echo "  Detach:    Ctrl+A then D"
echo "  Tail log:  tail -f logs/cloud_training.log"
echo "  S3 sync:   aws s3 sync models/ s3://$S3_BUCKET/alphagrid/models/"
echo "══════════════════════════════════════════════════"
