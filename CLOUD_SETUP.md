AlphaGrid — AWS Cloud Training Setup
═════════════════════════════════════

Total cost for 150 symbols × 10 years: ~$0.50–$1.50
Time: ~2–4 hours on g4dn.xlarge (NVIDIA T4 GPU)


STEP 1 — One-time AWS setup (do this from your Mac)
══════════════════════════════════════════════════════

1a. Install AWS CLI if you haven't:
    brew install awscli
    aws configure       # enter your Access Key ID, Secret, region (e.g. us-east-1)

1b. Create S3 bucket (pick a unique name):
    aws s3 mb s3://alphagrid-models-YOUR-NAME --region us-east-1

1c. Push your code to GitHub:
    cd /Users/saminathanadaikkappan/Documents/alphagrid-final
    git init
    git add .
    git commit -m "alphagrid v6"
    git remote add origin https://github.com/YOUR_USERNAME/alphagrid-final.git
    git push -u origin main


STEP 2 — Launch EC2 Spot Instance
══════════════════════════════════

Option A: AWS Console (easiest)
    1. Go to EC2 → Launch Instance
    2. Name: alphagrid-training
    3. AMI: search "Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2)"
       (It already has CUDA 12.x + Python 3.11 — saves 30 min of setup)
    4. Instance type: g4dn.xlarge  (T4 GPU, $0.16/hr spot, 16GB VRAM)
    5. Key pair: create or use existing (you need this to SSH in)
    6. Storage: 50 GB (gp3)
    7. Under "Advanced" → Request Spot Instance → check the box
    8. Launch

Option B: AWS CLI (one command)
    aws ec2 run-instances \
      --image-id ami-0f9b3b6de7e0e3bc4 \
      --instance-type g4dn.xlarge \
      --key-name YOUR_KEY_PAIR_NAME \
      --instance-market-options '{"MarketType":"spot"}' \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=alphagrid-training}]'

    Note: AMI ID varies by region. Find it at:
    https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami.html


STEP 3 — SSH In and Run Bootstrap
══════════════════════════════════

3a. Get your instance's public IP from EC2 Console or:
    aws ec2 describe-instances --filters "Name=tag:Name,Values=alphagrid-training" \
      --query "Reservations[*].Instances[*].PublicIpAddress" --output text

3b. SSH in:
    ssh -i ~/.ssh/YOUR_KEY.pem ec2-user@YOUR_PUBLIC_IP

3c. Edit bootstrap script with your settings, then run it:
    # On EC2:
    curl -o setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/alphagrid-final/main/scripts/cloud_bootstrap.sh
    # OR upload it:
    # scp -i ~/.ssh/YOUR_KEY.pem scripts/cloud_bootstrap.sh ec2-user@YOUR_IP:~/

    # Edit the two config lines at the top of setup.sh:
    nano setup.sh
    #   S3_BUCKET="alphagrid-models-YOUR-NAME"
    #   GITHUB_REPO="https://github.com/YOUR_USERNAME/alphagrid-final.git"

    bash setup.sh


STEP 4 — Monitor Training
══════════════════════════

# Reattach to training session:
screen -r alphagrid

# Tail live log:
tail -f logs/cloud_training.log

# Check GPU usage:
watch -n 2 nvidia-smi

# Check what's been uploaded to S3 so far:
aws s3 ls s3://alphagrid-models-YOUR-NAME/alphagrid/models/ --recursive | wc -l

# Detach from screen without stopping training:
Ctrl+A then D


STEP 5 — Download Your Trained Models
══════════════════════════════════════

From your Mac, after training completes:

    # Download all models:
    aws s3 sync s3://alphagrid-models-YOUR-NAME/alphagrid/models/ models/

    # Download training logs:
    aws s3 sync s3://alphagrid-models-YOUR-NAME/alphagrid/logs/ logs/


STEP 6 — Terminate Instance
════════════════════════════

IMPORTANT: Always terminate when done or you keep paying.

    aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID

    Or from the EC2 console: select instance → Instance State → Terminate.


INSTANCE OPTIONS AND COST
══════════════════════════

    Instance        GPU              VRAM    Spot/hr   150 symbols   Total cost
    ─────────────────────────────────────────────────────────────────────────────
    g4dn.xlarge     NVIDIA T4        16 GB   ~$0.16    ~3-4 hrs      ~$0.50-0.65
    g4dn.2xlarge    NVIDIA T4        16 GB   ~$0.28    ~2-3 hrs      ~$0.55-0.85
    g5.xlarge       NVIDIA A10G      24 GB   ~$0.50    ~1.5-2 hrs    ~$0.75-1.00
    g5.2xlarge      NVIDIA A10G      24 GB   ~$0.75    ~1-1.5 hrs    ~$0.75-1.15
    c5.4xlarge      CPU only (16c)   -       ~$0.17    ~8-12 hrs     ~$1.35-2.00

    Recommendation: g4dn.xlarge — best price/performance for this workload.
    The T4 handles 16GB model+data fine for single-symbol training.


IF SPOT INSTANCE IS INTERRUPTED
════════════════════════════════

No problem. Models are uploaded to S3 after each symbol completes.
To resume from where you left off:

    1. Check which symbols already have models in S3:
       aws s3 ls s3://YOUR-BUCKET/alphagrid/models/ | grep lstm | sed 's/.*models\///' | sed 's/_lstm.*//'

    2. Launch a new spot instance (same steps as above)

    3. Download already-trained models first:
       aws s3 sync s3://YOUR-BUCKET/alphagrid/models/ models/

    4. Run training only on remaining symbols:
       python scripts/train_models.py --symbols REMAINING,SYMBOLS,HERE --lookback 3650 --s3-bucket YOUR-BUCKET


AFTER TRAINING — RUN THE DASHBOARD
════════════════════════════════════

Once models are downloaded to your Mac:

    cd /Users/saminathanadaikkappan/Documents/alphagrid-final
    venv/bin/uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
    open http://localhost:8080/dashboard
