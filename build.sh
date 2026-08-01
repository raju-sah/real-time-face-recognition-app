#!/usr/bin/env bash
set -e

# OpenCV needs system libraries that are not in Render's Python image.
apt-get update -y
apt-get install -y --no-install-recommends libgl1 libglib2.0-0

pip install --upgrade pip
pip install -r requirements.txt

# Pre-download insightface models so the first request isn't slow.
# Build artifacts are baked into the Render image, so the ~120MB
# buffalo_s pack ships with the release instead of downloading at runtime.
python - <<'EOF'
from insightface.app import FaceAnalysis
FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
).prepare(ctx_id=-1, det_size=(640, 640))
EOF
