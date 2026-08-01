#!/usr/bin/env bash
set -e

pip install --upgrade pip

# insightface's metadata requires `opencv-python`, which needs system libGL.
# Render's build environment is read-only (no apt-get), so install insightface
# without deps and satisfy them via requirements.txt + opencv-python-headless.
pip install --no-deps insightface
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
