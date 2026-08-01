#!/usr/bin/env bash
set -e

# Deploys are triggered by .github/workflows/render-deploy.yml (Render
# auto-deploy is disabled; this file is in the workflow's paths filter).
pip install --upgrade pip

# insightface's metadata requires `opencv-python`, which needs system libGL.
# Render's build environment is read-only (no apt-get), so install insightface
# without deps and satisfy them via requirements.txt + opencv-python-headless.
pip install --no-deps insightface
pip install -r requirements.txt

# Pre-download insightface models so the first request isn't slow and the
# runtime doesn't re-download (+120MB) inside the 512MB memory cap. Keep them
# inside the project dir (repo root/.insightface) so they ship in the image;
# recognition_service.py loads them from the same relative path via `root=`.
python - <<'EOF'
import os
from insightface.app import FaceAnalysis
FaceAnalysis(
    name="buffalo_s",
    root=os.path.join(os.getcwd(), ".insightface"),
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
).prepare(ctx_id=-1, det_size=(640, 640))
EOF

# Runtime only needs detection + recognition. The extra buffalo_s models
# (2d106det, genderage, 1k3d68) are ignored by allowed_modules but their
# sessions are still constructed, and 1k3d68 alone spikes memory ~440MB
# (OOM on Render's 512MB free tier). Drop them so only det+rec ship.
rm -f .insightface/models/buffalo_s/1k3d68.onnx \
      .insightface/models/buffalo_s/2d106det.onnx \
      .insightface/models/buffalo_s/genderage.onnx
echo "trimmed buffalo_s models:" && ls -la .insightface/models/buffalo_s/
