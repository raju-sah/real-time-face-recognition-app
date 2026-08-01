---
name: render-cli-deploy
description: Deploy a Python/FastAPI backend to Render from the CLI. Use when deploying a backend to Render, creating a Render service, configuring Render auto-deploy/deploy hooks, or diagnosing Render free-tier build/runtime issues (RAM limits, OOM, read-only build env, .python-version, ONNX/insightface model memory).
---

# Deploy backend to Render via CLI

Deploys a Python backend (FastAPI/uvicorn) to Render's free tier using the
official `render` CLI. Covers service creation, path-filtered auto-deploy via
deploy hooks + GitHub Actions, and the free-tier gotchas that cause build or
runtime failures.

## 1. Install & authenticate

```bash
# CLI is NOT an npm package. Install via the official script:
curl -L https://render.com/render-cli/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

render login              # OAuth through browser
render workspace list
render workspace set <workspace-id>   # e.g. tea-...
render whoami
```

The CLI stores a **short-lived session key** in `~/.render/cli.yaml` (`api.key`,
has `expires_at`). Do NOT reuse it as a permanent CI secret — it expires.

## 2. Create the service

```bash
render services create \
  --name my-backend \
  --repo https://github.com/USER/REPO.git \
  --branch main \
  --plan free \
  --region singapore \
  --build-command "bash build.sh" \
  --start-command 'uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT' \
  --health-check-path "/"
```

Notes:
- `$PORT` must stay literal in the start command; quote it so the shell does
  not expand it.
- Point the build command at a committed `build.sh`, not inline pip (see §5).

## 3. Inspect state

```bash
render services get <service-id> -o json    # list response; filter for your id
render deploys list <service-id> -o json    # top-level dicts (NOT d.deploy nesting)
render logs --resources <service-id> -o json
render services update <service-id> --auto-deploy=false
```

`render services get` may return an array — extract the entry whose
`service.id` matches your service.

## 4. Path-filtered auto-deploy (skip backend rebuilds on frontend-only pushes)

Render auto-deploy is ON by default for any push to the branch and has **no
path filters**. For a monorepo with a separate frontend deploy, disable it and
drive deploys from a GitHub Action:

1. Disable auto-deploy:
   ```bash
   render services update <service-id> --auto-deploy=false
   ```
2. In the Render dashboard (Service → Settings → Deploy Hook) copy the hook URL:
   `https://api.render.com/deploy/srv-<id>?key=<key>` — it is dashboard-only,
   no public API to create/read it.
3. Set it as a GitHub secret:
   ```bash
   gh secret set RENDER_DEPLOY_HOOK --body "https://api.render.com/deploy/srv-<id>?key=<key>"
   ```
4. Add `.github/workflows/render-deploy.yml` with a paths filter, so only
   backend-relevant files trigger a deploy:

   ```yaml
   name: Deploy Backend to Render
   on:
     push:
       branches: [main]
       paths:
         - 'backend/**'
         - 'build.sh'
         - 'render.yaml'
         - 'requirements.txt'
         - '.python-version'
     workflow_dispatch:
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - name: Trigger Render deploy hook
           run: curl -sf -X POST "${{ secrets.RENDER_DEPLOY_HOOK }}"
   ```

   Deploy hooks always build the current default-branch HEAD.

## 5. Free-tier gotchas

- **512MB RAM.** Runtime OOM kills the process. Measure with RSS probes
  (`/proc/self/status` VmRSS) around model/session imports — find and eliminate
  the spike, not the steady state.
- **Read-only build env — no apt-get.** Anything needing system libraries
  (e.g. `opencv-python` → libGL) fails. Use `opencv-python-headless`.
- **Python version.** Render defaults to the newest Python (e.g. 3.14); pin with
  a committed `.python-version` (e.g. `3.12.3`).
- **ONNX/insightface memory.** `FaceAnalysis` constructs a session for every
  `.onnx` file in the model dir BEFORE applying `allowed_modules`. A single
  large landmark model (`1k3d68.onnx`, 143MB) spiked ~440MB and OOM'd. Fix:
  keep only needed models in the dir and/or trim in `build.sh`. Force
  single-threaded low-memory sessions:

  ```python
  original = onnxruntime.InferenceSession.__init__
  def patched(self, path_or_bytes, sess_options=None, **kwargs):
      if sess_options is None:
          o = onnxruntime.SessionOptions()
          o.intra_op_num_threads = 1
          o.inter_op_num_threads = 1
          o.enable_mem_pattern = False
          sess_options = o
      return original(self, path_or_bytes, sess_options, **kwargs)
  onnxruntime.InferenceSession.__init__ = patched
  ```

  Set `OMP_NUM_THREADS=1` in render.yaml env too.
- **Pre-download models at build.** Download large model packs in `build.sh`
  into a repo-relative dir and load via `root=...`, so the first request is
  fast and runtime never re-downloads inside the RAM cap. `.python-version`,
  `requirements.txt`, `build.sh`, `render.yaml` must all be committed.

## 6. Verify

```bash
curl -s https://<service>.onrender.com/ -o /dev/null -w "%{http_code}\n"
render deploys list <service-id> -o json | head
```
