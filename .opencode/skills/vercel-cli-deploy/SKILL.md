---
name: vercel-cli-deploy
description: Deploy a frontend (Vite/React) to Vercel from the CLI. Use when deploying a frontend to Vercel, linking a Vercel project, setting Vercel environment variables (VITE_*), or wiring a GitHub Actions deploy to Vercel (tokens, org/project ids, permissions).
---

# Deploy frontend to Vercel via CLI

Deploys a Vite/React frontend to Vercel and wires a GitHub Actions
auto-deploy. Covers project linking, build-time env vars, and the token/permission
gotchas that break CI deploys.

## 1. Link the project

```bash
cd frontend
npx vercel link        # creates project + frontend/.vercel/project.json
npx vercel whoami      # confirm account
npx vercel --prod      # optional: manual first deploy
```

`.vercel/` is written locally and should stay gitignored. `project.json`
contains `projectId` (`prj_...`) and `orgId` (`team_...`) — you need both for CI.
Vercel auto-detects Vite: build command `vite build`, output dir `dist`.

## 2. Build-time env vars

Vite bakes `VITE_*` vars in at build time — they are NOT read at runtime.

```bash
printf 'https://my-backend.onrender.com\n' | npx vercel env add VITE_API_URL production
```

Verify the built bundle references the right URL (no `localhost` leftovers):

```bash
curl -s https://<proj>.vercel.app/ | grep -oE '/assets/[^"]+\.js' \
  | while read a; do curl -s "https://<proj>.vercel.app$a"; done | grep -oE 'https://[a-zA-Z0-9.-]*onrender\.com|localhost:800[0-9]+'
```

## 3. Tokens

- `vercel tokens add` fails with `403 Cannot create tokens for this app` — token
  creation is blocked; do not rely on it.
- A manually-pasted token may be **invalid** — test it first:
  ```bash
  npx vercel whoami --token "$TOKEN"
  # "User not found" => invalid, do not use in CI
  ```
- The CLI's own auth token is valid and reusable for CI. On Linux it lives at
  `~/.local/share/com.vercel.cli/auth.json` (field `token`). Extract without
  echoing:
  ```bash
  gh secret set VERCEL_TOKEN --body "$(python3 -c "import json;print(json.load(open('$HOME/.local/share/com.vercel.cli/auth.json'))['token'])")"
  ```

## 4. GitHub Actions deploy

Secrets needed (values come from `frontend/.vercel/project.json` — run the
`node -p` from the `frontend` dir, not the repo root, or the secret is empty):

```bash
gh secret set VERCEL_TOKEN       --body "<valid token>"
gh secret set VERCEL_ORG_ID      --body "<orgId from project.json>"
gh secret set VERCEL_PROJECT_ID  --body "<projectId from project.json>"
```

Workflow `.github/workflows/vercel-deploy.yml`:

```yaml
name: Deploy Frontend to Vercel
on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'
      - '.github/workflows/vercel-deploy.yml'
  workflow_dispatch:
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      deployments: write
      statuses: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: ./frontend
          vercel-args: '--prod'
          github-token: ${{ secrets.GITHUB_TOKEN }}
          github-comment: false
```

## 5. Gotchas

- **`Resource not accessible by integration`** after a successful Vercel deploy:
  the action's post-deploy GitHub API call failed. Causes & fixes:
  - Missing `permissions` block → add the one above.
  - `github-comment: true` on a push (no PR) calls the issues-comments API and
    403s → set `github-comment: false` for push deploys.
  - The actual Vercel deployment still succeeds ("Ready in Ns") even when the
    workflow shows failure — check the alias output in the logs.
- **Node 20 deprecation warnings** from third-party actions are harmless.
- Deploy URL shows as `Production https://<proj>-<hash>-<org>.vercel.app` plus an
  aliased stable URL.

## 6. Verify

```bash
gh run watch <run-id> --exit-status
curl -s -o /dev/null -w "%{http_code}\n" https://<stable-alias>.vercel.app/
```
