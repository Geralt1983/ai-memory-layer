# Local Edit Agent (Codex-style direct edits)

## What this gives you
- Read and write files under the repo root
- Apply unified diffs safely (auto backup)
- Git status, branch, commit
- Auth via bearer token
- Path jail to repo root (no wandering)

## Quick start
1) npm install
2) echo "AUTH_TOKEN=changeme" > .env
3) npm run dev

By default it binds to 127.0.0.1:7373 (loopback only). To use from other devices, set HOST=0.0.0.0 and use a tunnel (ngrok, cloudflared).

## Example calls
# health
curl -s -H "Authorization: Bearer changeme" http://127.0.0.1:7373/health

# read a file
curl -s -H "Authorization: Bearer changeme" "http://127.0.0.1:7373/fs?path=src/app/page.tsx"

# write a file (create or overwrite)
curl -s -X POST -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"path":"src/foo.txt","content":"hello\n"}' http://127.0.0.1:7373/write

# apply a unified diff patch
curl -s -X POST -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  --data-binary @patch.json http://127.0.0.1:7373/patch

# git status
curl -s -H "Authorization: Bearer changeme" http://127.0.0.1:7373/git/status

# git commit all changes
curl -s -X POST -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"message":"feat: apply assistant patch"}' http://127.0.0.1:7373/git/commit

## patch.json format
{
  "patches": [
    {
      "path": "src/components/ChatUI.tsx",
      "unified": "diff --git a/src/components/ChatUI.tsx b/src/components/ChatUI.tsx\n--- a/src/components/ChatUI.tsx\n+++ b/src/components/ChatUI.tsx\n@@ -1,3 +1,4 @@\n+// example\n ..."
    }
  ]
}

## Notes
- The agent refuses paths outside the repo.
- Every write/patch creates a timestamped .bak file next to the target.
- If you want PRs, run this in a branch and push; or extend /git with your CI.