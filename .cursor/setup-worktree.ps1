# .cursor/setup-worktree.ps1

$ErrorActionPreference = "Stop"

if (Test-Path "$env:ROOT_WORKTREE_PATH\.env") {
    Copy-Item "$env:ROOT_WORKTREE_PATH\.env" ".env"
}

uv sync
