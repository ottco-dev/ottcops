"""Git update check helpers."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional, Sequence

from .config import BASE_DIR, UPSTREAM_REPO_BRANCH, UPSTREAM_REPO_URL

logger = logging.getLogger("ottcouture.app")


def _run_git_command(args: Sequence[str]) -> tuple[int, str, str]:
    """Execute a git command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            args,
            cwd=str(BASE_DIR),
            capture_output=True,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 1, "", "git executable not available"


def _get_local_commit_hash() -> Optional[str]:
    code, stdout, stderr = _run_git_command(["git", "rev-parse", "HEAD"])
    if code == 0:
        return stdout
    logger.debug("Could not determine local commit: %s", stderr)
    return None


def _get_remote_commit_hash() -> Optional[str]:
    code, stdout, stderr = _run_git_command(["git", "ls-remote", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH])
    if code == 0 and stdout:
        return stdout.split()[0]
    logger.debug("Could not determine remote commit: %s", stderr)
    return None


def _prompt_user_for_update(remote_hash: str) -> bool:
    """Ask the operator whether an update should be pulled."""
    prompt = f"A new version ({remote_hash[:7]}) is available. Update now? [y/N]: "
    try:
        response = input(prompt)
    except EOFError:
        logger.info("No interactive input available – skipping update.")
        return False
    return response.strip().lower() in {"y", "yes", "j", "ja"}


def _perform_git_update() -> None:
    logger.info("Starting git pull from %s (%s)...", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH)
    code, stdout, stderr = _run_git_command(["git", "pull", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH])
    if code != 0:
        logger.error("git pull failed: %s", stderr or stdout)
    else:
        logger.info("Repository updated: %s", stdout)


def ensure_latest_code_checked_out() -> None:
    """Check GitHub for new commits and prompt for update before app start."""
    if os.getenv("OTTC_SKIP_UPDATE_CHECK", "0") in {"1", "true", "True"}:
        logger.info("Update check skipped (OTTC_SKIP_UPDATE_CHECK=1).")
        return

    local_hash = _get_local_commit_hash()
    remote_hash = _get_remote_commit_hash()

    if not local_hash or not remote_hash:
        logger.info("Update check could not be completed (missing git info).")
        return

    if local_hash == remote_hash:
        logger.info("OPENCORE Analyzer is up to date (%s).", local_hash[:7])
        return

    logger.info("New version available. Local %s, remote %s.", local_hash[:7], remote_hash[:7])
    if _prompt_user_for_update(remote_hash):
        _perform_git_update()
    else:
        logger.info("Update skipped by operator.")
