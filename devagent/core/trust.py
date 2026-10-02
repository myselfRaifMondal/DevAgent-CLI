"""Approval of the exact commands DevAgent will run inside a workspace.

``run start`` executes whatever a project defines -- ``npm run dev`` runs the
``dev`` script in ``package.json``, a Python entry file runs that file. When the
project is a freshly cloned or recently updated repository, those commands are
written by someone else. Each launch spec therefore has to be approved once,
and is approved only for the exact command, directory and manifest contents it
had at that moment: change ``package.json`` and the approval no longer applies.

This also covers a tampered ``run_profiles.json``: a planted profile carries a
command nobody approved, so it asks before running instead of running silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Protocol

from devagent.config.settings import ConfigManager

MANIFEST_NAMES = (
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pyproject.toml",
    "requirements.txt",
    "manage.py",
)

Approver = Callable[[list[str]], bool]


class _SpecLike(Protocol):
    name: str
    cwd: Path
    command: list[str]
    display_command: str
    venv_dir: Path | None
    bootstrap_commands: tuple[tuple[str, ...], ...]


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def spec_fingerprint(spec: _SpecLike) -> str:
    """Digest of what a launch spec will run: command, directory and the manifests beside it."""
    cwd = Path(spec.cwd).expanduser().resolve()
    payload = {
        "cwd": str(cwd),
        "command": list(spec.command),
        "bootstrap": [list(command) for command in spec.bootstrap_commands],
        "venv": str(spec.venv_dir) if spec.venv_dir else None,
        "manifests": {name: _file_digest(cwd / name) for name in MANIFEST_NAMES if (cwd / name).is_file()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class TrustStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.expanduser().resolve()
        self.path = ConfigManager.workspace_cache_dir(self.workspace) / "trusted_launches.json"

    def _load(self) -> set[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        approved = data.get("approved") if isinstance(data, dict) else None
        return {item for item in approved if isinstance(item, str)} if isinstance(approved, list) else set()

    def is_trusted(self, spec: _SpecLike) -> bool:
        return spec_fingerprint(spec) in self._load()

    def untrusted(self, specs: Iterable[_SpecLike]) -> list[_SpecLike]:
        approved = self._load()
        return [spec for spec in specs if spec_fingerprint(spec) not in approved]

    def trust(self, specs: Iterable[_SpecLike]) -> None:
        approved = self._load() | {spec_fingerprint(spec) for spec in specs}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates the file readable and writable by this user only (0600).
        fd, temp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".trust-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"approved": sorted(approved)}, handle, indent=2)
            os.replace(temp_name, self.path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise


class LaunchNotApprovedError(RuntimeError):
    pass


def describe_specs(workspace: Path, specs: Iterable[_SpecLike]) -> list[str]:
    lines: list[str] = []
    for spec in specs:
        try:
            scope = Path(spec.cwd).resolve().relative_to(workspace).as_posix() or "."
        except ValueError:
            scope = str(spec.cwd)
        for command in spec.bootstrap_commands:
            lines.append(f"{scope}> {' '.join(command)}")
        lines.append(f"{scope}> {spec.display_command or ' '.join(spec.command)}")
    return lines


def interactive_approver(lines: list[str]) -> bool:
    """Ask on the terminal. Refuses outright when there is no terminal to ask on."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()
    console.print("[bold yellow]DevAgent is about to run commands defined by this project:[/bold yellow]")
    for line in lines:
        console.print(f"  {line}")
    console.print("These run with your user's permissions. Approve only code you trust.")
    return Confirm.ask("Run these commands?", default=False)


def require_launch_approval(workspace: Path, specs: list[_SpecLike], approver: Approver | None = None) -> None:
    """Raise unless every spec was approved before or is approved now."""
    store = TrustStore(workspace)
    pending = store.untrusted(specs)
    if not pending:
        return
    lines = describe_specs(workspace, pending)
    if not (approver or interactive_approver)(lines):
        raise LaunchNotApprovedError(
            "These commands have not been approved for this workspace, so nothing was started:\n"
            + "\n".join(f"  {line}" for line in lines)
            + "\nRun this from an interactive terminal to review and approve them."
        )
    store.trust(pending)
