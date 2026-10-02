from __future__ import annotations

from pathlib import Path
from typing import Iterable

from devagent.config.settings import ConfigManager

IGNORED_DIRS = {
    ".git",
    ".devagent",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".scss",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".php",
    ".rb",
    ".sql",
    ".sh",
    ".ps1",
    ".env.example",
}


def iter_source_files(root: Path) -> Iterable[Path]:
    resolved = root.expanduser().resolve()
    # DevAgent's own state (index cache, saved profiles) can live inside the project when
    # DEVAGENT_CONFIG_DIR points there. Indexing it would index the index.
    config_dir = ConfigManager.config_dir()
    for path in sorted(resolved.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(resolved)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if config_dir == path or config_dir in path.resolve().parents:
            continue
        if path.name.endswith(".env.example") or path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def read_text_safely(path: Path, max_bytes: int = 250_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    except OSError:
        return None
