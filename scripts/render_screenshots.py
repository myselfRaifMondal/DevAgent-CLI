#!/usr/bin/env python3
"""Regenerate the README screenshots from DevAgent's real renderers.

The images in ``docs/assets`` are not mock-ups: each scene below calls the same functions the CLI
uses, with sample data. Re-run this after changing the UI so the README never shows a screen the
product no longer produces.

    python scripts/render_screenshots.py            # writes docs/assets/*.svg (dark and light)
    python scripts/render_screenshots.py --check    # exit 1 if the committed files are out of date

Each scene is exported twice, once on a dark and once on a light terminal theme. DevAgent only uses
the terminal's ANSI palette, so the two exports use identical output and differ only in the palette
the terminal supplies; the README picks one by the viewer's colour scheme.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console, Group, RenderableType
from rich.terminal_theme import DEFAULT_TERMINAL_THEME, MONOKAI, TerminalTheme
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from devagent.cli import ui  # noqa: E402
from devagent.cli.renderers import ai_status_renderable, pr_preview_renderable  # noqa: E402
from devagent.core.actions import PullRequestPreview, WorkspaceSnapshot  # noqa: E402
from devagent.core.project import ProjectInfo  # noqa: E402
from devagent.core.shell import AgentShell  # noqa: E402
from devagent.tools.ai import AIProviderStatus, AIStatusSnapshot  # noqa: E402
from devagent.tools.runtime_tool import LaunchSpec, RunProfile  # noqa: E402

WIDTH = 100
THEMES: dict[str, TerminalTheme] = {"": MONOKAI, "-light": DEFAULT_TERMINAL_THEME}
WORKSPACE = Path("/home/dev/acme-api")

SAMPLE_DIFF = """--- a/README.md
+++ b/README.md
@@ -12,6 +12,11 @@
 ## Running locally

 ```bash
 uvicorn app.main:app --reload
 ```
+
+## Health check
+
+`GET /health` returns `{"status": "ok"}` and is used by the load balancer.
+
 ## Configuration
"""


def command_line(command: str) -> Text:
    line = Text()
    line.append("$ ", style="dim")
    line.append(command, style="bold")
    return line


def _shell_home() -> RenderableType:
    fake_actions = SimpleNamespace(
        workspace_status=lambda: WorkspaceSnapshot(
            project=ProjectInfo(path=WORKSPACE, project_types=["python", "node"], package_files=[], file_tree=[]),
            is_repo=True,
            branch="feature/health-check",
            dirty=True,
            changed_files=["README.md", "app/main.py", "tests/test_health.py"],
        ),
        run_inventory=lambda: SimpleNamespace(
            detected=[
                LaunchSpec("api python", WORKSPACE, ["python", "main.py"], "python main.py", "python"),
                LaunchSpec("web node dev", WORKSPACE / "web", ["npm", "run", "dev"], "npm run dev", "node"),
            ],
            profiles={"start the stack": RunProfile("start the stack", [], open_browser=True)},
        ),
        ai_summary=lambda: ("Gemini", "gemini-2.5-flash"),
    )
    shell = AgentShell.__new__(AgentShell)
    shell.actions = fake_actions

    menu = Text()
    menu.append("? ", style="bold cyan")
    menu.append("DevAgent Home ", style="bold")
    menu.append("(Use arrow keys)\n", style="dim")
    items = ["AI", "Chat", "Git", "Run", "Repo", "Setup", "Edit", "Watch", "Quick command / phrase", "Help", "Exit"]
    for index, item in enumerate(items):
        if index == 1:
            menu.append(" » ● ", style="bold cyan")
            menu.append(item + "\n", style="bold cyan")
        else:
            menu.append(f"   ○ {item}\n")
    return Group(
        ui.hero_panel("Agent Shell", "One menu-driven control room for chat, Git, runtime, setup, and repo work."),
        ui.app_panel(shell.welcome_renderable(), "Workspace linked", tone="success"),
        menu,
    )


def _ai_status() -> RenderableType:
    status = AIStatusSnapshot(
        selected_provider="gemini",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        embedding_model="gemini-embedding-001",
        providers=(
            AIProviderStatus(provider="gemini", api_source="GEMINI_API_KEY", selected=True, generation_models=14, embedding_models=2),
            AIProviderStatus(provider="xai", api_source="XAI_API_KEY", selected=False, error="HTTP 403: no credits on this team"),
        ),
    )
    return Group(command_line("devagent ai status"), Text(""), ai_status_renderable(status))


def _edit_preview() -> RenderableType:
    return Group(
        command_line('devagent edit "document the health endpoint in the README"'),
        Text(""),
        ui.app_panel(ui.diff_renderable(SAMPLE_DIFF), "Proposed change", tone="info"),
        Text("Apply this diff? [y/N]: ", style="bold"),
    )


def _pr_preview() -> RenderableType:
    preview = PullRequestPreview(
        summary="feature/health-check into main on acme/api",
        title="feat: add a health endpoint and document it",
        body="- Add GET /health for the load balancer.\n- Cover it with a test.\n- Document it in the README.",
        readiness=("2 commits ahead of main", "Working tree is clean"),
        ready_to_create=True,
    )
    return Group(command_line("devagent git pr preview --base main"), Text(""), pr_preview_renderable(preview))


SCENES = {
    "shell-home": _shell_home,
    "ai-status": _ai_status,
    "edit-diff-preview": _edit_preview,
    "git-pr-preview": _pr_preview,
}


def render_svg(name: str, theme: TerminalTheme) -> str:
    """Render one scene to SVG text. Pure: it writes nothing to disk."""
    recorder = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        color_system="standard",
        width=WIDTH,
        theme=ui.THEME,
        highlight=False,
        legacy_windows=False,
    )
    recorder.print(SCENES[name]())
    return recorder.export_svg(title="devagent", theme=theme)


def outputs() -> dict[Path, str]:
    return {
        ROOT / "docs" / "assets" / f"{name}{suffix}.svg": render_svg(name, theme)
        for name in SCENES
        for suffix, theme in THEMES.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed screenshots differ from a fresh render")
    args = parser.parse_args()

    if not ui.unicode_ok():
        print("The screenshots need a UTF-8 terminal; run with PYTHONUTF8=1 and unset DEVAGENT_ASCII.", file=sys.stderr)
        return 2

    stale = []
    for path, svg in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        for path in stale:
            print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
        print("Run `python scripts/render_screenshots.py` and commit the result.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
