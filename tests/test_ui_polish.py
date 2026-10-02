from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from devagent.cli.main import app
from devagent.cli.renderers import workspace_status_table
from devagent.config.settings import ConfigManager
from devagent.core.actions import snapshot_workspace
from devagent.core.shell import AgentShell

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "backend").mkdir(parents=True)
    (workspace / "backend" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return workspace


def _render_home(shell: AgentShell, width: int = 100) -> str:
    buffer = Console(width=width, record=True, force_terminal=False, color_system=None)
    with buffer.capture() as capture:
        buffer.print(shell.welcome_renderable())
    return capture.get()


def test_the_home_panel_no_longer_repeats_the_menu(tmp_path: Path) -> None:
    text = _render_home(AgentShell(_workspace(tmp_path)))

    for label in ("workspace", "git", "stack", "ai", "run"):
        assert label in text
    assert "Modes:" not in text
    for line in ("AI for provider selection", "Chat for repo-aware", "Watch for background"):
        assert line not in text


def test_run_list_prints_the_inventory_without_the_banner(tmp_path: Path) -> None:
    ConfigManager.bind_workspace(_workspace(tmp_path))

    result = runner.invoke(app, ["run", "list"])

    assert result.exit_code == 0
    assert "Detected Run Targets" in result.output
    assert "Runtime Agent" not in result.output


def test_bare_run_keeps_its_landing_banner(tmp_path: Path) -> None:
    ConfigManager.bind_workspace(_workspace(tmp_path))

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    assert "Runtime Agent" in result.output
    assert "Detected Run Targets" in result.output


def test_an_unbound_workspace_is_explained_not_reported_as_a_usage_error() -> None:
    result = runner.invoke(app, ["workspace", "status"])

    assert result.exit_code == 1
    assert "No workspace is bound yet" in result.output
    assert "devagent workspace bind <path>" in result.output
    assert "Usage:" not in result.output
    assert "Invalid value" not in result.output


def test_a_missing_workspace_lists_the_recovery_commands(tmp_path: Path) -> None:
    ConfigManager.bind_workspace(tmp_path / "gone")

    result = runner.invoke(app, ["workspace", "status"])

    assert result.exit_code == 1
    assert "The saved workspace path no longer exists" in result.output
    assert "devagent workspace bind <path>" in result.output
    assert "Usage:" not in result.output


def test_long_paths_fold_instead_of_being_cut_with_an_ellipsis(tmp_path: Path) -> None:
    deep = tmp_path / "a-rather-long-directory-name" / "another-long-directory-name" / "final-project-folder"
    deep.mkdir(parents=True)
    table = workspace_status_table(snapshot_workspace(deep))

    narrow = Console(width=44, record=True, force_terminal=False, color_system=None)
    with narrow.capture() as capture:
        narrow.print(table)
    rendered = capture.get()

    assert "…" not in rendered
    flattened = "".join(rendered.split())
    assert "final-project-folder" in flattened
    assert "another-long-directory-name" in flattened
