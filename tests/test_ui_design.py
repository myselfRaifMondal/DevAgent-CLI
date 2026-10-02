"""Guards for the terminal design system in devagent/cli/ui.py."""

import importlib.util
import io
import re
from pathlib import Path

import pytest
from rich.console import Console, Group
from rich.text import Text

from devagent import __version__
from devagent.cli import ui
from devagent.cli.renderers import ai_status_renderable, pr_preview_renderable, workspace_status_table
from devagent.core.actions import PullRequestPreview, WorkspaceSnapshot
from devagent.core.project import ProjectInfo
from devagent.tools.ai import AIProviderStatus, AIStatusSnapshot

SOURCE_ROOT = Path(ui.__file__).resolve().parent.parent
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\b")
ESCAPE = re.compile(r"\x1b\[([0-9;]*)m")


def _scenes():
    snapshot = WorkspaceSnapshot(
        project=ProjectInfo(path=Path("/work/acme"), project_types=["python"], package_files=["pyproject.toml"], file_tree=[]),
        is_repo=True,
        branch="main",
        dirty=True,
        changed_files=["a.py"],
    )
    status = AIStatusSnapshot(
        selected_provider="gemini",
        fast_model="m",
        deep_model="d",
        embedding_model="e",
        providers=(
            AIProviderStatus(provider="gemini", api_source="GEMINI_API_KEY", selected=True, generation_models=3),
            AIProviderStatus(provider="xai", api_source="XAI_API_KEY", selected=False, error="403"),
        ),
    )
    preview = PullRequestPreview(summary="s", title="t", body="b", readiness=("ok",), ready_to_create=False)
    return [
        ui.hero_panel("Agent Shell", "subtitle"),
        ui.app_panel("body", "Title", tone="warning"),
        ui.app_table("Table"),
        workspace_status_table(snapshot),
        ai_status_renderable(status),
        pr_preview_renderable(preview),
        ui.diff_renderable("--- a/x\n+++ b/x\n@@ -1 +1 @@\n context line\n-old\n+new\n"),
        ui.status_badge("yes", "success"),
    ]


def _ansi(renderable, *, colour_system: str = "truecolor") -> str:
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=True, color_system=colour_system, width=90, legacy_windows=False).print(renderable)
    return buffer.getvalue()


def test_no_fixed_hex_colours_anywhere_in_the_package() -> None:
    # Fixed hex values cannot adapt to the terminal's background: #dbeafe is 1.2:1 against white.
    offenders = [
        f"{path.relative_to(SOURCE_ROOT.parent)}:{number}"
        for path in SOURCE_ROOT.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if HEX_COLOUR.search(line)
    ]

    assert offenders == []


@pytest.mark.parametrize("index", range(8))
def test_every_component_uses_only_the_terminals_ansi_palette(index: int) -> None:
    rendered = _ansi(_scenes()[index])

    codes = [part for match in ESCAPE.findall(rendered) for part in match.split(";") if part]
    # 38;2;r;g;b is truecolor and 38;5;n is the 256-colour cube; neither adapts to a light theme.
    assert "38;2" not in ";".join(codes)
    assert "48;2" not in ";".join(codes)
    assert "38;5" not in ";".join(codes)
    assert "48;5" not in ";".join(codes)


def test_components_render_on_a_console_without_the_app_theme() -> None:
    # They must not rely on named theme styles that only the app's own console defines.
    for scene in _scenes():
        Console(file=io.StringIO(), width=80).print(scene)


def test_the_header_names_the_product_the_screen_and_the_version() -> None:
    buffer = Console(file=io.StringIO(), width=70, force_terminal=False)
    with buffer.capture() as capture:
        buffer.print(ui.hero_panel("Runtime Agent", "Spin up services."))
    text = capture.get()

    assert "DEVAGENT" in text
    assert "Runtime Agent" in text
    assert f"v{__version__}" in text
    assert "Spin up services." in text


@pytest.mark.parametrize("tone, word", [("success", "ok"), ("warning", "careful"), ("error", "failed"), ("info", "note")])
def test_status_is_never_carried_by_colour_alone(tone: str, word: str) -> None:
    badge = ui.status_badge(word, tone)
    plain = Text(badge.plain)

    assert plain.plain.endswith(word)
    assert plain.plain.split(" ")[0] != ""  # a glyph precedes the word
    assert ui.tone_glyph(tone) in badge.plain


def test_each_tone_has_a_distinct_glyph() -> None:
    glyphs = {ui.tone_glyph(tone) for tone in ("success", "warning", "error", "info")}

    assert len(glyphs) == 4


def test_ascii_fallback_is_pure_ascii(monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_ASCII", "1")
    raw = io.BytesIO()
    # A stream that really is ASCII, as on a LANG=C shell: writing any other character would raise.
    wrapper = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    stream = Console(file=wrapper, width=80, force_terminal=False)
    stream.print(Group(*[scene for scene in _scenes()]))
    wrapper.flush()

    assert not ui.unicode_ok()
    assert raw.getvalue().decode("ascii")


def test_the_glyph_set_follows_the_streams_encoding(monkeypatch) -> None:
    monkeypatch.delenv("DEVAGENT_ASCII", raising=False)
    monkeypatch.setattr(type(ui.console), "encoding", property(lambda self: "ascii"))

    assert not ui.unicode_ok()
    assert ui.glyph("success") == "+"
    assert ui.glyph("brand") == "#"


def test_the_glyph_set_is_unicode_for_utf8(monkeypatch) -> None:
    monkeypatch.delenv("DEVAGENT_ASCII", raising=False)
    monkeypatch.setattr(type(ui.console), "encoding", property(lambda self: "utf-8"))

    assert ui.unicode_ok()
    assert ui.glyph("success") == "✓"


@pytest.mark.parametrize("width", [30, 40, 60, 80, 120])
def test_nothing_overflows_the_terminal_width(width: int) -> None:
    for scene in _scenes():
        buffer = Console(file=io.StringIO(), width=width, force_terminal=False, color_system=None)
        with buffer.capture() as capture:
            buffer.print(scene)
        assert max(len(line) for line in capture.get().splitlines()) <= width


def test_long_plain_values_in_key_value_tables_fold_instead_of_being_cut() -> None:
    # A plain string has no overflow setting of its own, so this exercises the column's.
    table = ui.kv_table("Details")
    table.add_row("Files", "src/" + "very-long-directory-name/" * 4 + "module.py")
    buffer = Console(file=io.StringIO(), width=34, force_terminal=False, color_system=None)
    with buffer.capture() as capture:
        buffer.print(table)
    rendered = capture.get()

    assert "…" not in rendered
    assert "module.py" in "".join(rendered.split())


def test_markup_in_messages_is_shown_literally_not_interpreted() -> None:
    body = "see [/etc/hosts] and the [red]error[/red] in list[int]"
    buffer = Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)
    with buffer.capture() as capture:
        buffer.print(ui.app_panel(body, "Proposed change"))

    assert body in " ".join(capture.get().replace("│", " ").split())


def test_diff_lines_are_coloured_by_kind_and_text_is_unchanged() -> None:
    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n context\n-old\n+new\n"
    rendered = ui.diff_renderable(diff)
    styles = {rendered.plain.splitlines()[i]: None for i in range(len(rendered.plain.splitlines()))}
    by_prefix = {}
    for span in rendered.spans:
        by_prefix[rendered.plain[span.start]] = str(span.style)

    assert rendered.plain == diff.rstrip("\n")
    assert by_prefix["+"] == "green"
    assert by_prefix["-"] in {"red", "bold"}
    assert by_prefix["@"] == "cyan"
    assert styles  # every line is present


def test_shorten_path_uses_tilde_and_keeps_both_ends(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project = tmp_path / "code" / "a" / "b" / "c" / "my-project"

    assert ui.shorten_path(project).startswith("~")
    shortened = ui.shorten_path(project, max_width=22)
    assert len(shortened) <= 22
    assert shortened.startswith("~")
    assert shortened.endswith("my-project")


def _load_screenshot_script():
    path = SOURCE_ROOT.parent / "scripts" / "render_screenshots.py"
    spec = importlib.util.spec_from_file_location("render_screenshots", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_screenshot_scene_renders_on_both_themes() -> None:
    script = _load_screenshot_script()

    for name in script.SCENES:
        for theme in script.THEMES.values():
            svg = script.render_svg(name, theme)
            assert svg.startswith("<svg")
            assert "devagent" in svg


def test_the_screenshots_show_real_ui_not_old_chrome() -> None:
    script = _load_screenshot_script()

    svg = script.render_svg("shell-home", script.MONOKAI)

    assert "Workspace&#160;linked" in svg
    assert "feature/health-check" in svg
