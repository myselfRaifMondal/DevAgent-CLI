from __future__ import annotations

import os
import re
from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from devagent import __version__

# Colour policy
# -------------
# Everything here is drawn from the terminal's own 16-colour ANSI palette plus bold/dim, never from
# fixed hex values. A terminal theme remaps those 16 colours to suit its own background, so the same
# output stays readable on light and dark themes alike. (Pastel fixed colours, such as the light blue this theme used for table rows, measure
# 1.2:1 against white, which is unreadable, and no single hex value passes on both backgrounds.)
# Blue is avoided as a text colour: ANSI blue is dark on most dark themes.
THEME = Theme({"status.spinner": "cyan"})

console = Console(theme=THEME, highlight=False)

_TONE_STYLES = {
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
}
_TONE_BORDER = {"info": "cyan", "success": "green", "warning": "yellow", "error": "red"}

# Status is never carried by colour alone: every tone has a glyph, and badges also spell the word.
_GLYPHS = {
    "unicode": {"info": "●", "success": "✓", "warning": "▲", "error": "✗", "brand": "◆", "sep": "·", "dot": "●", "prompt": "›"},
    "ascii": {"info": "*", "success": "+", "warning": "!", "error": "x", "brand": "#", "sep": "-", "dot": "*", "prompt": ">"},
}
_MARKDOWN_RE = (
    r"(^#{1,6}\s+\S)|"
    r"(^\s*(?:[-*+]|\d+\.)\s+\S)|"
    r"(```)|"
    r"(`[^`]+`)|"
    r"(\*\*[^*]+\*\*)|"
    r"(^>\s+\S)"
)


def unicode_ok() -> bool:
    """False when the output cannot show box-drawing or symbol characters.

    ``DEVAGENT_ASCII=1`` forces the plain set; otherwise it follows the stream's encoding, so
    ``LANG=C`` shells and legacy consoles get ``+ x !`` instead of mojibake.
    """
    if os.environ.get("DEVAGENT_ASCII", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return (console.encoding or "").lower().replace("_", "-").startswith("utf")


def glyph(name: str) -> str:
    return _GLYPHS["unicode" if unicode_ok() else "ascii"][name]


def tone_glyph(tone: str) -> str:
    return glyph(tone if tone in _TONE_STYLES else "info")


def divider() -> Rule:
    # Rich's untitled Rule draws with its own default character even on an ASCII stream (it works
    # out the substitute and then does not use it), so the character is chosen here.
    return Rule(characters="─" if unicode_ok() else "-", style="dim")


def brand_text(label: str = "DEVAGENT") -> Text:
    text = Text(no_wrap=True)
    text.append(f"{glyph('brand')} ", style="bold cyan")
    text.append(label, style="bold cyan")
    return text


def hero_panel(title: str, subtitle: str) -> RenderableType:
    """Compact screen header: brand and screen name on one line, version at the right, then a rule.

    (The name is kept from the old boxed banner so call sites do not change.)
    """
    left = Text(no_wrap=True)
    left.append(f"{glyph('brand')} DEVAGENT", style="bold cyan")
    left.append(f"  {glyph('sep')}  ", style="dim")
    left.append(title, style="bold")
    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(justify="right", no_wrap=True)
    top.add_row(left, Text(f"v{__version__}", style="dim"))
    return Group(top, Text(subtitle, style="dim"), divider())


def app_panel(
    body: RenderableType,
    title: str,
    *,
    tone: str = "info",
    expand: bool = True,
    padding: tuple[int, int] = (0, 1),
) -> Panel:
    tone = tone if tone in _TONE_STYLES else "info"
    title_text = Text(f"{tone_glyph(tone)} {title}", style=_TONE_STYLES[tone])
    if isinstance(body, str):
        # Messages are data (diffs, file text, model answers), not Rich markup. As a string,
        # "[/etc/hosts]" raises MarkupError and "list[int]" silently loses "[int]".
        body = Text(body)
    return Panel(
        body,
        title=title_text,
        title_align="left",
        box=box.ROUNDED,
        border_style=_TONE_BORDER[tone],
        padding=padding,
        expand=expand,
    )


def app_table(title: str, *, expand: bool = False) -> Table:
    return Table(
        title=Text(title, style="bold"),
        title_justify="left",
        header_style="bold",
        border_style="dim",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        expand=expand,
        show_lines=False,
    )


def kv_table(title: str) -> Table:
    """Two-column label/value table with no header row; labels are dimmed."""
    table = Table(
        title=Text(title, style="bold"),
        title_justify="left",
        show_header=False,
        box=None,
        show_edge=False,
        pad_edge=False,
        padding=(0, 2, 0, 0),
        expand=False,
    )
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    return table


def status_badge(label: str, tone: str) -> Text:
    tone = tone if tone in _TONE_STYLES else "info"
    return Text(f"{tone_glyph(tone)} {label}", style=_TONE_STYLES[tone])


def muted(value: str) -> Text:
    """Placeholder text such as ``none`` or ``No saved phrases yet``: present, but visually quiet."""
    return Text(value, style="dim")


def styled_path(value: str) -> Text:
    return Text(value, style="cyan", overflow="fold")


def toned_message(value: str, tone: str) -> Text:
    return Text(value, style=_TONE_STYLES.get(tone, _TONE_STYLES["info"]))


def shorten_path(path: str | Path, max_width: int | None = None) -> str:
    """Show the home directory as ``~`` and, if still too wide, keep both ends of the path."""
    text = str(path)
    home = str(Path.home())
    if text == home:
        text = "~"
    elif text.startswith(home.rstrip("/\\") + os.sep):
        text = "~" + text[len(home.rstrip("/\\")):]
    if max_width is None or len(text) <= max_width or max_width < 12:
        return text
    ellipsis = "…" if unicode_ok() else "..."
    keep = max_width - len(ellipsis)
    head = keep // 3
    tail = keep - head
    return f"{text[:head]}{ellipsis}{text[-tail:]}"


def status_strip(rows: list[tuple[str, RenderableType]]) -> Table:
    """Borderless label/value block used for at-a-glance state, such as the shell home."""
    grid = Table.grid(padding=(0, 2, 0, 0))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")
    for label, value in rows:
        grid.add_row(label, value)
    return grid


def diff_renderable(diff: str) -> Text:
    """A unified diff with additions, removals and hunk headers coloured, as plain text underneath."""
    text = Text(no_wrap=False)
    lines = diff.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("+++", "---")) or line.startswith(("diff --git", "index ")):
            style = "bold"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = ""
        text.append(line, style=style)
        if index < len(lines) - 1:
            text.append("\n")
    return text


def render_chat_markdown(value: str) -> RenderableType:
    if not value.strip():
        return Text("")
    if not re.search(_MARKDOWN_RE, value, re.MULTILINE):
        return Text(value)
    # ansi_dark maps code colours onto the terminal's own palette instead of fixing them.
    return Markdown(value, code_theme="ansi_dark")
