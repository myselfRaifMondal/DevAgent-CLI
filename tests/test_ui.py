from rich.console import Console
from rich.text import Text

from devagent.cli.ui import render_chat_markdown


def test_render_chat_markdown_renders_markdown_syntax() -> None:
    console = Console(record=True, width=80)

    console.print(render_chat_markdown("## Heading\n\n- first item\n- second item\n\nUse `code` here."))
    output = console.export_text()

    assert "Heading" in output
    assert "## Heading" not in output
    assert "- first item" not in output
    assert "first item" in output


def test_render_chat_markdown_leaves_plain_text_alone() -> None:
    rendered = render_chat_markdown("plain answer")

    # Not run through the Markdown renderer, and returned as Text so Rich never parses it as markup.
    assert isinstance(rendered, Text)
    assert rendered.plain == "plain answer"
