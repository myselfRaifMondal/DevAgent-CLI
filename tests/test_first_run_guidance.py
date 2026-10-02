from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import devagent.tools.edit_tool as edit_tool_module
from devagent.cli.renderers import ai_status_renderable
from devagent.cli.ui import console
from devagent.core.agent import RepoAgent
from devagent.core.shell import AgentShell
from devagent.tools.ai import AIStatusSnapshot, ai_setup_hint
from devagent.tools.edit_tool import EditAgent


class NoAI:
    available = False
    provider_label = "AI"

    def embed(self, texts):
        return None


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))


def _no_ai(monkeypatch) -> None:
    monkeypatch.setattr("devagent.tools.ai.AIClient.from_env", classmethod(lambda cls: NoAI()))


def _render(renderable) -> str:
    buffer = Console(width=120, record=True, force_terminal=False)
    with buffer.capture() as capture:
        buffer.print(renderable)
    return capture.get()


def test_the_hint_names_every_provider_key_and_the_check_command() -> None:
    hint = ai_setup_hint()

    # Literal names, not PROVIDER_KEY_NAMES: iterating the constant under test cannot fail.
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        assert name in hint
    assert "devagent ai status" in hint
    assert "README" in hint


def test_chat_with_no_ai_and_no_match_says_ai_is_missing_not_to_reindex(tmp_path: Path, monkeypatch) -> None:
    _no_ai(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    answer = RepoAgent(workspace).answer("Where is the zebra module?")

    assert ai_setup_hint() in answer
    assert "devagent index" not in answer


def test_chat_with_ai_and_no_match_still_suggests_indexing(tmp_path: Path, monkeypatch) -> None:
    fake = SimpleNamespace(
        available=True,
        provider_label="Gemini",
        embed=lambda texts: None,
        generate=lambda *a, **k: SimpleNamespace(text=None, final_error="busy", succeeded=False),
    )
    monkeypatch.setattr("devagent.tools.ai.AIClient.from_env", classmethod(lambda cls: fake))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    answer = RepoAgent(workspace).answer("Where is the zebra module?")

    assert "devagent index" in answer
    assert "No AI provider is configured" not in answer


def test_chat_with_no_ai_but_a_match_explains_why_the_answer_is_not_ai_written(tmp_path: Path, monkeypatch) -> None:
    _no_ai(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "billing.py").write_text("def charge_customer(card):\n    return card\n", encoding="utf-8")

    answer = RepoAgent(workspace).answer("Where is charge_customer?")

    assert "billing.py" in answer
    assert "Why the AI answer fell back: No AI provider is configured" in answer


def test_edit_without_ai_tells_you_how_to_configure_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(edit_tool_module.AIClient, "from_env", classmethod(lambda cls: NoAI()))
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    proposal = EditAgent(tmp_path).propose("change x")

    assert proposal.diff is None
    assert ai_setup_hint() in proposal.message


def test_ai_status_with_no_provider_shows_a_setup_row() -> None:
    status = AIStatusSnapshot(selected_provider=None, fast_model=None, deep_model=None, embedding_model=None, providers=())

    text = _render(ai_status_renderable(status))

    assert "Setup" in text
    assert "GEMINI_API_KEY" in text


def test_ai_status_with_a_provider_has_no_setup_row() -> None:
    from devagent.tools.ai import AIProviderStatus

    provider = AIProviderStatus(provider="gemini", api_source="GEMINI_API_KEY", selected=True, generation_models=3)
    status = AIStatusSnapshot(selected_provider="gemini", fast_model="m", deep_model="d", embedding_model="e", providers=(provider,))

    assert "Setup" not in _render(ai_status_renderable(status))


def test_edit_mode_without_ai_explains_and_does_not_ask_for_an_instruction(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shell = AgentShell(workspace)
    shell.actions.ai_available = lambda: False
    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *a, **k: pytest.fail("must not ask for an instruction"))
    shell.actions.edit_propose = lambda *a, **k: pytest.fail("must not propose")

    with console.capture() as captured:
        shell.edit_mode()

    assert "Edit needs an AI provider" in captured.get()
    assert "GEMINI_API_KEY" in captured.get()


def test_edit_mode_with_ai_still_asks_for_an_instruction(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shell = AgentShell(workspace)
    shell.actions.ai_available = lambda: True
    asked: list[str] = []
    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *a, **k: asked.append("asked") or "")

    shell.edit_mode()

    assert asked == ["asked"]


@pytest.mark.parametrize("available, expect_notice", [(False, True), (True, False)])
def test_chat_mode_warns_up_front_only_when_no_ai_is_configured(tmp_path: Path, monkeypatch, available: bool, expect_notice: bool) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    shell = AgentShell(workspace)
    shell.actions.ai_available = lambda: available

    def leave(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr(console, "input", leave)

    with console.capture() as captured:
        shell.chat_mode()

    assert ("keyword search only" in captured.get()) is expect_notice
