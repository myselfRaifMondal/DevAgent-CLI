"""Make the suite independent of the machine it runs on.

Typer forces coloured output when ``GITHUB_ACTIONS``, ``FORCE_COLOR`` or ``PY_COLORS`` is set, and
Rich styles each word separately, so a substring such as ``--deep`` can be split by escape codes.
Tests that read CLI output assert on plain text, so colour is switched off for every test.
"""

import pytest

PLAIN_OUTPUT_ENV_REMOVED = ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS", "FORCE_TERMINAL", "CLICOLOR_FORCE")
PROVIDER_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _plain_output_and_no_provider_keys(monkeypatch):
    for name in PLAIN_OUTPUT_ENV_REMOVED + PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    try:
        import typer.rich_utils as rich_utils

        # Evaluated once at import, so the environment change above is too late for it.
        monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", None, raising=False)
    except ImportError:
        pass
