from pathlib import Path

import pytest

from devagent.context.indexer import CodeIndexer
from devagent.context.secrets import REDACTED, is_sensitive_path, redact_secrets
from devagent.core.agent import render_chunk
from devagent.tools.edit_tool import EditAgent

AWS_KEY = "AKIAABCDEFGHIJKLMNOP"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "backend/.env.production",
        ".npmrc",
        "deploy/id_rsa",
        "certs/server.pem",
        "certs/server.key",
        "config/credentials.json",
        "gcp/service-account.json",
        "client_secret_123.json",
        "infra/prod.tfvars",
        ".aws/config",
        "home/.ssh/config",
        "config\\secrets.yaml",
    ],
)
def test_sensitive_paths_are_detected(path: str) -> None:
    assert is_sensitive_path(path)


@pytest.mark.parametrize(
    "path",
    [".env.example", ".env.sample", "app.py", "README.md", "src/auth.py", "config/settings.json", "docs/secrets-policy.md"],
)
def test_ordinary_paths_are_not_sensitive(path: str) -> None:
    assert not is_sensitive_path(path)


@pytest.mark.parametrize(
    "text, leaked",
    [
        (f'aws_id = "{AWS_KEY}"', AWS_KEY),
        (f"token: {GITHUB_TOKEN}", GITHUB_TOKEN),
        ('API_KEY = "sk_live_abcdefghijklmnop"', "sk_live_abcdefghijklmnop"),
        ('password = "hunter2hunter2"', "hunter2hunter2"),
        ("export STRIPE_SECRET=abcdef1234567890", "abcdef1234567890"),
        ("DATABASE: postgres://admin:s3cretPass@db.internal:5432/app", "s3cretPass"),
        ("-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\n-----END PRIVATE KEY-----", "MIIEvQIBADANBgkq"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345", "abcdefghijklmnopqrstuvwxyz012345"),
    ],
)
def test_redact_secrets_removes_credential_values(text: str, leaked: str) -> None:
    redacted = redact_secrets(text)

    assert leaked not in redacted
    assert REDACTED in redacted


def test_redact_secrets_keeps_the_name_so_context_stays_useful() -> None:
    redacted = redact_secrets('client_secret = "abcdefghijkl"')

    assert redacted.startswith("client_secret = ")


def test_redact_secrets_leaves_ordinary_code_alone() -> None:
    code = "def login(user, password):\n    return check(user.token, password)\nMAX_TOKENS = 100\n"

    assert redact_secrets(code) == code


def test_indexer_skips_sensitive_files_and_redacts_the_rest(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "credentials.json").write_text('{"private_key": "TOPSECRETKEYMATERIAL"}', encoding="utf-8")
    (workspace / "settings.yml").write_text(f"github_token: {GITHUB_TOKEN}\nname: demo\n", encoding="utf-8")
    (workspace / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    index = CodeIndexer(workspace, chunk_lines=20).build()
    paths = {record.path for record in index.records}
    all_text = "\n".join(record.text for record in index.records)

    assert "credentials.json" not in paths
    assert "TOPSECRETKEYMATERIAL" not in all_text
    assert GITHUB_TOKEN not in all_text
    assert "name: demo" in all_text
    assert "app.py" in paths


def test_index_cache_on_disk_contains_no_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "settings.yml").write_text(f"github_token: {GITHUB_TOKEN}\n", encoding="utf-8")

    indexer = CodeIndexer(workspace, chunk_lines=20)
    indexer.build()

    assert GITHUB_TOKEN not in indexer.index_file.read_text(encoding="utf-8")


def test_loading_an_old_index_drops_records_for_sensitive_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    indexer = CodeIndexer(workspace, chunk_lines=20)
    indexer.build()

    import json

    data = json.loads(indexer.index_file.read_text(encoding="utf-8"))
    data["records"].append(
        {"path": ".env", "start_line": 1, "end_line": 1, "text": "API_KEY=oldleakedvalue"}
    )
    indexer.index_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = indexer.load()

    assert ".env" not in {record.path for record in loaded.records}


def test_chat_prompt_rendering_redacts_chunk_text() -> None:
    class Chunk:
        path = "stale.py"
        start_line = 1
        end_line = 1
        text = f'token = "{GITHUB_TOKEN}"'
        symbols = None
        imports = None
        headings = None

    assert GITHUB_TOKEN not in render_chunk(Chunk())


def test_edit_prompt_never_contains_secrets_from_a_stale_index(tmp_path: Path, monkeypatch) -> None:
    import devagent.tools.edit_tool as edit_tool_module
    from devagent.context.indexer import CodeChunk, CodeIndex

    prompts: list[str] = []

    class FakeAI:
        available = True
        provider_label = "Fake"

        def embed(self, texts):
            return None

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            from types import SimpleNamespace

            return SimpleNamespace(text="NO_PATCH", final_error=None, fallback_notes=())

    stale = CodeIndex(
        root=tmp_path,
        records=[CodeChunk("app.py", 1, 1, f'token = "{GITHUB_TOKEN}"  # change the greeting')],
    )
    monkeypatch.setattr(edit_tool_module.AIClient, "from_env", classmethod(lambda cls: FakeAI()))
    monkeypatch.setattr(edit_tool_module.CodeIndexer, "load_or_build", lambda self: stale)

    EditAgent(tmp_path).propose("change the greeting")

    assert prompts
    assert GITHUB_TOKEN not in prompts[0]
