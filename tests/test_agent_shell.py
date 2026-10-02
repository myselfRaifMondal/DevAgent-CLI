from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from devagent.cli.main import app
from devagent.config.settings import ConfigManager
from devagent.core.actions import AISelectionResult, PullOutcome, PullRequestPreview, PushOutcome, WorkspaceSnapshot
from devagent.core.agent import RepoAgent
from devagent.core.project import ProjectInfo
from devagent.core.shell import AgentShell, GitIntent, home_menu_choices, interactive_terminal
from devagent.tools.ai import AIProviderStatus, AIStatusSnapshot, DiscoveredModel, GenerationResult, ProviderModelListing
from devagent.tools.git_tool import GitRemote
from devagent.tools.runtime_tool import LaunchSpec
from devagent.tools.setup_tool import SetupResult


class FakeAI:
    def __init__(self):
        self.available = True
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, *, deep: bool = False, system_instruction: str | None = None, progress_callback=None) -> GenerationResult:
        self.calls.append({"prompt": prompt, "deep": deep, "system_instruction": system_instruction})
        return GenerationResult(
            text="Detailed repo answer",
            provider="gemini",
            model="gemini-2.5-flash",
            used_deep_mode=deep,
            attempts=1,
        )

    def complete(self, prompt: str, *, deep: bool = False, system_instruction: str | None = None) -> str:
        generated = self.generate(prompt, deep=deep, system_instruction=system_instruction)
        return generated.text or ""

    def embed(self, texts):
        return None


class FailingAI:
    def __init__(self):
        self.available = True
        self.provider_label = "Gemini"

    def generate(self, prompt: str, *, deep: bool = False, system_instruction: str | None = None, progress_callback=None) -> GenerationResult:
        return GenerationResult(
            text=None,
            provider="gemini",
            model="gemini-2.5-pro" if deep else "gemini-2.5-flash",
            used_deep_mode=deep,
            attempts=3,
            error_kind="transient_server",
            final_error="Gemini stayed busy after retries.",
        )

    def complete(self, prompt: str, *, deep: bool = False, system_instruction: str | None = None) -> str | None:
        return None

    def embed(self, texts):
        return None


class FakeRepoAgent:
    def __init__(self):
        self.deep_calls: list[bool] = []
        self.cleared = False

    def answer(self, question: str, *, deep: bool = False, new_session: bool = False, progress_callback=None) -> str:
        self.deep_calls.append(deep)
        return f"chat:{question}"

    def clear_session(self) -> None:
        self.cleared = True


def test_repo_agent_uses_session_memory_and_deep_mode(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "backend").mkdir()
    (workspace / "backend" / "auth.py").write_text(
        "def login_user(user):\n    return create_session(user)\n",
        encoding="utf-8",
    )
    (workspace / "backend" / "routes.py").write_text(
        "from backend.auth import login_user\n\n"
        "def login_route(payload):\n    return login_user(payload)\n",
        encoding="utf-8",
    )

    fake_ai = FakeAI()
    monkeypatch.setattr("devagent.tools.ai.AIClient.from_env", classmethod(lambda cls: fake_ai))

    agent = RepoAgent(workspace)
    first = agent.answer("Where is authentication implemented?")
    second = agent.answer("How does it connect to routes?", deep=True)
    third = agent.answer("Explain the login flow again.", new_session=True)

    assert first == "Detailed repo answer"
    assert second == "Detailed repo answer"
    assert third == "Detailed repo answer"
    assert fake_ai.calls[1]["deep"] is True
    assert "Where is authentication implemented?" in str(fake_ai.calls[1]["prompt"])
    assert "Where is authentication implemented?" not in str(fake_ai.calls[2]["prompt"])
    assert "backend/auth.py" in str(fake_ai.calls[1]["prompt"])
    assert "backend/routes.py" in str(fake_ai.calls[1]["prompt"])


def test_repo_agent_returns_grounded_fallback_when_generation_fails(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "backend").mkdir()
    (workspace / "backend" / "auth.py").write_text(
        "def login_user(user):\n    return create_session(user)\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("devagent.tools.ai.AIClient.from_env", classmethod(lambda cls: FailingAI()))

    agent = RepoAgent(workspace)
    answer = agent.answer("Where is authentication implemented?", deep=True)

    assert "could not finish the AI synthesis" in answer
    assert "backend/auth.py" in answer
    assert "Gemini stayed busy after retries." in answer


def test_shell_routes_saved_phrase_runtime_and_chat(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "frontend").mkdir()
    (workspace / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite"}}', encoding="utf-8")

    shell = AgentShell(workspace)
    shell.repo_agent = FakeRepoAgent()
    saved_profile = shell.run_tool.save_detected_profile("Start I Command You", open_browser=True)

    launched: list[tuple[str, bool | None]] = []
    shell.run_tool.launch_profile = lambda profile, open_browser=None: launched.append((profile.phrase, open_browser)) or profile.specs

    saved_result = shell.handle_input(" start i command you ")
    assert saved_result is not None
    assert saved_result.title == "Saved Run Phrase"
    assert launched == [("Start I Command You", None)]

    runtime_calls: list[bool] = []
    shell.run_tool.launch_detected = lambda open_browser=False: runtime_calls.append(open_browser) or [
        LaunchSpec(
            name="frontend node dev",
            cwd=workspace / "frontend",
            command=["npm", "run", "dev"],
            display_command="npm run dev",
            kind="node",
            browser_url="http://localhost:5173",
        )
    ]
    runtime_result = shell.handle_input("start the app and open the browser")
    assert runtime_result is not None
    assert runtime_calls == [True]
    assert "Opened browser at http://localhost:5173" in runtime_result.message

    deep_toggle = shell.handle_input("/deep")
    assert deep_toggle is not None
    assert shell.deep_mode is True

    chat_result = shell.handle_input("Explain the login flow")
    assert chat_result is not None
    assert chat_result.message == "chat:Explain the login flow"
    assert shell.repo_agent.deep_calls == [True]

    clear_result = shell.handle_input("/clear")
    assert clear_result is not None
    assert shell.repo_agent.cleared is True


def test_shell_quick_command_routes_git_intent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    commit_calls: list[tuple[str | None, bool]] = []
    shell.actions.git_commit = lambda message=None, all_files=True: commit_calls.append((message, all_files)) or type(
        "Outcome",
        (),
        {"commit_id": "abc123", "message": "feat: update project changes"},
    )()

    result = shell.handle_input("commit my changes")

    assert result is not None
    assert result.title == "Commit Complete"
    assert commit_calls == [(None, True)]
    assert "abc123" in str(result.message)


def test_shell_workspace_property_tracks_actions_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    third = tmp_path / "third"
    third.mkdir()

    shell = AgentShell(workspace)

    shell.actions.refresh_workspace(other)
    assert shell.workspace == other

    shell.workspace = third
    assert shell.workspace == third
    assert shell.actions.workspace == third


def test_shell_clone_action_shows_workspace_snapshot_and_updates_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    clone_parent = tmp_path / "clones"
    clone_parent.mkdir()
    cloned = clone_parent / "demo"
    cloned.mkdir()
    (cloned / "package.json").write_text('{"name":"demo"}', encoding="utf-8")

    shell = AgentShell(workspace)

    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *args, **kwargs: "https://github.com/example/demo")
    monkeypatch.setattr("devagent.core.shell.choose_directory", lambda *args, **kwargs: clone_parent)
    confirms = iter([True, False])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    def fake_clone(repo_url, *, target=None, install_deps=False, open_code=False):
        assert repo_url == "https://github.com/example/demo"
        assert target == clone_parent
        assert install_deps is True
        assert open_code is False
        shell.actions.refresh_workspace(cloned)
        return SetupResult(path=cloned, message="Cloned repo.\nSuggested dependency commands:\n.> npm install")

    shell.actions.clone_repo = fake_clone

    result = shell.clone_setup_action()

    assert shell.workspace == cloned
    assert result.title == "Clone Complete"
    assert result.use_panel is False
    renderables = getattr(result.message, "renderables", ())
    assert len(renderables) == 2
    assert "Suggested dependency commands" in str(renderables[0])


def test_chat_mode_preserves_phrase_match_and_menu_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "frontend").mkdir()
    (workspace / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite"}}', encoding="utf-8")

    shell = AgentShell(workspace)
    shell.repo_agent = FakeRepoAgent()
    shell.run_tool.save_detected_profile("Start the site", open_browser=True)
    launches: list[str] = []
    shell.run_tool.launch_profile = lambda profile, open_browser=None: launches.append(profile.phrase) or profile.specs

    phrase_result = shell.handle_chat_input("start the site")
    menu_result = shell.handle_chat_input("/menu")

    assert phrase_result is not None
    assert phrase_result.title == "Saved Run Phrase"
    assert launches == ["Start the site"]
    assert menu_result is not None
    assert menu_result.return_to_menu is True


def test_edit_mode_stays_active_after_failed_proposal(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)
    calls: list[str] = []

    def fake_propose(instruction: str, *, progress_callback=None):
        calls.append(instruction)
        if progress_callback:
            progress_callback("Thinking...")
        return type("Proposal", (), {"diff": None, "message": "Still busy. Please try again."})()

    shell.actions.edit_propose = fake_propose
    shell.actions.ai_available = lambda: True
    prompts = iter(["first attempt", ""])
    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *args, **kwargs: next(prompts))

    shell.edit_mode()

    assert calls == ["first attempt"]


def test_edit_mode_stays_active_after_apply_failure(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)
    proposal = type("Proposal", (), {"diff": "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-Hello\n+Hi\n", "message": "Patch generated."})()
    propose_calls: list[str] = []
    apply_calls: list[str] = []

    def fake_propose(instruction: str, *, progress_callback=None):
        propose_calls.append(instruction)
        return proposal

    def fake_apply(current_proposal, *, progress_callback=None):
        apply_calls.append(current_proposal.message)
        raise RuntimeError("Patch repair exhausted.")

    shell.actions.edit_propose = fake_propose
    shell.actions.edit_apply = fake_apply
    shell.actions.ai_available = lambda: True
    prompts = iter(["first attempt", ""])
    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *args, **kwargs: next(prompts))
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    shell.edit_mode()

    assert propose_calls == ["first attempt"]
    assert apply_calls == ["Patch generated."]


def test_home_menu_choices_cover_all_modes() -> None:
    labels = [choice.label for choice in home_menu_choices()]

    assert labels == [
        "AI",
        "Chat",
        "Git",
        "Run",
        "Repo",
        "Setup",
        "Edit",
        "Watch",
        "Quick command / phrase",
        "Help",
        "Exit",
    ]


def test_devagent_no_args_prints_help_when_not_interactive(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Local-first agentic AI developer assistant." in result.stdout


def test_devagent_help_catalogs_command_families(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Command families:" in result.stdout
    assert "`ai`" in result.stdout
    assert "Tip: running `devagent` with no subcommand opens the interactive shell." in result.stdout
    assert "devagent run start --open-browser" in result.stdout


def test_ai_help_catalog_is_richer(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["ai", "--help"])

    assert result.exit_code == 0
    assert "Discover configured AI providers" in result.stdout
    assert "Groq" in result.stdout
    assert "save defaults" in result.stdout


def test_git_help_catalog_is_richer(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["git", "--help"])

    assert result.exit_code == 0
    assert "Guided Git workflows" in result.stdout
    assert "Common workflows:" in result.stdout


def test_chat_help_mentions_deep_and_session_flags(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["chat", "--help"])

    assert result.exit_code == 0
    assert "--deep" in result.stdout
    assert "--new-session" in result.stdout
    assert "Ask a repo-aware question" in result.stdout


def test_ai_status_command_renders_without_a_bound_workspace(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    class FakeActions:
        def ai_status(self, refresh=False):
            return AIStatusSnapshot(
                selected_provider="xai",
                fast_model="grok-3-mini",
                deep_model="grok-3-mini",
                embedding_model=None,
                providers=(
                    AIProviderStatus("xai", "XAI_API_KEY", True, generation_models=2, embedding_models=0),
                ),
            )

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "status"])

    assert result.exit_code == 0
    assert "xai" in result.stdout
    assert "grok-3-mini" in result.stdout


def test_ai_status_command_keeps_unavailable_provider_summary_compact(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    class FakeActions:
        def ai_status(self, refresh=False):
            return AIStatusSnapshot(
                selected_provider="groq",
                fast_model="llama-3.1-8b-instant",
                deep_model="llama-3.3-70b-versatile",
                embedding_model=None,
                providers=(
                    AIProviderStatus("groq", "GROQ_API_KEY", True, generation_models=11, embedding_models=0),
                    AIProviderStatus("xai", "XAI_API_KEY", False, generation_models=0, embedding_models=0, error="HTTP 403 Forbidden: credits or licenses yet."),
                ),
                warnings=("xAI: HTTP 403 Forbidden: credits or licenses yet.",),
            )

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "status"])

    assert result.exit_code == 0
    assert "xAI | XAI_API_KEY | unavailable" in result.stdout
    assert "credits or licenses yet" not in result.stdout


def test_ai_models_command_renders_partial_success(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    class FakeActions:
        def ai_models(self, provider=None, refresh=False):
            return [
                ProviderModelListing(
                    provider="gemini",
                    models=(
                        DiscoveredModel("gemini", "gemini-2.5-flash", "Gemini 2.5 Flash", ("generate",)),
                    ),
                ),
                ProviderModelListing(
                    provider="xai",
                    error="HTTP 403 Forbidden: Your newly created team doesn't have any credits or licenses yet.",
                ),
            ]

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "models"])

    assert result.exit_code == 0
    assert "gemini Models" in result.stdout
    assert "gemini-2.5-flash" in result.stdout
    assert "Unavailable providers hidden: xAI" in result.stdout
    assert "xai Models" not in result.stdout
    assert "credits or licenses yet" not in result.stdout


def test_ai_models_command_shows_provider_specific_failure_for_broken_provider(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    class FakeActions:
        def ai_models(self, provider=None, refresh=False):
            return [
                ProviderModelListing(
                    provider="xai",
                    error="HTTP 403 Forbidden: Your newly created team doesn't have any credits or licenses yet.",
                ),
            ]

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "models", "--provider", "xai"])

    assert result.exit_code == 1
    assert "DevAgent could not load live models" in result.stdout
    assert "credits or licenses yet" in result.stdout


def test_ai_use_provider_only_saves_when_discovery_is_unavailable(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)
    calls: list[str] = []

    class FakeActions:
        def ai_status(self, refresh=False):
            return AIStatusSnapshot(
                selected_provider="gemini",
                fast_model="gemini-2.5-flash",
                deep_model="gemini-2.5-pro",
                embedding_model="gemini-embedding-001",
                providers=(
                    AIProviderStatus("gemini", "GEMINI_API_KEY", True, generation_models=2, embedding_models=1),
                    AIProviderStatus("xai", "XAI_API_KEY", False, generation_models=0, embedding_models=0, error="HTTP 403 Forbidden"),
                ),
            )

        def ai_models(self, provider=None, refresh=False):
            return [ProviderModelListing(provider="xai", error="HTTP 403 Forbidden: Your newly created team doesn't have any credits or licenses yet.")]

        def save_ai_selection(self, **kwargs):
            calls.append(kwargs["provider"])
            return AISelectionResult(
                provider=kwargs["provider"],
                fast_model="grok-3-mini",
                deep_model="grok-3-mini",
                embedding_model=None,
            )

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "use", "--provider", "xai"])

    assert result.exit_code == 0
    assert calls == ["xai"]
    assert "AI Selection Saved" in result.stdout
    assert "grok-3-mini" in result.stdout
    assert "credits or licenses yet" in result.stdout


def test_ai_use_rejects_embedding_model_for_groq(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    class FakeActions:
        def ai_status(self, refresh=False):
            return AIStatusSnapshot(
                selected_provider="groq",
                fast_model="llama-3.1-8b-instant",
                deep_model="llama-3.1-8b-instant",
                embedding_model=None,
                providers=(
                    AIProviderStatus("groq", "GROQ_API_KEY", True, generation_models=2, embedding_models=0),
                ),
            )

        def ai_models(self, provider=None, refresh=False):
            return [
                ProviderModelListing(
                    provider="groq",
                    models=(
                        DiscoveredModel("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant", ("generate",)),
                    ),
                ),
            ]

    monkeypatch.setattr("devagent.cli.main._ai_actions", lambda: FakeActions())

    result = runner.invoke(app, ["ai", "use", "--provider", "groq", "--embedding-model", "ghost-embed"])

    assert result.exit_code == 2


def test_git_pull_help_shows_explicit_remote_and_branch_flags(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["git", "pull", "--help"])

    assert result.exit_code == 0
    assert "--remote" in result.stdout
    assert "--branch" in result.stdout
    assert "tracked remote" in result.stdout


def test_interactive_terminal_helper(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, tty: bool):
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr("devagent.core.shell.sys.stdin", FakeStream(True))
    monkeypatch.setattr("devagent.core.shell.sys.stdout", FakeStream(True))
    assert interactive_terminal() is True


def test_shell_pull_wizard_uses_tracking_branch_without_extra_prompts(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.git_remotes = lambda: [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
        GitRemote("upstream", "https://github.com/base/repo.git", "https://github.com/base/repo.git", "base/repo"),
    ]
    shell.actions.git_remote_branches = lambda remote: ["main", "release"] if remote == "upstream" else ["main"]
    shell.actions.git_upstream_for = lambda branch=None: "upstream/main"
    shell.actions.git_tracked_remote_target = lambda branch=None: ("upstream", "main")
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="feature/git-upgrade",
        dirty=True,
        changed_files=[" M devagent/core/shell.py"],
    )
    shell.actions.git_pull = lambda remote, branch, rebase=False: PullOutcome(
        local_branch="feature/git-upgrade",
        remote=remote,
        remote_branch=branch or "main",
        rebase=rebase,
    )

    shell.choose_named_value = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tracked pull should not ask for remote or branch"))
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    result = shell.pull_with_prompts()

    assert result.remote == "upstream"
    assert result.remote_branch == "main"
    assert result.rebase is False


def test_shell_can_save_ai_provider_selection(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.ai_status = lambda refresh=False: AIStatusSnapshot(
        selected_provider="gemini",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        embedding_model="gemini-embedding-001",
        providers=(
            AIProviderStatus("gemini", "GEMINI_API_KEY", True, generation_models=2, embedding_models=1),
            AIProviderStatus("xai", "XAI_API_KEY", False, generation_models=2, embedding_models=0),
        ),
    )
    calls: list[str] = []
    shell.actions.save_ai_selection = lambda **kwargs: calls.append(kwargs["provider"]) or AISelectionResult(
        provider=kwargs["provider"],
        fast_model="grok-3-mini",
        deep_model="grok-3-mini",
        embedding_model=None,
    )
    shell.choose_named_value = lambda *args, **kwargs: "xai"

    result = shell.set_default_provider_action()

    assert result.title == "AI Selection Saved"
    assert result.use_panel is False
    assert calls == ["xai"]


def test_shell_saving_unavailable_provider_shows_the_full_reason() -> None:
    workspace = Path("codex_test_tmp/shell-ai-provider-warning").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    shell = AgentShell(workspace)

    shell.actions.ai_status = lambda refresh=False: AIStatusSnapshot(
        selected_provider="gemini",
        fast_model="gemini-2.5-flash",
        deep_model="gemini-2.5-pro",
        embedding_model="gemini-embedding-001",
        providers=(
            AIProviderStatus("gemini", "GEMINI_API_KEY", True, generation_models=2, embedding_models=1),
            AIProviderStatus("xai", "XAI_API_KEY", False, generation_models=0, embedding_models=0, error="HTTP 403 Forbidden: credits or licenses yet."),
        ),
    )
    shell.actions.save_ai_selection = lambda **kwargs: AISelectionResult(
        provider=kwargs["provider"],
        fast_model="grok-3-mini",
        deep_model="grok-3-mini",
        embedding_model=None,
    )
    shell.choose_named_value = lambda *args, **kwargs: "xai"

    result = shell.set_default_provider_action()
    console = Console(record=True, width=120)
    console.print(result.message)
    rendered = console.export_text()

    assert result.title == "AI Selection Saved"
    assert "credits or licenses yet" in rendered


def test_devagent_no_args_with_missing_bound_workspace_shows_recovery(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_dir = tmp_path / "config-home"
    missing = tmp_path / "missing-workspace"
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(config_dir))
    ConfigManager.bind_workspace(missing)
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: True)
    launched: list[Path] = []
    monkeypatch.setattr("devagent.cli.main.AgentShell", lambda path: launched.append(path))

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert not launched
    assert "The saved workspace path no longer exists" in result.stdout
    assert "devagent workspace bind <path>" in result.stdout


def test_workspace_status_with_missing_bound_workspace_shows_actionable_error(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_dir = tmp_path / "config-home"
    missing = tmp_path / "missing-workspace"
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(config_dir))
    ConfigManager.bind_workspace(missing)
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)

    result = runner.invoke(app, ["workspace", "status"])

    assert result.exit_code != 0
    assert "The saved workspace path no longer exists" in result.stdout
    assert "devagent workspace bind <path>" in result.stdout


def test_ai_status_still_works_when_saved_workspace_path_is_missing(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_dir = tmp_path / "config-home"
    missing = tmp_path / "missing-workspace"
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(config_dir))
    ConfigManager.bind_workspace(missing)
    monkeypatch.setattr("devagent.cli.main.interactive_terminal", lambda: False)
    captured: list[Path] = []

    class FakeActions:
        def __init__(self, workspace):
            captured.append(Path(workspace))

        def ai_status(self, refresh=False):
            return AIStatusSnapshot(
                selected_provider="gemini",
                fast_model="gemini-2.5-flash",
                deep_model="gemini-2.5-pro",
                embedding_model="gemini-embedding-001",
                providers=(
                    AIProviderStatus("gemini", "GEMINI_API_KEY", True, generation_models=2, embedding_models=1),
                ),
            )

    monkeypatch.setattr("devagent.cli.main.DevAgentActions", FakeActions)

    result = runner.invoke(app, ["ai", "status"])

    assert result.exit_code == 0
    assert captured == [Path.cwd()]
    assert "gemini-2.5-flash" in result.stdout


def test_shell_pull_wizard_asks_only_for_remote_and_branch_when_untracked(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.git_remotes = lambda: [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
        GitRemote("upstream", "https://github.com/base/repo.git", "https://github.com/base/repo.git", "base/repo"),
    ]
    shell.actions.git_remote_branches = lambda remote: ["main", "release"] if remote == "upstream" else ["main"]
    shell.actions.git_upstream_for = lambda branch=None: None
    shell.actions.git_tracked_remote_target = lambda branch=None: None
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="feature/git-upgrade",
        dirty=True,
        changed_files=[" M devagent/core/shell.py"],
    )
    shell.actions.git_pull = lambda remote, branch, rebase=False: PullOutcome(
        local_branch="feature/git-upgrade",
        remote=remote,
        remote_branch=branch or "main",
        rebase=rebase,
    )

    picks = iter(["upstream", "release"])
    shell.choose_named_value = lambda *args, **kwargs: next(picks)
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    result = shell.pull_with_prompts()

    assert result.remote == "upstream"
    assert result.remote_branch == "release"
    assert result.rebase is False


def test_shell_push_wizard_uses_tracking_branch_without_extra_prompts(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.git_remotes = lambda: [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
    ]
    shell.actions.git_remote_branches = lambda remote: ["main", "feature/git-upgrade"]
    shell.actions.git_local_branches = lambda: ["main", "feature/git-upgrade"]
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="feature/git-upgrade",
        dirty=True,
        changed_files=[" M devagent/cli/main.py"],
    )
    shell.actions.git_upstream_for = lambda branch=None: "origin/feature/git-upgrade"
    shell.actions.git_tracked_remote_target = lambda branch=None: ("origin", "feature/git-upgrade")
    shell.actions.git_push = lambda **kwargs: PushOutcome(
        remote=kwargs["remote"],
        local_branch=kwargs["local_branch"],
        remote_branch=kwargs["remote_branch"],
        set_upstream=kwargs["set_upstream"],
        force_with_lease=kwargs.get("force_with_lease", False),
    )

    shell.choose_named_value = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tracked push should not ask for remote or branch"))
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    result = shell.push_with_prompts()

    assert result.remote == "origin"
    assert result.local_branch == "feature/git-upgrade"
    assert result.remote_branch == "feature/git-upgrade"
    assert result.set_upstream is False
    assert result.force_with_lease is False


def test_shell_push_wizard_asks_only_for_remote_and_branch_when_untracked(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.git_remotes = lambda: [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
        GitRemote("upstream", "https://github.com/base/repo.git", "https://github.com/base/repo.git", "base/repo"),
    ]
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="feature/git-upgrade",
        dirty=True,
        changed_files=[" M devagent/cli/main.py"],
    )
    shell.actions.git_upstream_for = lambda branch=None: None
    shell.actions.git_tracked_remote_target = lambda branch=None: None
    shell.actions.git_push = lambda **kwargs: PushOutcome(
        remote=kwargs["remote"],
        local_branch=kwargs["local_branch"],
        remote_branch=kwargs["remote_branch"],
        set_upstream=kwargs["set_upstream"],
        force_with_lease=kwargs.get("force_with_lease", False),
    )

    picks = iter(["origin"])
    shell.choose_named_value = lambda *args, **kwargs: next(picks)
    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *args, **kwargs: "feature/git-upgrade")
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    result = shell.push_with_prompts()

    assert result.remote == "origin"
    assert result.remote_branch == "feature/git-upgrade"
    assert result.set_upstream is True


def test_shell_push_wizard_ignores_broken_tracking_branch(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    shell.actions.git_remotes = lambda: [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
    ]
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="bug",
        dirty=True,
        changed_files=[" M README.md"],
    )
    shell.actions.git_upstream_for = lambda branch=None: "origin/origin"
    shell.actions.git_tracked_remote_target = lambda branch=None: None
    shell.actions.git_push = lambda **kwargs: PushOutcome(
        remote=kwargs["remote"],
        local_branch=kwargs["local_branch"],
        remote_branch=kwargs["remote_branch"],
        set_upstream=kwargs["set_upstream"],
        force_with_lease=kwargs.get("force_with_lease", False),
    )

    monkeypatch.setattr("devagent.core.shell.Prompt.ask", lambda *args, **kwargs: "bug")
    confirms = iter([True])
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: next(confirms))

    result = shell.push_with_prompts()

    assert result.remote == "origin"
    assert result.remote_branch == "bug"
    assert result.set_upstream is True


def test_shell_pr_wizard_auto_detects_repos_and_only_asks_for_base_branch(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)

    remotes = [
        GitRemote("origin", "https://github.com/me/repo.git", "https://github.com/me/repo.git", "me/repo"),
        GitRemote("upstream", "https://github.com/base/repo.git", "https://github.com/base/repo.git", "base/repo"),
    ]
    shell.actions.git_remotes = lambda: remotes
    shell.actions.git_remote_branches = lambda remote: ["main", "release"]
    shell.actions.git_local_branches = lambda: ["feature/git-upgrade", "main"]
    shell.actions.workspace_status = lambda: WorkspaceSnapshot(
        project=ProjectInfo(path=workspace, project_types=["python"], package_files=[], file_tree=[]),
        is_repo=True,
        branch="feature/git-upgrade",
        dirty=True,
        changed_files=[" M devagent/tools/git_tool.py"],
    )
    shell.actions.pr_preview = lambda **kwargs: PullRequestPreview(
        summary="Open PR from `feature/git-upgrade` into `release`.",
        title="feat: upgrade guided Git workflows",
        body=f"Base {kwargs['base_repo']} -> Head {kwargs['head_repo']}",
    )

    shell.choose_repo_slug = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PR flow should auto-detect repos"))
    value_picks = iter(["release"])
    shell.choose_named_value = lambda *args, **kwargs: next(value_picks)
    monkeypatch.setattr("devagent.core.shell.Confirm.ask", lambda *args, **kwargs: True)

    preview, options = shell.pr_preview_with_prompts(return_options=True)

    assert preview.title == "feat: upgrade guided Git workflows"
    assert options["base_repo"] == "base/repo"
    assert options["base_branch"] == "release"
    assert options["head_repo"] == "me/repo"
    assert options["head_branch"] == "feature/git-upgrade"
    assert options["draft"] is True


def test_shell_merge_continue_is_blocked_when_conflicts_remain(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shell = AgentShell(workspace)
    shell.actions.git_merge_in_progress = lambda: True
    shell.actions.merge_conflicts = lambda: [type("Conflict", (), {"path": "README.md", "markers": 1})()]

    result = shell.perform_git_intent(GitIntent(action="merge_continue"))

    assert result.title == "Merge Continue"
    assert result.tone == "warning"
    assert "Resolve all merge conflicts" in str(result.message)
