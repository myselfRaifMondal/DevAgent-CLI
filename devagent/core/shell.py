from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.prompt import Confirm, Prompt

from devagent.cli.prompts import MenuChoice, choose_directory, choose_menu_action
from devagent.cli.renderers import (
    ai_models_collection_renderable,
    ai_selection_renderable,
    ai_status_renderable,
    commit_suggestion_renderable,
    git_pull_summary_renderable,
    git_push_summary_renderable,
    git_remotes_renderable,
    insight_lines,
    insights_renderable,
    merge_conflicts_renderable,
    package_lines,
    packages_renderable,
    pr_preview_renderable,
    run_inventory_renderable,
    run_launch_message,
    workspace_status_table,
)
from devagent.cli.ui import app_panel, console, hero_panel, render_chat_markdown
from devagent.core.actions import AISelectionResult, DevAgentActions, PullOutcome, PullRequestPreview, PushOutcome, RunProfile, RunLaunchResult, WorkspaceSnapshot
from devagent.tools.ai import ai_setup_hint
from devagent.tools.git_tool import GitError, GitRemote


@dataclass(frozen=True)
class ShellResult:
    title: str
    message: RenderableType
    tone: str = "info"
    exit_shell: bool = False
    return_to_menu: bool = False
    use_panel: bool = True


@dataclass(frozen=True)
class GitIntent:
    action: str
    name: str | None = None
    path: str | None = None
    message: str | None = None


def interactive_terminal() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)()) and bool(getattr(sys.stdout, "isatty", lambda: False)())


def home_menu_choices() -> list[MenuChoice]:
    return [
        MenuChoice("AI", "ai"),
        MenuChoice("Chat", "chat"),
        MenuChoice("Git", "git"),
        MenuChoice("Run", "run"),
        MenuChoice("Repo", "repo"),
        MenuChoice("Setup", "setup"),
        MenuChoice("Edit", "edit"),
        MenuChoice("Watch", "watch"),
        MenuChoice("Quick command / phrase", "quick"),
        MenuChoice("Help", "help"),
        MenuChoice("Exit", "exit"),
    ]


def ai_menu_choices() -> list[MenuChoice]:
    return [
        MenuChoice("Show current AI status", "status"),
        MenuChoice("Show available models", "models"),
        MenuChoice("Choose the default provider", "provider"),
        MenuChoice("Choose the default chat model", "chat_model"),
        MenuChoice("Choose the default deep model", "deep_model"),
        MenuChoice("Choose the embedding model", "embedding_model"),
        MenuChoice("Reset saved AI settings", "reset"),
        MenuChoice("Back to home", "back"),
    ]


def git_menu_choices(*, merge_in_progress: bool) -> list[MenuChoice]:
    choices = [
        MenuChoice("See what changed and which branch you're on", "status"),
        MenuChoice("Stage everything for the next commit", "add_all"),
        MenuChoice("Stage a specific file or folder", "add_path"),
        MenuChoice("Create a branch for new work", "branch_create"),
        MenuChoice("Switch to another branch safely", "branch_switch"),
        MenuChoice("Commit with an auto-generated message", "commit_auto"),
        MenuChoice("Suggest a commit message without committing", "commit_suggest"),
        MenuChoice("Pull the latest changes into this branch", "pull"),
        MenuChoice("Push this branch to GitHub", "push"),
        MenuChoice("Preview the PR title and description", "pr_preview"),
        MenuChoice("Open a PR for this branch", "pr_create"),
        MenuChoice("Check merge conflicts", "merge_conflicts"),
    ]
    if merge_in_progress:
        choices.append(MenuChoice("Abort the current merge", "merge_abort"))
        choices.append(MenuChoice("Continue the current merge after resolution", "merge_continue"))
    choices.append(MenuChoice("Back to home", "back"))
    return choices


def run_menu_choices() -> list[MenuChoice]:
    return [
        MenuChoice("Start detected stack", "start_detected"),
        MenuChoice("Start a saved phrase", "start_saved"),
        MenuChoice("Save detected stack as a phrase", "save_detected"),
        MenuChoice("Save a custom command as a phrase", "save_manual"),
        MenuChoice("List detected targets and saved phrases", "list"),
        MenuChoice("Forget a saved phrase", "forget"),
        MenuChoice("Back to home", "back"),
    ]


def repo_menu_choices() -> list[MenuChoice]:
    return [
        MenuChoice("Show workspace status", "status"),
        MenuChoice("Reindex the workspace", "index"),
        MenuChoice("Show package dependencies", "packages"),
        MenuChoice("Run inspect", "inspect"),
        MenuChoice("Rebind workspace", "bind"),
        MenuChoice("Back to home", "back"),
    ]


def setup_menu_choices() -> list[MenuChoice]:
    return [
        MenuChoice("Clone a GitHub repo", "clone"),
        MenuChoice("Publish a local project to GitHub", "publish"),
        MenuChoice("Run the guided new project flow", "guided"),
        MenuChoice("Back to home", "back"),
    ]


class AgentShell:
    def __init__(self, workspace: Path):
        resolved = workspace.expanduser().resolve()
        self.actions = DevAgentActions(resolved)
        self.deep_mode = False

    @property
    def workspace(self) -> Path:
        return self.actions.workspace

    @workspace.setter
    def workspace(self, value: Path) -> None:
        self.actions.refresh_workspace(value)

    @property
    def repo_agent(self):
        return self.actions.repo_agent

    @repo_agent.setter
    def repo_agent(self, value) -> None:
        self.actions.repo_agent = value

    @property
    def run_tool(self):
        return self.actions.run_tool

    def welcome_message(self) -> str:
        snapshot = self.actions.workspace_status()
        inventory = self.actions.run_inventory()
        lines = [
            f"Workspace: {snapshot.project.path}",
            f"Project types: {', '.join(snapshot.project.project_types) or 'unknown'}",
            f"Saved run phrases: {len(inventory.profiles)}",
            "",
            "Modes:",
            "- AI for provider selection and model discovery",
            "- Chat for repo-aware Q&A",
            "- Git for version-control workflows",
            "- Run for launching services and saved phrases",
            "- Repo for status, indexing, packages, and inspect",
            "- Setup for clone/publish/onboarding",
            "- Edit for diff-first code changes",
            "- Watch for background file-change suggestions",
            "",
            "Quick command can route saved phrases, runtime requests, repo actions, Git requests, and chat.",
        ]
        return "\n".join(lines)

    def run(self) -> None:
        console.print(hero_panel("Agent Shell", "One menu-driven control room for chat, Git, runtime, setup, and repo work."))
        console.print(app_panel(self.welcome_message(), "Workspace Linked", tone="info", expand=False))
        while True:
            choice = choose_menu_action(console, "DevAgent Home", home_menu_choices())
            if not choice or choice == "exit":
                return
            if choice == "ai":
                self.ai_mode()
            elif choice == "chat":
                self.chat_mode()
            elif choice == "git":
                self.git_mode()
            elif choice == "run":
                self.run_mode()
            elif choice == "repo":
                self.repo_mode()
            elif choice == "setup":
                self.setup_mode()
            elif choice == "edit":
                self.edit_mode()
            elif choice == "watch":
                self.watch_mode()
            elif choice == "quick":
                self.quick_command_mode()
            elif choice == "help":
                self.display_result(self.help_result())

    def ai_mode(self) -> None:
        while True:
            action = choose_menu_action(console, "AI Mode", ai_menu_choices())
            if not action or action == "back":
                return
            try:
                if action == "status":
                    result = self.ai_status_result()
                elif action == "models":
                    result = self.ai_models_result()
                elif action == "provider":
                    result = self.set_default_provider_action()
                elif action == "chat_model":
                    result = self.set_default_model_action(kind="chat")
                elif action == "deep_model":
                    result = self.set_default_model_action(kind="deep")
                elif action == "embedding_model":
                    result = self.set_default_model_action(kind="embed")
                elif action == "reset":
                    result = self.reset_ai_settings_action()
                else:
                    result = None
            except (RuntimeError, ValueError) as exc:
                result = ShellResult("AI Mode Failed", str(exc), "error")
            self.display_result(result)

    def display_result(self, result: ShellResult | None) -> None:
        if not result:
            return
        if result.use_panel:
            console.print(app_panel(result.message, result.title, tone=result.tone, expand=False))
        else:
            console.print(result.message)

    def chat_mode(self) -> None:
        console.print(app_panel("Chat mode is ready. Ask repo questions, or use /help for shell controls.", "Chat Mode", tone="info", expand=False))
        if not self.actions.ai_available():
            console.print(app_panel(f"Answers will come from keyword search only.\n\n{ai_setup_hint()}", "No AI Provider", tone="warning", expand=False))
        while True:
            try:
                user_input = console.input("[bold bright_cyan]chat[/bold bright_cyan] [bright_black]>[/bright_black] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            if self._chat_uses_generation(user_input):
                with console.status("Thinking...") as status:
                    result = self.handle_chat_input(user_input, progress_callback=status.update)
            else:
                result = self.handle_chat_input(user_input)
            if not result:
                continue
            self.display_result(result)
            if result.exit_shell:
                raise SystemExit
            if result.return_to_menu:
                return

    def git_mode(self) -> None:
        while True:
            action = choose_menu_action(console, "Git Mode", git_menu_choices(merge_in_progress=self.actions.git_merge_in_progress()))
            if not action or action == "back":
                return
            try:
                result = self.perform_git_menu_action(action)
            except (GitError, ValueError, RuntimeError) as exc:
                result = ShellResult("Git Action Failed", str(exc), "error")
            self.display_result(result)

    def run_mode(self) -> None:
        while True:
            action = choose_menu_action(console, "Run Mode", run_menu_choices())
            if not action or action == "back":
                return
            try:
                if action == "start_detected":
                    result = self.run_detected_action()
                elif action == "start_saved":
                    result = self.run_saved_action()
                elif action == "save_detected":
                    result = self.save_detected_run_action()
                elif action == "save_manual":
                    result = self.save_manual_run_action()
                elif action == "list":
                    result = self.run_inventory_result()
                elif action == "forget":
                    result = self.forget_run_phrase_action()
                else:
                    result = None
            except (GitError, ValueError, RuntimeError) as exc:
                result = ShellResult("Run Mode Failed", str(exc), "error")
            self.display_result(result)

    def repo_mode(self) -> None:
        while True:
            action = choose_menu_action(console, "Repo Mode", repo_menu_choices())
            if not action or action == "back":
                return
            try:
                if action == "status":
                    result = self.workspace_status_result()
                elif action == "index":
                    count = self.actions.index_workspace()
                    result = ShellResult("Index Complete", f"Indexed {count} chunks from\n{self.workspace}", "success")
                elif action == "packages":
                    packages = self.actions.packages()
                    result = ShellResult("Package Scan", packages_renderable(self.workspace, packages), "info", use_panel=False)
                elif action == "inspect":
                    findings = self.actions.inspect()
                    result = ShellResult("DevAgent Insights", insights_renderable(findings), "info", use_panel=False)
                elif action == "bind":
                    result = self.rebind_workspace_action()
                else:
                    result = None
            except (GitError, ValueError, RuntimeError) as exc:
                result = ShellResult("Repo Mode Failed", str(exc), "error")
            self.display_result(result)

    def setup_mode(self) -> None:
        while True:
            action = choose_menu_action(console, "Setup Mode", setup_menu_choices())
            if not action or action == "back":
                return
            try:
                if action == "clone":
                    result = self.clone_setup_action()
                elif action == "publish":
                    result = self.publish_setup_action()
                elif action == "guided":
                    result = self.guided_project_action()
                else:
                    result = None
            except (GitError, ValueError, RuntimeError) as exc:
                result = ShellResult("Setup Failed", str(exc), "error")
            self.display_result(result)

    def edit_mode(self) -> None:
        if not self.actions.ai_available():
            # Say so before asking for an instruction that can only be refused.
            console.print(app_panel(f"Edit needs an AI provider.\n\n{ai_setup_hint()}", "Edit Mode", tone="warning", expand=False))
            return
        console.print(app_panel("Describe the edit you want. DevAgent will propose a diff before changing files.", "Edit Mode", tone="info", expand=False))
        while True:
            instruction = Prompt.ask("Edit instruction").strip()
            if not instruction:
                return
            with console.status("Thinking...") as status:
                proposal = self.actions.edit_propose(instruction, progress_callback=status.update)
            self.display_result(ShellResult("Proposed Change", proposal.diff or proposal.message, "info"))
            if not proposal.diff:
                continue
            if Confirm.ask("Apply this diff?", default=False):
                try:
                    with console.status("Applying patch...") as status:
                        self.actions.edit_apply(proposal, progress_callback=status.update)
                except RuntimeError as exc:
                    self.display_result(ShellResult("Edit Failed", str(exc), "error"))
                    continue
                else:
                    self.display_result(ShellResult("Edit Applied", "Applied the proposed change.", "success"))
                return
            self.display_result(ShellResult("Edit Skipped", "No files were changed.", "warning"))
            continue

    def watch_mode(self) -> None:
        interval = float(Prompt.ask("Polling interval", default="1.0"))
        console.print(app_panel(f"Watching {self.workspace}\nPress Ctrl+C to stop.", "Watch Mode", tone="info", expand=False))
        self.actions.watch_workspace(interval=interval)
        self.display_result(ShellResult("Watch Mode", "Watch mode stopped and returned to the shell.", "success"))

    def quick_command_mode(self) -> None:
        text = Prompt.ask("Quick command").strip()
        if not text:
            return
        try:
            result = self.handle_input(text)
        except (GitError, ValueError, RuntimeError) as exc:
            result = ShellResult("Quick Command Failed", str(exc), "error")
        self.display_result(result)

    def handle_input(self, raw_text: str, *, progress_callback=None) -> ShellResult | None:
        text = raw_text.strip()
        if not text:
            return None
        if text.startswith("/"):
            return self.handle_chat_command(text)

        matched_profile = self.actions.find_run_profile(text)
        if matched_profile:
            return self.run_profile_result(matched_profile)

        if is_runtime_intent(text):
            open_browser = wants_browser(text)
            launched = self.actions.run_start(open_browser=open_browser)
            return self.run_launch_result("Runtime Agent", launched)

        repo_action = infer_repo_action(text)
        if repo_action:
            return self.perform_repo_intent(repo_action)

        git_intent = infer_git_intent(text)
        if git_intent:
            return self.perform_git_intent(git_intent)

        answer = self.actions.chat(text, deep=self.deep_mode, progress_callback=progress_callback)
        return ShellResult("DevAgent", render_chat_markdown(answer), "info")

    def handle_chat_input(self, raw_text: str, *, progress_callback=None) -> ShellResult | None:
        text = raw_text.strip()
        if not text:
            return None
        if text.startswith("/"):
            return self.handle_chat_command(text)
        matched_profile = self.actions.find_run_profile(text)
        if matched_profile:
            return self.run_profile_result(matched_profile)
        answer = self.actions.chat(text, deep=self.deep_mode, progress_callback=progress_callback)
        return ShellResult("DevAgent", render_chat_markdown(answer), "info")

    def _chat_uses_generation(self, raw_text: str) -> bool:
        text = raw_text.strip()
        if not text or text.startswith("/"):
            return False
        return self.actions.find_run_profile(text) is None

    def handle_chat_command(self, text: str) -> ShellResult:
        command = text.strip().casefold()
        if command == "/help":
            return ShellResult(
                "Chat Mode",
                "\n".join(
                    [
                        "Ask repo questions naturally in this mode.",
                        "",
                        "Controls:",
                        "/help      Show chat controls",
                        "/deep      Toggle deeper answer mode",
                        "/clear     Clear stored chat memory",
                        "/workspace Show the active workspace snapshot",
                        "/menu      Return to the home menu",
                        "/exit      Leave DevAgent entirely",
                    ]
                ),
                "info",
            )
        if command == "/deep":
            self.deep_mode = not self.deep_mode
            state = "enabled" if self.deep_mode else "disabled"
            return ShellResult("Deep Mode", f"Deep repo-answer mode is now {state}.", "success")
        if command == "/clear":
            self.actions.clear_chat_session()
            return ShellResult("Chat Memory Cleared", "Cleared the stored workspace conversation context.", "success")
        if command == "/workspace":
            return self.workspace_status_result()
        if command == "/menu":
            return ShellResult("Chat Mode", "Returned to the home menu.", "success", return_to_menu=True)
        if command == "/exit":
            return ShellResult("Agent Shell", "See you soon.", "success", exit_shell=True)
        return ShellResult("Chat Mode", f"Unknown chat command: {text}", "warning")

    def help_result(self) -> ShellResult:
        return ShellResult(
            "Agent Shell",
            "\n".join(
                [
                    "Use the Home menu to move between modes.",
                    "",
                    "Modes:",
                    "AI     -> provider selection, model discovery, and saved defaults",
                    "Chat   -> repo-aware conversation with session memory",
                    "Git    -> branch, commit, push, PR, and merge helpers",
                    "Run    -> start services and manage saved phrases",
                    "Repo   -> status, index, packages, inspect, rebind",
                    "Setup  -> clone, publish, or guided onboarding",
                    "Edit   -> diff-first code changes with confirmation",
                    "Watch  -> file-change watcher that returns to the shell",
                    "Quick  -> one-line routing for phrases, runtime, repo, Git, or chat",
                    "",
                    "Explicit commands like `devagent git commit` still work outside the shell.",
                ]
            ),
            "info",
        )

    def ai_status_result(self) -> ShellResult:
        return ShellResult("AI Status", ai_status_renderable(self.actions.ai_status()), "info", use_panel=False)

    def ai_models_result(self, provider: str | None = None) -> ShellResult:
        listings = self.actions.ai_models(provider=provider, refresh=True)
        if not listings:
            return ShellResult("AI Models", "No AI providers are configured yet. Add a supported API key first.", "warning")
        renderable = ai_models_collection_renderable(listings, show_error_detail=provider is not None)
        return ShellResult("AI Models", renderable, "info", use_panel=False)

    def workspace_status_result(self) -> ShellResult:
        snapshot = self.actions.workspace_status()
        return ShellResult("Workspace Status", workspace_status_table(snapshot), "info", use_panel=False)

    def run_inventory_result(self) -> ShellResult:
        return ShellResult("Run Inventory", run_inventory_renderable(self.workspace, self.actions.run_inventory()), "info", use_panel=False)

    def run_launch_result(self, title: str, result: RunLaunchResult) -> ShellResult:
        return ShellResult(title, run_launch_message(self.workspace, result), "success")

    def run_profile_result(self, profile: RunProfile) -> ShellResult:
        launched = self.actions.run_launch_profile(profile)
        return self.run_launch_result("Saved Run Phrase", launched)

    def rebind_workspace_action(self) -> ShellResult:
        selected = choose_directory(console, self.workspace, "Choose a workspace to bind")
        snapshot = self.actions.bind_workspace(selected)
        return ShellResult("Workspace Linked", workspace_status_table(snapshot), "success", use_panel=False)

    def set_default_provider_action(self) -> ShellResult:
        status = self.actions.ai_status(refresh=True)
        provider_names = [item.provider for item in status.providers]
        if not provider_names:
            return ShellResult("AI Provider", "No AI providers are configured yet. Add a supported API key first.", "warning")
        current = status.selected_provider or provider_names[0]
        provider = self.choose_named_value("Choose the default provider", provider_names, default=current)
        selection = self.actions.save_ai_selection(provider=provider)
        provider_status = next((item for item in status.providers if item.provider == provider), None)
        if provider_status and provider_status.error:
            selection = AISelectionResult(
                provider=selection.provider,
                fast_model=selection.fast_model,
                deep_model=selection.deep_model,
                embedding_model=selection.embedding_model,
                warnings=(
                    f"Saved provider `{provider}`, but DevAgent could not verify its visible models right now. {provider_status.error}",
                ),
            )
        return ShellResult("AI Selection Saved", ai_selection_renderable(selection), "success", use_panel=False)

    def set_default_model_action(self, *, kind: str) -> ShellResult:
        status = self.actions.ai_status(refresh=True)
        provider_names = [item.provider for item in status.providers]
        if not provider_names:
            return ShellResult("AI Models", "No AI providers are configured yet. Add a supported API key first.", "warning")
        default_provider = status.selected_provider or provider_names[0]
        provider = self.choose_named_value("Choose the provider to configure", provider_names, default=default_provider)
        listing = self.actions.ai_models(provider=provider, refresh=True)[0]
        if listing.error:
            return ShellResult("AI Models", f"Could not load the visible models for {provider} right now.\n\n{listing.error}", "warning")
        models = list(listing.models)
        if kind == "embed":
            models = [model for model in models if "embed" in model.capabilities]
            if not models:
                return ShellResult("Embedding Models", f"{provider} does not currently expose embedding models in DevAgent.", "warning")
            default_model = status.embedding_model or models[0].id
            chosen = self.choose_named_value("Choose the embedding model", [model.id for model in models], default=default_model)
            selection = self.actions.save_ai_selection(provider=provider, embedding_model=chosen)
            return ShellResult("AI Selection Saved", ai_selection_renderable(selection), "success", use_panel=False)

        generation_models = [model for model in models if "generate" in model.capabilities]
        if not generation_models:
            return ShellResult("AI Models", f"No generation models are currently visible for {provider}.", "warning")
        if kind == "deep":
            default_model = status.deep_model or generation_models[0].id
            chosen = self.choose_named_value("Choose the default deep model", [model.id for model in generation_models], default=default_model)
            selection = self.actions.save_ai_selection(provider=provider, deep_model=chosen)
        else:
            default_model = status.fast_model or generation_models[0].id
            chosen = self.choose_named_value("Choose the default chat model", [model.id for model in generation_models], default=default_model)
            selection = self.actions.save_ai_selection(provider=provider, fast_model=chosen)
        return ShellResult("AI Selection Saved", ai_selection_renderable(selection), "success", use_panel=False)

    def reset_ai_settings_action(self) -> ShellResult:
        if not Confirm.ask("Reset saved AI provider and model selections?", default=False):
            return ShellResult("AI Settings", "Kept the current AI settings.", "warning")
        self.actions.reset_ai_settings()
        return ShellResult("AI Settings Reset", ai_status_renderable(self.actions.ai_status(refresh=False)), "success", use_panel=False)

    def clone_setup_action(self) -> ShellResult:
        repo_url = Prompt.ask("Paste the GitHub repository page URL")
        target = choose_directory(console, self.workspace.parent if self.workspace.parent.exists() else Path.cwd(), "Choose where to clone the repo")
        install_deps = Confirm.ask("Install dependencies if DevAgent detects them?", default=False)
        open_code = Confirm.ask("Open the project in VS Code after setup?", default=False)
        result = self.actions.clone_repo(repo_url, target=target, install_deps=install_deps, open_code=open_code)
        return self.setup_result_with_workspace("Clone Complete", result.message)

    def publish_setup_action(self) -> ShellResult:
        local_path = choose_directory(console, self.workspace, "Choose your local project folder")
        repo_name = Prompt.ask("GitHub repository name", default=local_path.name)
        private = Confirm.ask("Create the GitHub repo as private?", default=False)
        push = Confirm.ask("Push the local project after creating the remote?", default=True)
        result = self.actions.publish_repo(local_path, repo_name=repo_name, private=private, push=push)
        return self.setup_result_with_workspace("Publish Complete", result.message)

    def guided_project_action(self) -> ShellResult:
        mode = Prompt.ask(
            "Do you already have a GitHub repo, or do you have a local project to publish?",
            choices=["github", "local"],
            default="github",
        )
        start = self.workspace.parent if self.workspace.parent.exists() else Path.cwd()
        if mode == "github":
            repo_url = Prompt.ask("Paste the GitHub repository page URL")
            target = choose_directory(console, start, "Choose where to clone the repo")
            install_deps = Confirm.ask("Install dependencies if DevAgent detects them?", default=False)
            open_code = Confirm.ask("Open the project in VS Code after setup?", default=False)
            result = self.actions.clone_repo(repo_url, target=target, install_deps=install_deps, open_code=open_code)
            return self.setup_result_with_workspace("Clone Complete", result.message)

        local_path = choose_directory(console, start, "Choose your local project folder")
        repo_name = Prompt.ask("GitHub repository name", default=local_path.name)
        private = Confirm.ask("Create the GitHub repo as private?", default=False)
        push = Confirm.ask("Push the local project after creating the remote?", default=True)
        result = self.actions.publish_repo(local_path, repo_name=repo_name, private=private, push=push)
        return self.setup_result_with_workspace("Publish Complete", result.message)

    def run_detected_action(self) -> ShellResult:
        open_browser = Confirm.ask("Open the app in the browser after launch?", default=True)
        launched = self.actions.run_start(open_browser=open_browser)
        return self.run_launch_result("Services Started", launched)

    def run_saved_action(self) -> ShellResult:
        profile = self.choose_saved_profile("Choose a saved phrase to launch")
        if not profile:
            return ShellResult("Run Mode", "No saved run phrases are available yet.", "warning")
        launched = self.actions.run_launch_profile(profile)
        return self.run_launch_result("Saved Run Phrase", launched)

    def save_detected_run_action(self) -> ShellResult:
        phrase = Prompt.ask("Saved phrase")
        open_browser = Confirm.ask("Should this phrase open the browser after launch?", default=False)
        description = Prompt.ask("Description (optional)", default="").strip() or None
        profile = self.actions.save_run_profile(phrase, open_browser=open_browser, description=description)
        body = (
            f"Saved phrase {profile.phrase} for the detected stack.\n\n"
            + "\n".join(f"- {spec.scope(self.workspace)}: {spec.display_command}" for spec in profile.specs)
            + f"\n\nOpen browser: {'yes' if profile.open_browser else 'no'}"
        )
        return ShellResult("Run Phrase Saved", body, "success")

    def save_manual_run_action(self) -> ShellResult:
        phrase = Prompt.ask("Saved phrase")
        use_workspace = Confirm.ask("Use the current workspace as the command folder?", default=True)
        cwd = self.workspace if use_workspace else choose_directory(console, self.workspace, "Choose the working directory for this command")
        command = Prompt.ask("Command to launch")
        open_browser = Confirm.ask("Should this phrase open the browser after launch?", default=False)
        description = Prompt.ask("Description (optional)", default="").strip() or None
        profile = self.actions.save_run_profile(phrase, command=command, cwd=cwd, open_browser=open_browser, description=description)
        spec = profile.specs[0]
        body = (
            f"Saved phrase {profile.phrase}\n\n"
            f"Folder: {spec.scope(self.workspace)}\n"
            f"Command: {spec.display_command}\n"
            f"Open browser: {'yes' if profile.open_browser else 'no'}"
        )
        return ShellResult("Run Phrase Saved", body, "success")

    def forget_run_phrase_action(self) -> ShellResult:
        profile = self.choose_saved_profile("Choose a saved phrase to forget")
        if not profile:
            return ShellResult("Run Mode", "No saved run phrases are available yet.", "warning")
        deleted = self.actions.delete_run_profile(profile.phrase)
        if not deleted:
            return ShellResult("Run Mode", f"No saved run phrase found: {profile.phrase}", "warning")
        return ShellResult("Run Phrase Removed", f"Removed saved run phrase {profile.phrase}.", "success")

    def setup_result_with_workspace(self, title: str, message: str) -> ShellResult:
        snapshot = self.actions.workspace_status()
        return ShellResult(title, Group(message, workspace_status_table(snapshot)), "success", use_panel=False)

    def choose_saved_profile(self, title: str) -> RunProfile | None:
        profiles = list(self.actions.run_inventory().profiles.values())
        if not profiles:
            return None
        choices = [MenuChoice(profile.phrase, profile.phrase) for profile in profiles]
        picked = choose_menu_action(console, title, choices)
        if not picked:
            return None
        return self.actions.find_run_profile(picked)

    def perform_repo_intent(self, action: str) -> ShellResult:
        if action == "status":
            return self.workspace_status_result()
        if action == "index":
            count = self.actions.index_workspace()
            return ShellResult("Index Complete", f"Indexed {count} chunks from\n{self.workspace}", "success")
        if action == "packages":
            packages = self.actions.packages()
            return ShellResult("Package Scan", packages_lines_or_table(packages, self.workspace), "info", use_panel=not packages)
        if action == "inspect":
            findings = self.actions.inspect()
            renderable = insights_renderable(findings)
            return ShellResult("DevAgent Insights", renderable, "info", use_panel=not findings)
        raise RuntimeError(f"Unknown repo action: {action}")

    def perform_git_intent(self, intent: GitIntent) -> ShellResult:
        action = intent.action
        if action == "status":
            return ShellResult("Git Status", self.actions.git_status(), "info")
        if action == "add_all":
            self.actions.git_add(".")
            return ShellResult("Git Add", "Staged the whole workspace.", "success")
        if action == "add_path":
            path = intent.path or Prompt.ask("Path to stage", default=".")
            self.actions.git_add(path)
            return ShellResult("Git Add", f"Staged {path}.", "success")
        if action == "branch_create":
            name = intent.name or Prompt.ask("New branch name")
            self.actions.git_create_branch(name)
            return ShellResult("Branch Created", f"Created and switched to branch {name}.", "success")
        if action == "branch_switch":
            name = intent.name or Prompt.ask("Branch to switch to")
            force = Confirm.ask("Allow switching with uncommitted changes?", default=False)
            self.actions.git_switch_branch(name, force=force)
            return ShellResult("Branch Switched", f"Switched to branch {name}.", "success")
        if action == "commit":
            outcome = self.actions.git_commit(message=intent.message, all_files=True)
            return ShellResult("Commit Complete", f"Created commit {outcome.commit_id}\n\n{outcome.message}", "success")
        if action == "commit_suggest":
            suggestion = self.actions.suggest_commit(conventional=True)
            return ShellResult("Commit Suggestion", commit_suggestion_renderable(suggestion), "info", use_panel=False)
        if action == "pull":
            result = self.pull_with_prompts()
            return ShellResult("Pull Complete", git_pull_summary_renderable(result), "success", use_panel=False)
        if action == "push":
            result = self.push_with_prompts()
            return ShellResult("Push Complete", git_push_summary_renderable(result), "success", use_panel=False)
        if action == "pr_preview":
            preview = self.pr_preview_with_prompts(draft=False)
            return ShellResult("Pull Request Preview", pr_preview_renderable(preview), "info", use_panel=False)
        if action == "pr_create":
            preview, options = self.pr_preview_with_prompts(return_options=True)
            console.print(pr_preview_renderable(preview))
            if not preview.ready_to_create:
                guidance = "\n".join(f"- {line}" for line in preview.readiness) or "This branch needs another commit or push before opening a PR."
                return ShellResult("Pull Request Not Ready", guidance, "warning")
            if not Confirm.ask("Create this pull request now?", default=True):
                return ShellResult("Pull Request", "Cancelled PR creation.", "warning")
            url = self.actions.pr_create(
                base=options["base_branch"],
                base_repo=options["base_repo"],
                head_branch=options["head_branch"],
                head_repo=options["head_repo"],
                title=preview.title,
                body=preview.body,
                draft=options["draft"],
            )
            return ShellResult("Pull Request", url or "Pull request created.", "success")
        if action == "merge_conflicts":
            conflicts = self.actions.merge_conflicts()
            merge_in_progress = self.actions.git_merge_in_progress()
            if not conflicts and not merge_in_progress:
                return ShellResult("Merge Conflicts", "No active merge conflicts or merge in progress.", "success")
            return ShellResult(
                "Merge Conflicts",
                merge_conflict_status_renderable(conflicts, merge_in_progress=merge_in_progress),
                "info",
                use_panel=False,
            )
        if action == "merge_abort":
            if not self.actions.git_merge_in_progress():
                return ShellResult("Merge Abort", "There is no active merge to abort.", "warning")
            self.actions.merge_abort()
            return ShellResult("Merge Abort", "Aborted the merge.", "success")
        if action == "merge_continue":
            if not self.actions.git_merge_in_progress():
                return ShellResult("Merge Continue", "There is no active merge to continue.", "warning")
            if self.actions.merge_conflicts():
                return ShellResult("Merge Continue", "Resolve all merge conflicts before continuing the merge.", "warning")
            self.actions.merge_continue()
            return ShellResult("Merge Continue", "Continued the merge.", "success")
        raise RuntimeError(f"Unknown Git action: {action}")

    def perform_git_menu_action(self, action: str) -> ShellResult:
        if action == "add_path":
            return self.perform_git_intent(GitIntent(action="add_path", path=Prompt.ask("Path to stage", default=".")))
        if action == "branch_create":
            return self.perform_git_intent(GitIntent(action="branch_create", name=Prompt.ask("New branch name")))
        if action == "branch_switch":
            name = Prompt.ask("Branch to switch to")
            force = Confirm.ask("Allow switching with uncommitted changes?", default=False)
            self.actions.git_switch_branch(name, force=force)
            return ShellResult("Branch Switched", f"Switched to branch {name}.", "success")
        if action == "commit_auto":
            custom = Confirm.ask("Write your own commit message?", default=False)
            message = Prompt.ask("Commit message") if custom else None
            outcome = self.actions.git_commit(message=message, all_files=True)
            return ShellResult("Commit Complete", f"Created commit {outcome.commit_id}\n\n{outcome.message}", "success")
        return self.perform_git_intent(GitIntent(action=action))

    def pr_preview_result(self, preview: PullRequestPreview) -> ShellResult:
        return ShellResult("Pull Request Preview", f"TITLE\n{preview.title}\n\nBODY\n{preview.body}", "info")

    def pull_with_prompts(self) -> PullOutcome:
        snapshot = self.actions.workspace_status()
        current_branch = snapshot.branch or "current branch"
        tracked_target = self.actions.git_tracked_remote_target(current_branch)
        raw_upstream = self.actions.git_upstream_for(current_branch)
        remotes = self.actions.git_remotes()
        if raw_upstream and tracked_target is None:
            console.print(
                app_panel(
                    f"The tracked upstream `{raw_upstream}` does not look usable, so DevAgent will ignore it for this pull.",
                    "Pull Guidance",
                    tone="warning",
                    expand=False,
                )
            )
        if tracked_target:
            remote, branch = tracked_target
        else:
            remote = remotes[0].name if len(remotes) == 1 else self.choose_named_value(
                "Choose the remote to pull from",
                [item.name for item in remotes],
                default=self.actions.git_tool.default_remote_name() or "origin",
            )
            branches = self.actions.git_remote_branches(remote)
            branch = self.choose_named_value(
                "Choose the branch to pull into this branch",
                branches,
                default=current_branch,
                allow_custom=True,
            )
        summary = PullOutcome(
            local_branch=current_branch,
            remote=remote,
            remote_branch=branch,
        )
        if not tracked_target:
            console.print(git_remotes_renderable(remotes))
        console.print(git_pull_summary_renderable(summary))
        if not Confirm.ask("Run this pull now?", default=True):
            raise RuntimeError("Pull cancelled.")
        return self.actions.git_pull(remote=remote, branch=branch)

    def push_with_prompts(self) -> PushOutcome:
        snapshot = self.actions.workspace_status()
        current_branch = snapshot.branch or "current branch"
        remotes = self.actions.git_remotes()
        tracked_target = self.actions.git_tracked_remote_target(current_branch)
        raw_upstream = self.actions.git_upstream_for(current_branch)
        if raw_upstream and tracked_target is None:
            console.print(
                app_panel(
                    f"The tracked upstream `{raw_upstream}` does not look usable, so DevAgent will publish `{current_branch}` with a clean target.",
                    "Push Guidance",
                    tone="warning",
                    expand=False,
                )
            )
        if tracked_target:
            remote, remote_branch = tracked_target
            set_upstream = False
        else:
            remote = remotes[0].name if len(remotes) == 1 else self.choose_named_value(
                "Choose where to publish this branch",
                [item.name for item in remotes],
                default=self.actions.git_tool.default_remote_name() or "origin",
            )
            remote_branch = Prompt.ask("Branch name to create on the remote", default=current_branch).strip() or current_branch
            set_upstream = True
        summary = PushOutcome(
            remote=remote,
            local_branch=current_branch,
            remote_branch=remote_branch,
            set_upstream=set_upstream,
        )
        if tracked_target is None:
            console.print(git_remotes_renderable(remotes))
        console.print(git_push_summary_renderable(summary))
        if not Confirm.ask("Run this push now?", default=True):
            raise RuntimeError("Push cancelled.")
        return self.actions.git_push(
            remote=remote,
            local_branch=current_branch,
            remote_branch=remote_branch,
            set_upstream=set_upstream,
        )

    def pr_preview_with_prompts(self, draft: bool | None = None, return_options: bool = False):
        remotes = self.actions.git_remotes()
        default_base_remote = next((item for item in remotes if item.name == "upstream"), None) or next((item for item in remotes if item.name == "origin"), None)
        default_head_remote = next((item for item in remotes if item.name == "origin"), None) or default_base_remote
        base_repo = default_base_remote.repo_slug if default_base_remote else None
        head_repo = default_head_remote.repo_slug if default_head_remote else base_repo
        base_remote_name = default_base_remote.name if default_base_remote else (self.actions.git_tool.default_remote_name() or "origin")
        base_branches = self.actions.git_remote_branches(base_remote_name)
        base_branch = self.choose_named_value("Choose the branch to open this PR into", base_branches, default="main", allow_custom=True)
        head_branch = self.actions.workspace_status().branch or "main"
        is_draft = Confirm.ask("Create or preview this as a draft PR?", default=False if draft is None else draft)
        preview = self.actions.pr_preview(
            base=base_branch,
            base_repo=base_repo,
            head_branch=head_branch,
            head_repo=head_repo,
            draft=is_draft,
        )
        if return_options:
            return preview, {
                "base_repo": base_repo,
                "base_branch": base_branch,
                "head_repo": head_repo,
                "head_branch": head_branch,
                "draft": is_draft,
            }
        return preview

    def choose_named_value(self, title: str, values: list[str], *, default: str, allow_custom: bool = False) -> str:
        cleaned = [value for value in values if value]
        if cleaned:
            choices = [MenuChoice(value, value) for value in cleaned]
            if allow_custom:
                choices.append(MenuChoice("Type a custom value", "__custom__"))
            picked = choose_menu_action(console, title, choices)
            if allow_custom and picked == "__custom__":
                return Prompt.ask(title, default=default).strip() or default
            if picked:
                return picked
        if allow_custom:
            return Prompt.ask(title, default=default).strip() or default
        return default

    def choose_repo_slug(self, title: str, remotes: list[GitRemote], default: str | None) -> str | None:
        options = [remote.repo_slug for remote in remotes if remote.repo_slug]
        unique = []
        for option in options:
            if option and option not in unique:
                unique.append(option)
        if unique:
            choices = [MenuChoice(option, option) for option in unique]
            choices.append(MenuChoice("Type a custom repo slug", "__custom__"))
            picked = choose_menu_action(console, title, choices)
            if picked == "__custom__":
                value = Prompt.ask(title, default=default or "").strip()
                return value or default
            if picked:
                return picked
        if default:
            value = Prompt.ask(title, default=default).strip()
            return value or default
        return None


def packages_lines_or_table(packages, workspace: Path) -> RenderableType:
    if packages:
        return packages_renderable(workspace, packages)
    return package_lines(packages)


def infer_repo_action(text: str) -> str | None:
    lowered = text.casefold()
    if any(token in lowered for token in ("inspect", "security scan", "scan repo", "check secrets")):
        return "inspect"
    if any(token in lowered for token in ("workspace status", "repo status", "show status", "what changed", "which branch")):
        return "status"
    if lowered in {"status", "show me the status"}:
        return "status"
    if any(token in lowered for token in ("reindex", "build index", "refresh index")) or lowered == "index":
        return "index"
    if any(token in lowered for token in ("packages", "dependencies", "node packages")):
        return "packages"
    return None


def is_runtime_intent(text: str) -> bool:
    lowered = text.casefold().strip()
    runtime_starts = ("start", "run", "launch", "boot", "spin up", "bring up", "open")
    if not lowered.startswith(runtime_starts):
        return False
    return any(token in lowered for token in ("app", "site", "website", "frontend", "backend", "services", "project"))


def wants_browser(text: str) -> bool:
    lowered = text.casefold()
    if any(token in lowered for token in ("no browser", "without browser", "don't open")):
        return False
    return any(token in lowered for token in ("browser", "website", "site", "frontend", "app"))


def infer_git_intent(text: str) -> GitIntent | None:
    lowered = " ".join(text.casefold().split())
    if "git status" in lowered or lowered in {"commit status", "show git status"}:
        return GitIntent("status")
    if any(phrase in lowered for phrase in ("stage everything", "stage all", "add all", "stage all changes")):
        return GitIntent("add_all")
    add_path = re.match(r"^(?:stage|add)\s+(.+)$", lowered)
    if add_path and add_path.group(1) not in {"everything", "all", "all changes"}:
        return GitIntent("add_path", path=add_path.group(1))
    branch_create = re.match(r"^(?:create|make)\s+(?:a\s+)?branch(?:\s+(?:called|named|for))?\s+(.+)$", lowered)
    if branch_create:
        return GitIntent("branch_create", name=branch_create.group(1).strip())
    branch_switch = re.match(r"^(?:switch|checkout)\s+(?:to\s+)?(?:branch\s+)?(.+)$", lowered)
    if branch_switch and any(token in lowered for token in ("switch", "checkout")):
        return GitIntent("branch_switch", name=branch_switch.group(1).strip())
    if lowered.startswith("commit"):
        message_match = re.search(r"(?:with message|message)\s+(.+)$", text, re.IGNORECASE)
        return GitIntent("commit", message=message_match.group(1).strip() if message_match else None)
    if "suggest commit" in lowered or "generate commit message" in lowered:
        return GitIntent("commit_suggest")
    if lowered.startswith("pull"):
        return GitIntent("pull")
    if lowered.startswith("push"):
        return GitIntent("push")
    if "preview pr" in lowered or "preview pull request" in lowered:
        return GitIntent("pr_preview")
    if "create pr" in lowered or "open pr" in lowered or "create pull request" in lowered:
        return GitIntent("pr_create")
    if "merge conflict" in lowered:
        return GitIntent("merge_conflicts")
    if "abort merge" in lowered:
        return GitIntent("merge_abort")
    if "continue merge" in lowered:
        return GitIntent("merge_continue")
    return None


def merge_conflict_status_renderable(conflicts, *, merge_in_progress: bool) -> RenderableType:
    if not conflicts:
        status = "Merge is in progress and all conflict markers are resolved."
        return Group(status, merge_conflicts_renderable(conflicts))
    heading = "Merge is in progress." if merge_in_progress else "Conflict markers were detected."
    return Group(heading, merge_conflicts_renderable(conflicts))


def remote_name_for_repo(remotes: list[GitRemote], repo_slug: str | None) -> str | None:
    if not repo_slug:
        return None
    for remote in remotes:
        if remote.repo_slug == repo_slug:
            return remote.name
    return None
