from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import NoReturn, Optional

import typer
from rich.prompt import Confirm, Prompt

from devagent.cli.prompts import MenuChoice, can_use_arrow_menu, choose_directory, choose_menu_action
from devagent.cli.renderers import (
    ai_models_collection_renderable,
    ai_selection_renderable,
    ai_status_renderable,
    commit_suggestion_renderable,
    git_pull_summary_renderable,
    git_push_summary_renderable,
    insights_renderable,
    merge_conflicts_renderable,
    packages_renderable,
    pr_preview_renderable,
    run_inventory_renderable,
    run_launch_message,
    workspace_status_table,
)
from devagent.cli.ui import app_panel, app_table, console, hero_panel, render_chat_markdown, status_badge, styled_path, toned_message
from devagent.config.settings import ConfigManager
from devagent.core.actions import DevAgentActions, bind_workspace_action, snapshot_workspace
from devagent.core.shell import AgentShell, git_menu_choices, interactive_terminal
from devagent.tools.git_tool import GitError

APP_HELP = dedent(
    """
    Local-first agentic AI developer assistant.

    Command families:
    - `ai`: discover configured providers, browse visible models, and save defaults
    - `chat`: ask grounded repo questions with session memory and deep mode
    - `git`: run guided Git, push, PR, and merge workflows
    - `run`: launch local services and saved startup phrases
    - `workspace` / `setup` / `new`: bind, clone, publish, and onboard projects
    - `edit`, `inspect`, `watch`, `commit`: change code safely and keep the repo healthy

    Typical flows:
    - `devagent workspace bind D:\\MyProject`
    - `devagent ai status`
    - `devagent chat "Explain the auth flow"`
    - `devagent git --help`
    - `devagent run start --open-browser`

    Tip: running `devagent` with no subcommand opens the interactive shell.
    """
).strip()

WORKSPACE_HELP = dedent(
    """
    Bind, inspect, and re-check the active workspace.

    Use this family when you want DevAgent to point at a different project,
    verify what it detected, or inspect the current Git/project snapshot.
    """
).strip()

SETUP_HELP = dedent(
    """
    Clone projects or publish local projects to GitHub.

    Use `clone` for existing GitHub repos, `publish` for local work, or
    `devagent new project` for the guided onboarding flow.
    """
).strip()

NEW_HELP = dedent(
    """
    Guided onboarding flows for new or existing projects.

    This is the friendlier setup surface when you want prompts instead of
    remembering the exact clone or publish command.
    """
).strip()

GIT_HELP = dedent(
    """
    Guided Git workflows for status, staging, commits, pulls, pushes, PRs, and merge recovery.

    Common workflows:
    - inspect repo state with `status`
    - stage and commit with richer generated messages
    - pull the current branch from its tracked remote
    - push the current branch to GitHub without extra Git plumbing
    - preview or open a PR from the current branch into `main`
    - inspect, abort, or continue merges with better context
    """
).strip()

RUN_HELP = dedent(
    """
    Launch workspace services and manage saved run phrases.

    Use this family to start detected stacks, open the browser, or save
    natural-language startup shortcuts for repeated workflows.
    """
).strip()

AI_HELP = dedent(
    """
    Discover configured AI providers, browse visible models, and save defaults.

    Use this family when you want to see which providers DevAgent can use from
    your current API keys, compare visible Gemini, Groq, or xAI models, or
    switch the default provider and model without editing environment variables
    by hand. Supported providers include Gemini, Groq, xAI, and OpenRouter.
    """
).strip()

COMMIT_HELP = dedent(
    """
    Commit message helpers.

    Use `suggest` to preview a detailed commit subject and body generated from
    the actual files, symbols, and impact of your current changes.
    """
).strip()

app = typer.Typer(help=APP_HELP, invoke_without_command=True, no_args_is_help=False)
ai_app = typer.Typer(help=AI_HELP, invoke_without_command=True)
workspace_app = typer.Typer(help=WORKSPACE_HELP)
setup_app = typer.Typer(help=SETUP_HELP)
new_app = typer.Typer(help=NEW_HELP)
git_app = typer.Typer(help=GIT_HELP, invoke_without_command=True)
run_app = typer.Typer(help=RUN_HELP, invoke_without_command=True)
branch_app = typer.Typer(help="Create and switch branches with safer guided flows.")
pr_app = typer.Typer(help="Preview and create pull requests across repos and branches.")
merge_app = typer.Typer(help="Inspect, abort, and continue merges with clearer conflict context.")
commit_app = typer.Typer(help=COMMIT_HELP)

app.add_typer(workspace_app, name="workspace")
app.add_typer(ai_app, name="ai")
app.add_typer(setup_app, name="setup")
app.add_typer(new_app, name="new")
app.add_typer(git_app, name="git")
app.add_typer(run_app, name="run")
app.add_typer(commit_app, name="commit")
git_app.add_typer(branch_app, name="branch")
git_app.add_typer(pr_app, name="pr")
git_app.add_typer(merge_app, name="merge")


@app.callback()
def app_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if interactive_terminal():
        state, bound_path = _bound_workspace_state()
        if state == "unbound":
            console.print(app_panel("No workspace is bound yet.\nRun `devagent workspace bind <path>` or `devagent new project` first.", "Agent Shell", tone="warning", expand=False))
            console.print(ctx.get_help())
            raise typer.Exit()
        if state == "missing":
            console.print(app_panel(_missing_workspace_message(bound_path), "Workspace Missing", tone="warning", expand=False))
            console.print(ctx.get_help())
            raise typer.Exit()
        shell = AgentShell(bound_path)
        shell.run()
        raise typer.Exit()
    console.print(ctx.get_help())
    raise typer.Exit()


def _workspace_path(explicit: Optional[Path] = None) -> Path:
    if explicit:
        resolved = explicit.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise typer.BadParameter(f"Workspace does not exist or is not a directory: {resolved}")
        return resolved
    state, bound_path = _bound_workspace_state()
    if state == "unbound":
        _exit_with_workspace_problem("No workspace is bound yet.\n\nRun `devagent workspace bind <path>` to choose one.", "Workspace Not Bound")
    if state == "missing":
        _exit_with_workspace_problem(_missing_workspace_message(bound_path), "Workspace Missing")
    return bound_path


def _exit_with_workspace_problem(message: str, title: str) -> NoReturn:
    """Explain a workspace problem and stop. Not a usage error, so no Usage/--help block."""
    console.print(app_panel(message, title, tone="warning", expand=False))
    raise typer.Exit(code=1)


def _print_project_status(path: Path) -> None:
    console.print(workspace_status_table(snapshot_workspace(path)))


def _actions(explicit: Optional[Path] = None) -> DevAgentActions:
    return DevAgentActions(_workspace_path(explicit))


def _ai_actions() -> DevAgentActions:
    state, workspace = _bound_workspace_state()
    if state == "ready" and workspace is not None:
        return DevAgentActions(workspace)
    return DevAgentActions(Path.cwd())


def _setup_actions() -> DevAgentActions:
    state, workspace = _bound_workspace_state()
    if state == "ready" and workspace is not None:
        return DevAgentActions(workspace)
    return DevAgentActions(Path.cwd())


def _bound_workspace_state() -> tuple[str, Path | None]:
    config = ConfigManager.load()
    if not config.workspace_path:
        return "unbound", None
    bound_path = config.workspace_path.expanduser().resolve()
    if not bound_path.exists() or not bound_path.is_dir():
        return "missing", bound_path
    return "ready", bound_path


def _missing_workspace_message(path: Path | None) -> str:
    missing_path = str(path) if path else "unknown path"
    return (
        f"The saved workspace path no longer exists:\n{missing_path}\n\n"
        "Use one of these to recover:\n"
        "- `devagent workspace bind <path>`\n"
        "- `devagent new project`\n"
        "- `devagent setup clone <repo-url>`"
    )


def _print_run_inventory() -> None:
    actions = _actions()
    console.print(run_inventory_renderable(actions.workspace, actions.run_inventory()))


def _print_run_menu() -> None:
    console.print(hero_panel("Runtime Agent", "Spin up services, attach environments, and open the app in one move."))
    _print_run_inventory()


def _print_git_menu() -> None:
    console.print(hero_panel("Git Assistant", "Command your repo with guided actions and neon telemetry."))
    table = app_table("DevAgent Git")
    table.add_column("What You Want")
    table.add_column("Command To Run")
    table.add_row("See what changed and which branch you're on", "devagent git status")
    table.add_row("Stage everything for the next commit", "devagent git add")
    table.add_row("Stage a specific file or folder", "devagent git add <path>")
    table.add_row("Create a branch for new work", "devagent git branch create <name>")
    table.add_row("Switch to another branch safely", "devagent git branch switch <name>")
    table.add_row("Commit with an auto-generated message", "devagent git commit --all")
    table.add_row("Suggest a commit message without committing", "devagent commit suggest")
    table.add_row("Pull the latest changes into this branch", "devagent git pull")
    table.add_row("Push this branch to GitHub", "devagent git push")
    table.add_row("Preview the PR title and description", "devagent git pr preview --base main")
    table.add_row("Open a PR for this branch", "devagent git pr create --base main")
    table.add_row("Check merge conflicts", "devagent git merge conflicts")
    table.add_row("Abort the current merge", "devagent git merge abort")
    table.add_row("Continue the current merge after resolution", "devagent git merge continue")
    console.print(table)


def git_action_choices(merge_in_progress: bool = True) -> list[MenuChoice]:
    choices = git_menu_choices(merge_in_progress=merge_in_progress)
    return [choice for choice in choices if choice.value != "back"] + [MenuChoice("Exit Git assistant", "exit")]


def _run_git_menu() -> None:
    shell = AgentShell(_workspace_path())
    shell.git_mode()


@git_app.callback()
def git_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        if can_use_arrow_menu():
            _run_git_menu()
        else:
            _print_git_menu()


@run_app.callback()
def run_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _print_run_menu()


@ai_app.callback()
def ai_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ai_status_renderable(_ai_actions().ai_status(refresh=True)))


@ai_app.command("status", help="Show which providers are configured, which one is selected, and which models DevAgent will use right now.")
def ai_status(refresh: bool = typer.Option(False, "--refresh", help="Refresh visible model data before rendering status.")) -> None:
    console.print(ai_status_renderable(_ai_actions().ai_status(refresh=refresh)))


@ai_app.command("models", help="List the models DevAgent can currently see for the configured providers.")
def ai_models(
    provider: Optional[str] = typer.Option(None, "--provider", help="Limit the model list to one provider, such as `gemini`, `groq`, `xai`, or `openrouter`."),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh visible model data before listing models."),
) -> None:
    listings = _ai_actions().ai_models(provider=provider, refresh=refresh)
    if not listings:
        console.print(app_panel("No AI providers are configured yet. Add a supported API key first.", "AI Models", tone="warning", expand=False))
        raise typer.Exit(code=1)
    if all(listing.error and not listing.models for listing in listings):
        if provider:
            details = "\n".join(f"- {listing.label}: {listing.error}" for listing in listings)
            console.print(app_panel(f"DevAgent could not load live models for the requested provider(s).\n\n{details}", "AI Models", tone="warning", expand=False))
        else:
            hidden = ", ".join(listing.label for listing in listings)
            console.print(
                app_panel(
                    f"No AI providers are currently available right now.\n\nUnavailable providers: {hidden}\nUse `devagent ai models --provider <name>` to inspect a specific provider.",
                    "AI Models",
                    tone="warning",
                    expand=False,
                )
            )
        raise typer.Exit(code=1)
    if provider:
        console.print(ai_models_collection_renderable(listings, show_error_detail=True))
    else:
        console.print(ai_models_collection_renderable(listings, show_error_detail=False))


@ai_app.command("use", help="Save the default provider and model selection DevAgent should use for chat, edit, and other AI-powered flows.")
def ai_use(
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to save as the default, such as `gemini`, `groq`, `xai`, or `openrouter`."),
    model: Optional[str] = typer.Option(None, "--model", help="Default chat model for the chosen provider."),
    deep_model: Optional[str] = typer.Option(None, "--deep-model", help="Default deep-synthesis model for the chosen provider."),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Default embedding model for the chosen provider when supported."),
) -> None:
    actions = _ai_actions()
    status = actions.ai_status(refresh=True)
    configured = [item.provider for item in status.providers]
    if not configured:
        console.print(app_panel("No AI providers are configured yet. Add a supported API key first.", "AI Selection", tone="warning", expand=False))
        raise typer.Exit(code=1)

    chosen_provider = provider or status.selected_provider or configured[0]
    if chosen_provider not in configured:
        raise typer.BadParameter(f"Provider `{chosen_provider}` is not currently configured.")

    listing = actions.ai_models(provider=chosen_provider, refresh=True)[0]
    generation_ids = {item.id for item in listing.models if "generate" in item.capabilities}
    embedding_ids = {item.id for item in listing.models if "embed" in item.capabilities}
    deferred_validation_warning: str | None = None

    if chosen_provider != "gemini" and embedding_model:
        raise typer.BadParameter(f"Provider `{chosen_provider}` does not currently expose embedding models in DevAgent.")

    if listing.error:
        if model or deep_model or embedding_model:
            deferred_validation_warning = (
                f"Could not verify the visible models for {chosen_provider} right now. "
                f"Saved the explicit model values as provided. {listing.error}"
            )
        else:
            deferred_validation_warning = (
                f"Saved provider `{chosen_provider}`, but DevAgent could not verify its visible models right now. "
                f"{listing.error}"
            )
    else:
        if model and generation_ids and model not in generation_ids:
            raise typer.BadParameter(f"Model `{model}` is not visible for provider `{chosen_provider}`.")
        if deep_model and generation_ids and deep_model not in generation_ids:
            raise typer.BadParameter(f"Deep model `{deep_model}` is not visible for provider `{chosen_provider}`.")
        if embedding_model and embedding_ids and embedding_model not in embedding_ids:
            raise typer.BadParameter(f"Embedding model `{embedding_model}` is not visible for provider `{chosen_provider}`.")
        if embedding_model and not embedding_ids:
            raise typer.BadParameter(f"Provider `{chosen_provider}` does not currently expose embedding models in DevAgent.")

    selection = actions.save_ai_selection(
        provider=chosen_provider,
        fast_model=model,
        deep_model=deep_model,
        embedding_model=embedding_model,
    )
    if deferred_validation_warning:
        selection = type(selection)(
            provider=selection.provider,
            fast_model=selection.fast_model,
            deep_model=selection.deep_model,
            embedding_model=selection.embedding_model,
            warnings=(deferred_validation_warning,),
        )
    console.print(ai_selection_renderable(selection))


@ai_app.command("reset", help="Clear the saved default provider and model selections so DevAgent falls back to environment-driven defaults again.")
def ai_reset() -> None:
    actions = _ai_actions()
    actions.reset_ai_settings()
    console.print(ai_status_renderable(actions.ai_status(refresh=False)))


@workspace_app.command("bind")
def bind_workspace(path: Path = typer.Argument(..., help="Project folder to attach DevAgent to.")) -> None:
    try:
        snapshot = bind_workspace_action(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(app_panel(f"Bound workspace:\n{snapshot.project.path}", "Workspace Linked", tone="success", expand=False))
    console.print(workspace_status_table(snapshot))


@workspace_app.command("status")
def workspace_status() -> None:
    _print_project_status(_workspace_path())


@setup_app.command("clone")
def clone_repo(
    repo_url: str = typer.Argument(..., help="GitHub repository URL or clone URL."),
    target: Optional[Path] = typer.Option(None, "--target", "-t", help="Target parent directory."),
    install_deps: bool = typer.Option(False, "--install-deps", help="Install detected dependencies after clone."),
    open_code: bool = typer.Option(False, "--open-code", help="Open the cloned project in VS Code."),
) -> None:
    actions = _setup_actions()
    try:
        result = actions.clone_repo(repo_url, target=target, install_deps=install_deps, open_code=open_code)
    except (RuntimeError, ValueError) as exc:
        console.print(app_panel(str(exc), "Setup Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(result.message, "Clone Complete", tone="success", expand=False))
    _print_project_status(result.path)


@app.command("clone")
def clone_repo_alias(
    repo_url: str = typer.Argument(..., help="GitHub repository URL or clone URL."),
    target: Optional[Path] = typer.Option(None, "--target", "-t", help="Target parent directory."),
    install_deps: bool = typer.Option(False, "--install-deps", help="Install detected dependencies after clone."),
    open_code: bool = typer.Option(False, "--open-code", help="Open the cloned project in VS Code."),
) -> None:
    """Shortcut for `devagent setup clone`."""
    clone_repo(repo_url=repo_url, target=target, install_deps=install_deps, open_code=open_code)


@setup_app.command("publish")
def publish_repo(
    path: Path = typer.Argument(..., help="Local project folder."),
    repo_name: Optional[str] = typer.Option(None, "--name", "-n", help="GitHub repository name."),
    private: bool = typer.Option(False, "--private", help="Create a private GitHub repository."),
    push: bool = typer.Option(True, "--push/--no-push", help="Push after creating the remote repository."),
) -> None:
    actions = _setup_actions()
    try:
        result = actions.publish_repo(path, repo_name=repo_name, private=private, push=push)
    except (RuntimeError, ValueError) as exc:
        console.print(app_panel(str(exc), "Publish Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(result.message, "Publish Complete", tone="success", expand=False))


@app.command("publish")
def publish_repo_alias(
    path: Path = typer.Argument(..., help="Local project folder."),
    repo_name: Optional[str] = typer.Option(None, "--name", "-n", help="GitHub repository name."),
    private: bool = typer.Option(False, "--private", help="Create a private GitHub repository."),
    push: bool = typer.Option(True, "--push/--no-push", help="Push after creating the remote repository."),
) -> None:
    """Shortcut for `devagent setup publish`."""
    publish_repo(path=path, repo_name=repo_name, private=private, push=push)


@new_app.command("project")
def new_project(
    start: Optional[Path] = typer.Option(None, "--start", "-s", help="Folder where the directory picker starts."),
) -> None:
    """Guided setup for cloning a GitHub repo or publishing a local project."""
    console.print(hero_panel("New Project", "Clone an existing repo or publish local work without losing the flow."))
    mode = Prompt.ask(
        "Do you already have a GitHub repo, or do you have a local project to publish?",
        choices=["github", "local"],
        default="github",
    )
    picker_start = (start or Path.cwd()).expanduser().resolve()
    if mode == "github":
        repo_url = Prompt.ask("Paste the GitHub repository page URL")
        target = choose_directory(console, picker_start, "Choose where to clone the repo")
        install_deps = Confirm.ask("Install dependencies if DevAgent detects them?", default=False)
        open_code = Confirm.ask("Open the project in VS Code after setup?", default=False)
        clone_repo(repo_url=repo_url, target=target, install_deps=install_deps, open_code=open_code)
        return

    local_path = choose_directory(console, picker_start, "Choose your local project folder")
    repo_name = Prompt.ask("GitHub repository name", default=local_path.name)
    private = Confirm.ask("Create the GitHub repo as private?", default=False)
    push = Confirm.ask("Push the local project after creating the remote?", default=True)
    publish_repo(path=local_path, repo_name=repo_name, private=private, push=push)


@app.command("index", help="Scan the active workspace and refresh DevAgent's local code index for chat and edit workflows.")
def index_workspace(path: Optional[Path] = typer.Option(None, "--path", "-p", help="Workspace path override.")) -> None:
    actions = _actions(path)
    count = actions.index_workspace()
    console.print(app_panel(f"Indexed {count} chunks from\n{actions.workspace}", "Index Complete", tone="success", expand=False))


@app.command(
    "chat",
    help=(
        "Ask a repo-aware question about the active workspace. "
        "DevAgent retrieves relevant files, cites them, and can switch into deeper synthesis with `--deep`."
    ),
)
def chat(
    question: str = typer.Argument(..., help="Question about the active workspace."),
    deep: bool = typer.Option(False, "--deep", help="Use broader retrieval and the saved deep model for the active provider when configured."),
    new_session: bool = typer.Option(False, "--new-session", help="Clear saved workspace chat context before answering."),
) -> None:
    actions = _actions()
    if interactive_terminal():
        with console.status("Thinking...") as status:
            answer = actions.chat(question, deep=deep, new_session=new_session, progress_callback=status.update)
    else:
        answer = actions.chat(question, deep=deep, new_session=new_session)
    console.print(app_panel(render_chat_markdown(answer), "DevAgent Response", tone="info"))


@app.command("packages")
def packages() -> None:
    """List direct Node packages from package.json files in the active workspace."""
    actions = _actions()
    node_packages = actions.packages()
    renderable = packages_renderable(actions.workspace, node_packages)
    if node_packages:
        console.print(renderable)
    else:
        console.print(app_panel(renderable, "Package Scan", tone="warning", expand=False))


@run_app.command("start")
def run_start(
    phrase: Optional[str] = typer.Argument(None, help="Saved run phrase to launch. Defaults to detected services."),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Open the detected local app URL after launching services."),
) -> None:
    actions = _actions()
    try:
        result = actions.run_start(phrase, open_browser=open_browser)
    except RuntimeError as exc:
        console.print(app_panel(str(exc), "Run Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(run_launch_message(actions.workspace, result), "Services Started", tone="success", expand=False))


@run_app.command("save")
def run_save(
    phrase: str = typer.Argument(..., help="Natural-language phrase to remember."),
    command: Optional[str] = typer.Option(None, "--command", "-c", help="Manual command to launch instead of the detected stack."),
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Working directory for --command. Defaults to the workspace root."),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser", help="Remember whether this phrase should open the browser after launch."),
    description: Optional[str] = typer.Option(None, "--description", help="Optional note about what this phrase launches."),
) -> None:
    actions = _actions()
    try:
        profile = actions.save_run_profile(phrase, command=command, cwd=cwd, open_browser=open_browser, description=description)
    except RuntimeError as exc:
        console.print(app_panel(str(exc), "Save Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    if command:
        spec = profile.specs[0]
        body = (
            f"Saved phrase {phrase}\n\n"
            f"Folder: {spec.scope(actions.workspace)}\n"
            f"Command: {spec.display_command}\n"
            f"Open browser: {'yes' if profile.open_browser else 'no'}"
        )
    else:
        body = (
            f"Saved phrase {phrase} for the detected stack.\n\n"
            + "\n".join(f"- {spec.scope(actions.workspace)}: {spec.display_command}" for spec in profile.specs)
            + f"\n\nOpen browser: {'yes' if profile.open_browser else 'no'}"
        )
    console.print(app_panel(body, "Run Phrase Saved", tone="success", expand=False))


@run_app.command("list")
def run_list() -> None:
    # A list command lists; the banner belongs to the bare `devagent run` landing screen.
    _print_run_inventory()


@run_app.command("forget")
def run_forget(
    phrase: str = typer.Argument(..., help="Saved run phrase to delete."),
) -> None:
    deleted = _actions().delete_run_profile(phrase)
    if not deleted:
        console.print(app_panel(f"No saved run phrase found: {phrase}", "Nothing Deleted", tone="warning", expand=False))
        raise typer.Exit(code=1)
    console.print(toned_message(f"Removed saved run phrase {phrase}.", "success"))


@app.command(
    "edit",
    help="Describe a code change in plain English and review the generated diff before anything is applied.",
)
def edit(
    instruction: str = typer.Argument(..., help="Natural language code edit instruction."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Apply the proposed diff without prompting."),
) -> None:
    actions = _actions()
    if interactive_terminal():
        with console.status("Thinking...") as status:
            proposal = actions.edit_propose(instruction, progress_callback=status.update)
    else:
        proposal = actions.edit_propose(instruction)
    console.print(app_panel(proposal.diff or proposal.message, "Proposed Change", tone="info"))
    if not proposal.diff:
        raise typer.Exit(code=1)
    if yes or typer.confirm("Apply this diff?"):
        try:
            if interactive_terminal():
                with console.status("Applying patch...") as status:
                    actions.edit_apply(proposal, progress_callback=status.update)
            else:
                actions.edit_apply(proposal)
        except RuntimeError as exc:
            console.print(app_panel(str(exc), "Edit Failed", tone="error", expand=False))
            console.print(toned_message("No files were changed.", "warning"))
            raise typer.Exit(code=1) from exc
        console.print(toned_message("Applied change.", "success"))
    else:
        console.print(toned_message("No files changed.", "warning"))


@git_app.command("status", help="Show the current branch and working-tree state in one place.")
def git_status() -> None:
    try:
        status = _actions().git_status()
    except GitError as exc:
        console.print(app_panel(str(exc), "Git Status Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(status, "Git Status", tone="info", expand=False))


@git_app.command("add", help="Stage either the whole workspace or a specific file/folder for the next commit.")
def git_add(path: str = typer.Argument(".", help="Path to stage. Defaults to the whole workspace.")) -> None:
    try:
        _actions().git_add(path)
    except (GitError, ValueError) as exc:
        console.print(app_panel(str(exc), "Git Add Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    if path == ".":
        console.print(toned_message("Staged the whole workspace.", "success"))
    else:
        console.print(toned_message(f"Staged {path}.", "success"))


@branch_app.command("create")
def create_branch(name: str = typer.Argument(..., help="New branch name.")) -> None:
    try:
        _actions().git_create_branch(name)
    except GitError as exc:
        console.print(app_panel(str(exc), "Branch Create Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(toned_message(f"Created and switched to branch {name}.", "success"))


@branch_app.command("switch")
def switch_branch(
    name: str = typer.Argument(..., help="Branch to switch to."),
    force: bool = typer.Option(False, "--force", help="Allow switching with uncommitted changes."),
) -> None:
    try:
        _actions().git_switch_branch(name, force=force)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except GitError as exc:
        console.print(app_panel(str(exc), "Branch Switch Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(toned_message(f"Switched to branch {name}.", "success"))


@git_app.command(
    "commit",
    help=(
        "Create a commit. When no manual message is provided, DevAgent generates a detailed subject and body from the real diff."
    ),
)
def git_commit(
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Commit message."),
    all_files: bool = typer.Option(True, "--all/--staged-only", "-a/-s", help="Stage all changed files before committing."),
) -> None:
    try:
        outcome = _actions().git_commit(message=message, all_files=all_files)
    except GitError as exc:
        console.print(app_panel(str(exc), "Commit Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(f"Created commit {outcome.commit_id}\n\n{outcome.message}", "Commit Complete", tone="success", expand=False))


@git_app.command(
    "pull",
    help=(
        "Pull the latest changes into the current branch. DevAgent uses the tracked remote first and only asks for more when needed."
    ),
)
def git_pull(
    remote: Optional[str] = typer.Option(None, "--remote", help="Override the tracked remote when this branch is not already linked."),
    branch: Optional[str] = typer.Option(None, "--branch", help="Override the tracked branch when this branch is not already linked."),
) -> None:
    try:
        result = _actions().git_pull(remote=remote, branch=branch)
    except GitError as exc:
        console.print(app_panel(str(exc), "Pull Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(git_pull_summary_renderable(result))


@git_app.command(
    "push",
    help=(
        "Push the current branch to GitHub. DevAgent uses the tracked branch first and keeps the flow focused on the common case."
    ),
)
def git_push(
    remote: Optional[str] = typer.Option(None, "--remote", help="Override the tracked remote when this branch is not already linked."),
    branch: Optional[str] = typer.Option(None, "--branch", help="Override the destination branch name. Defaults to the current branch."),
) -> None:
    try:
        result = _actions().git_push(
            remote=remote,
            branch=branch,
        )
    except GitError as exc:
        console.print(app_panel(str(exc), "Push Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(git_push_summary_renderable(result))


@pr_app.command(
    "preview",
    help=(
        "Preview the PR title and description for the current branch. DevAgent auto-detects the likely repo setup."
    ),
)
def pr_preview(
    base: str = typer.Option("main", "--base", help="Base branch for the pull request."),
    draft: bool = typer.Option(False, "--draft", help="Preview the PR as a draft flow."),
) -> None:
    preview = _actions().pr_preview(base=base, draft=draft)
    console.print(pr_preview_renderable(preview))


@pr_app.command(
    "create",
    help=(
        "Open a PR for the current branch. DevAgent auto-detects the usual repo setup and keeps the flow simple."
    ),
)
def pr_create(
    base: str = typer.Option("main", "--base", help="Base branch for the pull request."),
    title: Optional[str] = typer.Option(None, "--title", help="Override the generated PR title."),
    body: Optional[str] = typer.Option(None, "--body", help="Override the generated PR body."),
    draft: bool = typer.Option(False, "--draft", help="Create the pull request as a draft."),
) -> None:
    try:
        url = _actions().pr_create(
            base=base,
            title=title,
            body=body,
            draft=draft,
        )
    except GitError as exc:
        console.print(app_panel(str(exc), "PR Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(app_panel(url or "Pull request created.", "Pull Request", tone="success", expand=False))


@merge_app.command("conflicts")
def merge_conflicts() -> None:
    conflicts = _actions().merge_conflicts()
    renderable = merge_conflicts_renderable(conflicts)
    if conflicts:
        console.print(renderable)
    else:
        console.print(toned_message(str(renderable), "success"))


@merge_app.command("abort")
def merge_abort() -> None:
    try:
        _actions().merge_abort()
    except GitError as exc:
        console.print(app_panel(str(exc), "Merge Abort Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(toned_message("Aborted the merge.", "success"))


@merge_app.command("continue")
def merge_continue() -> None:
    try:
        _actions().merge_continue()
    except GitError as exc:
        console.print(app_panel(str(exc), "Merge Continue Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc
    console.print(toned_message("Continued the merge.", "success"))


@commit_app.command(
    "suggest",
    help=(
        "Preview a context-driven commit subject and body built from changed files, diff hunks, and likely project impact."
    ),
)
def suggest_commit(conventional: bool = typer.Option(True, "--conventional/--plain")) -> None:
    try:
        console.print(commit_suggestion_renderable(_actions().suggest_commit(conventional=conventional)))
    except GitError as exc:
        console.print(app_panel(str(exc), "Commit Suggestion Failed", tone="error", expand=False))
        raise typer.Exit(code=1) from exc


@app.command("watch", help="Watch the active workspace for file changes and print lightweight repo-aware prompts when things move.")
def watch_workspace(
    interval: float = typer.Option(1.0, "--interval", help="Polling interval when watchdog is unavailable."),
) -> None:
    actions = _actions()
    console.print(app_panel(f"Watching {actions.workspace}\nPress Ctrl+C to stop.", "Watch Mode", tone="info", expand=False))
    actions.watch_workspace(interval=interval)


@app.command("inspect", help="Run DevAgent's lightweight safety and repo-hygiene checks against the active workspace.")
def inspect_workspace() -> None:
    findings = _actions().inspect()
    renderable = insights_renderable(findings)
    if findings:
        console.print(renderable)
    else:
        console.print(app_panel(renderable, "DevAgent Insights", tone="success", expand=False))


if __name__ == "__main__":
    app()
