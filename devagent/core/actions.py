from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from devagent.config.settings import AISettings, ConfigManager, ProviderModelConfig
from devagent.context.indexer import CodeIndexer
from devagent.core.agent import RepoAgent
from devagent.core.project import ProjectInfo, detect_project
from devagent.tools.ai import AIClient, AIStatusSnapshot, ProviderModelListing
from devagent.tools.edit_tool import EditAgent, EditProposal
from devagent.tools.git_tool import (
    CommitSuggestion,
    GitError,
    GitRemote,
    GitTool,
    PullOptions,
    PullRequestOptions,
    PushOptions,
)
from devagent.tools.insights import Finding, Inspector
from devagent.tools.node_tool import NodePackage, find_node_packages
from devagent.tools.runtime_tool import LaunchSpec, RunProfile, RunTool
from devagent.tools.setup_tool import SetupResult, SetupTool
from devagent.watcher.file_watcher import WatchService


@dataclass(frozen=True)
class WorkspaceSnapshot:
    project: ProjectInfo
    is_repo: bool
    branch: str | None = None
    dirty: bool = False
    changed_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunInventory:
    detected: list[LaunchSpec]
    profiles: dict[str, RunProfile]


@dataclass(frozen=True)
class RunLaunchResult:
    specs: list[LaunchSpec]
    phrase: str | None = None
    browser_opened: bool = False
    browser_url: str | None = None


@dataclass(frozen=True)
class CommitOutcome:
    commit_id: str
    message: str


@dataclass(frozen=True)
class PullRequestPreview:
    summary: str
    title: str
    body: str
    readiness: tuple[str, ...] = ()
    ready_to_create: bool = True


@dataclass(frozen=True)
class PullOutcome:
    local_branch: str
    remote: str
    remote_branch: str
    rebase: bool = False


@dataclass(frozen=True)
class PushOutcome:
    remote: str
    local_branch: str
    remote_branch: str
    set_upstream: bool = True
    force_with_lease: bool = False


@dataclass(frozen=True)
class MergeConflictDetail:
    path: str
    markers: int


@dataclass(frozen=True)
class AISelectionResult:
    provider: str
    fast_model: str | None
    deep_model: str | None
    embedding_model: str | None
    warnings: tuple[str, ...] = ()


def validate_workspace_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Workspace does not exist or is not a directory: {resolved}")
    return resolved


def bind_workspace_action(path: Path) -> WorkspaceSnapshot:
    resolved = validate_workspace_path(path)
    ConfigManager.bind_workspace(resolved)
    return snapshot_workspace(resolved)


def snapshot_workspace(path: Path) -> WorkspaceSnapshot:
    resolved = path.expanduser().resolve()
    project = detect_project(resolved)
    git = GitTool(resolved)
    changed = git.changed_files() if git.is_repo else []
    return WorkspaceSnapshot(
        project=project,
        is_repo=git.is_repo,
        branch=git.current_branch() if git.is_repo else None,
        dirty=bool(changed),
        changed_files=changed,
    )


class DevAgentActions:
    def __init__(self, workspace: Path):
        self.workspace = workspace.expanduser().resolve()
        self.repo_agent = RepoAgent(self.workspace)
        self.run_tool = RunTool(self.workspace)
        self.git_tool = GitTool(self.workspace)

    def refresh_workspace(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.repo_agent = RepoAgent(self.workspace)
        self.run_tool = RunTool(self.workspace)
        self.git_tool = GitTool(self.workspace)

    def bind_workspace(self, path: Path) -> WorkspaceSnapshot:
        snapshot = bind_workspace_action(path)
        self.refresh_workspace(snapshot.project.path)
        return snapshot

    def workspace_status(self) -> WorkspaceSnapshot:
        return snapshot_workspace(self.workspace)

    def index_workspace(self) -> int:
        index = CodeIndexer(self.workspace).build()
        return len(index.records)

    def chat(
        self,
        question: str,
        *,
        deep: bool = False,
        new_session: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        return self.repo_agent.answer(
            question,
            deep=deep,
            new_session=new_session,
            progress_callback=progress_callback,
        )

    def clear_chat_session(self) -> None:
        self.repo_agent.clear_session()

    def ai_available(self) -> bool:
        return AIClient.from_env().available

    def ai_status(self, *, refresh: bool = False) -> AIStatusSnapshot:
        return AIClient.from_env().provider_status(refresh=refresh)

    def ai_models(self, *, provider: str | None = None, refresh: bool = False) -> list[ProviderModelListing]:
        client = AIClient.from_env()
        if provider:
            if provider not in client.available_providers:
                return [ProviderModelListing(provider=provider, error=f"Provider `{provider}` is not currently configured.")]
            return [client.list_models_safe(provider=provider, refresh=refresh)]
        return [client.list_models_safe(provider=name, refresh=refresh) for name in client.available_providers]

    def save_ai_selection(
        self,
        *,
        provider: str,
        fast_model: str | None = None,
        deep_model: str | None = None,
        embedding_model: str | None = None,
    ) -> AISelectionResult:
        config = ConfigManager.load()
        ai_settings = config.ai_settings
        provider_config = ai_settings.providers.get(provider, ProviderModelConfig())
        updated_provider = ProviderModelConfig(
            model=fast_model if fast_model is not None else provider_config.model,
            deep_model=deep_model if deep_model is not None else provider_config.deep_model,
            embedding_model=embedding_model if embedding_model is not None else provider_config.embedding_model,
        )
        providers = dict(ai_settings.providers)
        providers[provider] = updated_provider
        updated_settings = AISettings(selected_provider=provider, providers=providers)
        ConfigManager.save_ai_settings(updated_settings)
        self.refresh_workspace(self.workspace)
        client = AIClient.from_env()
        return AISelectionResult(
            provider=provider,
            fast_model=client.fast_model,
            deep_model=client.deep_model,
            embedding_model=client.embedding_model,
        )

    def reset_ai_settings(self) -> None:
        ConfigManager.save_ai_settings(AISettings())
        self.refresh_workspace(self.workspace)

    def packages(self) -> list[NodePackage]:
        return find_node_packages(self.workspace)

    def inspect(self) -> list[Finding]:
        return Inspector(self.workspace).run()

    def run_inventory(self) -> RunInventory:
        return RunInventory(detected=self.run_tool.detect_launch_specs(), profiles=self.run_tool.saved_profiles())

    def run_start(self, phrase: str | None = None, *, open_browser: bool = True) -> RunLaunchResult:
        specs = self.run_tool.launch_saved(phrase, open_browser=open_browser) if phrase else self.run_tool.launch_detected(open_browser=open_browser)
        browser_url = next((spec.browser_url for spec in specs if spec.browser_url), None) if open_browser else None
        return RunLaunchResult(specs=specs, phrase=phrase, browser_opened=open_browser and browser_url is not None, browser_url=browser_url)

    def run_launch_profile(self, profile: RunProfile, *, open_browser: bool | None = None) -> RunLaunchResult:
        specs = self.run_tool.launch_profile(profile, open_browser=open_browser)
        should_open = profile.open_browser if open_browser is None else open_browser
        browser_url = next((spec.browser_url for spec in specs if spec.browser_url), None) if should_open else None
        return RunLaunchResult(specs=specs, phrase=profile.phrase, browser_opened=should_open and browser_url is not None, browser_url=browser_url)

    def find_run_profile(self, phrase: str) -> RunProfile | None:
        return self.run_tool.find_profile(phrase)

    def save_run_profile(
        self,
        phrase: str,
        *,
        command: str | None = None,
        cwd: Path | None = None,
        open_browser: bool = False,
        description: str | None = None,
    ) -> RunProfile:
        if command:
            return self.run_tool.save_manual_profile(phrase, command, cwd=cwd, open_browser=open_browser, description=description)
        return self.run_tool.save_detected_profile(phrase, open_browser=open_browser, description=description)

    def delete_run_profile(self, phrase: str) -> bool:
        return self.run_tool.delete_profile(phrase)

    def clone_repo(self, repo_url: str, *, target: Path | None = None, install_deps: bool = False, open_code: bool = False) -> SetupResult:
        result = SetupTool.clone_from_github(repo_url, target, install_deps=install_deps, open_code=open_code)
        ConfigManager.bind_workspace(result.path)
        self.refresh_workspace(result.path)
        return result

    def publish_repo(self, path: Path, *, repo_name: str | None = None, private: bool = False, push: bool = True) -> SetupResult:
        result = SetupTool.publish_to_github(path, repo_name=repo_name, private=private, push=push)
        ConfigManager.bind_workspace(result.path)
        self.refresh_workspace(result.path)
        return result

    def edit_propose(
        self,
        instruction: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> EditProposal:
        return EditAgent(self.workspace).propose(instruction, progress_callback=progress_callback)

    def edit_apply(
        self,
        proposal: EditProposal,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        EditAgent(self.workspace).apply(proposal, progress_callback=progress_callback)

    def git_status(self) -> str:
        return self.git_tool.status_text()

    def git_add(self, path: str = ".") -> None:
        self.git_tool.add(path)

    def git_create_branch(self, name: str) -> None:
        self.git_tool.create_branch(name)

    def git_switch_branch(self, name: str, *, force: bool = False) -> None:
        if self.git_tool.has_changes() and not force:
            raise ValueError("Uncommitted changes exist. Commit/stash them or pass --force.")
        self.git_tool.switch_branch(name)

    def git_commit(self, *, message: str | None = None, all_files: bool = True) -> CommitOutcome:
        suggestion = self.git_tool.suggest_commit() if message is None else message
        commit_id = self.git_tool.commit(suggestion, all_files=all_files)
        final_message = suggestion.full_message if isinstance(suggestion, CommitSuggestion) else suggestion
        return CommitOutcome(commit_id=commit_id, message=final_message)

    def git_pull(self, *, remote: str | None = None, branch: str | None = None, rebase: bool = False) -> PullOutcome:
        result = self.git_tool.pull(PullOptions(remote=remote or self.git_tool.default_remote_name() or "origin", branch=branch or self.git_tool.current_branch() or "", rebase=rebase) if remote or branch or rebase else None)
        return PullOutcome(
            local_branch=result.local_branch,
            remote=result.remote,
            remote_branch=result.remote_branch,
            rebase=result.rebase,
        )

    def git_push(
        self,
        *,
        remote: str | None = None,
        branch: str | None = None,
        local_branch: str | None = None,
        remote_branch: str | None = None,
        set_upstream: bool = True,
        force_with_lease: bool = False,
    ) -> PushOutcome:
        result = self.git_tool.push(
            PushOptions(
                remote=remote or self.git_tool.default_remote_name() or "origin",
                local_branch=local_branch or branch or self.git_tool.current_branch() or "",
                remote_branch=remote_branch or branch or local_branch or self.git_tool.current_branch() or "",
                set_upstream=set_upstream,
                force_with_lease=force_with_lease,
            )
            if any(value is not None for value in (remote, branch, local_branch, remote_branch)) or force_with_lease is not False or set_upstream is not True
            else None
        )
        return PushOutcome(
            remote=result.remote,
            local_branch=result.local_branch,
            remote_branch=result.remote_branch,
            set_upstream=result.set_upstream,
            force_with_lease=result.force_with_lease,
        )

    def pr_preview(
        self,
        *,
        base: str = "main",
        base_repo: str | None = None,
        head_branch: str | None = None,
        head_repo: str | None = None,
        draft: bool = False,
    ) -> PullRequestPreview:
        readiness = self.git_tool.pr_readiness(base_branch=base, head_branch=head_branch)
        preview = self.git_tool.build_pr_preview(
            PullRequestOptions(
                base_repo=base_repo,
                base_branch=base,
                head_repo=head_repo,
                head_branch=head_branch or self.git_tool.current_branch() or "current-branch",
                draft=draft,
            )
        )
        return PullRequestPreview(
            summary=f"Open PR from `{head_branch or self.git_tool.current_branch() or 'current-branch'}` into `{base}`.",
            title=preview.subject,
            body=preview.body,
            readiness=readiness.blocking_reasons + readiness.notes,
            ready_to_create=readiness.can_create_pr,
        )

    def pr_create(
        self,
        *,
        base: str = "main",
        base_repo: str | None = None,
        head_branch: str | None = None,
        head_repo: str | None = None,
        title: str | None = None,
        body: str | None = None,
        draft: bool = False,
    ) -> str:
        return self.git_tool.create_pr(
            base_branch=base,
            base_repo=base_repo,
            head_branch=head_branch,
            head_repo=head_repo,
            title=title,
            body=body,
            draft=draft,
        )

    def merge_conflicts(self) -> list[MergeConflictDetail]:
        return [MergeConflictDetail(path=file, markers=self.git_tool.conflict_marker_count(file)) for file in self.git_tool.conflict_files()]

    def merge_abort(self) -> None:
        self.git_tool.merge_abort()

    def merge_continue(self) -> None:
        self.git_tool.merge_continue()

    def suggest_commit(self, *, conventional: bool = True) -> CommitSuggestion:
        return self.git_tool.suggest_commit(conventional=conventional)

    def git_remotes(self) -> list[GitRemote]:
        return self.git_tool.remotes()

    def git_tracked_remote_target(self, branch: str | None = None) -> tuple[str, str] | None:
        return self.git_tool.tracked_remote_target(branch)

    def git_remote_names(self) -> list[str]:
        return self.git_tool.remote_names()

    def git_local_branches(self) -> list[str]:
        return self.git_tool.local_branches()

    def git_remote_branches(self, remote: str) -> list[str]:
        return self.git_tool.remote_branches(remote)

    def git_upstream_for(self, branch: str | None = None) -> str | None:
        return self.git_tool.upstream_for(branch)

    def git_merge_in_progress(self) -> bool:
        return self.git_tool.merge_in_progress()

    def watch_workspace(self, *, interval: float = 1.0) -> None:
        WatchService(self.workspace, interval=interval).run()
