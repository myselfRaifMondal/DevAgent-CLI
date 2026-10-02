from __future__ import annotations

import locale
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from devagent.tools.ai import AIClient

COMMON_PATH_TOKENS = {
    "api",
    "app",
    "build",
    "cli",
    "code",
    "core",
    "devagent",
    "file",
    "files",
    "index",
    "main",
    "module",
    "page",
    "project",
    "readme",
    "service",
    "services",
    "src",
    "test",
    "tests",
    "tool",
    "tools",
    "utils",
}
FOCUS_PATTERNS = {
    "Git workflows": {"git", "branch", "merge", "commit", "push", "pull", "pr", "remote"},
    "CLI help": {"help", "cli", "command", "prompt", "shell"},
    "repo chat": {"chat", "agent", "prompt", "retriever", "session"},
    "runtime launch flows": {"runtime", "run", "launch", "browser", "venv"},
    "inspection checks": {"inspect", "insight", "security", "secret", "scan"},
    "workspace setup": {"workspace", "setup", "publish", "clone"},
    "test coverage": {"test", "tests", "pytest"},
}
TOPIC_SCOPE_LABELS = {
    "Git workflows": "guided Git workflows",
    "CLI help": "CLI help catalogs",
    "repo chat": "repo chat answers",
    "runtime launch flows": "runtime launch flows",
    "inspection checks": "inspection checks",
    "workspace setup": "workspace setup flows",
    "test coverage": "regression coverage",
}
SYMBOL_RE = re.compile(
    r"^[+-]\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)|"
    r"^[+-]\s*class\s+([A-Za-z_][A-Za-z0-9_]*)|"
    r"^[+-]\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)|"
    r"^[+-]\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
    re.MULTILINE,
)
DIFF_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)
GITHUB_HTTP_RE = re.compile(r"https://github\.com/([^/]+/[^/.]+)(?:\.git)?/?$", re.IGNORECASE)
GITHUB_SSH_RE = re.compile(r"git@github\.com:([^/]+/[^/.]+)(?:\.git)?$", re.IGNORECASE)


class GitError(RuntimeError):
    pass


_REF_FORBIDDEN_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")
_REPO_SLUG_RE = re.compile(r"^(?!-)[A-Za-z0-9_.-]+/(?!-)[A-Za-z0-9_.-]+$")


def validate_ref_name(name: str, *, kind: str = "branch") -> str:
    """Return ``name`` if it is safe to place on a git command line, else raise GitError.

    User-typed branch and remote names end up as positional git arguments, so a value
    such as ``--orphan`` or ``--upload-pack=...`` would be read as an option. Reject
    those, then apply git's own ref-format rules on top.
    """
    if not name or not name.strip():
        raise GitError(f"The {kind} name cannot be empty.")
    if name.startswith("-"):
        raise GitError(f"Invalid {kind} name {name!r}: it cannot start with '-'.")
    if (
        _REF_FORBIDDEN_CHARS.search(name)
        or ".." in name
        or "@{" in name
        or "//" in name
        or name.startswith(("/", "."))
        or name.endswith(("/", ".", ".lock"))
        or name == "@"
    ):
        raise GitError(f"Invalid {kind} name {name!r}.")
    return name


def validate_repo_slug(value: str) -> str:
    if not _REPO_SLUG_RE.match(value):
        raise GitError(f"Invalid repository {value!r}; expected OWNER/REPO.")
    return value


@dataclass(frozen=True)
class GitRemote:
    name: str
    fetch_url: str
    push_url: str
    repo_slug: str | None = None

    @property
    def owner(self) -> str | None:
        if not self.repo_slug or "/" not in self.repo_slug:
            return None
        return self.repo_slug.split("/", 1)[0]


@dataclass(frozen=True)
class CommitSuggestion:
    subject: str
    body: str
    body_bullets: tuple[str, ...] = ()
    project_area: str = ""
    changed_files: tuple[str, ...] = ()
    change_summary: tuple[str, ...] = ()
    impact_summary: tuple[str, ...] = ()
    conventional: bool = True

    @property
    def full_message(self) -> str:
        if not self.body.strip():
            return self.subject
        return f"{self.subject}\n\n{self.body.strip()}"


@dataclass(frozen=True)
class PullOptions:
    remote: str
    branch: str
    rebase: bool = False


@dataclass(frozen=True)
class PullResult:
    local_branch: str
    remote: str
    remote_branch: str
    rebase: bool = False


@dataclass(frozen=True)
class PushOptions:
    remote: str
    local_branch: str
    remote_branch: str
    set_upstream: bool = True
    force_with_lease: bool = False


@dataclass(frozen=True)
class PushResult:
    remote: str
    local_branch: str
    remote_branch: str
    set_upstream: bool = True
    force_with_lease: bool = False


@dataclass(frozen=True)
class BranchReadiness:
    current_branch: str
    base_branch: str
    upstream: str | None
    publish_remote: str | None
    publish_branch: str | None
    valid_upstream: bool
    ahead: int = 0
    behind: int = 0
    has_staged_changes: bool = False
    has_unstaged_changes: bool = False
    commits_ahead_of_base: int = 0
    head_branch_published: bool = False
    blocking_reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def can_create_pr(self) -> bool:
        return not self.blocking_reasons


@dataclass(frozen=True)
class PullRequestOptions:
    base_repo: str | None
    base_branch: str
    head_repo: str | None
    head_branch: str
    draft: bool = False
    title: str | None = None
    body: str | None = None


@dataclass(frozen=True)
class ChangeAnalysis:
    files: tuple[str, ...]
    diff: str
    staged_diff: str
    statuses: tuple[str, ...]
    symbols: tuple[str, ...]
    focus_topics: tuple[str, ...]
    surface_labels: tuple[str, ...]
    key_files: tuple[str, ...]
    prefix: str
    action: str
    project_area: str
    change_summary: tuple[str, ...]
    impact_summary: tuple[str, ...]
    body_bullets: tuple[str, ...]


class GitTool:
    def __init__(self, path: Path, ai: AIClient | None = None):
        self.path = path.expanduser().resolve()
        self.ai = ai or AIClient.from_env()

    @property
    def is_repo(self) -> bool:
        result = self._run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def init(self) -> None:
        self._run(["git", "init"])

    def add(self, *paths: str) -> None:
        targets = list(paths) or ["."]
        self._run(["git", "add", *targets])

    def current_branch(self) -> str | None:
        result = self._run(["git", "branch", "--show-current"], check=False)
        branch = result.stdout.strip()
        return branch or None

    def local_branches(self) -> list[str]:
        result = self._run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def remotes(self) -> list[GitRemote]:
        result = self._run(["git", "remote", "-v"], check=False)
        if result.returncode != 0:
            return []
        remotes: dict[str, dict[str, str]] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name, url, kind = parts[0], parts[1], parts[2].strip("()")
            remotes.setdefault(name, {})
            remotes[name][kind] = url
        ordered: list[GitRemote] = []
        for name in sorted(remotes.keys()):
            urls = remotes[name]
            fetch = urls.get("fetch") or urls.get("push") or ""
            push = urls.get("push") or fetch
            ordered.append(GitRemote(name=name, fetch_url=fetch, push_url=push, repo_slug=parse_github_repo_slug(fetch)))
        return ordered

    def remote_names(self) -> list[str]:
        return [remote.name for remote in self.remotes()]

    def remote_branches(self, remote: str) -> list[str]:
        result = self._run(
            ["git", "for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote}"],
            check=False,
        )
        if result.returncode != 0:
            return []
        branches: list[str] = []
        prefix = f"{remote}/"
        for line in result.stdout.splitlines():
            value = line.strip()
            if not value or value.endswith("/HEAD") or value.endswith("->"):
                continue
            if value.startswith(prefix):
                value = value[len(prefix) :]
            if value and value not in branches:
                branches.append(value)
        return branches

    def remote_exists(self, remote: str) -> bool:
        return any(item.name == remote for item in self.remotes())

    def remote_tracking_ref_exists(self, remote: str, branch: str) -> bool:
        validate_ref_name(remote, kind="remote")
        validate_ref_name(branch)
        result = self._run(["git", "show-ref", "--verify", f"refs/remotes/{remote}/{branch}"], check=False)
        return result.returncode == 0

    def upstream_for(self, branch: str | None = None) -> str | None:
        target = branch or self.current_branch()
        if not target:
            return None
        result = self._run(["git", "rev-parse", "--abbrev-ref", f"{target}@{{upstream}}"], check=False)
        upstream = result.stdout.strip()
        return upstream or None

    def status_entries(self) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        for line in self.changed_files():
            if len(line) < 3:
                continue
            index_status = line[:1]
            worktree_status = line[1:2]
            entries.append((index_status, worktree_status, normalize_status_path(line)))
        return entries

    def has_staged_changes(self) -> bool:
        return any(index not in {" ", "?"} for index, _, _ in self.status_entries())

    def has_unstaged_changes(self) -> bool:
        return any(worktree != " " for _, worktree, _ in self.status_entries())

    def compare_to_upstream(self, branch: str | None = None) -> tuple[int, int]:
        target = branch or self.current_branch()
        tracked = self.tracked_remote_target(target)
        if not target or not tracked:
            return 0, 0
        remote, remote_branch = tracked
        result = self._run(["git", "rev-list", "--left-right", "--count", f"{target}...{remote}/{remote_branch}"], check=False)
        if result.returncode != 0:
            return 0, 0
        counts = result.stdout.split()
        if len(counts) != 2:
            return 0, 0
        try:
            return int(counts[0]), int(counts[1])
        except ValueError:
            return 0, 0

    def resolve_base_ref(self, base_branch: str, remote: str | None = None) -> str | None:
        validate_ref_name(base_branch, kind="base branch")
        if remote:
            validate_ref_name(remote, kind="remote")
        candidates = [base_branch]
        if remote:
            candidates.append(f"{remote}/{base_branch}")
        default_remote = self.default_remote_name()
        if default_remote and default_remote != remote:
            candidates.append(f"{default_remote}/{base_branch}")
        for candidate in unique_limited(candidates, len(candidates)):
            result = self._run(["git", "rev-parse", "--verify", candidate], check=False)
            if result.returncode == 0:
                return candidate
        return None

    def merge_in_progress(self) -> bool:
        result = self._run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], check=False)
        return result.returncode == 0 and bool(result.stdout.strip())

    def has_changes(self) -> bool:
        return bool(self.changed_files())

    def changed_files(self) -> list[str]:
        result = self._run(["git", "status", "--short"], check=False)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def tracked_files(self) -> set[str]:
        result = self._run(["git", "ls-files"], check=False)
        if result.returncode != 0:
            return set()
        return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}

    def is_ignored(self, relative_path: str) -> bool:
        normalized = relative_path.replace("\\", "/")
        result = self._run(["git", "check-ignore", "-q", normalized], check=False)
        return result.returncode == 0

    def status_text(self) -> str:
        if not self.is_repo:
            return "This workspace is not a Git repository."
        result = self._run(["git", "status", "--short", "--branch"])
        return result.stdout.strip() or "Clean working tree."

    def create_branch(self, name: str) -> None:
        validate_ref_name(name)
        self._run(["git", "checkout", "-b", name])

    def switch_branch(self, name: str) -> None:
        validate_ref_name(name)
        # Trailing "--" makes git treat the name as a branch, never as a file path.
        self._run(["git", "checkout", name, "--"])

    def add_all(self) -> None:
        self._run(["git", "add", "."])

    def pull(self, options: PullOptions | None = None, *, remote: str | None = None, branch: str | None = None, rebase: bool = False) -> PullResult:
        tracked = self.tracked_remote_target()
        plan = options or PullOptions(
            remote=remote or (tracked[0] if tracked else self.default_remote_name() or "origin"),
            branch=branch or (tracked[1] if tracked else self.current_branch() or ""),
            rebase=rebase,
        )
        if not plan.branch:
            raise GitError("Could not determine which branch to pull.")
        validate_ref_name(plan.remote, kind="remote")
        validate_ref_name(plan.branch)
        args = ["git", "pull", plan.remote, plan.branch]
        if plan.rebase:
            args.insert(2, "--rebase")
        self._run(args)
        return PullResult(
            local_branch=self.current_branch() or plan.branch,
            remote=plan.remote,
            remote_branch=plan.branch,
            rebase=plan.rebase,
        )

    def commit(self, message: str | CommitSuggestion, all_files: bool = False) -> str:
        if all_files:
            self.add_all()
        args = ["git", "commit"]
        if isinstance(message, CommitSuggestion):
            args.extend(["-m", message.subject])
            if message.body.strip():
                args.extend(["-m", message.body.strip()])
        else:
            args.extend(["-m", message])
        self._run(args)
        result = self._run(["git", "rev-parse", "--short", "HEAD"])
        return result.stdout.strip()

    def push(
        self,
        options: PushOptions | None = None,
        *,
        remote: str | None = None,
        branch: str | None = None,
        local_branch: str | None = None,
        remote_branch: str | None = None,
        set_upstream: bool = True,
        force_with_lease: bool = False,
    ) -> PushResult:
        local = local_branch or branch or self.current_branch()
        if not local:
            raise GitError("Could not determine which local branch to push.")
        tracked = self.tracked_remote_target(local)
        resolved_remote = remote or (tracked[0] if tracked else self.default_remote_name() or "origin")
        destination = remote_branch or branch or (tracked[1] if tracked else local)
        plan = options or PushOptions(
            remote=resolved_remote,
            local_branch=local,
            remote_branch=destination,
            set_upstream=set_upstream if tracked is None else False,
            force_with_lease=force_with_lease,
        )
        validate_ref_name(plan.remote, kind="remote")
        validate_ref_name(plan.local_branch)
        validate_ref_name(plan.remote_branch)
        args = ["git", "push"]
        if plan.set_upstream:
            args.append("-u")
        if plan.force_with_lease:
            args.append("--force-with-lease")
        args.extend([plan.remote, f"{plan.local_branch}:{plan.remote_branch}"])
        self._run(args)
        return PushResult(
            remote=plan.remote,
            local_branch=plan.local_branch,
            remote_branch=plan.remote_branch,
            set_upstream=plan.set_upstream,
            force_with_lease=plan.force_with_lease,
        )

    def diff(self, staged: bool = False) -> str:
        args = ["git", "diff", "--cached"] if staged else ["git", "diff"]
        result = self._run(args, check=False)
        return result.stdout or ""

    def conflict_files(self) -> list[str]:
        result = self._run(["git", "diff", "--name-only", "--diff-filter=U"], check=False)
        if result.returncode != 0:
            return []
        return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]

    def has_conflicts(self) -> bool:
        return bool(self.conflict_files())

    def merge_abort(self) -> None:
        self._run(["git", "merge", "--abort"])

    def merge_continue(self) -> None:
        self._run(["git", "merge", "--continue"])

    def conflict_marker_count(self, relative_path: str) -> int:
        path = self.path / relative_path
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 0
        return sum(1 for line in text.splitlines() if line.startswith("<<<<<<<"))

    def pr_title(self, *, base_branch: str = "main") -> str:
        return self.suggest_commit(conventional=False).subject

    def tracked_remote_target(self, branch: str | None = None) -> tuple[str, str] | None:
        upstream = self.upstream_for(branch)
        if not upstream or "/" not in upstream:
            return None
        remote, remote_branch = upstream.split("/", 1)
        if not remote or not remote_branch:
            return None
        if not self.remote_exists(remote):
            return None
        if not self.remote_tracking_ref_exists(remote, remote_branch):
            return None
        return remote, remote_branch

    def resolve_push_target(self, branch: str | None = None) -> tuple[str | None, str | None, bool]:
        current = branch or self.current_branch()
        tracked = self.tracked_remote_target(current)
        if tracked:
            return tracked[0], tracked[1], False
        return self.default_remote_name(), current, True

    def commit_count_since(self, base_branch: str = "main", *, remote: str | None = None) -> int:
        base_ref = self.resolve_base_ref(base_branch, remote=remote)
        if not base_ref:
            return 0
        result = self._run(["git", "rev-list", "--count", f"{base_ref}..HEAD"], check=False)
        if result.returncode != 0:
            return 0
        try:
            return int(result.stdout.strip() or "0")
        except ValueError:
            return 0

    def pr_readiness(self, *, base_branch: str = "main", head_branch: str | None = None) -> BranchReadiness:
        current = head_branch or self.current_branch() or ""
        upstream = self.upstream_for(current) if current else None
        tracked = self.tracked_remote_target(current)
        publish_remote, publish_branch, _ = self.resolve_push_target(current)
        ahead, behind = self.compare_to_upstream(current) if tracked else (0, 0)
        has_staged = self.has_staged_changes()
        has_unstaged = self.has_unstaged_changes()
        commits_ahead = self.commit_count_since(base_branch)
        head_branch_published = bool(
            publish_remote
            and publish_branch
            and self.remote_tracking_ref_exists(publish_remote, publish_branch)
        )

        blocking: list[str] = []
        notes: list[str] = []
        if not current:
            blocking.append("DevAgent could not determine the current branch.")
        if upstream and tracked is None:
            notes.append(
                f"The tracked upstream `{upstream}` looks unusable, so DevAgent will ignore it for guided push and PR flows."
            )
        if has_staged or has_unstaged:
            blocking.append("You still have local changes that are not committed on this branch.")
        if commits_ahead <= 0 and current:
            blocking.append(f"`{current}` does not have any commits ahead of `{base_branch}` yet.")
        if not publish_remote:
            blocking.append("No Git remote is configured for this repository.")
        elif not head_branch_published and publish_branch:
            blocking.append(f"Push `{publish_branch}` to `{publish_remote}` before opening a pull request.")
        if tracked and behind > 0:
            notes.append(f"This branch is behind `{tracked[0]}/{tracked[1]}` by {behind} commit(s).")
        if tracked and ahead > 0:
            notes.append(f"This branch is ahead of `{tracked[0]}/{tracked[1]}` by {ahead} commit(s).")

        return BranchReadiness(
            current_branch=current or "current branch",
            base_branch=base_branch,
            upstream=upstream,
            publish_remote=publish_remote,
            publish_branch=publish_branch,
            valid_upstream=tracked is not None,
            ahead=ahead,
            behind=behind,
            has_staged_changes=has_staged,
            has_unstaged_changes=has_unstaged,
            commits_ahead_of_base=commits_ahead,
            head_branch_published=head_branch_published,
            blocking_reasons=tuple(blocking),
            notes=tuple(notes),
        )

    def default_remote_name(self) -> str | None:
        remotes = self.remotes()
        for preferred in ("origin", "upstream"):
            match = next((remote.name for remote in remotes if remote.name == preferred), None)
            if match:
                return match
        return remotes[0].name if remotes else None

    def default_base_repo(self) -> str | None:
        remotes = self.remotes()
        for preferred in ("upstream", "origin"):
            match = next((remote.repo_slug for remote in remotes if remote.name == preferred and remote.repo_slug), None)
            if match:
                return match
        return next((remote.repo_slug for remote in remotes if remote.repo_slug), None)

    def default_head_repo(self) -> str | None:
        remotes = self.remotes()
        for preferred in ("origin", "upstream"):
            match = next((remote.repo_slug for remote in remotes if remote.name == preferred and remote.repo_slug), None)
            if match:
                return match
        return next((remote.repo_slug for remote in remotes if remote.repo_slug), None)

    def pr_body(
        self,
        *,
        base_branch: str = "main",
        base_repo: str | None = None,
        head_repo: str | None = None,
        head_branch: str | None = None,
    ) -> str:
        preview = self.build_pr_preview(
            PullRequestOptions(
                base_repo=base_repo,
                base_branch=base_branch,
                head_repo=head_repo,
                head_branch=head_branch or self.current_branch() or "current-branch",
            )
        )
        return preview.body

    def build_pr_preview(self, options: PullRequestOptions) -> CommitSuggestion:
        base_repo = options.base_repo or self.default_base_repo()
        head_repo = options.head_repo or self.default_head_repo() or base_repo
        head_branch = options.head_branch or self.current_branch() or "current-branch"
        suggestion = self.suggest_commit(conventional=False)
        files = self.changed_files_since(options.base_branch) or [normalize_status_path(line) for line in self.changed_files()]
        summary_lines = [
            f"Open PR from `{head_branch}` into `{options.base_branch}`.",
        ]
        if base_repo and head_repo and base_repo != head_repo:
            summary_lines.append(f"Source: `{head_repo}` -> `{base_repo}`.")
        elif base_repo:
            summary_lines.append(f"Repo: `{base_repo}`.")
        if options.draft:
            summary_lines.append("This PR will open as a draft.")

        body_sections = ["## Summary", "", *summary_lines]
        if suggestion.body_bullets:
            body_sections.extend(["", "## What changed", ""])
            body_sections.extend(f"- {line}" for line in suggestion.body_bullets)
        elif suggestion.change_summary:
            body_sections.extend(["", "## What changed", ""])
            body_sections.extend(f"- {line}" for line in suggestion.change_summary)
        if files:
            body_sections.extend(["", "## Files changed", ""])
            body_sections.extend(f"- `{file}`" for file in files[:12])

        return CommitSuggestion(
            subject=options.title or suggestion.subject,
            body=options.body or "\n".join(body_sections).strip(),
            body_bullets=suggestion.body_bullets,
            project_area=suggestion.project_area,
            changed_files=tuple(files),
            change_summary=suggestion.change_summary,
            impact_summary=suggestion.impact_summary,
            conventional=False,
        )

    def create_pr(
        self,
        *,
        base_branch: str = "main",
        base_repo: str | None = None,
        head_branch: str | None = None,
        head_repo: str | None = None,
        title: str | None = None,
        body: str | None = None,
        draft: bool = False,
    ) -> str:
        readiness = self.pr_readiness(base_branch=base_branch, head_branch=head_branch)
        if not readiness.can_create_pr:
            reason_block = "\n".join(f"- {reason}" for reason in readiness.blocking_reasons)
            raise GitError(
                "This branch is not ready for a pull request yet:\n"
                f"{reason_block}"
            )
        options = PullRequestOptions(
            base_repo=base_repo or self.default_base_repo(),
            base_branch=base_branch,
            head_repo=head_repo or self.default_head_repo(),
            head_branch=head_branch or self.current_branch() or "current-branch",
            draft=draft,
            title=title,
            body=body,
        )
        validate_ref_name(options.base_branch, kind="base branch")
        validate_ref_name(options.head_branch, kind="head branch")
        for repo in (options.base_repo, options.head_repo):
            if repo:
                validate_repo_slug(repo)
        preview = self.build_pr_preview(options)
        args = ["gh", "pr", "create", "--base", options.base_branch, "--title", preview.subject, "--body", preview.body]
        if options.base_repo:
            args.extend(["--repo", options.base_repo])
        head_value = build_pr_head_value(options.head_repo, options.head_branch, options.base_repo)
        if head_value:
            args.extend(["--head", head_value])
        if options.draft:
            args.append("--draft")
        result = self._run(args)
        return result.stdout.strip()

    def changed_files_since(self, base: str = "main") -> list[str]:
        base_ref = self.resolve_base_ref(base)
        if not base_ref:
            return []
        result = self._run(["git", "diff", "--name-only", f"{base_ref}...HEAD"], check=False)
        if result.returncode != 0:
            return []
        return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]

    def diff_stat_since(self, base: str = "main") -> str:
        base_ref = self.resolve_base_ref(base)
        if not base_ref:
            return ""
        result = self._run(["git", "diff", "--stat", f"{base_ref}...HEAD"], check=False)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def suggest_commit(self, conventional: bool = True) -> CommitSuggestion:
        diff = self.diff(staged=False)
        staged_diff = self.diff(staged=True)
        working_diff = staged_diff or diff or ""
        changed = self.changed_files()
        if not working_diff and not changed:
            subject = "chore: no changes to commit" if conventional else "No changes to commit"
            return CommitSuggestion(subject=subject, body="", conventional=conventional)

        analysis = analyze_changes(changed, working_diff, staged_diff or "")
        fallback = build_deterministic_commit_suggestion(analysis, conventional=conventional)
        refined = self._refine_commit_suggestion_with_ai(analysis, fallback, conventional=conventional)
        return refined or fallback

    def suggest_commit_message(self, conventional: bool = True) -> str:
        return self.suggest_commit(conventional=conventional).full_message

    def _refine_commit_suggestion_with_ai(
        self,
        analysis: ChangeAnalysis,
        fallback: CommitSuggestion,
        *,
        conventional: bool,
    ) -> CommitSuggestion | None:
        if not self.ai.available:
            return None
        prompt = build_commit_prompt(analysis, fallback, conventional=conventional)
        response = self.ai.complete(
            prompt,
            system_instruction=(
                "You write precise Git commit messages for a developer CLI. "
                "Return exactly this shape:\n"
                "SUBJECT: <one line>\n"
                "BODY:\n"
                "- bullet one\n"
                "- bullet two\n"
                "Keep the subject under 72 characters if possible. "
                "Use concrete modules, files, and user-facing effects. "
                "Do not mention tests unless they changed. "
                "If conventional commits are requested, keep the prefix in the subject."
            ),
        )
        if not response:
            return None
        return parse_ai_commit_suggestion(response, fallback)

    def _run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                args,
                cwd=self.path,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            if check:
                raise GitError(f"Required command not found: {args[0]}") from exc
            return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=f"Command not found: {args[0]}")
        decoded = subprocess.CompletedProcess(
            args=result.args,
            returncode=result.returncode,
            stdout=decode_process_output(result.stdout),
            stderr=decode_process_output(result.stderr),
        )
        if check and decoded.returncode != 0:
            raise GitError(decoded.stderr.strip() or decoded.stdout.strip() or f"Command failed: {' '.join(args)}")
        return decoded


def normalize_status_path(line: str) -> str:
    return line[3:].strip() if len(line) > 3 else line.strip()


def infer_action(changed: list[str], diff: str | None) -> str:
    diff_text = diff or ""
    statuses = [line[:2] for line in changed]
    if any("A" in status or "??" in status for status in statuses):
        if any(token in diff_text.lower() for token in ("help", "support", "guide")):
            return "document"
        return "add"
    if any("D" in status for status in statuses):
        return "remove"
    lowered = diff_text.lower()
    if re.search(r"test_|describe\(|pytest|unittest", diff_text, re.IGNORECASE):
        return "cover"
    if any(token in lowered for token in ("fix", "bug", "error", "exception", "traceback")):
        return "fix"
    if any(token in lowered for token in ("refactor", "rename", "extract", "cleanup")):
        return "refine"
    if any(token in lowered for token in ("help", "description", "usage", "example")):
        return "improve"
    return "update"


def infer_area(files: list[str]) -> str:
    if not files:
        return "project changes"
    top_levels = [Path(file).parts[0] for file in files if Path(file).parts]
    if not top_levels:
        return "project changes"
    common = top_levels[0] if all(part == top_levels[0] for part in top_levels) else "project"
    return f"{common} changes"


def infer_conventional_prefix(files: list[str], extensions: set[str], diff: str | None) -> str:
    diff_text = (diff or "").lower()
    lowered = " ".join(files).lower()
    if files and all(is_test_path(file) for file in files):
        return "test"
    if files and all(is_docs_path(file) for file in files):
        return "docs"
    if "fix" in diff_text or "bug" in diff_text or "error" in diff_text:
        return "fix"
    if any(status_word in diff_text for status_word in ("add", "create", "new ")):
        return "feat"
    if any(token in lowered for token in ("config", "workflow", ".yml", ".yaml", ".toml")):
        return "chore"
    return "refactor" if any(token in diff_text for token in ("refactor", "cleanup", "rename")) else "chore"


def analyze_changes(changed: list[str], diff: str, staged_diff: str = "") -> ChangeAnalysis:
    files = tuple(normalize_status_path(line) for line in changed)
    statuses = tuple(line[:2].strip() or line[:2] for line in changed)
    extensions = {Path(file).suffix.lower() for file in files}
    action = infer_action(changed, diff)
    prefix = infer_conventional_prefix(list(files), extensions, diff)
    symbols = extract_symbols(diff or staged_diff)
    focus_topics = derive_focus_topics(files, diff, symbols)
    surface_labels = extract_surface_labels(files, diff or staged_diff)
    key_files = select_key_files(files)
    project_area = derive_project_area(files, focus_topics, symbols, diff, surface_labels)
    change_summary = build_change_summary(files, focus_topics, symbols, project_area, surface_labels)
    impact_summary = build_impact_summary(project_area, focus_topics, files, surface_labels)
    body_bullets = build_body_bullets(change_summary, impact_summary, key_files)
    return ChangeAnalysis(
        files=files,
        diff=diff,
        staged_diff=staged_diff,
        statuses=statuses,
        symbols=symbols,
        focus_topics=focus_topics,
        surface_labels=surface_labels,
        key_files=key_files,
        prefix=prefix,
        action=action,
        project_area=project_area,
        change_summary=change_summary,
        impact_summary=impact_summary,
        body_bullets=body_bullets,
    )


def build_deterministic_commit_suggestion(analysis: ChangeAnalysis, *, conventional: bool) -> CommitSuggestion:
    subject = build_subject_line(analysis, conventional=conventional)
    body = build_commit_body(analysis)
    return CommitSuggestion(
        subject=subject,
        body=body,
        body_bullets=analysis.body_bullets,
        project_area=analysis.project_area,
        changed_files=analysis.key_files,
        change_summary=analysis.change_summary,
        impact_summary=analysis.impact_summary,
        conventional=conventional,
    )


def build_subject_line(analysis: ChangeAnalysis, *, conventional: bool) -> str:
    action = analysis.action
    scope = analysis.project_area
    if should_adjust_wording(analysis):
        phrase = f"adjust {scope}"
    elif action == "document":
        phrase = f"document {scope}"
    elif action == "cover":
        phrase = f"cover {scope}"
    elif action == "fix":
        phrase = f"fix {scope}"
    elif action == "refine":
        phrase = f"refine {scope}"
    elif action == "improve":
        phrase = f"improve {scope}"
    elif action == "add":
        phrase = f"add {scope}"
    elif action == "remove":
        phrase = f"remove {scope}"
    else:
        phrase = f"update {scope}"
    phrase = phrase.strip()
    if not conventional:
        return phrase[:1].upper() + phrase[1:]
    return f"{analysis.prefix}: {phrase}"


def build_commit_body(analysis: ChangeAnalysis) -> str:
    return "\n".join(f"- {line}" for line in analysis.body_bullets).strip()


def derive_focus_topics(files: tuple[str, ...], diff: str, symbols: tuple[str, ...]) -> tuple[str, ...]:
    lowered_diff = diff.lower()
    token_counts: Counter[str] = Counter()
    for file in files:
        token_counts.update(path_tokens(file))
    token_counts.update(symbol.casefold() for symbol in symbols)
    token_counts.update(re.findall(r"[a-z][a-z0-9_-]{2,}", lowered_diff))

    matched_topics: list[str] = []
    for label, triggers in FOCUS_PATTERNS.items():
        if any(trigger in token_counts for trigger in triggers):
            matched_topics.append(label)

    if not matched_topics:
        if any(file.endswith((".md", ".mdx")) for file in files):
            matched_topics.append("documentation")
        elif any("/tests/" in file.lower() or file.lower().startswith("tests/") for file in files):
            matched_topics.append("test coverage")

    if not matched_topics:
        matched_topics.extend(humanize_token(token) for token, _ in token_counts.most_common(2) if token not in COMMON_PATH_TOKENS)
    return tuple(unique_limited(matched_topics, 2)) or ("project changes",)


def extract_surface_labels(files: tuple[str, ...], diff: str) -> tuple[str, ...]:
    labels: list[str] = []
    if re.search(r"^[+-].*<title>.*</title>", diff, re.MULTILINE | re.IGNORECASE):
        labels.append("page title")
    if re.search(r"^[+-]\s*#+\s+", diff, re.MULTILINE):
        if any(Path(file).stem.casefold() == "readme" for file in files):
            labels.append("README wording")
        else:
            labels.append("documentation headings")
    if any(is_ui_text_path(file) for file in files):
        if re.search(r"^[+-].*(?:<h[1-6][^>]*>|<p[^>]*>|<button[^>]*>|<span[^>]*>|<li[^>]*>)", diff, re.MULTILINE | re.IGNORECASE):
            labels.append("page copy")
        elif re.search(r"^[+-]\s*(?:title:|subtitle:|label:|text:|description:)", diff, re.MULTILINE | re.IGNORECASE):
            labels.append("page copy")
    if not labels and any(is_docs_path(file) for file in files) and any(is_ui_text_path(file) for file in files):
        labels.append("page and README wording")
    return tuple(unique_limited(labels, 2))


def derive_project_area(
    files: tuple[str, ...],
    focus_topics: tuple[str, ...],
    symbols: tuple[str, ...],
    diff: str,
    surface_labels: tuple[str, ...],
) -> str:
    if not files:
        return "project changes"
    primary_files = primary_source_files(files)
    if files and all(is_docs_path(file) for file in files):
        return derive_docs_area(files)
    if files and all(is_test_path(file) for file in files):
        return derive_test_area(files)
    if surface_labels and any(is_docs_path(file) for file in files) and any(is_ui_text_path(file) for file in files):
        return join_human_list(list(surface_labels[:2]))
    git_area = derive_git_area(primary_files, symbols, diff)
    if git_area:
        return git_area
    if surface_labels:
        return join_human_list(list(surface_labels[:2]))
    symbol_area = derive_symbol_area(symbols)
    if symbol_area:
        return symbol_area
    path_context = derive_path_context(primary_files)
    if path_context:
        return path_context
    file_context = derive_file_context(primary_files)
    if file_context:
        return file_context
    recognized = [TOPIC_SCOPE_LABELS[topic] for topic in focus_topics if topic in TOPIC_SCOPE_LABELS]
    if len(recognized) >= 2:
        return f"{recognized[0]} and {recognized[1]}"
    if recognized:
        return recognized[0]
    if len(focus_topics) == 1:
        return focus_topics[0]
    return "project changes"


def build_change_summary(
    files: tuple[str, ...],
    focus_topics: tuple[str, ...],
    symbols: tuple[str, ...],
    project_area: str,
    surface_labels: tuple[str, ...],
) -> tuple[str, ...]:
    summaries: list[str] = []
    primary_files = select_key_files(files)
    doc_files = [file for file in files if is_docs_path(file)]
    ui_files = [file for file in files if is_ui_text_path(file)]
    if surface_labels and doc_files and ui_files:
        ui_file = ui_files[0]
        doc_file = doc_files[0]
        if "page title" in surface_labels:
            summaries.append(f"Refreshes the page title in `{ui_file}` and matching wording in `{doc_file}`.")
        elif "page copy" in surface_labels:
            summaries.append(f"Refreshes on-page copy in `{ui_file}` and keeps `{doc_file}` aligned.")
        else:
            summaries.append(f"Keeps `{ui_file}` and `{doc_file}` in sync.")
    elif primary_files:
        if len(primary_files) == 1:
            summaries.append(f"Updates `{primary_files[0]}`.")
        else:
            summaries.append(f"Updates {', '.join(f'`{file}`' for file in primary_files[:3])}.")
    if project_area:
        summaries.append(f"Focuses on {project_area}.")
    if symbols:
        summaries.append(f"Touches {', '.join(f'`{symbol}`' for symbol in symbols[:4])}.")
    elif surface_labels:
        summaries.append(f"Centers the work on {join_human_list(list(surface_labels[:2])).casefold()}.")
    elif focus_topics and focus_topics != ("project changes",):
        summaries.append(f"Centers the work on {', '.join(topic.casefold() for topic in focus_topics[:2])}.")
    return tuple(unique_limited(summaries, 3))


def build_impact_summary(
    project_area: str,
    focus_topics: tuple[str, ...],
    files: tuple[str, ...],
    surface_labels: tuple[str, ...],
) -> tuple[str, ...]:
    impacts: list[str] = []
    lowered_area = project_area.casefold()
    if "coverage" in lowered_area or "test coverage" in focus_topics:
        impacts.append("Adds stronger regression protection for the updated behavior.")
    if surface_labels and any(is_docs_path(file) for file in files) and any(is_ui_text_path(file) for file in files):
        impacts.append("Keeps the public-facing wording consistent across the app and README.")
    if "commit suggestion" in lowered_area or ("commit" in lowered_area and "suggest" in lowered_area):
        impacts.append("Makes generated commit messages reflect the changed files and project area more clearly.")
    if any(token in lowered_area for token in ("pull", "push", "pr", "branch", "merge", "git ")) and "test coverage" not in focus_topics:
        impacts.append("Makes everyday Git actions easier to understand for normal project work.")
    if any(token in lowered_area for token in ("help", "guidance", "readme", "documentation")):
        impacts.append("Keeps the command surface clearer for people using DevAgent.")
    for topic in focus_topics:
        if topic == "repo chat":
            impacts.append("Improves how clearly DevAgent explains repository behavior.")
        elif topic == "runtime launch flows":
            impacts.append("Makes local project startup flows easier to run from the CLI.")
        elif topic == "inspection checks":
            impacts.append("Strengthens repo hygiene and safety feedback.")
        elif topic == "workspace setup":
            impacts.append("Smooths onboarding and workspace setup flows.")
    if not impacts and files:
        impacts.append(f"Affects {len(files)} changed file(s) across the workspace.")
    return tuple(unique_limited(impacts, 2))


def build_body_bullets(change_summary: tuple[str, ...], impact_summary: tuple[str, ...], key_files: tuple[str, ...]) -> tuple[str, ...]:
    bullets: list[str] = list(change_summary)
    bullets.extend(impact_summary)
    if key_files:
        if len(key_files) == 1:
            bullets.append(f"Primary file: `{key_files[0]}`.")
        else:
            bullets.append(f"Key files: {', '.join(f'`{file}`' for file in key_files[:3])}.")
    return tuple(unique_limited(bullets, 4))


def derive_path_context(files: tuple[str, ...]) -> str | None:
    tokens: list[str] = []
    for file in files[:4]:
        for token in path_tokens(file):
            humanized = humanize_token(token)
            if humanized not in tokens:
                tokens.append(humanized)
    if not tokens:
        for file in files[:4]:
            normalized = file.replace("\\", "/")
            for part in Path(normalized).parts:
                stem = Path(part).stem
                for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", stem):
                    lowered = raw.casefold()
                    if lowered in COMMON_PATH_TOKENS:
                        continue
                    humanized = humanize_token(lowered)
                    if humanized not in tokens:
                        tokens.append(humanized)
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0]} and {tokens[1]}"


def derive_file_context(files: tuple[str, ...]) -> str | None:
    if len(files) != 1:
        return None
    target = Path(files[0])
    stem = target.stem.casefold()
    if stem and stem not in COMMON_PATH_TOKENS:
        return humanize_token(stem)
    return target.name if target.name else None


def primary_source_files(files: tuple[str, ...]) -> tuple[str, ...]:
    non_tests = tuple(file for file in files if not is_test_path(file))
    non_docs = tuple(file for file in non_tests if not is_docs_path(file))
    if non_docs:
        return non_docs
    if non_tests:
        return non_tests
    return files


def derive_docs_area(files: tuple[str, ...]) -> str:
    if len(files) == 1:
        stem = Path(files[0]).stem
        if stem.casefold() == "readme":
            return "README guidance"
        return f"{humanize_token(stem)} guidance"
    return "documentation guidance"


def derive_test_area(files: tuple[str, ...]) -> str:
    if len(files) == 1:
        stem = Path(files[0]).stem
        if stem.startswith("test_"):
            stem = stem[len("test_") :]
        return f"{humanize_token(stem)} coverage"
    return "regression coverage"


def should_adjust_wording(analysis: ChangeAnalysis) -> bool:
    return bool(analysis.surface_labels) and not analysis.symbols and any(
        label in {"page title", "page copy", "README wording", "page and README wording"}
        for label in analysis.surface_labels
    )


def derive_git_area(files: tuple[str, ...], symbols: tuple[str, ...], diff: str) -> str | None:
    text = " ".join(files) + "\n" + diff + "\n" + " ".join(symbols)
    lowered = text.casefold()
    phrases: list[str] = []
    if "suggest_commit" in lowered or ("commit" in lowered and "message" in lowered):
        phrases.append("commit suggestions")
    operations = detect_git_operations(lowered)
    flow_ops = [operation for operation in operations if operation in {"pull", "push", "PR", "merge", "branch"}]
    if flow_ops:
        phrases.append(f"git {join_human_list(flow_ops)} flows")
    if "help" in lowered and not phrases:
        phrases.append("Git help guidance")
    if phrases:
        return join_human_list(phrases[:2])
    return None


def derive_symbol_area(symbols: tuple[str, ...]) -> str | None:
    cleaned = []
    for symbol in symbols:
        lowered = symbol.casefold()
        if lowered.startswith("test_"):
            continue
        cleaned.append(humanize_token(symbol))
    if not cleaned:
        return None
    return join_human_list(cleaned[:2])


def detect_git_operations(lowered_text: str) -> list[str]:
    operations: list[str] = []
    if any(token in lowered_text for token in ("pull_with_prompts", " git pull", "pull ")):
        operations.append("pull")
    if any(token in lowered_text for token in ("push_with_prompts", " git push", "push ")):
        operations.append("push")
    if any(token in lowered_text for token in ("pr_preview", "pr_create", "pull request", "gh pr", "pr ")):
        operations.append("PR")
    if any(token in lowered_text for token in ("merge_abort", "merge_continue", "merge conflict", "merge ")):
        operations.append("merge")
    if any(token in lowered_text for token in ("branch_create", "branch_switch", "checkout", "branch ")):
        operations.append("branch")
    return unique_limited(operations, 3)


def select_key_files(files: tuple[str, ...]) -> tuple[str, ...]:
    ordered = sorted(files, key=file_priority)
    return tuple(ordered[:5])


def file_priority(path: str) -> tuple[int, int, str]:
    lowered = path.casefold()
    if lowered.endswith((".md", ".txt")):
        weight = 4
    elif "/tests/" in lowered or lowered.startswith("tests/"):
        weight = 3
    elif lowered.endswith((".yml", ".yaml", ".toml", ".json")):
        weight = 2
    else:
        weight = 1
    return (weight, len(Path(path).parts), lowered)


def extract_symbols(diff: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in SYMBOL_RE.finditer(diff or ""):
        for group in match.groups():
            if group:
                found.append(group)
    return tuple(unique_limited(found, 6))


def path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    relative = path.replace("\\", "/")
    for part in Path(relative).parts:
        stem = Path(part).stem
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", stem):
            lowered = raw.casefold()
            if lowered not in COMMON_PATH_TOKENS and len(lowered) > 2:
                tokens.append(lowered)
    return tokens


def humanize_token(token: str) -> str:
    return token.replace("_", " ").replace("-", " ")


def is_docs_path(path: str) -> bool:
    lowered = path.casefold()
    return lowered.endswith((".md", ".mdx", ".rst", ".txt")) or "readme" in lowered


def is_ui_text_path(path: str) -> bool:
    lowered = path.casefold()
    return lowered.endswith((".html", ".htm", ".jsx", ".tsx", ".vue"))


def is_test_path(path: str) -> bool:
    lowered = path.casefold().replace("\\", "/")
    return lowered.startswith("tests/") or "/tests/" in lowered or Path(lowered).name.startswith("test_")


def build_commit_prompt(analysis: ChangeAnalysis, fallback: CommitSuggestion, *, conventional: bool) -> str:
    return (
        f"Conventional commit required: {'yes' if conventional else 'no'}\n"
        f"Fallback subject: {fallback.subject}\n"
        f"Fallback body:\n{fallback.body or '(none)'}\n\n"
        f"Changed files:\n- " + "\n- ".join(analysis.files or ("none",)) + "\n\n"
        "Focus topics:\n- " + "\n- ".join(analysis.focus_topics or ("project changes",)) + "\n\n"
        "Key symbols:\n- " + "\n- ".join(analysis.symbols or ("none",)) + "\n\n"
        "Change summary:\n- " + "\n- ".join(analysis.change_summary or ("none",)) + "\n\n"
        "Impact summary:\n- " + "\n- ".join(analysis.impact_summary or ("none",)) + "\n\n"
        "Diff excerpt:\n"
        f"{truncate_text(analysis.diff or analysis.staged_diff, 5000)}"
    )


def parse_ai_commit_suggestion(response: str, fallback: CommitSuggestion) -> CommitSuggestion | None:
    subject_match = re.search(r"^SUBJECT:\s*(.+)$", response, re.MULTILINE)
    body_match = re.search(r"^BODY:\s*(.*)$", response, re.MULTILINE | re.DOTALL)
    if not subject_match:
        return None
    subject = subject_match.group(1).strip()
    body = (body_match.group(1) if body_match else "").strip()
    if not subject:
        return None
    return CommitSuggestion(
        subject=subject,
        body=body,
        body_bullets=fallback.body_bullets,
        project_area=fallback.project_area,
        changed_files=fallback.changed_files,
        change_summary=fallback.change_summary,
        impact_summary=fallback.impact_summary,
        conventional=fallback.conventional,
    )


def build_pr_head_value(head_repo: str | None, head_branch: str, base_repo: str | None) -> str:
    if not head_repo:
        return head_branch
    if base_repo and head_repo == base_repo:
        return head_branch
    owner = head_repo.split("/", 1)[0] if "/" in head_repo else head_repo
    return f"{owner}:{head_branch}"


def parse_github_repo_slug(url: str) -> str | None:
    for pattern in (GITHUB_HTTP_RE, GITHUB_SSH_RE):
        match = pattern.search(url.strip())
        if match:
            return match.group(1)
    return None


def unique_limited(values: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for value in values:
        compact = " ".join(str(value).split())
        if not compact or compact in seen:
            continue
        seen.append(compact)
        if len(seen) >= limit:
            break
    return seen


def join_human_list(values: list[str] | tuple[str, ...]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def truncate_text(value: str, limit: int) -> str:
    compact = value.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    preferred = locale.getpreferredencoding(False) or "utf-8"
    for encoding in ("utf-8", preferred):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")
