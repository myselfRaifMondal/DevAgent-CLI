import subprocess
from pathlib import Path

import pytest

from devagent.tools.git_tool import GitError, GitTool, PullOptions, PushOptions, validate_ref_name, validate_repo_slug


class SilentAI:
    available = False

    def complete(self, *args, **kwargs):
        return None


def _repo(tmp_path: Path) -> GitTool:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)
    return GitTool(tmp_path, ai=SilentAI())


def _branches(path: Path) -> list[str]:
    out = subprocess.run(["git", "branch", "--format=%(refname:short)"], cwd=path, capture_output=True, text=True, check=True)
    return sorted(out.stdout.split())


BAD_REFS = [
    "",
    "   ",
    "-b",
    "--orphan",
    "--upload-pack=touch /tmp/pwned",
    "-f",
    "feature branch",
    "a..b",
    "a@{1}",
    "a~1",
    "a^",
    "a:b",
    "a?b",
    "a*b",
    "a[b",
    "a\\b",
    "/leading",
    ".hidden",
    "trailing/",
    "trailing.",
    "name.lock",
    "double//slash",
    "tab\tname",
    "new\nline",
    "@",
]


@pytest.mark.parametrize("name", BAD_REFS)
def test_unsafe_ref_names_are_rejected(name: str) -> None:
    with pytest.raises(GitError):
        validate_ref_name(name)


@pytest.mark.parametrize("name", ["main", "feature/login", "release-1.2", "fix_bug", "user/topic.v2", "origin", "ünï"])
def test_ordinary_ref_names_are_accepted(name: str) -> None:
    assert validate_ref_name(name) == name


@pytest.mark.parametrize("slug", ["", "owner", "-x/repo", "owner/-repo", "owner/repo/extra", "owner/re po", "--repo=evil/x", "a/b;rm"])
def test_bad_repo_slugs_are_rejected(slug: str) -> None:
    with pytest.raises(GitError):
        validate_repo_slug(slug)


@pytest.mark.parametrize("slug", ["owner/repo", "IndiQuant-by-CAWM/Meta-Model", "owner/.github", "a_b/c.d"])
def test_ordinary_repo_slugs_are_accepted(slug: str) -> None:
    assert validate_repo_slug(slug) == slug


def _forbid_network_git(tool: GitTool, monkeypatch) -> None:
    """Allow read-only lookups (tracked remote, branch name); fail if pull/push/etc. would run."""
    original = tool._run

    def guarded(args, check=True):
        if len(args) > 1 and args[1] in {"pull", "push", "fetch", "checkout"}:
            pytest.fail(f"git must not be invoked with unsafe input: {args}")
        return original(args, check)

    monkeypatch.setattr(tool, "_run", guarded)


def test_create_branch_refuses_an_option_and_changes_nothing(tmp_path: Path) -> None:
    tool = _repo(tmp_path)
    before = _branches(tmp_path)

    with pytest.raises(GitError):
        tool.create_branch("--orphan")

    assert _branches(tmp_path) == before


def test_create_branch_never_invokes_git_for_an_unsafe_name(tmp_path: Path, monkeypatch) -> None:
    # git rejects "--orphan" on its own, which would mask a missing check; assert it is never asked.
    tool = _repo(tmp_path)
    monkeypatch.setattr(tool, "_run", lambda *a, **k: pytest.fail("git must not be invoked for an unsafe name"))

    with pytest.raises(GitError):
        tool.create_branch("--orphan")


def test_switch_branch_refuses_an_option_and_never_runs_git(tmp_path: Path, monkeypatch) -> None:
    tool = _repo(tmp_path)
    monkeypatch.setattr(tool, "_run", lambda *a, **k: pytest.fail("git must not be invoked for an unsafe name"))

    with pytest.raises(GitError):
        tool.switch_branch("-f")


def test_switch_branch_passes_double_dash_so_a_name_is_never_a_path(tmp_path: Path) -> None:
    tool = _repo(tmp_path)
    tool.create_branch("topic")
    tool.switch_branch("main")
    calls: list[list[str]] = []
    original = tool._run
    tool._run = lambda args, check=True: calls.append(list(args)) or original(args, check)

    tool.switch_branch("topic")

    assert calls[-1] == ["git", "checkout", "topic", "--"]
    assert tool.current_branch() == "topic"


@pytest.mark.parametrize("remote, branch", [("--upload-pack=touch /tmp/pwned", "main"), ("origin", "--rebase"), ("origin", "a b")])
def test_pull_refuses_unsafe_remote_or_branch(tmp_path: Path, monkeypatch, remote: str, branch: str) -> None:
    tool = _repo(tmp_path)
    _forbid_network_git(tool, monkeypatch)

    with pytest.raises(GitError):
        tool.pull(PullOptions(remote=remote, branch=branch))


@pytest.mark.parametrize(
    "remote, local, destination",
    [("--receive-pack=evil", "main", "main"), ("origin", "--delete", "main"), ("origin", "main", "--force")],
)
def test_push_refuses_unsafe_remote_or_branches(tmp_path: Path, monkeypatch, remote: str, local: str, destination: str) -> None:
    tool = _repo(tmp_path)
    _forbid_network_git(tool, monkeypatch)

    with pytest.raises(GitError):
        tool.push(PushOptions(remote=remote, local_branch=local, remote_branch=destination))


def test_resolve_base_ref_refuses_an_option(tmp_path: Path, monkeypatch) -> None:
    tool = _repo(tmp_path)
    monkeypatch.setattr(tool, "_run", lambda *a, **k: pytest.fail("git must not be invoked"))

    with pytest.raises(GitError):
        tool.resolve_base_ref("--output=/tmp/x")


def test_create_pr_refuses_unsafe_head_branch_and_slugs(tmp_path: Path, monkeypatch) -> None:
    tool = _repo(tmp_path)
    ready = type("Ready", (), {"can_create_pr": True, "blocking_reasons": ()})()
    monkeypatch.setattr(tool, "pr_readiness", lambda **kwargs: ready)
    monkeypatch.setattr(tool, "build_pr_preview", lambda options: pytest.fail("must fail before building a preview"))

    with pytest.raises(GitError):
        tool.create_pr(base_branch="main", head_branch="--web", base_repo="owner/repo", head_repo="owner/repo")
    with pytest.raises(GitError):
        tool.create_pr(base_branch="main", head_branch="topic", base_repo="--repo=evil/x", head_repo="owner/repo")
