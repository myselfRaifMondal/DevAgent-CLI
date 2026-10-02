import subprocess
from pathlib import Path

import pytest

import devagent.tools.edit_tool as edit_tool_module
from devagent.tools.edit_tool import (
    EditAgent,
    EditProposal,
    UnsafePatchPathError,
    apply_unified_diff_fallback,
    validate_patch_paths,
)

NEW_FILE_TEMPLATE = """--- /dev/null
+++ {path}
@@ -0,0 +1 @@
+pwned
"""


def _git_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    (workspace / "README.md").write_text("Hello\n", encoding="utf-8")
    return workspace


def _reject_git_apply(monkeypatch) -> None:
    """Force the fallback applier, as happens whenever ``git apply`` rejects a model diff."""

    def fake_run(args, cwd=None, input=None, capture_output=None):
        return subprocess.CompletedProcess(args=args, returncode=1, stdout=b"", stderr=b"patch does not apply")

    monkeypatch.setattr(edit_tool_module.subprocess, "run", fake_run)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "b/../../escape.txt",
        "sub/../../escape.txt",
        "..\\escape.txt",
    ],
)
def test_fallback_refuses_parent_directory_traversal(tmp_path: Path, path: str) -> None:
    workspace = _git_workspace(tmp_path)

    with pytest.raises(UnsafePatchPathError):
        apply_unified_diff_fallback(NEW_FILE_TEMPLATE.format(path=path), workspace)

    assert not (tmp_path / "escape.txt").exists()


def test_dot_dot_is_refused_even_when_it_stays_inside_the_workspace(tmp_path: Path) -> None:
    # Resolves to <workspace>/README.md, so only an explicit ".." rule rejects it: the
    # string is ambiguous between git and the fallback applier and has no honest use.
    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(NEW_FILE_TEMPLATE.format(path="b/sub/../README.md"), tmp_path)


def test_fallback_refuses_absolute_path(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)
    outside = tmp_path / "absolute.txt"

    with pytest.raises(UnsafePatchPathError):
        apply_unified_diff_fallback(NEW_FILE_TEMPLATE.format(path=outside), workspace)

    assert not outside.exists()


@pytest.mark.parametrize("path", ["C:/Windows/evil.txt", "C:\\Windows\\evil.txt"])
def test_validation_refuses_windows_drive_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(NEW_FILE_TEMPLATE.format(path=path), tmp_path)


@pytest.mark.parametrize("path", [".git/hooks/pre-commit", "b/.git/config", ".devagent/run_profiles.json"])
def test_validation_refuses_protected_directories(tmp_path: Path, path: str) -> None:
    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(NEW_FILE_TEMPLATE.format(path=path), tmp_path)


def test_validation_refuses_symlink_that_points_outside_workspace(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs extra privileges on this platform")

    with pytest.raises(UnsafePatchPathError):
        apply_unified_diff_fallback(NEW_FILE_TEMPLATE.format(path="b/link/planted.txt"), workspace)

    assert not (outside / "planted.txt").exists()


def test_validation_covers_rename_and_git_headers(tmp_path: Path) -> None:
    rename = "diff --git a/README.md b/README.md\nrename from README.md\nrename to ../moved.md\n"
    header = "diff --git a/README.md b/../../README.md\n"

    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(rename, tmp_path)
    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(header, tmp_path)


def test_validation_does_not_depend_on_the_fallback_parser(tmp_path: Path) -> None:
    # No hunk header at all: parse_unified_diff would reject this, git might not.
    malformed = "--- a/README.md\n+++ ../../etc/passwd\n"

    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(malformed, tmp_path)


def test_hunk_content_that_looks_like_a_header_is_not_a_path(tmp_path: Path) -> None:
    diff = """--- a/notes.sql
+++ b/notes.sql
@@ -1,2 +1,2 @@
 select 1;
--- ../comment
+++ /also/a/comment
"""
    # The pair above sits where a header could legitimately start, so it is validated; the
    # point is that ordinary removed/added lines elsewhere are not mistaken for paths.
    ordinary = """--- a/notes.sql
+++ b/notes.sql
@@ -1,2 +1,2 @@
 select 1;
-select 2;
+select 3;
"""
    validate_patch_paths(ordinary, tmp_path)
    with pytest.raises(UnsafePatchPathError):
        validate_patch_paths(diff, tmp_path)


def test_apply_refuses_traversal_even_when_git_rejects_the_diff(tmp_path: Path, monkeypatch) -> None:
    workspace = _git_workspace(tmp_path)
    _reject_git_apply(monkeypatch)
    proposal = EditProposal(
        instruction="write a file",
        diff=NEW_FILE_TEMPLATE.format(path="b/../../escape.txt"),
        message="Patch generated.",
    )

    with pytest.raises(UnsafePatchPathError):
        EditAgent(workspace).apply(proposal)

    assert not (tmp_path / "escape.txt").exists()


def test_apply_does_not_ask_the_model_to_repair_an_unsafe_diff(tmp_path: Path, monkeypatch) -> None:
    workspace = _git_workspace(tmp_path)
    _reject_git_apply(monkeypatch)
    agent = EditAgent(workspace)
    repair_calls: list[str] = []
    monkeypatch.setattr(agent, "_repair_diff", lambda **kwargs: repair_calls.append("called"))

    with pytest.raises(UnsafePatchPathError):
        agent.apply(EditProposal("x", NEW_FILE_TEMPLATE.format(path="../escape.txt"), "Patch generated."))

    assert repair_calls == []


def test_a_repaired_diff_is_validated_before_it_is_applied(tmp_path: Path, monkeypatch) -> None:
    workspace = _git_workspace(tmp_path)
    _reject_git_apply(monkeypatch)
    agent = EditAgent(workspace)
    monkeypatch.setattr(
        agent, "_repair_diff", lambda **kwargs: NEW_FILE_TEMPLATE.format(path="b/../../escape.txt")
    )
    broken = "--- a/README.md\n+++ b/README.md\n@@ -9 +9 @@\n-nothing\n+something\n"
    agent.ai = type("AI", (), {"available": True})()

    with pytest.raises(UnsafePatchPathError):
        agent.apply(EditProposal("x", broken, "Patch generated."))

    assert not (tmp_path / "escape.txt").exists()


def test_safe_relative_diff_still_applies(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)
    proposal = EditProposal(
        instruction="edit readme",
        diff="--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n Hello\n+World\n",
        message="Patch generated.",
    )

    EditAgent(workspace).apply(proposal)

    assert (workspace / "README.md").read_text(encoding="utf-8") == "Hello\nWorld\n"


def test_safe_new_file_in_subdirectory_still_applies_via_fallback(tmp_path: Path) -> None:
    workspace = _git_workspace(tmp_path)

    apply_unified_diff_fallback(NEW_FILE_TEMPLATE.format(path="b/docs/notes.txt"), workspace)

    assert (workspace / "docs" / "notes.txt").read_text(encoding="utf-8") == "pwned\n"
