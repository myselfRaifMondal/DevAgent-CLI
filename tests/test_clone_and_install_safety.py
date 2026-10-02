import subprocess
from pathlib import Path

import pytest

import devagent.tools.setup_tool as setup_tool_module
from devagent.tools.setup_tool import SetupTool, is_js_install, normalize_github_clone_url, run, validate_github_clone_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/",
        "git@github.com:owner/repo.git",
        "git@github.com:owner/repo",
        "https://github.com/owner/.github",
    ],
)
def test_github_clone_urls_are_accepted(url: str) -> None:
    assert validate_github_clone_url(normalize_github_clone_url(url))


def test_plain_http_github_urls_are_upgraded_to_https() -> None:
    assert validate_github_clone_url(normalize_github_clone_url("http://github.com/owner/repo")) == "https://github.com/owner/repo.git"


@pytest.mark.parametrize(
    "url",
    [
        "ext::sh -c touch% /tmp/pwned",
        "ext::sh -c 'touch /tmp/pwned'.git",
        "--upload-pack=touch /tmp/pwned",
        "-oProxyCommand=evil.git",
        "file:///etc/passwd",
        "file:///home/user/repo.git",
        "/local/path/repo.git",
        "../repo.git",
        "https://evil.example/owner/repo.git",
        "https://github.com.evil.example/owner/repo.git",
        "https://user:pass@github.com/owner/repo.git",
        "git@evil.example:owner/repo.git",
        "ssh://git@github.com/owner/repo.git",
        "https://github.com/owner",
        "https://github.com/owner/..",
        "git@github.com:owner/..",
        "",
    ],
)
def test_other_clone_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        validate_github_clone_url(normalize_github_clone_url(url))


@pytest.mark.parametrize("url", ["ext::sh -c 'touch /tmp/pwned'.git", "--upload-pack=x", "file:///etc/passwd"])
def test_clone_never_reaches_git_with_an_unsafe_url(tmp_path: Path, monkeypatch, url: str) -> None:
    monkeypatch.setattr(setup_tool_module.subprocess, "run", lambda *a, **k: pytest.fail("git clone must not run"))

    with pytest.raises(ValueError):
        SetupTool.clone_from_github(url, target=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["--public", "-x", "has space", "a/b/c", "", ".hidden", "semi;colon"])
def test_publish_refuses_an_unsafe_repository_name(tmp_path: Path, monkeypatch, name: str) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    monkeypatch.setattr(setup_tool_module, "run", lambda *a, **k: pytest.fail("gh must not run"))

    with pytest.raises(ValueError):
        SetupTool.publish_to_github(project, repo_name=name or "--", private=True)


@pytest.mark.parametrize("args", [["npm", "install"], ["pnpm", "install"], ["yarn", "install"], ["/usr/bin/npm", "install"], ["npm.cmd", "install"]])
def test_js_installs_are_recognised(args: list[str]) -> None:
    assert is_js_install(args)


@pytest.mark.parametrize("args", [["npm", "run", "dev"], ["pip", "install", "-r", "requirements.txt"], ["git", "clone", "x"], []])
def test_other_commands_are_not_js_installs(args: list[str]) -> None:
    assert not is_js_install(args)


def test_js_installs_run_with_lifecycle_scripts_disabled(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run(args, cwd=None, text=None, capture_output=None, env=None):
        captured["env"] = env
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup_tool_module.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_tool_module, "resolve_command", lambda args: args)

    run(["npm", "install"], cwd=tmp_path)

    assert captured["env"]["npm_config_ignore_scripts"] == "true"
    assert captured["env"]["YARN_ENABLE_SCRIPTS"] == "false"


def test_other_commands_do_not_get_the_scripts_override(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run(args, cwd=None, text=None, capture_output=None, env=None):
        captured["env"] = env
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup_tool_module.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_tool_module, "resolve_command", lambda args: args)

    run(["git", "status"], cwd=tmp_path)

    assert captured["env"] is None
