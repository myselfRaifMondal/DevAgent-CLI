import os
from pathlib import Path

import pytest

from devagent.tools.ai import dotenv_candidates, load_dotenv_if_available

KEY = "XAI_API_KEY"


@pytest.fixture(autouse=True)
def _clean_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("DEVAGENT_ENV_FILE", raising=False)
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))


def test_a_dotenv_in_the_working_directory_is_never_read(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "untrusted-repo"
    project.mkdir()
    (project / ".env").write_text(f"{KEY}=planted-by-the-repo\n", encoding="utf-8")
    monkeypatch.chdir(project)

    load_dotenv_if_available()

    assert KEY not in os.environ


def test_the_config_directory_dotenv_is_read(tmp_path: Path) -> None:
    config = tmp_path / "config-home"
    config.mkdir()
    (config / ".env").write_text(f"{KEY}=from-config-dir\n", encoding="utf-8")

    load_dotenv_if_available()

    assert os.environ[KEY] == "from-config-dir"
    os.environ.pop(KEY, None)


def test_an_explicit_env_file_is_read(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "keys.env"
    explicit.write_text(f"{KEY}=from-explicit-file\n", encoding="utf-8")
    monkeypatch.setenv("DEVAGENT_ENV_FILE", str(explicit))

    load_dotenv_if_available()

    assert os.environ[KEY] == "from-explicit-file"
    os.environ.pop(KEY, None)


def test_an_already_set_variable_wins_over_any_file(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "keys.env"
    explicit.write_text(f"{KEY}=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DEVAGENT_ENV_FILE", str(explicit))
    monkeypatch.setenv(KEY, "from-shell")

    load_dotenv_if_available()

    assert os.environ[KEY] == "from-shell"


def test_only_two_locations_are_ever_considered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_ENV_FILE", str(tmp_path / "keys.env"))

    assert dotenv_candidates() == [tmp_path / "keys.env", tmp_path / "config-home" / ".env"]


def test_a_missing_or_broken_file_is_not_an_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_ENV_FILE", str(tmp_path / "does-not-exist.env"))

    load_dotenv_if_available()

    assert KEY not in os.environ
