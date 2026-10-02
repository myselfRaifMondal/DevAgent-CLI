from pathlib import Path

import devagent.tools.runtime_tool as runtime_tool_module
from devagent.core import host
from devagent.tools.runtime_tool import RunTool, build_windows_terminal_command, write_windows_launcher


def test_detect_launch_specs_for_mixed_workspace(tmp_path: Path) -> None:
    (tmp_path / "frontend").mkdir()
    (tmp_path / "backend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text(
        '{"scripts": {"dev": "vite", "start": "vite preview"}}',
        encoding="utf-8",
    )
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "backend" / "main.py").write_text("print('hello')\n", encoding="utf-8")

    specs = RunTool(tmp_path).detect_launch_specs()

    assert len(specs) == 2
    node_spec = next(spec for spec in specs if spec.kind == "node")
    python_spec = next(spec for spec in specs if spec.kind == "python")

    assert node_spec.cwd == tmp_path / "frontend"
    assert node_spec.command == ["npm", "run", "dev"]
    assert node_spec.browser_url == "http://localhost:5173"
    assert python_spec.cwd == tmp_path / "backend"
    assert python_spec.command == ["python", "main.py"]
    assert python_spec.venv_dir == tmp_path / "backend" / ".venv"
    assert python_spec.bootstrap_commands


def test_detect_launch_specs_reuses_existing_named_venv(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "backend" / "run.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "backend" / "venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "backend" / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "venv" / "pyvenv.cfg").write_text("home = C:\\Python311\n", encoding="utf-8")

    specs = RunTool(tmp_path).detect_launch_specs()

    python_spec = next(spec for spec in specs if spec.kind == "python")
    assert python_spec.venv_dir == tmp_path / "backend" / "venv"
    assert python_spec.bootstrap_commands == ()


def test_build_windows_terminal_command_activates_venv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(host, "is_windows", lambda: True)
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (backend / "main.py").write_text("print('hello')\n", encoding="utf-8")
    venv_dir = backend / ".venv"
    profile = RunTool(tmp_path).save_manual_profile("start backend", "python main.py", cwd=backend)
    command = build_windows_terminal_command(profile.specs[0])

    assert "activate.bat" in command
    assert "python main.py" in command
    assert str(venv_dir / "Scripts" / "python.exe") in command


def test_write_windows_launcher_uses_batch_file_for_bootstrap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(host, "is_windows", lambda: True)
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (backend / "run.py").write_text("print('hello')\n", encoding="utf-8")

    workspace_tool = RunTool(tmp_path)
    profile = workspace_tool.save_manual_profile("start backend", "python run.py", cwd=backend)
    launcher = write_windows_launcher(tmp_path, profile.specs[0])
    content = launcher.read_text(encoding="utf-8")

    assert launcher.suffix == ".cmd"
    assert "cd /d" in content
    assert "call" in content
    assert "activate.bat" in content
    assert "python.exe" in content
    assert "python run.py" in content


def test_launch_opens_browser_for_detected_frontend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(host, "is_windows", lambda: True)
    opened_urls: list[str] = []
    spawned: list[list[str]] = []

    class DummyProcess:
        pass

    def fake_popen(args, cwd=None, creationflags=0):
        spawned.append(list(args))
        return DummyProcess()

    monkeypatch.setattr(runtime_tool_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime_tool_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(runtime_tool_module.webbrowser, "open", lambda url: opened_urls.append(url))

    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite --port 3001"}}', encoding="utf-8")

    specs = RunTool(tmp_path).launch_detected(open_browser=True)

    assert specs[0].browser_url == "http://localhost:3001"
    assert opened_urls == ["http://localhost:3001"]
    assert spawned and spawned[0][0:2] == ["cmd.exe", "/k"]
    assert spawned[0][2].endswith(".cmd")


def test_saved_profiles_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "frontend").mkdir()
    (workspace / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite"}}', encoding="utf-8")

    tool = RunTool(workspace)
    saved = tool.save_detected_profile("I order you to start in the name of jesus", open_browser=True)

    profiles = tool.saved_profiles()

    assert saved.specs
    assert "I order you to start in the name of jesus" in profiles
    assert profiles["I order you to start in the name of jesus"].specs[0].command == ["npm", "run", "dev"]
    assert profiles["I order you to start in the name of jesus"].specs[0].browser_url == "http://localhost:5173"
    assert profiles["I order you to start in the name of jesus"].open_browser is True

    assert tool.delete_profile("I order you to start in the name of jesus")
    assert tool.saved_profiles() == {}


def test_saved_profiles_match_normalized_phrase(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "frontend").mkdir()
    (workspace / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite"}}', encoding="utf-8")

    tool = RunTool(workspace)
    tool.save_detected_profile("Start   I Command You", open_browser=True)

    profile = tool.find_profile("  start i command you ")

    assert profile is not None
    assert profile.phrase == "Start   I Command You"
    assert profile.open_browser is True


def test_launch_saved_uses_profile_browser_preference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setattr(host, "is_windows", lambda: True)
    opened_urls: list[str] = []

    class DummyProcess:
        pass

    monkeypatch.setattr(runtime_tool_module.subprocess, "Popen", lambda *args, **kwargs: DummyProcess())
    monkeypatch.setattr(runtime_tool_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(runtime_tool_module.webbrowser, "open", lambda url: opened_urls.append(url))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "frontend").mkdir()
    (workspace / "frontend" / "package.json").write_text('{"scripts": {"dev": "vite --port 4173"}}', encoding="utf-8")

    tool = RunTool(workspace)
    tool.save_detected_profile("Start the site", open_browser=True)
    tool.launch_saved("start the site")

    assert opened_urls == ["http://localhost:4173"]
