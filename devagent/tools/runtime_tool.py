from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devagent.config.settings import ConfigManager
from devagent.context.scanner import IGNORED_DIRS
from devagent.core import host
from devagent.core.trust import Approver, require_launch_approval
from devagent.tools.setup_tool import is_python_venv_dir, preferred_python_venv_dir, python_venv_executable


PYTHON_ENTRY_FILES = ("manage.py", "main.py", "app.py", "server.py", "run.py")


@dataclass(frozen=True)
class LaunchSpec:
    name: str
    cwd: Path
    command: list[str]
    display_command: str
    kind: str
    source: str = "detected"
    venv_dir: Path | None = None
    bootstrap_commands: tuple[tuple[str, ...], ...] = ()
    browser_url: str | None = None

    def scope(self, workspace: Path) -> str:
        try:
            relative = self.cwd.relative_to(workspace)
        except ValueError:
            return str(self.cwd)
        return relative.as_posix() if relative.parts else "."

    def to_dict(self, workspace: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "cwd": self.scope(workspace),
            "command": self.command,
            "display_command": self.display_command,
            "kind": self.kind,
            "source": self.source,
            "venv_dir": serialize_optional_path(workspace, self.venv_dir),
            "bootstrap_commands": [list(command) for command in self.bootstrap_commands],
            "browser_url": self.browser_url,
        }

    @classmethod
    def from_dict(cls, workspace: Path, data: dict[str, Any]) -> "LaunchSpec":
        return cls(
            name=str(data["name"]),
            cwd=resolve_serialized_path(workspace, str(data["cwd"])),
            command=[str(part) for part in data.get("command", [])],
            display_command=str(data.get("display_command") or ""),
            kind=str(data.get("kind") or "custom"),
            source=str(data.get("source") or "saved"),
            venv_dir=resolve_optional_path(workspace, data.get("venv_dir")),
            bootstrap_commands=tuple(tuple(str(part) for part in command) for command in data.get("bootstrap_commands", [])),
            browser_url=str(data["browser_url"]) if data.get("browser_url") else None,
        )


@dataclass(frozen=True)
class RunProfile:
    phrase: str
    specs: list[LaunchSpec]
    open_browser: bool = False
    description: str | None = None

    def normalized_phrase(self) -> str:
        return normalize_phrase(self.phrase)

    def to_dict(self, workspace: Path) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "open_browser": self.open_browser,
            "description": self.description,
            "specs": [spec.to_dict(workspace) for spec in self.specs],
        }

    @classmethod
    def from_dict(cls, workspace: Path, phrase: str, data: dict[str, Any] | list[dict[str, Any]]) -> "RunProfile":
        if isinstance(data, list):
            specs = [LaunchSpec.from_dict(workspace, item) for item in data]
            return cls(phrase=phrase, specs=specs)
        return cls(
            phrase=str(data.get("phrase") or phrase),
            specs=[LaunchSpec.from_dict(workspace, item) for item in data.get("specs", [])],
            open_browser=bool(data.get("open_browser", False)),
            description=str(data["description"]) if data.get("description") else None,
        )


class RunTool:
    def __init__(self, workspace: Path, *, approver: Approver | None = None):
        self.workspace = workspace.expanduser().resolve()
        self.approver = approver

    def detect_launch_specs(self) -> list[LaunchSpec]:
        specs: list[LaunchSpec] = []
        for directory in candidate_directories(self.workspace):
            node_spec = detect_node_launch_spec(directory, self.workspace)
            if node_spec:
                specs.append(node_spec)
            python_spec = detect_python_launch_spec(directory, self.workspace)
            if python_spec:
                specs.append(python_spec)
        return sorted(specs, key=lambda item: (item.scope(self.workspace), item.kind, item.name))

    def launch_detected(self, *, open_browser: bool = False) -> list[LaunchSpec]:
        specs = self.detect_launch_specs()
        if not specs:
            raise RuntimeError("No launchable services were detected in the active workspace.")
        self.launch(specs, open_browser=open_browser)
        return specs

    def launch_saved(self, phrase: str, *, open_browser: bool | None = None) -> list[LaunchSpec]:
        profile = self.find_profile(phrase)
        if not profile:
            raise RuntimeError(f"No saved launch phrase found: {phrase}")
        return self.launch_profile(profile, open_browser=open_browser)

    def launch_profile(self, profile: RunProfile, *, open_browser: bool | None = None) -> list[LaunchSpec]:
        should_open_browser = profile.open_browser if open_browser is None else open_browser
        self.launch(profile.specs, open_browser=should_open_browser)
        return profile.specs

    def save_detected_profile(self, phrase: str, *, open_browser: bool = False, description: str | None = None) -> RunProfile:
        specs = self.detect_launch_specs()
        if not specs:
            raise RuntimeError("No launchable services were detected to save.")
        profiles = self.saved_profiles()
        profile = RunProfile(phrase=phrase, specs=specs, open_browser=open_browser, description=description)
        profiles = replace_profile(profiles, profile)
        self._save_profiles(profiles)
        return profile

    def save_manual_profile(
        self,
        phrase: str,
        command_text: str,
        cwd: Path | None = None,
        *,
        open_browser: bool = False,
        description: str | None = None,
    ) -> RunProfile:
        target_dir = (cwd or self.workspace).expanduser().resolve()
        spec = build_manual_launch_spec(phrase, target_dir, command_text)
        profiles = self.saved_profiles()
        profile = RunProfile(phrase=phrase, specs=[spec], open_browser=open_browser, description=description)
        profiles = replace_profile(profiles, profile)
        self._save_profiles(profiles)
        return profile

    def saved_profiles(self) -> dict[str, RunProfile]:
        file_path = self._profiles_file()
        if not file_path.exists():
            return {}
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        profiles: dict[str, RunProfile] = {}
        for phrase, items in raw.get("profiles", {}).items():
            profile = RunProfile.from_dict(self.workspace, phrase, items)
            profiles[profile.phrase] = profile
        return profiles

    def find_profile(self, phrase: str) -> RunProfile | None:
        normalized = normalize_phrase(phrase)
        for profile in self.saved_profiles().values():
            if profile.normalized_phrase() == normalized:
                return profile
        return None

    def delete_profile(self, phrase: str) -> bool:
        profiles = self.saved_profiles()
        target_key = next((key for key, profile in profiles.items() if profile.normalized_phrase() == normalize_phrase(phrase)), None)
        if not target_key:
            return False
        profiles.pop(target_key)
        self._save_profiles(profiles)
        return True

    def launch(self, specs: list[LaunchSpec], *, open_browser: bool = False) -> None:
        require_launch_approval(self.workspace, specs, self.approver)
        if host.is_windows():
            for spec in specs:
                launcher = write_windows_launcher(self.workspace, spec)
                subprocess.Popen(
                    ["cmd.exe", "/k", str(launcher)],
                    cwd=spec.cwd,
                    creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                )
        else:
            for spec in specs:
                subprocess.Popen(spec.command, cwd=spec.cwd)
        if open_browser:
            url = preferred_browser_url(specs)
            if not url:
                raise RuntimeError("Started the services, but DevAgent could not infer a local app URL to open.")
            time.sleep(2.0)
            webbrowser.open(url)

    def _profiles_file(self) -> Path:
        return ConfigManager.workspace_cache_dir(self.workspace) / "run_profiles.json"

    def _save_profiles(self, profiles: dict[str, RunProfile]) -> None:
        file_path = self._profiles_file()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiles": {
                phrase: profile.to_dict(self.workspace)
                for phrase, profile in sorted(profiles.items())
            }
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_directories(workspace: Path, max_depth: int = 3) -> list[Path]:
    directories = [workspace]
    for path in sorted(workspace.rglob("*")):
        if not path.is_dir():
            continue
        relative = path.relative_to(workspace)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if len(relative.parts) > max_depth:
            continue
        directories.append(path)
    return directories


def detect_node_launch_spec(directory: Path, workspace: Path) -> LaunchSpec | None:
    package_json = directory / "package.json"
    if not package_json.exists():
        return None
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        return None
    script_name = next((name for name in ("dev", "start") if isinstance(scripts.get(name), str)), None)
    if not script_name:
        return None
    script_body = str(scripts[script_name])
    package_manager = detect_package_manager(directory)
    command, display = package_manager_command(package_manager, script_name)
    scope = relative_scope(directory, workspace)
    label = f"{scope} node {script_name}" if scope != "." else f"node {script_name}"
    return LaunchSpec(
        name=label,
        cwd=directory,
        command=command,
        display_command=display,
        kind="node",
        browser_url=infer_browser_url_from_script(script_name, script_body),
    )


def detect_python_launch_spec(directory: Path, workspace: Path) -> LaunchSpec | None:
    entry = python_entry_command(directory)
    if not entry:
        return None

    requirements = directory / "requirements.txt"
    pyproject = directory / "pyproject.toml"
    venv_dir = preferred_python_venv_dir(directory)
    bootstrap_commands: list[tuple[str, ...]] = []
    if requirements.exists():
        if not is_python_venv_dir(venv_dir):
            bootstrap_commands.append((sys.executable, "-m", "venv", venv_dir.name))
            bootstrap_commands.append((str(python_venv_executable(venv_dir)), "-m", "pip", "install", "-r", "requirements.txt"))
    elif pyproject.exists():
        if not is_python_venv_dir(venv_dir):
            bootstrap_commands.append((sys.executable, "-m", "venv", venv_dir.name))
            bootstrap_commands.append((str(python_venv_executable(venv_dir)), "-m", "pip", "install", "-e", "."))
    else:
        venv_dir = venv_dir if is_python_venv_dir(venv_dir) else None

    scope = relative_scope(directory, workspace)
    label = f"{scope} python" if scope != "." else "python app"
    return LaunchSpec(
        name=label,
        cwd=directory,
        command=entry,
        display_command=" ".join(entry),
        kind="python",
        venv_dir=venv_dir,
        bootstrap_commands=tuple(bootstrap_commands),
        browser_url=infer_browser_url_from_python_command(entry),
    )


def python_entry_command(directory: Path) -> list[str] | None:
    manage = directory / "manage.py"
    if manage.exists():
        return ["python", "manage.py", "runserver"]
    for filename in PYTHON_ENTRY_FILES[1:]:
        candidate = directory / filename
        if candidate.exists():
            return ["python", filename]
    return None


def detect_package_manager(directory: Path) -> str:
    if (directory / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (directory / "yarn.lock").exists():
        return "yarn"
    return "npm"


def package_manager_command(package_manager: str, script_name: str) -> tuple[list[str], str]:
    if package_manager == "yarn":
        return [package_manager, script_name], f"{package_manager} {script_name}"
    return [package_manager, "run", script_name], f"{package_manager} run {script_name}"


def build_manual_launch_spec(phrase: str, cwd: Path, command_text: str) -> LaunchSpec:
    command = shlex.split(command_text, posix=not host.is_windows())
    if not command:
        raise RuntimeError("The custom launch command was empty.")

    requirements = cwd / "requirements.txt"
    pyproject = cwd / "pyproject.toml"
    venv_dir = preferred_python_venv_dir(cwd)
    bootstrap_commands: list[tuple[str, ...]] = []
    if command[0].lower() in {"python", "py"} and (requirements.exists() or pyproject.exists() or venv_dir.exists()):
        if requirements.exists():
            if not is_python_venv_dir(venv_dir):
                bootstrap_commands = [
                    (sys.executable, "-m", "venv", venv_dir.name),
                    (str(python_venv_executable(venv_dir)), "-m", "pip", "install", "-r", "requirements.txt"),
                ]
        elif pyproject.exists():
            if not is_python_venv_dir(venv_dir):
                bootstrap_commands = [
                    (sys.executable, "-m", "venv", venv_dir.name),
                    (str(python_venv_executable(venv_dir)), "-m", "pip", "install", "-e", "."),
                ]
        else:
            venv_dir = venv_dir if is_python_venv_dir(venv_dir) else None
    else:
        venv_dir = None

    return LaunchSpec(
        name=phrase,
        cwd=cwd,
        command=["python", *command[1:]] if command[0].lower() == "py" else command,
        display_command=command_text,
        kind="custom",
        source="saved",
        venv_dir=venv_dir,
        bootstrap_commands=tuple(bootstrap_commands),
        browser_url=infer_browser_url_from_manual_command(command_text),
    )


def build_windows_terminal_command(spec: LaunchSpec) -> str:
    steps: list[str] = [f"title DevAgent - {sanitize_console_title(spec.name)}"]
    if spec.venv_dir:
        activate_script = spec.venv_dir / "Scripts" / "activate.bat"
        if spec.bootstrap_commands:
            bootstrap = " && ".join(format_windows_command(command) for command in spec.bootstrap_commands)
            steps.append(f'if not exist "{activate_script}" ({bootstrap})')
        steps.append(f'call "{activate_script}"')
    steps.append(format_windows_command(spec.command))
    return " && ".join(steps)


def relative_scope(directory: Path, workspace: Path) -> str:
    try:
        relative = directory.relative_to(workspace)
    except ValueError:
        return str(directory)
    return relative.as_posix() if relative.parts else "."


def sanitize_console_title(value: str) -> str:
    return value.replace("&", "and").replace("|", " ").strip() or "DevAgent"


def serialize_optional_path(workspace: Path, value: Path | None) -> str | None:
    if value is None:
        return None
    try:
        return value.relative_to(workspace).as_posix()
    except ValueError:
        return str(value)


def resolve_optional_path(workspace: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return resolve_serialized_path(workspace, value)


def resolve_serialized_path(workspace: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.expanduser().resolve()
    return (workspace / raw).expanduser().resolve()


def preferred_browser_url(specs: list[LaunchSpec]) -> str | None:
    for kind in ("node", "custom", "python"):
        for spec in specs:
            if spec.kind == kind and spec.browser_url:
                return spec.browser_url
    return next((spec.browser_url for spec in specs if spec.browser_url), None)


def infer_browser_url_from_script(script_name: str, script_body: str) -> str | None:
    lower = script_body.lower()
    port = extract_port(script_body)
    if "vite" in lower:
        return f"http://localhost:{port or 5173}"
    if "next" in lower:
        return f"http://localhost:{port or 3000}"
    if "react-scripts" in lower or "webpack-dev-server" in lower:
        return f"http://localhost:{port or 3000}"
    if "ng serve" in lower:
        return f"http://localhost:{port or 4200}"
    if script_name == "dev":
        return f"http://localhost:{port or 5173}"
    if script_name == "start":
        return f"http://localhost:{port or 3000}"
    return None


def infer_browser_url_from_python_command(command: list[str]) -> str | None:
    joined = " ".join(command).lower()
    port = extract_port(joined)
    if "runserver" in joined:
        return f"http://127.0.0.1:{port or 8000}"
    if any(token in joined for token in ("uvicorn", "fastapi", "flask", "streamlit", "gradio")):
        return f"http://127.0.0.1:{port or 8000}"
    return None


def infer_browser_url_from_manual_command(command_text: str) -> str | None:
    lowered = command_text.lower()
    if lowered.startswith(("npm", "yarn", "pnpm")):
        return infer_browser_url_from_script("dev" if " dev" in lowered else "start", lowered)
    if lowered.startswith(("python", "py", "uvicorn", "flask", "streamlit")):
        return infer_browser_url_from_python_command(shlex.split(command_text, posix=not host.is_windows()))
    return None


def extract_port(text: str) -> int | None:
    patterns = (
        r"--port(?:=|\s+)(\d{2,5})",
        r"\bport(?:=|:)\s*(\d{2,5})",
        r"\bPORT=(\d{2,5})",
        r"0\.0\.0\.0:(\d{2,5})",
        r"localhost:(\d{2,5})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def write_windows_launcher(workspace: Path, spec: LaunchSpec) -> Path:
    launcher_dir = ConfigManager.workspace_cache_dir(workspace) / "launchers"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    script_path = launcher_dir / f"{safe_filename(spec.name)}.cmd"
    lines = [
        "@echo off",
        f"title DevAgent - {sanitize_console_title(spec.name)}",
        f"cd /d {quote_cmd_arg(str(spec.cwd))}",
    ]
    if spec.venv_dir:
        activate_script = spec.venv_dir / "Scripts" / "activate.bat"
        if spec.bootstrap_commands:
            lines.append(f"if not exist {quote_cmd_arg(str(activate_script))} (")
            for command in spec.bootstrap_commands:
                lines.append(f"  {format_windows_command(command)}")
            lines.append(")")
        lines.append(f"call {quote_cmd_arg(str(activate_script))}")
    lines.append(format_windows_command(spec.command))
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def format_windows_command(command: list[str] | tuple[str, ...]) -> str:
    return " ".join(quote_cmd_arg(str(part)) for part in command)


def quote_cmd_arg(value: str) -> str:
    escaped = value.replace('"', '""')
    if not value or any(char in value for char in ' \t&()[]{}^=;!\'"+,`~'):
        return f'"{escaped}"'
    return escaped


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "devagent-service"


def normalize_phrase(value: str) -> str:
    return " ".join(value.split()).casefold()


def replace_profile(profiles: dict[str, RunProfile], profile: RunProfile) -> dict[str, RunProfile]:
    normalized = profile.normalized_phrase()
    updated = {key: existing for key, existing in profiles.items() if existing.normalized_phrase() != normalized}
    updated[profile.phrase] = profile
    return updated
