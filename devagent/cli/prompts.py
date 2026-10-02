from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from devagent.cli.ui import app_table


@dataclass(frozen=True)
class MenuChoice:
    label: str
    value: str


def can_use_arrow_menu() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def choose_menu_action(console: Console, title: str, choices: list[MenuChoice]) -> str | None:
    if not choices:
        return None

    if can_use_arrow_menu():
        try:
            import questionary

            answer = questionary.select(
                title,
                choices=[
                    questionary.Choice(title=choice.label, value=choice.value)
                    for choice in choices
                ],
                use_indicator=True,
                use_shortcuts=False,
                style=questionary.Style(
                    [
                        # ANSI colour names, so the terminal theme picks readable shades for its background.
                        ("qmark", "fg:ansicyan bold"),
                        ("pointer", "fg:ansicyan bold"),
                        ("highlighted", "fg:ansicyan bold"),
                        ("answer", "fg:ansigreen bold"),
                    ]
                ),
            ).ask()
            if answer is not None:
                return str(answer)
        except Exception:
            pass

    table = app_table(title)
    table.add_column("#", justify="right")
    table.add_column("Action")
    for index, choice in enumerate(choices, start=1):
        table.add_row(str(index), choice.label)
    console.print(table)
    picked = Prompt.ask("Choose action", choices=[str(index) for index in range(1, len(choices) + 1)], default="1")
    return choices[int(picked) - 1].value


def visible_directories(path: Path, limit: int = 50) -> list[Path]:
    directories = [item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")]
    return sorted(directories, key=lambda item: item.name.lower())[:limit]


def choose_directory(console: Console, start: Path, title: str) -> Path:
    current = start.expanduser().resolve()
    while True:
        table = app_table(f"{title}: {current}")
        table.add_column("#", justify="right")
        table.add_column("Folder")
        folders = visible_directories(current)
        for index, folder in enumerate(folders, start=1):
            table.add_row(str(index), folder.name)
        console.print(table)
        console.print("[dim]Enter a number to open a folder, [bold].[/bold] to choose this folder, [bold]..[/bold] to go up, or paste/type a path.[/dim]")
        choice = Prompt.ask("Directory").strip()
        if choice == ".":
            return current
        if choice == "..":
            current = current.parent
            continue
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(folders):
                current = folders[selected - 1]
                continue
            console.print("[yellow]That folder number is not in the list.[/yellow]")
            continue
        candidate = Path(choice).expanduser()
        if not candidate.is_absolute():
            candidate = current / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        if Confirm.ask(f"Create folder {candidate}?", default=True):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
