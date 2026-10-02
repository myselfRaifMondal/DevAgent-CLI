from pathlib import Path

from devagent.context.indexer import CodeIndexer
from devagent.context.retriever import Retriever


def test_indexer_ignores_dependency_folders(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def login():\n    return True\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("def login_secret(): pass\n", encoding="utf-8")

    index = CodeIndexer(tmp_path, chunk_lines=10).build()
    paths = {record.path for record in index.records}

    assert "app.py" in paths
    assert "node_modules/ignored.py" not in paths


def test_retriever_finds_matching_chunk(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def login_user():\n    validate_password()\n", encoding="utf-8")
    index = CodeIndexer(tmp_path, chunk_lines=10).build()

    results = Retriever(index).search("where is login validation", limit=1)

    assert results
    assert results[0].path == "auth.py"


def test_load_or_build_rebuilds_when_source_files_change(tmp_path: Path, monkeypatch) -> None:
    config_home = tmp_path / "config-home"
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(config_home))

    readme = tmp_path / "README.md"
    readme.write_text("Old ending\nAritrajit Guha\n", encoding="utf-8")
    indexer = CodeIndexer(tmp_path, chunk_lines=10)
    original = indexer.build()

    assert any("Aritrajit Guha" in record.text for record in original.records)

    readme.write_text("New ending\nagentic and personal.\n", encoding="utf-8")

    refreshed = indexer.load_or_build()

    assert any("agentic and personal." in record.text for record in refreshed.records)
    assert all("Aritrajit Guha" not in record.text for record in refreshed.records)


def test_retriever_matches_a_natural_word_to_an_abbreviated_file_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    workspace = tmp_path / "workspace"
    (workspace / "backend").mkdir(parents=True)
    (workspace / "backend" / "auth.py").write_text("def login_user(user):\n    return create_session(user)\n", encoding="utf-8")
    (workspace / "backend" / "billing.py").write_text("def charge(card):\n    return card\n", encoding="utf-8")

    index = CodeIndexer(workspace, chunk_lines=10).build()
    results = Retriever(index).search("Where is authentication implemented?", limit=1)

    assert [chunk.path for chunk in results] == ["backend/auth.py"]


def test_short_words_never_get_a_prefix_match() -> None:
    from collections import Counter

    from devagent.context.retriever import prefix_match_credit

    assert prefix_match_credit("is", Counter({"issue": 1})) == 0.0
    assert prefix_match_credit("id", Counter({"identity": 1})) == 0.0
    assert prefix_match_credit("auth", Counter({"login": 1})) == 0.0
    assert prefix_match_credit("authentication", Counter({"auth": 1})) == 0.5


def test_exact_match_outranks_a_prefix_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / "config-home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "exact.py").write_text("def payment():\n    return 1\n", encoding="utf-8")
    (workspace / "prefix.py").write_text("def payments_summary():\n    return 2\n", encoding="utf-8")

    index = CodeIndexer(workspace, chunk_lines=10).build()
    results = Retriever(index).search("payment", limit=2)

    assert results[0].path == "exact.py"


def test_the_config_directory_is_never_indexed_even_inside_the_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_CONFIG_DIR", str(tmp_path / ".agent-state"))
    (tmp_path / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    indexer = CodeIndexer(tmp_path, chunk_lines=10)
    indexer.build()

    rebuilt = indexer.build()

    assert {record.path for record in rebuilt.records} == {"app.py"}
