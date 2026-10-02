from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from devagent.config.settings import ConfigManager
from devagent.context.scanner import iter_source_files, read_text_safely
from devagent.context.secrets import is_sensitive_path, redact_secrets
from devagent.tools.ai import AIClient


@dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    text: str
    embedding: list[float] | None = None
    headings: list[str] | None = None
    symbols: list[str] | None = None
    imports: list[str] | None = None

    def lexical_text(self) -> str:
        metadata = []
        metadata.extend(self.headings or [])
        metadata.extend(self.symbols or [])
        metadata.extend(self.imports or [])
        return " ".join([self.path, *metadata, self.text])


@dataclass(frozen=True)
class CodeIndex:
    root: Path
    records: list[CodeChunk]
    source_state: list["SourceFileState"] | None = None


@dataclass(frozen=True)
class SourceFileState:
    path: str
    size: int
    mtime_ns: int


class CodeIndexer:
    def __init__(self, root: Path, chunk_lines: int = 80, overlap: int = 12):
        self.root = root.expanduser().resolve()
        self.chunk_lines = chunk_lines
        self.overlap = overlap
        self.index_dir = ConfigManager.workspace_cache_dir(self.root)
        self.index_file = self.index_dir / "index.json"
        self.ai = AIClient.from_env()

    def build(self) -> CodeIndex:
        source_state = self.current_source_state()
        chunks: list[CodeChunk] = []
        for state in source_state:
            if is_sensitive_path(state.path):
                continue
            path = self.root / state.path
            text = read_text_safely(path)
            if not text:
                continue
            chunks.extend(self._chunk_file(path, redact_secrets(text)))

        embeddings = self.ai.embed([chunk.text for chunk in chunks])
        if embeddings and len(embeddings) == len(chunks):
            chunks = [
                CodeChunk(
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    embedding=embedding,
                    headings=chunk.headings,
                    symbols=chunk.symbols,
                    imports=chunk.imports,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]

        index = CodeIndex(root=self.root, records=chunks, source_state=source_state)
        self.save(index)
        return index

    def load_or_build(self) -> CodeIndex:
        if self.index_file.exists():
            index = self.load()
            if not self.is_current(index):
                return self.build()
            return index
        return self.build()

    def load(self) -> CodeIndex:
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        # Indexes written before secret filtering existed may still hold sensitive files.
        records = [CodeChunk(**item) for item in data.get("records", []) if not is_sensitive_path(item.get("path", ""))]
        source_state = [SourceFileState(**item) for item in data.get("source_state", [])] or None
        return CodeIndex(root=self.root, records=records, source_state=source_state)

    def save(self, index: CodeIndex) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(index.root),
            "records": [asdict(record) for record in index.records],
            "source_state": [asdict(state) for state in index.source_state or []],
        }
        self.index_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def current_source_state(self) -> list[SourceFileState]:
        states: list[SourceFileState] = []
        for path in iter_source_files(self.root):
            stat = path.stat()
            states.append(
                SourceFileState(
                    path=path.relative_to(self.root).as_posix(),
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
        return states

    def is_current(self, index: CodeIndex) -> bool:
        if not index.source_state:
            return False
        return index.source_state == self.current_source_state()

    def _chunk_file(self, path: Path, text: str) -> list[CodeChunk]:
        lines = text.splitlines()
        if not lines:
            return []
        relative = path.relative_to(self.root).as_posix()
        chunks: list[CodeChunk] = []
        step = max(1, self.chunk_lines - self.overlap)
        for start in range(0, len(lines), step):
            end = min(len(lines), start + self.chunk_lines)
            chunk_text = "\n".join(lines[start:end]).strip()
            if chunk_text:
                chunks.append(
                    CodeChunk(
                        path=relative,
                        start_line=start + 1,
                        end_line=end,
                        text=chunk_text,
                        headings=extract_headings(chunk_text),
                        symbols=extract_symbols(chunk_text),
                        imports=extract_imports(chunk_text),
                    )
                )
            if end >= len(lines):
                break
        return chunks


HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+[\w., ]+|const\s+\w+\s*=\s*require\(.+?\))", re.MULTILINE)
SYMBOL_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE),
)


def extract_headings(text: str, limit: int = 4) -> list[str]:
    return unique_limited((match.group(1).strip() for match in HEADING_RE.finditer(text)), limit=limit)


def extract_imports(text: str, limit: int = 6) -> list[str]:
    imports = []
    for match in IMPORT_RE.finditer(text):
        imports.append(" ".join(match.group(0).split()))
    return unique_limited(imports, limit=limit)


def extract_symbols(text: str, limit: int = 8) -> list[str]:
    symbols: list[str] = []
    for pattern in SYMBOL_PATTERNS:
        for match in pattern.finditer(text):
            symbols.append(match.group(1))
    return unique_limited(symbols, limit=limit)


def unique_limited(values, *, limit: int) -> list[str]:
    seen: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.append(value)
        if len(seen) >= limit:
            break
    return seen
