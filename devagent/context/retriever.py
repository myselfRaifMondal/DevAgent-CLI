from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from devagent.context.indexer import CodeChunk, CodeIndex
from devagent.tools.ai import AIClient

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Retriever:
    def __init__(self, index: CodeIndex):
        self.index = index
        self.ai = AIClient.from_env()
        self.records_by_file: dict[str, list[CodeChunk]] = {}
        for record in self.index.records:
            self.records_by_file.setdefault(record.path, []).append(record)
        for records in self.records_by_file.values():
            records.sort(key=lambda item: item.start_line)

    def search(self, query: str, limit: int = 5) -> list[CodeChunk]:
        return self.search_hybrid([query], limit=limit)

    def search_hybrid(self, queries: list[str], limit: int = 5, *, intent: str | None = None) -> list[CodeChunk]:
        normalized_queries = [query.strip() for query in queries if query and query.strip()]
        if not normalized_queries:
            return []

        query_embeddings = self.ai.embed(normalized_queries)
        has_embeddings = bool(query_embeddings and any(record.embedding for record in self.index.records))
        hits_by_key: dict[tuple[str, int, int], SearchHit] = {}

        for position, query in enumerate(normalized_queries):
            weight = max(0.45, 1.0 - position * 0.18)
            vector_scores = self._vector_scores(query_embeddings[position]) if has_embeddings and query_embeddings else {}
            keyword_scores = self._keyword_scores(query)
            for record in self.index.records:
                key = (record.path, record.start_line, record.end_line)
                vector_score = vector_scores.get(key, 0.0)
                keyword_score = keyword_scores.get(key, 0.0)
                metadata_boost = metadata_match_boost(record, query)
                combined = (vector_score * 0.68) + (keyword_score * 0.32) + metadata_boost
                if combined <= 0:
                    continue
                previous = hits_by_key.get(key)
                scaled = combined * weight
                if previous is None or scaled > previous.score:
                    hits_by_key[key] = SearchHit(record=record, score=scaled)

        ranked = sorted(hits_by_key.values(), key=lambda item: item.score, reverse=True)
        if intent in {"count", "list", "enumerate"}:
            return self._enumeration_slice(ranked, limit)
        return self._diverse_slice(ranked, limit)

    def _vector_scores(self, query_embedding: list[float]) -> dict[tuple[str, int, int], float]:
        scored: dict[tuple[str, int, int], float] = {}
        for record in self.index.records:
            if not record.embedding:
                continue
            score = cosine_similarity(query_embedding, record.embedding)
            if score > 0:
                scored[(record.path, record.start_line, record.end_line)] = score
        return scored

    def _keyword_scores(self, query: str) -> dict[tuple[str, int, int], float]:
        query_tokens = tokenize(query)
        scored: dict[tuple[str, int, int], float] = {}
        for record in self.index.records:
            text_tokens = tokenize(record.lexical_text())
            overlap = sum(
                min(text_tokens[token], weight) if token in text_tokens else prefix_match_credit(token, text_tokens)
                for token, weight in query_tokens.items()
            )
            score = overlap / max(len(query_tokens), 1)
            if score:
                scored[(record.path, record.start_line, record.end_line)] = score
        return scored

    def _diverse_slice(self, ranked_hits: list["SearchHit"], limit: int) -> list[CodeChunk]:
        selected: list[CodeChunk] = []
        deferred: list[CodeChunk] = []
        seen_files: set[str] = set()

        for hit in ranked_hits:
            if len(selected) >= limit:
                break
            if hit.record.path in seen_files:
                deferred.append(hit.record)
                continue
            selected.append(hit.record)
            seen_files.add(hit.record.path)

        if len(selected) < limit:
            for record in deferred:
                if len(selected) >= limit:
                    break
                selected.append(record)
        return selected

    def _enumeration_slice(self, ranked_hits: list["SearchHit"], limit: int) -> list[CodeChunk]:
        selected: list[CodeChunk] = []
        seen: set[tuple[str, int, int]] = set()

        for hit in ranked_hits:
            if len(selected) >= max(3, limit // 2):
                break
            key = (hit.record.path, hit.record.start_line, hit.record.end_line)
            if key in seen:
                continue
            seen.add(key)
            selected.append(hit.record)

        for seed in list(selected):
            if len(selected) >= limit:
                break
            for neighbor in self._adjacent_records(seed):
                key = (neighbor.path, neighbor.start_line, neighbor.end_line)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(neighbor)
                if len(selected) >= limit:
                    break

        if len(selected) < limit:
            for hit in ranked_hits:
                key = (hit.record.path, hit.record.start_line, hit.record.end_line)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(hit.record)
                if len(selected) >= limit:
                    break
        return selected

    def _adjacent_records(self, record: CodeChunk) -> list[CodeChunk]:
        records = self.records_by_file.get(record.path, [])
        if not records:
            return []
        try:
            index = records.index(record)
        except ValueError:
            return []

        neighbors: list[CodeChunk] = []
        if index > 0:
            neighbors.append(records[index - 1])
        if index + 1 < len(records):
            neighbors.append(records[index + 1])
        return neighbors


@dataclass(frozen=True)
class SearchHit:
    record: CodeChunk
    score: float


PREFIX_MATCH_MIN_LENGTH = 4
PREFIX_MATCH_CREDIT = 0.5


def prefix_match_credit(query_token: str, text_tokens: Counter[str]) -> float:
    """Half credit when a query word and a code word share a prefix ("authentication" ~ "auth").

    Only words of at least four letters take part, so "is" or "id" never match anything, and
    an exact match always scores higher than a prefix match.
    """
    if len(query_token) < PREFIX_MATCH_MIN_LENGTH:
        return 0.0
    for token in text_tokens:
        if len(token) < PREFIX_MATCH_MIN_LENGTH:
            continue
        if token.startswith(query_token) or query_token.startswith(token):
            return PREFIX_MATCH_CREDIT
    return 0.0


def tokenize(text: str) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for raw in TOKEN_RE.findall(text):
        for token in expand_token(raw):
            tokens[token] += 1
    return tokens


def expand_token(raw: str) -> set[str]:
    lowered = raw.lower()
    pieces = {lowered}
    pieces.update(part for part in lowered.split("_") if part)
    for piece in list(pieces):
        if piece.endswith("tion") and len(piece) > 5:
            pieces.add(f"{piece[:-3]}e")
        if piece.endswith("ing") and len(piece) > 5:
            pieces.add(piece[:-3])
        if piece.endswith("ed") and len(piece) > 4:
            pieces.add(piece[:-2])
    return pieces


def metadata_match_boost(record: CodeChunk, query: str) -> float:
    lowered_query = query.casefold()
    boost = 0.0
    for value in (record.symbols or []):
        if value.casefold() in lowered_query:
            boost += 0.18
    for value in (record.headings or []):
        if value.casefold() in lowered_query:
            boost += 0.12
    for value in (record.imports or []):
        if value.casefold() in lowered_query:
            boost += 0.08
    if record.path.casefold() in lowered_query:
        boost += 0.22
    return boost


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
