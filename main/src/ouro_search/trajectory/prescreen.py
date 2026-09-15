from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CANDIDATE_TYPES = (
    "over_search",
    "search_stagnation",
    "bad_next_hop",
    "evidence_use_failure",
)
_TYPE_PRIORITY = {name: index for index, name in enumerate(CANDIDATE_TYPES)}
_TRUNCATION_REASONS = {"length", "length_no_action", "max_tokens"}
_UNRELIABLE_ALIASES = {"yes", "no", "true", "false", "unknown"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(slots=True)
class PrescreenStats:
    records_seen: int = 0
    excluded_truncation: int = 0
    excluded_retriever_anomaly: int = 0
    records_without_candidate: int = 0
    classified: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "excluded_truncation": self.excluded_truncation,
            "excluded_retriever_anomaly": self.excluded_retriever_anomaly,
            "records_without_candidate": self.records_without_candidate,
            "classified": dict(self.classified),
        }


def _tokens(value: object, *, remove_stopwords: bool = False) -> list[str]:
    words = re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE)
    if remove_stopwords:
        return [word for word in words if word not in _STOPWORDS]
    return words


def normalize_text(value: object) -> str:
    return " ".join(_tokens(value))


def token_jaccard(left: object, right: object) -> float:
    left_tokens = set(_tokens(left, remove_stopwords=True))
    right_tokens = set(_tokens(right, remove_stopwords=True))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _usable_aliases(values: Iterable[object]) -> list[str]:
    aliases: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if len(normalized) < 3 or normalized in _UNRELIABLE_ALIASES:
            continue
        if normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _document_text(document: dict[str, Any]) -> str:
    return f"{document.get('title', '')} {document.get('text', '')}"


def gold_string_hits(documents: Iterable[dict[str, Any]], aliases: Iterable[object]) -> list[str]:
    evidence = f" {normalize_text(' '.join(_document_text(doc) for doc in documents))} "
    return [alias for alias in _usable_aliases(aliases) if f" {alias} " in evidence]


def _doc_id(document: dict[str, Any]) -> str:
    value = document.get("docid", document.get("id"))
    if value is not None and str(value).strip():
        return str(value)
    content = _document_text(document).encode("utf-8")
    return "content:" + hashlib.sha1(content).hexdigest()[:16]


def _turn_documents(turn: dict[str, Any]) -> list[dict[str, Any]]:
    documents = turn.get("retrieved_docs", turn.get("retrieved_documents", []))
    if not isinstance(documents, list):
        return []
    return [document for document in documents if isinstance(document, dict)]


def _query(turn: dict[str, Any]) -> str:
    return str(turn.get("query", turn.get("parsed_query", "")) or "").strip()


def _trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("trajectory", record.get("turns", []))
    return [turn for turn in value if isinstance(turn, dict)] if isinstance(value, list) else []


def _baseline_em(record: dict[str, Any]) -> int:
    value = record.get("em", record.get("reward", 0))
    try:
        return int(float(value) == 1.0)
    except (TypeError, ValueError):
        return 0


def _reference_answers(
    record: dict[str, Any], reference_index: dict[tuple[str, int], list[str]] | None
) -> list[str]:
    answers: list[object] = []
    raw = record.get("reference_answers", record.get("answer_aliases", []))
    if isinstance(raw, list):
        answers.extend(raw)
    elif raw:
        answers.append(raw)
    for key in ("answer", "reference_answer"):
        if record.get(key):
            answers.append(record[key])
    if reference_index is not None:
        try:
            lookup = (str(record.get("source", "")).lower(), int(record["orig_index"]))
        except (KeyError, TypeError, ValueError):
            lookup = ("", -1)
        answers.extend(reference_index.get(lookup, []))
    result: list[str] = []
    for answer in answers:
        text = str(answer)
        if text and text not in result:
            result.append(text)
    return result


def load_reference_index(path: Path | None) -> dict[tuple[str, int], list[str]]:
    if path is None:
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["dataset", "reference_answer", "answer_aliases", "extra_info"],
    )
    index: dict[tuple[str, int], list[str]] = {}
    for row in table.to_pylist():
        extra = row.get("extra_info") or {}
        values = list(row.get("answer_aliases") or [])
        if row.get("reference_answer") and row["reference_answer"] not in values:
            values.append(row["reference_answer"])
        key = (str(row.get("dataset", "")).lower(), int(extra.get("index", -1)))
        index[key] = [str(value) for value in values]
    return index


def iter_jsonl_records(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for source_file in files:
        with source_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                yield (
                    record,
                    {
                        "path": str(source_file.resolve()),
                        "line_number": line_number,
                    },
                )


def _turn_signal_rows(
    turns: list[dict[str, Any]], question: str, aliases: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_documents: list[dict[str, Any]] = []
    cumulative_ids: set[str] = set()
    previous_query = ""
    previous_ids: set[str] = set()
    search_count = 0
    for turn_index, turn in enumerate(turns):
        query = _query(turn)
        documents = _turn_documents(turn)
        ids = {_doc_id(document) for document in documents}
        before_documents = list(cumulative_documents)
        before_ids = set(cumulative_ids)
        before_hits = gold_string_hits(before_documents, aliases)
        current_hits = gold_string_hits(documents, aliases)
        exact = bool(previous_query and normalize_text(query) == normalize_text(previous_query))
        adjacent_jaccard = token_jaccard(query, previous_query) if previous_query else None
        title_context = " ".join(str(doc.get("title", "")) for doc in before_documents)
        row = {
            "turn_index": turn_index,
            "is_search": bool(query),
            "query": query,
            "documents": documents,
            "doc_ids": sorted(ids),
            "cumulative_doc_ids_before": sorted(before_ids),
            "prefix_search_count": search_count,
            "search_number": search_count + 1 if query else search_count,
            "query_exact_match_previous": exact,
            "query_token_jaccard_previous": adjacent_jaccard,
            "query_question_token_jaccard": token_jaccard(query, question),
            "query_prior_titles_token_jaccard": token_jaccard(query, title_context),
            "previous_doc_id_overlap": len(ids & previous_ids),
            "cumulative_doc_id_overlap": len(ids & before_ids),
            "new_document_count": len(ids - before_ids),
            "gold_answer_string_hit_before": bool(before_hits),
            "gold_answer_string_hits_before": before_hits,
            "gold_answer_string_hit_current": bool(current_hits),
            "gold_answer_string_hits_current": current_hits,
        }
        rows.append(row)
        if query:
            search_count += 1
            previous_query = query
            previous_ids = ids
            cumulative_documents.extend(documents)
            cumulative_ids.update(ids)

    first_hit_turn = next(
        (
            row["turn_index"]
            for row in rows
            if row["is_search"] and row["gold_answer_string_hit_current"]
        ),
        None,
    )
    search_indices = [row["turn_index"] for row in rows if row["is_search"]]
    for row in rows:
        later_searches = [index for index in search_indices if index > row["turn_index"]]
        next_search_turn = later_searches[0] if later_searches else None
        row["first_gold_answer_string_hit_turn"] = first_hit_turn
        row["current_search_first_brought_gold_string"] = bool(first_hit_turn == row["turn_index"])
        row["next_search_turn"] = next_search_turn
        row["next_search_first_brought_gold_string"] = bool(
            next_search_turn is not None and first_hit_turn == next_search_turn
        )
        row["later_search_first_brought_gold_string"] = bool(
            first_hit_turn is not None and first_hit_turn > row["turn_index"]
        )
    return rows


def _candidate_option(
    candidate_type: str,
    row: dict[str, Any],
    baseline_em: int,
    source: str,
) -> tuple[float, str] | None:
    if candidate_type == "over_search":
        if not row["is_search"] or not row["gold_answer_string_hit_before"]:
            return None
        confidence = 0.72
        if row["new_document_count"] == 0:
            confidence += 0.08
        if baseline_em:
            confidence += 0.03
        return min(confidence, 0.88), (
            "A prior retrieved document contains a reference-answer string, but the current "
            "action is another Search. This is a sufficient-evidence proxy, not proof."
        )

    if candidate_type == "search_stagnation":
        if not row["is_search"] or row["turn_index"] == 0:
            return None
        exact = row["query_exact_match_previous"]
        similar = (row["query_token_jaccard_previous"] or 0.0) >= 0.78
        no_new = row["new_document_count"] == 0
        if not (exact or similar or no_new):
            return None
        confidence = 0.93 if exact else 0.80 if similar else 0.70
        if no_new:
            confidence = min(confidence + 0.05, 0.96)
        return confidence, (
            "The current Search repeats or closely matches the preceding query, or retrieves no "
            "new document IDs relative to the accumulated prefix."
        )

    if candidate_type == "bad_next_hop":
        if not row["is_search"] or row["turn_index"] == 0:
            return None
        if row["gold_answer_string_hit_before"] or row["gold_answer_string_hit_current"]:
            return None
        if not row["later_search_first_brought_gold_string"]:
            return None
        context_overlap = max(
            row["query_question_token_jaccard"],
            row["query_prior_titles_token_jaccard"],
        )
        if context_overlap > 0.24:
            return None
        confidence = 0.48 + (0.06 if source == "hotpotqa" else 0.0)
        if row["next_search_first_brought_gold_string"]:
            confidence += 0.07
        return min(confidence, 0.64), (
            "The current Search has weak lexical grounding in the question/prior titles and misses "
            "the answer string; a later Search first retrieves that string. This is a "
            "low-confidence bad-next-hop proxy."
        )

    return None


def build_candidate(
    record: dict[str, Any],
    raw_reference: dict[str, Any],
    reference_index: dict[tuple[str, int], list[str]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    finish_reason = str(record.get("finish_reason", "")).lower()
    if finish_reason in _TRUNCATION_REASONS or "length_no_action" in finish_reason:
        return None, "truncation"
    turns = _trajectory(record)
    if not turns:
        return None, None
    searched_turns = [turn for turn in turns if _query(turn)]
    if any(not _turn_documents(turn) for turn in searched_turns):
        return None, "retriever_anomaly"

    source = str(record.get("source", record.get("dataset", "unknown"))).lower()
    baseline_em = _baseline_em(record)
    aliases = _reference_answers(record, reference_index)
    rows = _turn_signal_rows(turns, str(record.get("question", "")), aliases)
    options: list[tuple[int, int, str, float, str, dict[str, Any]]] = []
    for row in rows:
        for candidate_type in CANDIDATE_TYPES[:3]:
            option = _candidate_option(candidate_type, row, baseline_em, source)
            if option is not None:
                confidence, reason = option
                options.append(
                    (
                        int(row["turn_index"]),
                        _TYPE_PRIORITY[candidate_type],
                        candidate_type,
                        confidence,
                        reason,
                        row,
                    )
                )

    final_turn = next((row for row in reversed(rows) if not row["is_search"]), None)
    if baseline_em == 0 and final_turn is not None and final_turn["gold_answer_string_hit_before"]:
        options.append(
            (
                int(final_turn["turn_index"]),
                _TYPE_PRIORITY["evidence_use_failure"],
                "evidence_use_failure",
                0.74,
                "A reference-answer string appears in accumulated retrieved documents, but the "
                "baseline final answer has EM=0. This is an evidence-presence proxy, not a factual "
                "sufficiency judgment.",
                final_turn,
            )
        )
    if not options:
        return None, None

    _, _, candidate_type, confidence, reason, row = min(options, key=lambda value: value[:2])
    target_turn = int(row["turn_index"])
    signal_keys = [
        "prefix_search_count",
        "search_number",
        "query_exact_match_previous",
        "query_token_jaccard_previous",
        "query_question_token_jaccard",
        "query_prior_titles_token_jaccard",
        "previous_doc_id_overlap",
        "cumulative_doc_id_overlap",
        "new_document_count",
        "gold_answer_string_hit_before",
        "gold_answer_string_hits_before",
        "gold_answer_string_hit_current",
        "gold_answer_string_hits_current",
        "first_gold_answer_string_hit_turn",
        "current_search_first_brought_gold_string",
        "next_search_turn",
        "next_search_first_brought_gold_string",
        "later_search_first_brought_gold_string",
    ]
    candidate = {
        "sample_id": str(record.get("sample_id", record.get("id", ""))),
        "source": source,
        "orig_index": record.get("orig_index"),
        "orig_id": record.get("orig_id"),
        "baseline_em": baseline_em,
        "baseline_answer": record.get("pred_answer", record.get("prediction", "")),
        "baseline_finish_reason": record.get("finish_reason"),
        "num_searches": len(searched_turns),
        "candidate_type": candidate_type,
        "target_turn": target_turn,
        "target_action": "search" if row["is_search"] else "answer",
        "question": record.get("question", ""),
        "reference_answers": aliases,
        "current_query": row["query"],
        "prefix_history": turns[:target_turn],
        "current_turn": turns[target_turn],
        "next_turn": turns[target_turn + 1] if target_turn + 1 < len(turns) else None,
        "evidence_doc_ids": row["cumulative_doc_ids_before"],
        "rule_signals": {key: row[key] for key in signal_keys},
        "selection_reason": reason,
        "confidence": round(float(confidence), 3),
        "raw_trajectory_ref": {
            **raw_reference,
            "sample_id": str(record.get("sample_id", record.get("id", ""))),
            "source": source,
            "orig_index": record.get("orig_index"),
        },
    }
    return candidate, None


def _stable_rank(candidate: dict[str, Any], seed: int) -> tuple[float, int]:
    identity = f"{seed}:{candidate.get('sample_id')}:{candidate.get('target_turn')}"
    tie_breaker = int(hashlib.sha1(identity.encode("utf-8")).hexdigest()[:15], 16)
    return float(candidate["confidence"]), tie_breaker


def _balanced_selection(
    pools: dict[tuple[str, str, int], list[dict[str, Any]]], per_type: int, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate_type in CANDIDATE_TYPES:
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for (pool_type, source, em), values in pools.items():
            if pool_type != candidate_type:
                continue
            buckets[(source, em)] = sorted(
                values, key=lambda row: _stable_rank(row, seed), reverse=True
            )
        type_selected = 0
        keys = sorted(buckets, key=lambda key: (key[0], key[1]))
        while type_selected < per_type:
            made_progress = False
            for key in keys:
                if not buckets[key]:
                    continue
                selected.append(buckets[key].pop(0))
                type_selected += 1
                made_progress = True
                if type_selected >= per_type:
                    break
            if not made_progress:
                break
    return sorted(
        selected,
        key=lambda row: (CANDIDATE_TYPES.index(row["candidate_type"]), row["sample_id"]),
    )


def prescreen(
    records: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    reference_index: dict[tuple[str, int], list[str]] | None = None,
    per_type: int = 50,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], PrescreenStats]:
    stats = PrescreenStats()
    pool_limit = max(per_type, 1)
    heaps: dict[tuple[str, str, int], list[tuple[float, int, int, dict[str, Any]]]] = defaultdict(
        list
    )
    serial = 0
    for record, raw_reference in records:
        stats.records_seen += 1
        candidate, exclusion = build_candidate(record, raw_reference, reference_index)
        if exclusion == "truncation":
            stats.excluded_truncation += 1
            continue
        if exclusion == "retriever_anomaly":
            stats.excluded_retriever_anomaly += 1
            continue
        if candidate is None:
            stats.records_without_candidate += 1
            continue
        candidate_type = str(candidate["candidate_type"])
        stats.classified[candidate_type] += 1
        key = (candidate_type, str(candidate["source"]), int(candidate["baseline_em"]))
        confidence, tie_breaker = _stable_rank(candidate, seed)
        item = (confidence, tie_breaker, serial, candidate)
        serial += 1
        if len(heaps[key]) < pool_limit:
            heapq.heappush(heaps[key], item)
        elif item[:2] > heaps[key][0][:2]:
            heapq.heapreplace(heaps[key], item)
    pools = {key: [item[3] for item in heap] for key, heap in heaps.items()}
    return _balanced_selection(pools, per_type, seed), stats


def write_candidates(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
