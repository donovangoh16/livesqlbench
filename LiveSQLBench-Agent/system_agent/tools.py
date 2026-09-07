"""ADK tools for single-turn text-to-SQL mode.

Tools allow the agent to interact with the DB Environment (port 6002):
  - Execute SQL, get schema, get column meanings, get knowledge
  - Submit final SQL for evaluation
"""

import json
import logging
import httpx
import re
import hashlib
from collections import deque
from difflib import SequenceMatcher
from typing import Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from shared.config import settings

logger = logging.getLogger(__name__)

_current_task_id: str = ""

MAX_RANKED_TABLES = 5
MAX_RANKED_COLUMNS_PER_TABLE = 8
MAX_RANKED_KNOWLEDGE = 5
MAX_JOIN_PATHS = 10
MAX_SELECTED_TABLES = 8
MAX_MEANING_CHARS = 180
MAX_MEANING_HINT_CHARS = 96
MAX_RESULT_SAMPLE_ROWS = 3


def set_current_task_id(task_id: str):
    global _current_task_id
    _current_task_id = task_id


def _get_task_id(tool_context: Optional[ToolContext] = None) -> str:
    if tool_context:
        tid = tool_context.state.get("task_id", "")
        if tid:
            return tid
    return _current_task_id


def _db_url(path: str) -> str:
    return f"http://localhost:{settings.db_env_port}{path}"


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1
    }


def _similarity(left: str, right: str) -> float:
    left_text, right_text = str(left).lower().strip(), str(right).lower().strip()
    if not left_text or not right_text:
        return 0.0
    left_tokens, right_tokens = _tokens(left_text), _tokens(right_text)
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens))
    substring = 1.0 if left_text in right_text or right_text in left_text else 0.0
    fuzzy = SequenceMatcher(None, left_text, right_text).ratio()
    return min(1.0, 0.55 * overlap + 0.30 * substring + 0.15 * fuzzy)


def _post_json(path: str, payload: dict, timeout: float = 30.0) -> dict:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.post(_db_url(path), json=payload)
        response.raise_for_status()
        return response.json()


def _column_metadata(task_id: str) -> dict:
    raw = _post_json("/all_column_meanings", {"task_id": task_id}).get(
        "column_meanings", "{}"
    )
    return json.loads(raw) if isinstance(raw, str) else raw


def _knowledge_names(task_id: str) -> list[str]:
    return _post_json("/knowledge_names", {"task_id": task_id}).get("names", [])


def _schema_text(task_id: str) -> str:
    return _post_json("/schema", {"task_id": task_id}).get("schema", "")


def _parse_schema(schema: str) -> dict[str, dict]:
    """Parse enough CREATE TABLE structure for compact retrieval tools."""
    tables: dict[str, dict] = {}
    pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:"?[\w]+"?\.)?'
        r'"?([\w]+)"?\s*\((.*?)\)\s*;',
        re.I | re.S,
    )
    for match in pattern.finditer(schema):
        table = match.group(1).lower()
        columns, primary_keys, foreign_keys = [], [], []
        body = match.group(2)
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            foreign_key = re.search(
                r'FOREIGN\s+KEY\s*\(\s*"?([\w]+)"?\s*\)\s*REFERENCES\s*'
                r'(?:"?[\w]+"?\.)?"?([\w]+)"?\s*\(\s*"?([\w]+)"?\s*\)',
                line, re.I,
            )
            if foreign_key:
                column, target_table, target_column = foreign_key.groups()
                foreign_keys.append({
                    "column": column.lower(),
                    "references": f"{target_table.lower()}.{target_column.lower()}",
                })
                continue
            primary_key = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', line, re.I)
            if primary_key:
                primary_keys.extend(
                    value.strip().strip('"').lower()
                    for value in primary_key.group(1).split(",")
                )
                continue
            if re.match(r'(?:CONSTRAINT|UNIQUE|CHECK|EXCLUDE)\b', line, re.I):
                continue
            column_match = re.match(r'"?([\w]+)"?\s+(.+)', line, re.S)
            if column_match:
                column, definition = column_match.groups()
                columns.append({"name": column.lower(), "definition": definition.strip()})
                if re.search(r'\bPRIMARY\s+KEY\b', definition, re.I):
                    primary_keys.append(column.lower())
        tables[table] = {
            "columns": columns,
            "primary_keys": list(dict.fromkeys(primary_keys)),
            "foreign_keys": foreign_keys,
        }
    return tables


def _parse_knowledge(raw, fallback_name: str = "") -> dict:
    """Normalize the DB environment's sometimes double-encoded KB payload."""
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = {"definition": value}
    if not isinstance(value, dict):
        value = {"definition": str(value)}
    return {
        "id": value.get("id"),
        "name": value.get("knowledge", value.get("name", fallback_name)),
        "definition": str(value.get("definition", value.get("description", ""))).strip(),
    }


def _knowledge_aliases(name: str) -> set[str]:
    aliases = {str(name).strip().lower()}
    aliases.update(value.lower() for value in re.findall(r"\(([A-Z][A-Z0-9]{1,})\)", str(name)))
    words = re.findall(r"[A-Za-z]+", re.sub(r"\([^)]*\)", "", str(name)))
    if len(words) > 1:
        aliases.add("".join(word[0] for word in words).lower())
    return {alias for alias in aliases if alias}


def _rounding_places(value) -> Optional[int]:
    """Accept integer or compact LLM forms such as '2_decimal_places'."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _structured_table_result(raw: str, sample_limit: int = MAX_RESULT_SAMPLE_ROWS) -> dict:
    """Convert the DB environment's pipe table into a compact JSON summary."""
    text = str(raw).strip()
    if not text:
        return {"row_count": 0, "columns": [], "sample_rows": [], "empty": True}
    if text.lower().startswith(("sql error:", "error calling")):
        return {"success": False, "error": text[:500]}
    if "empty result set" in text.lower():
        return {"row_count": 0, "columns": [], "sample_rows": [], "empty": True}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    separator = len(lines) >= 2 and set(lines[1]) <= {"-", "+", " ", "|"}
    if len(lines) < 2 or ("|" not in lines[0] and not separator):
        return {"row_count": None, "columns": [], "sample_rows": [], "message": text[:500]}
    columns = [part.strip() for part in lines[0].split("|")]
    data_lines = lines[2:] if separator else lines[1:]
    rows = []
    for line in data_lines:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == len(columns):
            rows.append(dict(zip(columns, parts)))
    numeric_ranges = {}
    for column in columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                pass
        if values and len(values) == len(rows):
            numeric_ranges[column] = [min(values), max(values)]
    return {
        "row_count": len(rows),
        "columns": columns,
        "sample_rows": rows[:sample_limit],
        "empty": not rows,
        "truncated_sample": len(rows) > sample_limit,
        "numeric_ranges": numeric_ranges,
    }


# ── DB Environment Tools ──

def execute_sql(sql: str, tool_context: ToolContext) -> str:
    """Execute a SQL query against the PostgreSQL database and return the results.
    Use this to explore the database, test queries, or verify your SQL before submitting.

    Args:
        sql: The PostgreSQL SQL query to execute.

    Returns:
        The query results formatted as a table, or an error message.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            if resp.status_code != 200:
                return f"SQL Error: Server returned status {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            if data.get("success"):
                return data.get("result", "Query executed successfully.")
            else:
                return f"SQL Error: {data.get('error') or 'Execution failed (no details)'}"
    except Exception as e:
        return f"Error calling DB environment: {type(e).__name__}: {e}"


def get_schema(tool_context: ToolContext) -> str:
    """Get the full database schema (CREATE TABLE statements) for the current task's database.

    Returns:
        The database schema as text.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/schema"),
                               json={"task_id": task_id})
            return resp.json().get("schema", "Schema not available")
    except Exception as e:
        return f"Error: {e}"


def get_schema_summary(tool_context: ToolContext) -> str:
    """Return compact table, key, and relationship metadata."""
    task_id = _get_task_id(tool_context)
    try:
        tables = _parse_schema(_schema_text(task_id))
        summary = []
        for table, details in sorted(tables.items()):
            summary.append({
                "table": table,
                "column_count": len(details["columns"]),
                "primary_keys": details["primary_keys"],
                "foreign_keys": details["foreign_keys"],
            })
        return json.dumps({"table_count": len(summary), "tables": summary})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def get_selected_schema(
    table_names: list[str], tool_context: ToolContext
) -> str:
    """Return columns and meanings only for selected tables.

    Args:
        table_names: Ranked table names to inspect, normally no more than five.
    """
    task_id = _get_task_id(tool_context)
    try:
        requested = list(dict.fromkeys(
            str(name).strip().lower() for name in table_names if str(name).strip()
        ))[:MAX_SELECTED_TABLES]
        tables = _parse_schema(_schema_text(task_id))
        metadata = _column_metadata(task_id)
        meanings = {}
        for key, value in metadata.items():
            parts = str(key).split("|")
            if len(parts) >= 3:
                meanings[(parts[-2].lower(), parts[-1].lower())] = str(value)

        selected = []
        compact_selected = []
        missing = []
        for table in requested:
            details = tables.get(table)
            if not details:
                missing.append(table)
                continue
            columns = []
            for column in details["columns"]:
                meaning = meanings.get((table, column["name"]), "")
                columns.append({
                    **column,
                    "meaning": meaning[:MAX_MEANING_CHARS],
                })
            selected.append({
                "table": table,
                "columns": columns,
                "primary_keys": details["primary_keys"],
                "foreign_keys": details["foreign_keys"],
            })
            compact_selected.append({
                "table": table,
                "columns": [column["name"] for column in columns],
                "primary_keys": details["primary_keys"],
                "foreign_keys": details["foreign_keys"],
            })
        tool_context.state["selected_schema_details"] = {
            "tables": selected,
            "missing_tables": missing,
        }
        return json.dumps({
            "stored_as": "selected_schema_details",
            "tables": compact_selected,
            "missing_tables": missing,
            "truncated_request": len(requested) < len(set(table_names)),
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def get_all_column_meanings(tool_context: ToolContext) -> str:
    """Get the meanings/descriptions of all columns in the database.

    Returns:
        JSON string with column meanings for all tables.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/all_column_meanings"),
                               json={"task_id": task_id})
            return resp.json().get("column_meanings", "{}")
    except Exception as e:
        return f"Error: {e}"


def get_column_meaning(table_name: str, column_name: str, tool_context: ToolContext) -> str:
    """Get the meaning/description of a specific column in a table.

    Args:
        table_name: Name of the table.
        column_name: Name of the column.

    Returns:
        The column meaning/description.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/column_meaning"),
                               json={"task_id": task_id,
                                     "table_name": table_name,
                                     "column_name": column_name})
            return resp.json().get("meaning", "Column meaning not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_external_knowledge_names(tool_context: ToolContext) -> str:
    """Get the names of all available external knowledge entries for this database.
    Use this to discover what domain knowledge is available.

    Returns:
        JSON list of knowledge entry names.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge_names"),
                               json={"task_id": task_id})
            return json.dumps(resp.json().get("names", []))
    except Exception as e:
        return f"Error: {e}"


def get_knowledge_definition(knowledge_name: str, tool_context: ToolContext) -> str:
    """Get the definition/details of a specific external knowledge entry.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id,
                                     "knowledge_name": knowledge_name})
            return resp.json().get("knowledge", "Knowledge not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_knowledge_definitions(tool_context: ToolContext) -> str:
    """Get all external knowledge definitions for this database.

    Returns:
        JSON string with all knowledge definitions.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id})
            return resp.json().get("knowledge", "[]")
    except Exception as e:
        return f"Error: {e}"


# ── Improved Pre-processing Tools (variants 1 and 3) ──

def rank_relevant_tables(
    question: str, top_k: int, tool_context: ToolContext
) -> str:
    """Rank database tables relevant to the natural-language task.

    Ranking uses table names, column names, and column descriptions. Use this
    after get_schema_summary and before selecting columns.

    Args:
        question: The complete user request.
        top_k: Requested result count; the tool applies a small hard cap.

    Returns:
        JSON with ranked tables, scores, matched columns, and ranking evidence.
    """
    task_id = _get_task_id(tool_context)
    try:
        metadata = _column_metadata(task_id)
        tables: dict[str, list[tuple[str, str]]] = {}
        for key, meaning in metadata.items():
            parts = str(key).split("|")
            if len(parts) >= 3:
                tables.setdefault(parts[-2], []).append((parts[-1], str(meaning)))

        ranked = []
        for table, columns in tables.items():
            table_score = _similarity(question, table)
            column_scores = [
                (_similarity(question, f"{column} {meaning}"), column)
                for column, meaning in columns
            ]
            column_scores.sort(reverse=True)
            best_columns = [name for score, name in column_scores[:5] if score > 0]
            score = min(1.0, 0.35 * table_score + 0.65 * sum(
                value for value, _ in column_scores[:3]
            ) / max(1, min(3, len(column_scores))))
            ranked.append({
                "table": table,
                "score": round(score, 4),
                "matched_columns": best_columns,
                "reason": "Ranked from table name, column names, and column descriptions",
            })
        ranked.sort(key=lambda item: (-item["score"], item["table"]))
        limit = max(1, min(top_k, MAX_RANKED_TABLES))
        selected = ranked[:limit]
        tool_context.state["ranked_table_candidates"] = selected
        return json.dumps({
            "stored_as": "ranked_table_candidates",
            "ranked_tables": [{
                "table": item["table"],
                "score": item["score"],
                "matched_columns": item["matched_columns"],
            } for item in selected],
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def rank_relevant_columns(
    question: str,
    selected_tables: list[str],
    top_k_per_table: int,
    tool_context: ToolContext,
) -> str:
    """Rank relevant columns within selected tables.

    The result includes likely semantic roles. Always retain primary/foreign
    keys needed to connect selected tables even when their lexical score is low.

    Args:
        question: The complete user request.
        selected_tables: Tables retained from table ranking.
        top_k_per_table: Maximum semantic matches returned per table.

    Returns:
        JSON containing ranked columns with scores, meanings, and likely roles.
    """
    task_id = _get_task_id(tool_context)
    try:
        metadata = _column_metadata(task_id)
        wanted = {table.lower() for table in selected_tables}
        results = []
        for key, meaning in metadata.items():
            parts = str(key).split("|")
            if len(parts) < 3 or parts[-2].lower() not in wanted:
                continue
            table, column = parts[-2], parts[-1]
            score = _similarity(question, f"{column} {meaning}")
            meaning_lower = str(meaning).lower()
            roles = []
            if "primary key" in meaning_lower:
                roles.append("entity_key")
            if "foreign key" in meaning_lower or "fk " in meaning_lower:
                roles.append("join")
            question_lower = question.lower()
            if any(word in question_lower for word in ("sort", "order", "highest", "lowest")):
                if score >= 0.25:
                    roles.append("ordering")
            if any(word in question_lower for word in ("where", "with", "above", "below", "exceed")):
                if score >= 0.25:
                    roles.append("filter")
            if score >= 0.25:
                roles.append("candidate_output")
            results.append({
                "table": table, "column": column, "score": round(score, 4),
                "roles": sorted(set(roles)),
                "meaning": str(meaning)[:MAX_MEANING_CHARS],
            })

        chosen = []
        limit = max(1, min(top_k_per_table, MAX_RANKED_COLUMNS_PER_TABLE))
        for table in selected_tables:
            matches = [item for item in results if item["table"].lower() == table.lower()]
            matches.sort(key=lambda item: (-item["score"], item["column"]))
            semantic = matches[:limit]
            keys = [item for item in matches if set(item["roles"]) & {"entity_key", "join"}]
            seen = set()
            for item in semantic + keys:
                identity = (item["table"].lower(), item["column"].lower())
                if identity not in seen:
                    chosen.append(item)
                    seen.add(identity)
        tool_context.state["ranked_column_candidates"] = chosen
        compact = [{
            "table": item["table"],
            "column": item["column"],
            "score": item["score"],
            "roles": item["roles"],
            "meaning_hint": item["meaning"][:MAX_MEANING_HINT_CHARS],
        } for item in chosen]
        return json.dumps({
            "stored_as": "ranked_column_candidates",
            "ranked_columns": compact,
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def find_join_paths(selected_tables: list[str], tool_context: ToolContext) -> str:
    """Find shortest foreign-key paths connecting selected tables.

    Args:
        selected_tables: Physical tables that the planned SQL may require.

    Returns:
        JSON with candidate paths, qualified join edges, bridge tables, and any
        disconnected table pairs. Join type selection remains a planning task.
    """
    task_id = _get_task_id(tool_context)
    try:
        schema = _post_json("/schema", {"task_id": task_id}).get("schema", "")
        edges = []
        current_table = None
        for line in schema.splitlines():
            table_match = re.search(r'CREATE\s+TABLE\s+"?([\w]+)"?', line, re.I)
            if table_match:
                current_table = table_match.group(1).lower()
            foreign_key = re.search(
                r'FOREIGN\s+KEY\s*\(\s*"?([\w]+)"?\s*\)\s*REFERENCES\s*'
                r'"?([\w]+)"?\s*\(\s*"?([\w]+)"?\s*\)', line, re.I,
            )
            if current_table and foreign_key:
                source_column, target_table, target_column = foreign_key.groups()
                edges.append({
                    "left": f"{current_table}.{source_column.lower()}",
                    "right": f"{target_table.lower()}.{target_column.lower()}",
                    "left_table": current_table,
                    "right_table": target_table.lower(),
                })

        graph: dict[str, list[tuple[str, dict]]] = {}
        for edge in edges:
            graph.setdefault(edge["left_table"], []).append((edge["right_table"], edge))
            graph.setdefault(edge["right_table"], []).append((edge["left_table"], edge))

        requested = [table.lower() for table in selected_tables]
        paths, disconnected = [], []
        for index, start in enumerate(requested):
            for end in requested[index + 1:]:
                queue = deque([(start, [], {start})])
                found = None
                while queue and found is None:
                    node, path, seen = queue.popleft()
                    if node == end:
                        found = path
                        break
                    for neighbour, edge in graph.get(node, []):
                        if neighbour not in seen:
                            queue.append((neighbour, path + [edge], seen | {neighbour}))
                if found is None:
                    disconnected.append([start, end])
                else:
                    path_tables = {
                        table for edge in found
                        for table in (edge["left_table"], edge["right_table"])
                    }
                    paths.append({
                        "from": start, "to": end,
                        "edges": [{"left": e["left"], "right": e["right"]} for e in found],
                        "bridge_tables": sorted(path_tables - set(requested)),
                        "path_length": len(found),
                    })
        tool_context.state["join_path_candidates"] = {
            "candidate_paths": paths,
            "disconnected_pairs": disconnected,
        }
        return json.dumps({
            "stored_as": "join_path_candidates",
            "candidate_paths": paths[:MAX_JOIN_PATHS],
            "disconnected_pairs": disconnected,
            "paths_truncated": len(paths) > MAX_JOIN_PATHS,
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def identify_knowledge_requirements(
    question: str,
    candidate_phrases: list[str],
    selected_columns: list[str],
    tool_context: ToolContext,
) -> str:
    """Check which natural-language concepts require external knowledge.

    Args:
        question: The complete user request.
        candidate_phrases: Acronyms, named metrics, classifications, formulas,
            or business terms extracted from the question by the agent.
        selected_columns: Qualified selected columns, such as table.column.

    Returns:
        JSON separating phrases covered by schema names from phrases that should
        be checked against the knowledge base, with candidate KB-name matches.
    """
    task_id = _get_task_id(tool_context)
    try:
        phrases = list(dict.fromkeys(
            [phrase.strip() for phrase in candidate_phrases if phrase.strip()]
            + re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)
        ))
        sql_native = {
            "median", "average", "avg", "count", "sum", "minimum", "maximum",
            "postgres", "postgresql", "group by", "order by", "join", "cte",
            "subquery", "distinct", "limit", "percentile",
        }
        phrases = [
            phrase for phrase in phrases
            if not any(term in phrase.lower() for term in sql_native)
        ]
        names = _knowledge_names(task_id)
        schema_text = " ".join(selected_columns)
        requires_kb, covered_by_schema = [], []
        for phrase in phrases:
            schema_score = _similarity(phrase, schema_text)
            matches = sorted(
                ({"knowledge_name": name, "score": round(_similarity(phrase, name), 4)}
                 for name in names),
                key=lambda item: (-item["score"], item["knowledge_name"]),
            )[:3]
            record = {"phrase": phrase, "schema_score": round(schema_score, 4),
                      "knowledge_candidates": matches}
            if schema_score >= 0.75 and (not matches or matches[0]["score"] < schema_score):
                covered_by_schema.append(record)
            else:
                requires_kb.append(record)
        result = {
            "requires_kb_check": requires_kb,
            "covered_by_schema": covered_by_schema,
        }
        tool_context.state["knowledge_requirements"] = result
        return json.dumps({"stored_as": "knowledge_requirements", **result})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def rank_relevant_knowledge(
    phrases: list[str], top_k: int, tool_context: ToolContext
) -> str:
    """Rank available knowledge names against required natural-language phrases.

    Args:
        phrases: Domain phrases that require a KB check.
        top_k: Maximum number of KB names to return.

    Returns:
        JSON containing ranked exact/fuzzy name matches and their covered phrases.
    """
    task_id = _get_task_id(tool_context)
    try:
        ranked = []
        for name in _knowledge_names(task_id):
            phrase_scores = [(_similarity(phrase, name), phrase) for phrase in phrases]
            phrase_scores.sort(reverse=True)
            best_score, best_phrase = phrase_scores[0] if phrase_scores else (0.0, "")
            ranked.append({
                "knowledge_name": name,
                "score": round(best_score, 4),
                "matched_phrase": best_phrase,
            })
        ranked.sort(key=lambda item: (-item["score"], item["knowledge_name"]))
        limit = max(1, min(top_k, MAX_RANKED_KNOWLEDGE))
        selected = ranked[:limit]
        tool_context.state["ranked_knowledge_candidates"] = selected
        return json.dumps({
            "stored_as": "ranked_knowledge_candidates",
            "ranked_knowledge": selected,
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def check_kb_completeness(
    required_phrases: list[str],
    retrieved_knowledge_names: list[str],
    tool_context: ToolContext,
) -> str:
    """Check whether retrieved KB names cover every identified knowledge phrase.

    This checks runtime coverage, not hidden ground-truth correctness.

    Args:
        required_phrases: Phrases previously identified as requiring KB lookup.
        retrieved_knowledge_names: KB definitions already retrieved by name.

    Returns:
        JSON with covered and missing phrases plus suggested KB names.
    """
    task_id = _get_task_id(tool_context)
    try:
        available_names = _knowledge_names(task_id)
        covered, missing, suggestions = [], [], []
        for phrase in required_phrases:
            retrieved_matches = [
                (name, _similarity(phrase, name)) for name in retrieved_knowledge_names
            ]
            best_retrieved = max(retrieved_matches, key=lambda item: item[1], default=("", 0.0))
            if best_retrieved[1] >= 0.55:
                covered.append({"phrase": phrase, "knowledge_name": best_retrieved[0]})
            else:
                missing.append(phrase)
                candidates = sorted(
                    ((name, _similarity(phrase, name)) for name in available_names),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
                suggestions.append({
                    "phrase": phrase,
                    "candidates": [
                        {"knowledge_name": name, "score": round(score, 4)}
                        for name, score in candidates
                    ],
                })
        result = {
            "complete": not missing,
            "covered_phrases": covered,
            "missing_phrases": missing,
            "suggested_knowledge": suggestions,
        }
        tool_context.state["kb_completeness"] = result
        return json.dumps({"stored_as": "kb_completeness", **result})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def prepare_schema_context(
    question: str,
    candidate_table_count: int,
    columns_per_table: int,
    tool_context: ToolContext,
) -> str:
    """Rank tables/columns and find joins in one preprocessing call."""
    tool_context.state["task_question"] = question
    table_result = json.loads(rank_relevant_tables(
        question, candidate_table_count, tool_context,
    ))
    table_names = [
        item["table"] for item in table_result.get("ranked_tables", [])
    ]
    schema_result = json.loads(get_selected_schema(table_names, tool_context))
    column_result = json.loads(rank_relevant_columns(
        question, table_names, columns_per_table, tool_context,
    ))
    join_result = (
        json.loads(find_join_paths(table_names, tool_context))
        if len(table_names) > 1 else {"candidate_paths": [], "disconnected_pairs": []}
    )
    result = {
        "tables": table_result.get("ranked_tables", []),
        "selected_schema": schema_result.get("tables", []),
        "columns": column_result.get("ranked_columns", []),
        "join_paths": join_result.get("candidate_paths", []),
        "disconnected_pairs": join_result.get("disconnected_pairs", []),
    }
    tool_context.state["prepared_schema_context"] = result
    paths = sorted(
        result["join_paths"],
        key=lambda item: (item.get("path_length", 999), item.get("from", ""), item.get("to", "")),
    )[:5]
    return json.dumps({
        "tables": result["tables"],
        "schema": result["selected_schema"],
        "ranked_columns": result["columns"],
        "recommended_join_paths": [
            {"id": f"jp_{index + 1}", "edges": path.get("edges", []),
             "bridge_tables": path.get("bridge_tables", [])}
            for index, path in enumerate(paths)
        ],
        "disconnected_pairs": result["disconnected_pairs"],
        "details_stored": True,
    }, separators=(",", ":"))


def prepare_knowledge_context(
    question: str,
    candidate_phrases: list[str],
    selected_columns: list[str],
    tool_context: ToolContext,
) -> str:
    """Identify, rank, retrieve, and check relevant KB entries in one call."""
    requirements = json.loads(identify_knowledge_requirements(
        question, candidate_phrases, selected_columns, tool_context,
    ))
    required = requirements.get("requires_kb_check", [])
    selected_names = []
    for item in required:
        candidates = item.get("knowledge_candidates", [])
        if candidates and candidates[0].get("score", 0) >= 0.35:
            selected_names.append(candidates[0]["knowledge_name"])
    selected_names = list(dict.fromkeys(selected_names))[:MAX_RANKED_KNOWLEDGE]

    definitions = []
    task_id = _get_task_id(tool_context)
    available_names = _knowledge_names(task_id)
    alias_to_name = {
        alias: name for name in available_names for alias in _knowledge_aliases(name)
    }
    pending = list(selected_names)
    seen = set()
    while pending and len(definitions) < MAX_RANKED_KNOWLEDGE:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        raw = _post_json(
            "/knowledge", {"task_id": task_id, "knowledge_name": name}
        ).get("knowledge", "Knowledge not found")
        item = _parse_knowledge(raw, name)
        definition_lower = item["definition"].lower()
        dependencies = []
        for alias, dependency_name in alias_to_name.items():
            if dependency_name != name and re.search(rf"\b{re.escape(alias)}\b", definition_lower):
                dependencies.append(dependency_name)
        dependencies = list(dict.fromkeys(dependencies))
        item["depends_on"] = dependencies
        definitions.append(item)
        pending.extend(dependency for dependency in dependencies if dependency not in seen)
    selected_names = [item["name"] for item in definitions]
    formula_tokens = {
        token.lower() for item in definitions
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", item.get("definition", ""))
    }
    required_schema = {}
    for key in _column_metadata(task_id):
        parts = str(key).split("|")
        if len(parts) >= 3 and parts[-1].lower() in formula_tokens:
            required_schema.setdefault(parts[-2].lower(), []).append(parts[-1].lower())
    required_schema = {
        table: sorted(set(columns)) for table, columns in sorted(required_schema.items())
    }
    selected_column_set = {str(column).lower() for column in selected_columns}
    missing_schema_columns = [
        f"{table}.{column}" for table, columns in required_schema.items() for column in columns
        if f"{table}.{column}" not in selected_column_set
    ]
    completeness = json.loads(check_kb_completeness(
        [item.get("phrase", "") for item in required], selected_names, tool_context,
    ))
    result = {
        "required_phrases": [item.get("phrase", "") for item in required],
        "knowledge": definitions,
        "dependency_order": [item.get("id") for item in reversed(definitions) if item.get("id") is not None],
        "required_schema": required_schema,
        "missing_schema_columns": missing_schema_columns,
        "complete": completeness.get("complete", False),
        "missing_phrases": completeness.get("missing_phrases", []),
        "suggested_knowledge": completeness.get("suggested_knowledge", []),
    }
    tool_context.state["prepared_knowledge_context"] = result
    return json.dumps(result, separators=(",", ":"))


def inspect_database(sql: str, tool_context: ToolContext) -> str:
    """Run one small read-only query to resolve preprocessing ambiguity."""
    raw = execute_sql(sql, tool_context)
    result = _structured_table_result(raw, sample_limit=50)
    result.setdefault("success", "error" not in result)
    if len(result.get("columns", [])) == 1:
        column = result["columns"][0]
        result["values"] = [row[column] for row in result.get("sample_rows", [])]
        result.pop("sample_rows", None)
    result["selection_rule"] = "exact_match_only_no_inferred_synonyms"
    inspected_columns = sorted(set(re.findall(
        r"\b([A-Za-z_]\w*\.[A-Za-z_]\w*)\b", sql.lower()
    )))
    result["inspected_columns"] = inspected_columns
    tool_context.state["last_database_inspection"] = {**result, "sql": sql}
    return json.dumps(result, separators=(",", ":"))


def finalize_preprocessing_context(
    selected_tables: list[str],
    selected_columns: list[dict],
    selected_join_edges: list[dict],
    required_knowledge_phrases: list[str],
    selected_knowledge: list[dict],
    unresolved_items: list[str],
    tool_context: ToolContext,
) -> str:
    """Validate and store selected schema, joins, and KB as Phase 1 context."""
    task_id = _get_task_id(tool_context)
    errors = []
    try:
        metadata = _column_metadata(task_id)
        schema = _post_json("/schema", {"task_id": task_id}).get("schema", "")
        available_knowledge = set(_knowledge_names(task_id))

        available_columns = set()
        available_tables = set()
        for key in metadata:
            parts = str(key).split("|")
            if len(parts) >= 3:
                table, column = parts[-2].lower(), parts[-1].lower()
                available_tables.add(table)
                available_columns.add(f"{table}.{column}")

        normalized_tables = list(dict.fromkeys(
            str(table).strip().lower() for table in selected_tables if str(table).strip()
        ))
        for table in normalized_tables:
            if table not in available_tables:
                errors.append(f"Selected table does not exist: {table}")

        expanded_columns = []
        for item in selected_columns:
            if isinstance(item, dict) and isinstance(item.get("columns"), list):
                for nested in item["columns"]:
                    nested = {"name": nested} if isinstance(nested, str) else nested
                    expanded_columns.append({
                        "table": item.get("table", ""),
                        "column": nested.get("column", nested.get("name", "")),
                        "role": nested.get("role", nested.get("roles", [])),
                    })
            else:
                expanded_columns.append(item)
        normalized_columns = []
        for item in expanded_columns:
            table = str(item.get("table", "")).strip().lower()
            column = str(item.get("column", "")).strip().lower()
            role = item.get("role", item.get("roles", []))
            if not table or not column:
                errors.append(f"Selected column requires table and column: {item}")
                continue
            qualified = f"{table}.{column}"
            if qualified not in available_columns:
                errors.append(f"Selected column does not exist: {qualified}")
            if table not in normalized_tables:
                errors.append(f"Selected column belongs to an unselected table: {qualified}")
            normalized_columns.append({"table": table, "column": column, "role": role})

        schema_edges = set()
        current_table = None
        for line in schema.splitlines():
            table_match = re.search(r'CREATE\s+TABLE\s+"?([\w]+)"?', line, re.I)
            if table_match:
                current_table = table_match.group(1).lower()
            foreign_key = re.search(
                r'FOREIGN\s+KEY\s*\(\s*"?([\w]+)"?\s*\)\s*REFERENCES\s*'
                r'"?([\w]+)"?\s*\(\s*"?([\w]+)"?\s*\)', line, re.I,
            )
            if current_table and foreign_key:
                source_column, target_table, target_column = foreign_key.groups()
                schema_edges.add(frozenset({
                    f"{current_table}.{source_column.lower()}",
                    f"{target_table.lower()}.{target_column.lower()}",
                }))

        normalized_edges = []
        for edge in selected_join_edges:
            left = str(edge.get("left", "")).strip().lower()
            right = str(edge.get("right", "")).strip().lower()
            if not left and edge.get("left_table") and edge.get("left_column"):
                left = f"{edge['left_table']}.{edge['left_column']}".lower()
            if not right and edge.get("right_table") and edge.get("right_column"):
                right = f"{edge['right_table']}.{edge['right_column']}".lower()
            if not left or not right:
                errors.append(f"Join edge requires left and right columns: {edge}")
                continue
            if left not in available_columns or right not in available_columns:
                errors.append(f"Join edge references an unknown column: {left} = {right}")
            if frozenset({left, right}) not in schema_edges:
                errors.append(f"Join edge is not a declared foreign-key relationship: {left} = {right}")
            edge_tables = {endpoint.split(".", 1)[0] for endpoint in (left, right)}
            if not edge_tables.issubset(set(normalized_tables)):
                errors.append(f"Join edge uses an unselected table: {left} = {right}")
            normalized_edges.append({"left": left, "right": right})

        normalized_knowledge = []
        selected_names = []
        for item in selected_knowledge:
            name = str(item.get("name", item.get("knowledge_name", ""))).strip()
            knowledge_id = item.get("id", item.get("knowledge_id"))
            if not name:
                errors.append(f"Selected knowledge requires a name: {item}")
                continue
            if name not in available_knowledge:
                errors.append(f"Selected knowledge entry does not exist: {name}")
            selected_names.append(name)
            normalized_knowledge.append({"id": knowledge_id, "name": name})

        normalized_phrases = list(dict.fromkeys(
            str(phrase).strip() for phrase in required_knowledge_phrases
            if str(phrase).strip()
        ))
        for phrase in normalized_phrases:
            if not any(_similarity(phrase, name) >= 0.55 for name in selected_names):
                errors.append(f"Required knowledge phrase is not covered: {phrase}")

        normalized_unresolved = [
            str(item).strip() for item in unresolved_items if str(item).strip()
        ]
        if normalized_unresolved:
            errors.append("Unresolved preprocessing items remain: " + "; ".join(normalized_unresolved))
        if not normalized_tables:
            errors.append("At least one table must be selected")
        if not normalized_columns:
            errors.append("At least one column must be selected")

        context = {
            "selected_tables": normalized_tables,
            "selected_columns": normalized_columns,
            "selected_join_edges": normalized_edges,
            "required_knowledge_phrases": normalized_phrases,
            "selected_knowledge": normalized_knowledge,
            "unresolved_items": normalized_unresolved,
        }
        validation = {"valid": not errors, "errors": errors}
        tool_context.state["preprocessing_validation"] = validation
        counts = {
            "tables": len(normalized_tables),
            "columns": len(normalized_columns),
            "join_edges": len(normalized_edges),
            "knowledge_entries": len(normalized_knowledge),
        }
        if errors:
            tool_context.state["preprocessing_completed"] = False
            return json.dumps({
                **validation,
                "counts": counts,
                "recommendation": "Resolve each error and finalize preprocessing again",
            })

        tool_context.state["preprocessing_context"] = context
        tool_context.state["preprocessing_completed"] = True
        edge_text = "|".join(
            sorted(f"{edge['left']}={edge['right']}" for edge in normalized_edges)
        )
        return json.dumps({
            "valid": True,
            "counts": counts,
            "tables": normalized_tables,
            "join_path_id": "jp_" + hashlib.sha1(edge_text.encode()).hexdigest()[:8],
            "kb_ids": [item.get("id") for item in normalized_knowledge if item.get("id") is not None],
            "next": "generate_plan",
        }, separators=(",", ":"))
    except Exception as exc:
        tool_context.state["preprocessing_completed"] = False
        return json.dumps({
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        })


# ── Improved Query-planning Tools (variants 1 and 3) ──

def _physical_table_and_alias(value) -> tuple[str, str]:
    if isinstance(value, dict):
        table = str(
            value.get("table") or value.get("table_name") or value.get("name") or ""
        ).strip()
        alias = str(value.get("alias") or "").strip()
    else:
        text = str(value).strip()
        match = re.match(r'^([^\s]+)(?:\s+(?:AS\s+)?([\w]+))?$', text, re.I)
        table = match.group(1) if match else text
        alias = match.group(2) if match and match.group(2) else ""
    table = table.strip('"').split(".")[-1].strip('"').lower()
    return table, alias.strip('"').lower()


def _normalize_plan(plan: dict, category: str) -> tuple[dict, list[str]]:
    """Accept common LLM field variants and store one canonical plan shape."""
    normalized = dict(plan)
    changes = []

    field_aliases = {
        "sources": "source_tables",
        "outputs": "output_columns",
        "filters": "conditions",
        "group_by": "grouping",
        "order_by": "ordering",
        "where": "global_filters",
    }
    for alternate, canonical in field_aliases.items():
        if canonical not in normalized and alternate in normalized:
            normalized[canonical] = normalized[alternate]
            changes.append(f"{alternate}->{canonical}")

    raw_tables = normalized.get("source_tables") or []
    tables, aliases = [], {}
    for value in raw_tables:
        table, alias = _physical_table_and_alias(value)
        if table:
            tables.append(table)
            aliases[table] = table
            if alias:
                aliases[alias] = table
    normalized["source_tables"] = list(dict.fromkeys(tables))
    if raw_tables and normalized["source_tables"] != raw_tables:
        changes.append("source_tables normalized to physical names")

    def endpoint(value: str) -> str:
        text = str(value or "").strip().strip('"').lower()
        if "." not in text:
            return text
        prefix, column = text.split(".", 1)
        return f"{aliases.get(prefix.strip(chr(34)), prefix)}.{column.strip(chr(34))}"

    joins = []
    for value in normalized.get("joins") or []:
        if not isinstance(value, dict):
            joins.append(value)
            continue
        join = dict(value)
        left = join.get("left") or join.get("left_column")
        right = join.get("right") or join.get("right_column")
        if join.get("left_table") and left and "." not in str(left):
            left = f"{join['left_table']}.{join['left_column']}"
        if join.get("right_table") and right and "." not in str(right):
            right = f"{join['right_table']}.{join['right_column']}"
        joins.append({
            **join,
            "left": endpoint(left),
            "right": endpoint(right),
            "type": str(join.get("type") or join.get("join_type") or "INNER").upper(),
        })
    normalized["joins"] = joins

    outputs = []
    moved_output_filters = []
    for value in normalized.get("output_columns") or []:
        if not isinstance(value, dict):
            expression = str(value)
            outputs.append({
                "name": expression.split(".")[-1],
                "expression": expression,
                "expose": True,
            })
            continue
        item = dict(value)
        if "filter" not in item and "condition" in item:
            item["filter"] = item["condition"]
        if "knowledge_id" not in item and item.get("knowledge_ids"):
            item["knowledge_id"] = item["knowledge_ids"][0]
        item.setdefault("expose", True)
        scale = str(item.get("scale", ""))
        if "rounding" in item:
            normalized_rounding = _rounding_places(item["rounding"])
            if normalized_rounding is not None:
                item["rounding"] = normalized_rounding
                changes.append("output rounding normalized")
        else:
            rounding = re.search(r"(?:round(?:ed)?(?:\s+to)?|decimal(?:\s+places?)?)\D*(\d+)", scale, re.I)
            if rounding:
                item["rounding"] = int(rounding.group(1))
        expression = str(item.get("expression", ""))
        aggregate = str(item.get("aggregate", ""))
        is_aggregate = bool(aggregate or re.search(
            r"\b(count|sum|avg|min|max|percentile_cont|percentile_disc)\s*\(",
            expression, re.I,
        ))
        if item.get("filter") and not is_aggregate:
            moved_output_filters.append({
                "expression": item.pop("filter"), "stage": "WHERE",
                "phrase": f"restriction for {item.get('name', 'output')}",
            })
        outputs.append(item)
    normalized["output_columns"] = outputs

    if normalized.get("global_filters") and not normalized.get("conditions"):
        normalized["conditions"] = [
            {"expression": value, "stage": "WHERE"}
            if isinstance(value, str) else value
            for value in normalized["global_filters"]
        ]
    if "population_constraints" not in normalized:
        alternatives = normalized.get("entity_filters") or normalized.get("row_filters")
        if alternatives is None:
            alternatives = normalized.get("global_filters")
        if alternatives is not None:
            normalized["population_constraints"] = [
                {"phrase": "global restriction", "predicate": value}
                if isinstance(value, str) else value
                for value in alternatives
            ]
        else:
            normalized["population_constraints"] = []

    for field in ("calculations", "conditions"):
        items = []
        for value in normalized.get(field) or []:
            if not isinstance(value, dict):
                items.append(value)
                continue
            item = dict(value)
            if "knowledge_id" not in item and "kb_id" in item:
                item["knowledge_id"] = item["kb_id"]
            if "knowledge_id" not in item and item.get("knowledge_ids"):
                item["knowledge_id"] = item["knowledge_ids"][0]
            if field == "conditions" and "expression" not in item:
                item["expression"] = item.get("predicate", item.get("condition", item.get("filter", "")))
            if field == "conditions" and not item.get("stage"):
                item["stage"] = (
                    "HAVING" if re.search(
                        r"\b(count|sum|avg|min|max|percentile_cont|percentile_disc)\s*\(",
                        str(item.get("expression", "")), re.I,
                    ) else "WHERE"
                )
            items.append(item)
        normalized[field] = items

    if moved_output_filters:
        normalized.setdefault("conditions", []).extend(moved_output_filters)
        normalized.setdefault("population_constraints", []).extend({
            "phrase": item["phrase"], "predicate": item["expression"]
        } for item in moved_output_filters)
        changes.append("non-aggregate output filters moved to WHERE")

    for calculation in normalized.get("calculations") or []:
        if isinstance(calculation, dict):
            calculation.setdefault("expose", False)

    if not normalized.get("steps"):
        steps = ["read grounded sources and apply joins"]
        if normalized.get("conditions") or normalized.get("population_constraints"):
            steps.append("apply row and group restrictions")
        if normalized.get("calculations"):
            steps.append("compute KB-backed expressions")
        steps.append("project the exact output contract")
        if normalized.get("ordering") or normalized.get("limit"):
            steps.append("apply ordering and limit")
        normalized["steps"] = steps
        changes.append("ordered steps generated")

    if category == "Query" and str(normalized.get("operation", "")).upper() != "SELECT":
        normalized["operation"] = "SELECT"
        changes.append("operation normalized to SELECT for Query")
    return normalized, changes

def _derive_plan_difficulty(category: str, plan: dict) -> str:
    joins = plan.get("joins") or []
    if category == "Query":
        nested = any(bool(plan.get(field)) for field in (
            "requires_cte", "requires_subquery", "requires_set_operation",
        ))
        if nested:
            return "nested_complex"
        return "non_nested_complex" if joins else "easy"

    procedural = bool(plan.get("requires_procedural_logic"))
    multi_object = len(plan.get("target_objects") or []) > 1
    multi_statement = len(plan.get("statement_sequence") or []) > 1
    relational = bool(
        joins
        or plan.get("requires_cte")
        or plan.get("requires_subquery")
        or plan.get("requires_set_operation")
        or len(plan.get("source_tables") or []) > 1
    )
    if procedural or multi_object or multi_statement:
        return "nested_complex"
    return "non_nested_complex" if relational else "easy"


def _generation_strategy(category: str, difficulty: str) -> dict:
    strategies = {
        ("Query", "easy"): {
            "shape": "single_select",
            "checks": ["one_table", "exact_outputs", "filters"],
        },
        ("Query", "non_nested_complex"): {
            "shape": "joined_select",
            "checks": ["fk_join_path", "result_grain", "filter_stage", "exact_outputs"],
        },
        ("Query", "nested_complex"): {
            "shape": "staged_cte_pipeline",
            "stages": ["base_population", "component_metrics", "combined_metrics", "final_projection"],
            "checks": ["key_preservation", "grain_per_stage", "dependency_order",
                       "no_row_multiplication", "exact_outputs"],
        },
        ("Management", "easy"): {
            "shape": "single_statement",
            "checks": ["target", "operation", "affected_rows"],
        },
        ("Management", "non_nested_complex"): {
            "shape": "source_then_mutation",
            "checks": ["source_target_join", "deduplication", "affected_rows"],
        },
        ("Management", "nested_complex"): {
            "shape": "ordered_statement_pipeline",
            "checks": ["statement_order", "object_dependencies", "predicate_scope", "definitions"],
        },
    }
    return strategies[(category, difficulty)]


def _compact_plan_contract(plan: dict) -> dict:
    expressions = [
        str(item.get("expression", ""))
        for field in ("calculations", "output_columns")
        for item in (plan.get(field) or []) if isinstance(item, dict)
    ]
    knowledge_ids = list(dict.fromkeys(
        item.get("knowledge_id", item.get("kb_id"))
        for field in ("calculations", "conditions", "output_columns")
        for item in (plan.get(field) or []) if isinstance(item, dict)
        if item.get("knowledge_id", item.get("kb_id")) is not None
    ))
    dependencies = list(dict.fromkeys(
        str(dependency)
        for field in ("calculations", "output_columns")
        for item in (plan.get(field) or []) if isinstance(item, dict)
        for dependency in (item.get("formula_dependencies") or [])
    ))
    edge_text = "|".join(sorted(
        f"{item.get('left', '')}={item.get('right', '')}"
        for item in (plan.get("joins") or []) if isinstance(item, dict)
    ))
    return {
        "output_count": len(plan.get("output_columns") or []),
        "knowledge_ids": knowledge_ids,
        "formula_dependencies": dependencies,
        "population": [
            item.get("predicate", item.get("expression", ""))
            for item in (plan.get("population_constraints") or []) if isinstance(item, dict)
        ],
        "join_path_id": "jp_" + hashlib.sha1(edge_text.encode()).hexdigest()[:8],
        "decimal_divisors": list(dict.fromkeys(
            re.findall(r"/\s*(\d+\.\d+)", " ".join(expressions))
        )),
    }


def _transitive_kb_ids(direct_ids: set[str], tool_context: ToolContext) -> set[str]:
    knowledge = tool_context.state.get("prepared_knowledge_context", {}).get("knowledge", [])
    name_to_item = {str(item.get("name", "")): item for item in knowledge}
    id_to_item = {
        str(item.get("id")): item for item in knowledge if item.get("id") is not None
    }
    used = set(direct_ids)
    pending = list(direct_ids)
    while pending:
        item = id_to_item.get(pending.pop())
        if not item:
            continue
        for dependency_name in item.get("depends_on", []) or []:
            dependency = name_to_item.get(str(dependency_name))
            if dependency and dependency.get("id") is not None:
                dependency_id = str(dependency["id"])
                if dependency_id not in used:
                    used.add(dependency_id)
                    pending.append(dependency_id)
    return used


def _semantic_plan_contract(
    category: str, difficulty: str, plan: dict, context: dict, tool_context: ToolContext
) -> dict:
    question = str(tool_context.state.get("task_question", ""))
    outputs = [
        str(item.get("name", item.get("expression", "output")))
        if isinstance(item, dict) else str(item)
        for item in (plan.get("output_columns") or [])
        if not isinstance(item, dict) or item.get("expose", True)
    ]
    populations = [
        str(item.get("predicate", item.get("expression", "")))
        for item in (plan.get("population_constraints") or []) if isinstance(item, dict)
    ]
    joins = [
        f"{item.get('left', '')}={item.get('right', '')}"
        for item in (plan.get("joins") or []) if isinstance(item, dict)
    ]
    direct_kb = {
        str(item.get("knowledge_id", item.get("kb_id")))
        for field in ("calculations", "conditions", "output_columns")
        for item in (plan.get(field) or []) if isinstance(item, dict)
        if item.get("knowledge_id", item.get("kb_id")) is not None
    }
    used_kb = _transitive_kb_ids(direct_kb, tool_context)
    selected_kb = {
        str(item.get("id")) for item in context.get("selected_knowledge", [])
        if item.get("id") is not None
    }
    selected_edges = {
        frozenset((str(item.get("left", "")).lower(), str(item.get("right", "")).lower()))
        for item in context.get("selected_join_edges", [])
    }
    categorical_grounded = all(
        not re.findall(r"'([^']+)'", predicate)
        or item.get("value_verified") is True
        for item, predicate in (
            (item, str(item.get("predicate", item.get("expression", ""))))
            for item in (plan.get("population_constraints") or []) if isinstance(item, dict)
        )
    )
    ordering_requested = bool(re.search(r"\b(sort|order|highest|lowest|ascending|descending)\b", question, re.I))
    checks = {
        "population_grounded": all(populations) and categorical_grounded
        if plan.get("population_constraints") else True,
        "outputs_defined": bool(outputs) if category == "Query" else bool(plan.get("target_objects")),
        "kb_dependencies_complete": selected_kb.issubset(used_kb),
        "join_path_grounded": all(
            frozenset((str(item.get("left", "")).lower(), str(item.get("right", "")).lower()))
            in selected_edges for item in (plan.get("joins") or []) if isinstance(item, dict)
        ),
        "filter_stages_defined": all(
            isinstance(item, dict) and bool(item.get("expression")) and bool(item.get("stage"))
            for item in (plan.get("conditions") or [])
        ),
        "ordering_covered": not ordering_requested or bool(plan.get("ordering")),
    }
    summary_parts = [f"{category}/{difficulty}"]
    if plan.get("result_grain"):
        summary_parts.append("grain=" + str(plan["result_grain"])[:80])
    summary_parts.append("population=" + (", ".join(value[:100] for value in populations) or "all rows"))
    summary_parts.append("outputs=" + (", ".join(outputs[:12]) or "none"))
    if joins:
        summary_parts.append("joins=" + ", ".join(joins[:5]))
    if selected_kb:
        summary_parts.append("KB=" + ",".join(sorted(selected_kb)))
    if plan.get("ordering"):
        summary_parts.append("ordered=yes")
    return {
        "summary": "; ".join(summary_parts)[:700],
        "checks": checks,
        "semantic_valid": all(checks.values()),
        "next": "review_summary_then_generate_sql" if all(checks.values()) else "regenerate_plan",
    }


def generate_query_plan(
    category: str, plan: dict, tool_context: ToolContext
) -> str:
    """Record a new structured query-plan candidate for validation.

    Use PREPROCESSING_CONTEXT to construct a complete Query or Management plan.
    If validation later fails, correct the reported issues and call this tool
    again to create a new version.

    Args:
        category: Task category, exactly Query or Management.
        plan: Structured plan containing sources, targets/outputs, joins,
            calculations, conditions, operations, and ordered steps.

    Returns:
        JSON with the candidate version, derived difficulty, and next action.
    """
    normalized_category = str(category).strip().title()
    if normalized_category not in {"Query", "Management"}:
        return json.dumps({
            "status": "rejected",
            "errors": ["category must be Query or Management"],
            "recommendation": "Call generate_query_plan again with a valid category",
        })
    if not isinstance(plan, dict) or not plan:
        return json.dumps({
            "status": "rejected",
            "errors": ["plan must be a non-empty object"],
            "recommendation": "Call generate_query_plan again with a structured plan",
        })

    plan, normalizations = _normalize_plan(plan, normalized_category)
    difficulty = _derive_plan_difficulty(normalized_category, plan)
    history = list(tool_context.state.get("query_plan_history", []))
    version = len(history) + 1
    candidate = {
        "version": version,
        "category": normalized_category,
        "difficulty": difficulty,
        "plan": plan,
    }
    history.append(candidate)
    tool_context.state["query_plan_history"] = history
    tool_context.state["query_plan_candidate"] = candidate
    tool_context.state["task_category"] = normalized_category
    tool_context.state["task_difficulty"] = difficulty
    tool_context.state["query_plan_validated"] = False
    tool_context.state["query_plan_completed"] = False
    return json.dumps({
        "status": "generated",
        "version": version,
        "category": normalized_category,
        "difficulty": difficulty,
        "normalizations": normalizations,
        "next_action": "Call validate_query_plan",
    })


def validate_query_plan(tool_context: ToolContext) -> str:
    """Validate the latest plan against the finalized preprocessing context.

    Returns errors and recommends generate_query_plan when invalid. Only a valid
    candidate is promoted to the official QUERY_PLAN handoff.

    Returns:
        JSON containing validity, errors, warnings, and the required next action.
    """
    errors, warnings = [], []
    context = tool_context.state.get("preprocessing_context")
    candidate = tool_context.state.get("query_plan_candidate")
    if not context or not tool_context.state.get("preprocessing_completed", False):
        errors.append("PREPROCESSING_CONTEXT is missing or not finalized")
    if not candidate:
        errors.append("No query-plan candidate exists")
    if errors:
        result = {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "recommendation": "Complete preprocessing, then call generate_query_plan",
        }
        tool_context.state["query_plan_validation"] = result
        tool_context.state["query_plan_validated"] = False
        tool_context.state["query_plan_completed"] = False
        return json.dumps(result)

    category = candidate["category"]
    plan = candidate["plan"]
    selected_tables = {str(table).lower() for table in context.get("selected_tables", [])}
    selected_columns = {
        f"{str(item.get('table', '')).lower()}.{str(item.get('column', '')).lower()}"
        for item in context.get("selected_columns", [])
    }
    selected_edges = {
        frozenset({str(edge.get("left", "")).lower(), str(edge.get("right", "")).lower()})
        for edge in context.get("selected_join_edges", [])
    }
    selected_knowledge_ids = {
        str(item.get("id")) for item in context.get("selected_knowledge", [])
        if item.get("id") is not None
    }
    used_knowledge_ids = {
        str(item.get("knowledge_id", item.get("kb_id")))
        for field in ("calculations", "conditions", "output_columns")
        for item in (plan.get(field) or []) if isinstance(item, dict)
        if item.get("knowledge_id", item.get("kb_id")) is not None
    }
    used_knowledge_ids = _transitive_kb_ids(used_knowledge_ids, tool_context)
    for knowledge_id in sorted(selected_knowledge_ids - used_knowledge_ids):
        errors.append(f"Required knowledge ID is not used by a calculation or condition: {knowledge_id}")

    source_tables = [str(table).lower() for table in plan.get("source_tables", [])]
    if not source_tables:
        errors.append("source_tables must not be empty")
    for table in source_tables:
        if table not in selected_tables:
            errors.append(f"Plan source table is not in PREPROCESSING_CONTEXT: {table}")

    for join in plan.get("joins", []) or []:
        left = str(join.get("left", "")).lower()
        right = str(join.get("right", "")).lower()
        if not left or not right:
            errors.append(f"Join requires left and right qualified columns: {join}")
        elif frozenset({left, right}) not in selected_edges:
            errors.append(f"Join edge is not in PREPROCESSING_CONTEXT: {left} = {right}")
        if category == "Query" and not join.get("type"):
            errors.append(f"Query join requires a join type: {left} = {right}")

    for calculation in plan.get("calculations", []) or []:
        if not calculation.get("expression"):
            errors.append(f"Calculation requires an expression: {calculation}")
        knowledge_id = calculation.get("knowledge_id")
        if knowledge_id is not None and str(knowledge_id) not in selected_knowledge_ids:
            errors.append(f"Calculation references unselected knowledge ID: {knowledge_id}")

    for condition in plan.get("conditions", []) or []:
        if not condition.get("expression"):
            errors.append(f"Condition requires an expression: {condition}")
        if not condition.get("stage"):
            errors.append(f"Condition requires an application stage: {condition}")

    if category == "Query" and "population_constraints" not in plan:
        errors.append("Query plan must explicitly provide population_constraints (use [] when none)")
    for constraint in plan.get("population_constraints", []) or []:
        if not isinstance(constraint, dict):
            errors.append(f"Population constraint must be an object: {constraint}")
            continue
        if not constraint.get("phrase"):
            errors.append(f"Population constraint requires source phrase: {constraint}")
        predicate = str(constraint.get("predicate", constraint.get("expression", "")))
        if not predicate:
            errors.append(f"Population constraint requires predicate: {constraint}")
            continue
        refs = set(re.findall(r"\b([a-zA-Z_]\w*\.[a-zA-Z_]\w*)\b", predicate.lower()))
        for ref in refs:
            if ref not in selected_columns:
                errors.append(f"Population constraint column is not selected: {ref}")
        literals = re.findall(r"'([^']+)'", predicate)
        if literals:
            inspection = tool_context.state.get("last_database_inspection", {})
            observed_values = {
                str(value).lower() for value in inspection.get("values", [])
            }
            for row in inspection.get("sample_rows", []) or []:
                observed_values.update(str(value).lower() for value in row.values())
            if constraint.get("value_verified") is not True:
                errors.append(
                    f"Categorical constraint requires value_verified=true: {constraint.get('phrase', '')}"
                )
            unobserved = [value for value in literals if value.lower() not in observed_values]
            if unobserved:
                errors.append(
                    f"Categorical values were not observed by inspect_database: {unobserved}"
                )
            phrase_tokens = {
                token[:-1] if token.endswith("s") else token
                for token in re.findall(r"[a-z0-9]+", str(constraint.get("phrase", "")).lower())
            }
            directly_named = {
                value for value in observed_values
                if any(
                    (token[:-1] if token.endswith("s") else token) in phrase_tokens
                    for token in re.findall(r"[a-z0-9]+", value)
                )
            }
            unsupported = [
                value for value in literals
                if directly_named and value.lower() not in directly_named
                and not constraint.get("equivalence_reason")
            ]
            if unsupported:
                errors.append(
                    "Categorical values are broader than the request's exact wording: "
                    f"{unsupported}; add equivalence_reason only when the request or KB equates them"
                )

    for output in plan.get("output_columns", []) or []:
        if not isinstance(output, dict) or not output.get("filter"):
            continue
        expression = str(output.get("expression", ""))
        aggregate = str(output.get("aggregate", ""))
        if not aggregate and not re.search(
            r"\b(count|sum|avg|min|max|percentile_cont|percentile_disc)\s*\(",
            expression, re.I,
        ):
            errors.append(f"Output filter requires an aggregate output: {output.get('name', output)}")

    referenced_columns = []
    for field in ("output_columns", "target_objects"):
        for value in plan.get(field, []) or []:
            if isinstance(value, str) and re.fullmatch(r"[\w]+\.[\w]+", value):
                referenced_columns.append(value.lower())
    for column in referenced_columns:
        if column not in selected_columns:
            errors.append(f"Plan column is not in PREPROCESSING_CONTEXT: {column}")

    if category == "Query":
        if str(plan.get("operation", "")).upper() != "SELECT":
            errors.append("Query plan operation must be SELECT")
        if not plan.get("result_grain"):
            errors.append("Query plan requires result_grain")
        if not plan.get("output_columns"):
            errors.append("Query plan requires output_columns")
    else:
        operation = str(plan.get("operation", "")).upper()
        if not operation:
            errors.append("Management plan requires an operation")
        if not plan.get("target_objects"):
            errors.append("Management plan requires target_objects")
        if operation == "UPDATE" and not plan.get("mutations"):
            errors.append("UPDATE plan requires mutations")
        if operation in {"CREATE FUNCTION", "CREATE PROCEDURE", "CREATE TRIGGER"}:
            if not plan.get("object_definitions"):
                errors.append(f"{operation} plan requires object_definitions")

    if not plan.get("steps"):
        errors.append("Plan requires ordered decomposition steps")
    if context.get("unresolved_items"):
        errors.append("PREPROCESSING_CONTEXT still contains unresolved items")

    derived_difficulty = _derive_plan_difficulty(category, plan)
    if derived_difficulty != candidate.get("difficulty"):
        warnings.append(
            f"Difficulty normalized from {candidate.get('difficulty')} to {derived_difficulty}"
        )
        candidate["difficulty"] = derived_difficulty

    semantic_contract = _semantic_plan_contract(
        category, derived_difficulty, plan, context, tool_context,
    )
    for check, passed in semantic_contract["checks"].items():
        if not passed:
            errors.append(f"Semantic plan check failed: {check}")

    if errors:
        result = {
            "valid": False,
            "version": candidate["version"],
            "category": category,
            "difficulty": derived_difficulty,
            "errors": errors,
            "warnings": warnings,
            "semantic_contract": semantic_contract,
            "recommendation": "Fix errors; call generate_query_plan again",
            "next": "fix_errors_then_regenerate_plan",
        }
        tool_context.state["query_plan_validation"] = result
        tool_context.state["query_plan_validated"] = False
        tool_context.state["query_plan_completed"] = False
        return json.dumps(result)

    official_plan = {
        "version": candidate["version"],
        "category": category,
        "difficulty": derived_difficulty,
        "plan": plan,
    }
    result = {
        "valid": True,
        "version": candidate["version"],
        "category": category,
        "difficulty": derived_difficulty,
        "errors": [],
        "warnings": warnings,
        "contract": _compact_plan_contract(plan),
        "semantic_contract": semantic_contract,
        "generation_strategy": _generation_strategy(category, derived_difficulty),
        "next": "generate_sql",
    }
    tool_context.state["query_plan"] = official_plan
    tool_context.state["query_plan_validation"] = result
    tool_context.state["query_plan_validated"] = True
    tool_context.state["query_plan_completed"] = True
    return json.dumps(result)


def generate_and_validate_query_plan(
    category: str, plan: dict, tool_context: ToolContext
) -> str:
    """Normalize, generate, and validate one plan in a single call."""
    generated = json.loads(generate_query_plan(category, plan, tool_context))
    if generated.get("status") != "generated":
        return json.dumps(generated)
    validation = json.loads(validate_query_plan(tool_context))
    validation["normalizations"] = generated.get("normalizations", [])
    return json.dumps(validation)


# ── Improved SQL-generation Tool (variants 1 and 3) ──

def validate_sql_to_plan(
    sql: str, plan_version: int, tool_context: ToolContext
) -> str:
    """Validate draft SQL against the stored plan and store it when valid."""
    from sqlglot import exp, parse

    official = tool_context.state.get("query_plan")
    diagnoses = []

    def add_diagnosis(kind: str, message: str, component: str, change: str):
        diagnoses.append({
            "type": kind,
            "code": kind.upper(),
            "message": message,
            "path": component,
            "action": change,
        })

    if not tool_context.state.get("query_plan_validated", False) or not official:
        add_diagnosis(
            "missing_validated_plan",
            "No validated QUERY_PLAN is available for this draft.",
            "QUERY_PLAN",
            "Generate and validate a query plan before generating SQL.",
        )
        trees = []
    elif int(plan_version) != int(official.get("version", -1)):
        add_diagnosis(
            "plan_version_mismatch",
            f"Draft uses plan version {plan_version}, but version {official.get('version')} is validated.",
            "QUERY_PLAN.version",
            "Regenerate the SQL from the currently validated plan version.",
        )
        trees = []
    else:
        try:
            trees = [tree for tree in parse(sql, dialect="postgres") if tree is not None]
            if not trees:
                raise ValueError("no SQL statement was parsed")
        except Exception as exc:
            trees = []
            add_diagnosis(
                "sql_parse_error",
                f"The draft is not valid parseable PostgreSQL: {exc}",
                "DRAFT_SQL",
                "Correct the PostgreSQL syntax and validate the complete draft again.",
            )

    if official and trees and not diagnoses:
        plan = official.get("plan", {})
        category = official.get("category")
        expected_operation = str(plan.get("operation", "")).upper()
        root_types = {type(tree).__name__.upper() for tree in trees}
        has_select = any(tree.find(exp.Select) is not None for tree in trees)
        ddl_command_matches = (
            expected_operation.startswith(("CREATE", "ALTER", "DROP"))
            and bool(root_types & {"CREATE", "ALTER", "DROP", "COMMAND"})
        )
        actual_operation = (
            "SELECT" if category == "Query" and has_select
            else expected_operation if expected_operation in root_types or ddl_command_matches
            else next(iter(root_types), "UNKNOWN")
        )
        if expected_operation and actual_operation != expected_operation:
            add_diagnosis(
                "operation_mismatch",
                f"The plan requires {expected_operation}, but the draft uses {actual_operation}.",
                "operation",
                f"Generate a {expected_operation} statement as specified by the plan.",
            )

        cte_names = {
            cte.alias_or_name.lower()
            for tree in trees for cte in tree.find_all(exp.CTE)
            if cte.alias_or_name
        }
        actual_tables = {
            table.name.lower()
            for tree in trees for table in tree.find_all(exp.Table)
            if table.name and table.name.lower() not in cte_names
        }
        planned_tables = {str(table).lower() for table in plan.get("source_tables", [])}
        preprocessing_tables = {
            str(table).lower()
            for table in tool_context.state.get("preprocessing_context", {}).get(
                "selected_tables", []
            )
        }
        for table in sorted(actual_tables - preprocessing_tables):
            add_diagnosis(
                "ungrounded_table",
                f"The draft references {table}, which is not in PREPROCESSING_CONTEXT.",
                "source_tables",
                "Use only grounded tables or return to preprocessing to add the required table.",
            )
        for table in sorted(planned_tables - actual_tables):
            add_diagnosis(
                "missing_planned_table",
                f"The planned source table {table} is missing from the draft.",
                "source_tables",
                f"Add {table} using the relationship specified by QUERY_PLAN.",
            )

        actual_join_types = []
        for tree in trees:
            for join in tree.find_all(exp.Join):
                side = str(join.args.get("side") or "").upper()
                kind = str(join.args.get("kind") or "").upper()
                actual_join_types.append(side or kind or "INNER")
        planned_join_types = [
            str(join.get("type", "INNER")).upper().replace(" JOIN", "")
            for join in plan.get("joins", []) or []
        ]
        if sorted(actual_join_types) != sorted(planned_join_types):
            add_diagnosis(
                "join_type_mismatch",
                f"Planned join types are {planned_join_types}, but draft join types are {actual_join_types}.",
                "joins",
                "Use the join types from QUERY_PLAN, including required outer joins.",
            )

        structural_requirements = {
            "requires_cte": any(tree.find(exp.CTE) is not None for tree in trees),
            "requires_subquery": any(tree.find(exp.Subquery) is not None for tree in trees),
            "requires_set_operation": any(
                tree.find(exp.Union, exp.Intersect, exp.Except) is not None
                for tree in trees
            ),
        }
        for field, present in structural_requirements.items():
            if bool(plan.get(field)) and not present:
                add_diagnosis(
                    "missing_planned_structure",
                    f"The plan sets {field}=true, but the draft does not contain that structure.",
                    field,
                    f"Implement the planned structure represented by {field}.",
                )

        sql_lower = sql.lower()
        for output in plan.get("output_columns", []) or []:
            output_text = str(output).lower()
            if re.fullmatch(r"[\w]+\.[\w]+", output_text):
                column_name = output_text.split(".")[-1]
                if column_name not in sql_lower:
                    add_diagnosis(
                        "missing_output_column",
                        f"The planned output column {output} is not represented in the draft.",
                        "output_columns",
                        f"Include the planned output column {output} in the projection.",
                    )
        exposed_outputs = [
            output for output in (plan.get("output_columns") or [])
            if not isinstance(output, dict) or output.get("expose", True)
        ]
        outer_selects = [
            tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
            for tree in trees
        ]
        actual_projections = [
            projection for select in outer_selects if select is not None
            for projection in select.expressions
        ]
        if category == "Query" and len(actual_projections) != len(exposed_outputs):
            add_diagnosis(
                "output_count_mismatch",
                f"The output contract exposes {len(exposed_outputs)} columns, but SQL projects {len(actual_projections)}.",
                "output_columns",
                "Project exactly the exposed outputs; keep intermediate calculations inside CTEs/subqueries.",
            )
        for index, output in enumerate(exposed_outputs[:len(actual_projections)]):
            if not isinstance(output, dict):
                continue
            projection_sql = actual_projections[index].sql(dialect="postgres").lower()
            raw_rounding = output.get("rounding")
            rounding = _rounding_places(raw_rounding)
            if raw_rounding is not None and rounding is None:
                add_diagnosis(
                    "invalid_rounding_contract",
                    f"Output {index + 1} has unsupported rounding value {raw_rounding!r}.",
                    f"output_columns[{index}].rounding",
                    "Use an integer number of decimal places.",
                )
            elif rounding is not None and not re.search(
                rf"\bround\s*\(.*?,\s*{rounding}\s*\)", projection_sql, re.I,
            ):
                add_diagnosis(
                    "output_rounding_mismatch",
                    f"Output {index + 1} requires rounding to {rounding} decimals.",
                    f"output_columns[{index}].rounding",
                    f"Apply ROUND(expression, {rounding}) to this output.",
                )
            scale = str(output.get("scale", "")).lower().replace("-", "_").replace(" ", "_")
            if "percent_rank" in projection_sql:
                multiplied_by_100 = bool(re.search(r"percent_rank\s*\(\).*?\*\s*100(?:\.0+)?", projection_sql))
                if scale in {"0_to_1", "ratio", "fraction"} and multiplied_by_100:
                    add_diagnosis(
                        "output_scale_mismatch", "PERCENT_RANK must remain on a 0-to-1 scale.",
                        f"output_columns[{index}].scale", "Remove multiplication by 100.",
                    )
                if scale in {"0_to_100", "percentage", "percent"} and not multiplied_by_100:
                    add_diagnosis(
                        "output_scale_mismatch", "PERCENT_RANK must be converted to a 0-to-100 scale.",
                        f"output_columns[{index}].scale", "Multiply PERCENT_RANK by 100.",
                    )
        for target in plan.get("target_objects", []) or []:
            target_text = str(target).lower()
            if target_text not in sql_lower and target_text.split(".")[-1] not in sql_lower:
                add_diagnosis(
                    "missing_target_object",
                    f"The planned target object {target} is not represented in the draft.",
                    "target_objects",
                    f"Include the planned target object {target}.",
                )

        for field in ("calculations", "conditions", "output_columns"):
            for item in plan.get(field, []) or []:
                if not isinstance(item, dict) or item.get("knowledge_id") is None:
                    continue
                expression = str(item.get("expression", ""))
                required = set(re.findall(r"\b[a-zA-Z_][\w]*\b|\b\d+(?:\.\d+)?\b", expression.lower()))
                required -= {"select", "where", "and", "or", "as", "null"}
                required -= planned_tables
                missing = sorted(token for token in required if token not in sql_lower)
                if missing:
                    add_diagnosis(
                        "kb_expression_mismatch",
                        f"Knowledge-backed {field[:-1]} {item.get('knowledge_id')} is missing: {missing}",
                        field,
                        "Implement the complete knowledge-backed expression from QUERY_PLAN.",
                    )

        filtered_outputs = [
            output for output in plan.get("output_columns", []) or []
            if isinstance(output, dict) and output.get("filter")
        ]
        if filtered_outputs:
            has_local_filter = any(
                tree.find(exp.Filter) is not None or tree.find(exp.Case) is not None
                for tree in trees
            )
            if not has_local_filter:
                add_diagnosis(
                    "output_filter_scope_mismatch",
                    "A filtered output was implemented without FILTER or CASE.",
                    "output_columns.filter",
                    "Apply the condition inside only the named aggregate output.",
                )

        population_constraints = plan.get("population_constraints", []) or []
        where_sql = " ".join(
            where.sql(dialect="postgres").lower()
            for tree in trees for where in tree.find_all(exp.Where)
        )
        calculation_expressions = {
            str(item.get("name", "")).lower(): str(item.get("expression", "")).lower()
            for item in (plan.get("calculations") or []) if isinstance(item, dict)
            and item.get("name") and item.get("expression")
        }
        for constraint in population_constraints:
            if not isinstance(constraint, dict):
                continue
            predicate = str(constraint.get("predicate", constraint.get("expression", ""))).lower()
            expanded_predicate = predicate
            for name, expression in calculation_expressions.items():
                expanded_predicate = re.sub(
                    rf"\b{re.escape(name)}\b", f"({expression})", expanded_predicate,
                )
            required = set(re.findall(
                r"\b[a-zA-Z_]\w*\b|'[^']*'|\b\d+(?:\.\d+)?\b", expanded_predicate
            ))
            required -= planned_tables | {
                "and", "or", "not", "is", "null", "true", "false", "numeric",
                "decimal", "double", "precision", "round", "cast",
            }
            missing = sorted(token for token in required if token.strip("'") not in where_sql)
            if missing:
                add_diagnosis(
                    "population_constraint_missing",
                    f"Population restriction '{constraint.get('phrase', '')}' is missing from WHERE.",
                    "population_constraints",
                    "Add the grounded population predicate without applying it only to one output.",
                )
                diagnoses[-1].update({
                    "expected_predicate": expanded_predicate[:500],
                    "missing_tokens": missing,
                    "observed_where": where_sql[:500],
                })


    history = list(tool_context.state.get("sql_validation_history", []))
    sql_version = len(history) + 1
    record = {
        "sql_version": sql_version,
        "plan_version": plan_version,
        "sql": sql,
        "valid": not diagnoses,
        "diagnoses": diagnoses,
    }
    history.append(record)
    tool_context.state["sql_validation_history"] = history
    tool_context.state["sql_generation_completed"] = not diagnoses

    if diagnoses:
        signature = tuple(sorted(
            (item.get("code", item.get("type", "")), item.get("path", ""))
            for item in diagnoses
        ))
        same_diagnosis_count = 1 + sum(
            tuple(sorted(
                (item.get("code", item.get("type", "")), item.get("path", item.get("plan_component", "")))
                for item in record.get("diagnoses", [])
            )) == signature
            for record in history[:-1]
        )
        return json.dumps({
            "valid": False,
            "sql_version": sql_version,
            "plan_version": plan_version,
            "diagnoses": diagnoses,
            "same_diagnosis_count": same_diagnosis_count,
            "next": "revise_plan" if same_diagnosis_count >= 2 else "revise_sql",
        }, separators=(",", ":"))

    tool_context.state["draft_sql"] = sql
    tool_context.state["draft_sql_version"] = sql_version
    tool_context.state["draft_sql_plan_version"] = plan_version
    category = official.get("category") if official else ""
    return json.dumps({
        "valid": True,
        "sql_version": sql_version,
        "plan_version": plan_version,
        "diagnoses": [],
        "next_action": (
            "Call execute_validated_sql"
            if category == "Query"
            else "Perform final checks, then call submit_validated_sql"
        ),
    }, separators=(",", ":"))


# ── Improved Post-processing Tool (variants 1 and 3) ──

def diagnose_execution_error(
    sql: str,
    error_message: str,
    plan_version: int,
    tool_context: ToolContext,
) -> str:
    """Classify a failed Query execution and recommend a scoped correction.

    This tool diagnoses but does not rewrite SQL. After correcting the draft,
    call validate_sql_to_plan before the next execute_sql retry.

    Args:
        sql: The complete SQL statement that failed.
        error_message: The error returned by execute_sql.
        plan_version: Validated QUERY_PLAN version used by the failed SQL.
    """
    message = str(error_message).strip()
    lowered = message.lower()
    official = tool_context.state.get("query_plan") or {}
    validated_version = official.get("version")
    category = official.get("category", "")

    patterns = (
        ("syntax_error", ("syntax error", "unterminated", "parse error"),
         "SQL syntax", "Correct the reported syntax near the database error position."),
        ("missing_table", ("relation", "does not exist"),
         "table reference", "Replace the table with a grounded table from PREPROCESSING_CONTEXT."),
        ("missing_column", ("column", "does not exist"),
         "column reference", "Use the qualified column selected in PREPROCESSING_CONTEXT."),
        ("ambiguous_column", ("column reference", "ambiguous"),
         "column reference", "Qualify the ambiguous column with its planned table alias."),
        ("grouping_error", ("must appear in the group by",),
         "GROUP BY", "Align projected non-aggregated columns with the planned grouping."),
        ("type_mismatch", ("operator does not exist", "invalid input syntax", "cannot cast"),
         "expression or predicate", "Correct the operand types without changing the planned condition."),
        ("invalid_function", ("function", "does not exist"),
         "function call", "Use a PostgreSQL function compatible with the planned calculation and types."),
        ("constraint_violation", ("violates", "constraint"),
         "management constraint", "Recheck affected-row conditions and the violated constraint before submission."),
        ("read_only_restriction", ("read-only",),
         "execution policy", "Do not execute modifying SQL; validate it statically and reserve it for submit_sql."),
        ("permission_error", ("permission denied", "not authorized"),
         "database permission", "Do not retry unchanged SQL; verify whether the planned operation is permitted."),
        ("timeout", ("timeout", "timed out", "statement timeout"),
         "query shape", "Reduce unnecessary work while preserving the validated plan."),
    )
    error_type = "unknown_error"
    component = "failed SQL"
    action = "Use the database message and validated plan to make the smallest grounded correction."
    for candidate, markers, candidate_component, candidate_action in patterns:
        if all(marker in lowered for marker in markers):
            error_type = candidate
            component = candidate_component
            action = candidate_action
            break

    issues = []
    if not message:
        issues.append("error_message is empty")
    if not official:
        issues.append("no validated QUERY_PLAN is available")
    elif int(plan_version) != int(validated_version):
        issues.append(
            f"plan version {plan_version} does not match validated version {validated_version}"
        )

    retry_allowed = not issues and category == "Query" and error_type not in {
        "permission_error", "read_only_restriction", "constraint_violation",
    }
    result = {
        "error_type": error_type,
        "failing_component": component,
        "likely_cause": message[:240] or "No database error was supplied.",
        "recommended_action": action,
        "return_to_phase": "sql_generation",
        "retry_allowed": retry_allowed,
    }
    if issues:
        result["diagnostic_issues"] = issues
    if category == "Management":
        result["recommended_action"] = (
            "Do not execute modifying SQL. Recheck it against QUERY_PLAN and submit only after static validation."
        )

    history = list(tool_context.state.get("execution_error_diagnoses", []))
    history.append({
        "plan_version": plan_version,
        "sql": sql,
        "error_message": message,
        **result,
    })
    tool_context.state["execution_error_diagnoses"] = history
    tool_context.state["last_execution_error_diagnosis"] = result
    return json.dumps(result)


def execute_validated_sql(tool_context: ToolContext) -> str:
    """Execute a bounded preview of the latest validated Query SQL."""
    sql = tool_context.state.get("draft_sql")
    if not sql or not tool_context.state.get("sql_generation_completed", False):
        return json.dumps({
            "success": False,
            "error": "No validated DRAFT_SQL is available",
            "next_action": "Call validate_sql_to_plan first",
        })
    preview_sql = (
        "SELECT * FROM (\n" + str(sql).strip().rstrip(";")
        + "\n) AS _candidate_preview LIMIT 3;"
    )
    result = execute_sql(preview_sql, tool_context)
    tool_context.state["last_execution_sql"] = sql
    tool_context.state["last_execution_result"] = result
    succeeded = not str(result).lower().startswith(("sql error:", "error calling"))
    tool_context.state["last_execution_succeeded"] = succeeded
    if not succeeded:
        return json.dumps({"success": False, "error": str(result)[:500]}, separators=(",", ":"))
    summary = _structured_table_result(result)
    summary["success"] = True
    summary["sample_row_count"] = summary.pop("row_count", None)
    summary["has_rows"] = not summary.get("empty", False)
    alerts = []
    plan = (tool_context.state.get("query_plan") or {}).get("plan", {})
    ranges = summary.get("numeric_ranges", {})
    for output in plan.get("output_columns", []) or []:
        if not isinstance(output, dict) or not output.get("expected_range"):
            continue
        name = str(output.get("name", ""))
        expected = output.get("expected_range")
        observed = ranges.get(name)
        if observed and isinstance(expected, (list, tuple)) and len(expected) == 2:
            if observed[0] < expected[0] or observed[1] > expected[1]:
                alerts.append({
                    "code": "VALUE_OUTSIDE_EXPECTED_RANGE",
                    "column": name,
                    "expected": list(expected),
                    "observed": observed,
                })
    if alerts:
        summary["alerts"] = alerts
        summary["next"] = "semantic_review"
    elif summary.get("empty"):
        summary["next"] = "semantic_review"
    else:
        summary["next"] = "submit"
    tool_context.state["last_execution_summary"] = summary
    return json.dumps(summary, separators=(",", ":"))


def diagnose_last_execution_error(tool_context: ToolContext) -> str:
    """Diagnose the last failed state-backed SQL execution."""
    if tool_context.state.get("last_execution_succeeded") is not False:
        return json.dumps({"error": "No failed validated execution is available"})
    return diagnose_execution_error(
        tool_context.state.get("last_execution_sql", ""),
        tool_context.state.get("last_execution_result", ""),
        tool_context.state.get("draft_sql_plan_version", -1),
        tool_context,
    )


# ── Submit Tool ──

def submit_sql(sql: str, tool_context: ToolContext) -> str:
    """Submit your final SQL query for evaluation. You only get ONE attempt.
    Only call this when you are confident in your answer.

    Args:
        sql: The final PostgreSQL SQL query to submit.

    Returns:
        Evaluation result: pass or fail with details.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/submit"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()

            # Always end the task after submit (one attempt only)
            tool_context.state["task_done"] = True

            if data.get("passed"):
                reward = data.get("reward", 0.0)
                tool_context.state["total_reward"] = tool_context.state.get("total_reward", 0.0) + reward
                tool_context.state["phase1_completed"] = True

            raw_msg = data.get("message", "")
            agent_msg = raw_msg.replace("[exec_err_flg] ", "")
            parts = [agent_msg]
            if data.get("reward", 0) > 0:
                parts.append(f"Reward: {data['reward']}")
            return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


def submit_validated_sql(tool_context: ToolContext) -> str:
    """Submit the latest SQL that passed validate_sql_to_plan exactly once."""
    sql = tool_context.state.get("draft_sql")
    if not sql or not tool_context.state.get("sql_generation_completed", False):
        return json.dumps({
            "submitted": False,
            "error": "No validated DRAFT_SQL is available",
        })
    return submit_sql(sql, tool_context)


# ── Build tool list ──

def get_tools(profile: str = "baseline"):
    """Return the isolated baseline or improved tool set."""
    baseline = [
        execute_sql,
        get_schema,
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ]
    improved = [
        prepare_schema_context,
        prepare_knowledge_context,
        finalize_preprocessing_context,
        generate_and_validate_query_plan,
        validate_sql_to_plan,
        inspect_database,
        execute_validated_sql,
        diagnose_last_execution_error,
        submit_validated_sql,
    ]
    profiles = {"baseline": baseline, "improved": improved}
    if profile not in profiles:
        raise ValueError(f"Unsupported active tool profile: {profile}")
    return [FunctionTool(tool) for tool in profiles[profile]]
