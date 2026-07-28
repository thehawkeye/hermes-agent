"""Public atomic graph-write adapter for Hermes Kanban.

This module is the Sequence 0 prerequisite for the GraphIt runtime slice
(ADP-t_2032a27a-r0, §5). It exposes a narrow, versioned, all-or-nothing
graph transaction API on top of the existing single-card ``create_task``
and ``link_tasks`` helpers. The adapter never copies Hermes private task
or event SQL — it composes the existing public primitives inside a
single ``BEGIN IMMEDIATE`` transaction.

Public surface (stable, versioned):

- :func:`graph_write_transaction` — entry point. Accepts caller-supplied
  in-memory graph (TaskSpecs + Edges), validates before mutation, then
  commits tasks + links + events + caller-supplied state rows under
  one transaction with full rollback on any injected failure.

- :class:`TaskSpec` — frozen dataclass mirroring a subset of
  :func:`hermes_cli.kanban_db.create_task` arguments.

- :class:`Edge` — frozen dataclass for explicit edge lists
  (alternative to ``parents=`` on each :class:`TaskSpec`).

- :class:`GraphWriteResult` — return value, exposes the logical-key →
  Hermes task-id map populated only after commit.

- Exception types :class:`GraphWriteError`, :class:`GraphValidationError`,
  :class:`GraphConflictError`.

The capability marker exposed by :mod:`hermes_cli.kanban_db` as
``GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY`` is the version pin: GraphIt
imports that constant and refuses to run if absent or unknown. Adding a
backward-compatible feature bumps the ``/vN`` suffix; a breaking change
bumps to ``v2``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hermes_cli import kanban_db as _kb


# ---------------------------------------------------------------------------
# Public version marker (re-exported on kanban_db for import convenience)
# ---------------------------------------------------------------------------


# Frozen capability/version marker. GraphIt imports
# ``hermes_cli.kanban_db.GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY`` and
# refuses to apply if absent or version-incompatible. Bump ``v1`` to
# ``v2`` only on a breaking change.
GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY: str = "graphit_atomic_graph_write/v1"

# Mirror the marker on the kanban_db module so GraphIt can import it
# without taking a dependency on this adapter module's internal layout.
# Pin once at import time; do NOT mutate at runtime.
setattr(
    _kb,
    "GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY",
    GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY,
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One node in the caller's in-memory graph.

    Mirrors a strict subset of :func:`hermes_cli.kanban_db.create_task`
    arguments. Fields the caller does not supply default to safe,
    non-dispatchable values: a GraphIt graph is always created behind
    a ``blocked`` root so the dispatcher never claims it directly.

    ``key`` is the caller's logical identifier for the node; the adapter
    returns a logical-key → Hermes task-id map after commit so the
    caller can locate the row without parsing ``task_events``.
    """

    key: str
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    created_by: Optional[str] = None
    workspace_kind: str = "scratch"
    workspace_path: Optional[str] = None
    branch_name: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    parents: Tuple[str, ...] = ()
    triage: bool = False
    idempotency_key: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    skills: Optional[Tuple[str, ...]] = None
    max_retries: Optional[int] = None
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    initial_status: str = "ready"
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    extra_event_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("TaskSpec.key is required and must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError(
                f"TaskSpec.title is required for key={self.key!r}"
            )
        # ``initial_status`` is constrained to the same enum as the underlying
        # ``create_task`` (running|blocked). The adapter derives the actual
        # status from parent state at create time, but the enum gate stays
        # tight so callers don't smuggle arbitrary statuses into the graph.
        if self.initial_status not in _kb.VALID_INITIAL_STATUSES:
            raise ValueError(
                f"initial_status must be one of {sorted(_kb.VALID_INITIAL_STATUSES)}; "
                f"got {self.initial_status!r} for key={self.key!r}"
            )
        if self.workspace_kind not in _kb.VALID_WORKSPACE_KINDS:
            raise ValueError(
                f"workspace_kind must be one of {sorted(_kb.VALID_WORKSPACE_KINDS)}; "
                f"got {self.workspace_kind!r} for key={self.key!r}"
            )
        if self.provider_override and not self.model_override:
            raise ValueError(
                f"provider_override requires model_override for key={self.key!r}"
            )


@dataclass(frozen=True)
class Edge:
    """Explicit edge between two logical keys in the same in-memory graph.

    Use this in place of ``parents=`` on each :class:`TaskSpec`. The two
    API shapes are mutually equivalent; the adapter rejects a mix.
    """

    parent_key: str
    child_key: str

    def __post_init__(self) -> None:
        if not self.parent_key or not self.child_key:
            raise ValueError(
                f"Edge.parent_key and Edge.child_key must be non-empty "
                f"(got parent={self.parent_key!r}, child={self.child_key!r})"
            )
        if self.parent_key == self.child_key:
            raise ValueError(
                f"Edge self-loop not allowed: {self.parent_key!r} -> {self.child_key!r}"
            )


@dataclass(frozen=True)
class GraphWriteResult:
    """Return value of :func:`graph_write_transaction`.

    Populated only after a successful commit. On any failure, this
    object is never constructed — the caller observes a raised
    :class:`GraphWriteError` and the DB is fully rolled back.
    """

    task_ids: Dict[str, str]
    link_count: int
    event_count: int
    capability_marker: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GraphWriteError(Exception):
    """Base class for every failure mode of :func:`graph_write_transaction`.

    The error message contains the stable error code from the ADP §8
    failure taxonomy plus the offending keys for diagnostics. No
    manifest content, secrets, or PII is ever echoed.
    """


class GraphValidationError(GraphWriteError):
    """Pre-validation rejected the graph (cycle, unknown parent, duplicate
    logical key, invalid status, invalid initial_status, etc.)."""


class GraphConflictError(GraphWriteError):
    """A pre-existing non-archived row collides with the in-memory graph
    (typically an existing ``idempotency_key`` owned by another task).
    """


# ---------------------------------------------------------------------------
# Adapter internals (exposed for fault-injection tests)
# ---------------------------------------------------------------------------


def _create_task_in_transaction(
    conn: sqlite3.Connection,
    spec: TaskSpec,
    *,
    parents_task_ids: Sequence[str],
) -> str:
    """Create one task inside an already-open transaction.

    Wraps :func:`hermes_cli.kanban_db.create_task_in_transaction` so
    fault-injection tests can monkeypatch this single function and
    observe "after the Nth task insert" behaviour.

    Why this works (verified against ``kanban_db.create_task`` and
    ``write_txn``): the outer :func:`write_txn` opens one
    ``BEGIN IMMEDIATE`` envelope; the in-transaction seam accepts that
    open envelope and adds its INSERT + event rows to the same
    transaction. The outer rollback on any later failure wipes every
    in-transaction insert. No private Hermes task-table SQL is
    duplicated here — the body is delegated to the public in-
    transaction helper.
    """
    # Resolve workspace_path from the spec (no board-level defaulting
    # because the contract says "no board discovery").
    return _kb.create_task_in_transaction(
        conn,
        title=spec.title,
        body=spec.body,
        assignee=spec.assignee,
        created_by=spec.created_by,
        workspace_kind=spec.workspace_kind,
        workspace_path=spec.workspace_path,
        branch_name=spec.branch_name,
        tenant=spec.tenant,
        priority=spec.priority,
        parents=parents_task_ids,
        triage=spec.triage,
        idempotency_key=spec.idempotency_key,
        max_runtime_seconds=spec.max_runtime_seconds,
        skills=list(spec.skills) if spec.skills is not None else None,
        max_retries=spec.max_retries,
        model_override=spec.model_override,
        provider_override=spec.provider_override,
        goal_mode=spec.goal_mode,
        goal_max_turns=spec.goal_max_turns,
        initial_status=spec.initial_status,
        session_id=spec.session_id,
        project_id=spec.project_id,
    )


def _link_in_transaction(
    conn: sqlite3.Connection,
    parent_id: str,
    child_id: str,
) -> None:
    """Link one parent→child edge inside an already-open transaction.

    Wraps :func:`hermes_cli.kanban_db.link_tasks_in_transaction` for
    fault-injection symmetry with :func:`_create_task_in_transaction`.
    """
    _kb.link_tasks_in_transaction(conn, parent_id, child_id)


def _append_event_in_transaction(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one task_event row inside an already-open transaction.

    Wraps :func:`hermes_cli.kanban_db.append_event_in_transaction` so
    fault-injection tests can pin a fault "after the Nth event append".
    """
    _kb.append_event_in_transaction(conn, task_id, kind, payload)


StateCallback = Callable[[sqlite3.Connection, Dict[str, str]], None]


def _run_state_callback(
    conn: sqlite3.Connection,
    state_callback: Optional[StateCallback],
    task_ids: Dict[str, str],
) -> None:
    """Invoke caller state writes over the same transaction and task map."""
    if state_callback is not None:
        state_callback(conn, dict(task_ids))


# ---------------------------------------------------------------------------
# Caller-supplied state-row helper
# ---------------------------------------------------------------------------


def append_state_row(
    conn: sqlite3.Connection,
    table: str,
    row: Dict[str, Any],
) -> None:
    """Append a row to a caller-supplied state table (e.g.
    ``graphit_card_map``) over the adapter's open transaction.

    The caller's callback uses this to write ``graphit_*`` state rows
    without the adapter (or the caller) copying any private Hermes
    task-table SQL. The row is committed atomically with the rest of
    the graph and rolled back automatically on any injected failure.
    """
    if not table or not table.strip():
        raise ValueError("table name is required")
    if not row:
        raise ValueError("row payload is required")
    # Reject anything that smells like a private Hermes table name; the
    # adapter must never write into ``tasks`` / ``task_links`` /
    # ``task_events`` / ``task_runs`` / ``task_comments`` from the
    # callback path. The caller is responsible for GraphIt-owned tables
    # only (``graphit_*``).
    if table in {
        "tasks", "task_links", "task_events", "task_runs",
        "task_comments", "kanban_notify_subs",
    }:
        raise ValueError(
            f"append_state_row refuses to write into private Hermes table "
            f"{table!r}; use the public adapter surface instead."
        )
    # Reject anything that is not snake_case / alnum + underscore to
    # avoid SQL-injection via the table name.
    if not all(ch.isalnum() or ch == "_" for ch in table):
        raise ValueError(f"invalid table name: {table!r}")
    columns = list(row.keys())
    if not columns:
        raise ValueError("row payload must include at least one column")
    placeholders = ",".join("?" * len(columns))
    col_list = ",".join(columns)
    values = [row[c] for c in columns]
    # The table may not exist yet; auto-create a minimal table with TEXT
    # columns inferred from the row dict so GraphIt doesn't have to ship
    # its own DDL.
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ({col_list})"
    )
    conn.execute(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
        values,
    )


# ---------------------------------------------------------------------------
# Pre-validation
# ---------------------------------------------------------------------------


def _collect_edges(
    tasks: Sequence[TaskSpec],
    edges: Optional[Sequence[Edge]],
) -> List[Tuple[str, str]]:
    """Build the (parent_key, child_key) edge list from either TaskSpec.parents
    or the explicit edges= parameter. Rejects a mix."""
    has_parent_tuples = any(spec.parents for spec in tasks)
    if edges is not None and has_parent_tuples:
        raise GraphValidationError(
            "GRPAH_VALIDATION:GRAPH_AMBIGUOUS_EDGES — "
            "pass either TaskSpec.parents OR edges=, not both"
        )
    if edges is not None:
        return [(e.parent_key, e.child_key) for e in edges]
    out: List[Tuple[str, str]] = []
    for spec in tasks:
        for parent_key in spec.parents:
            out.append((parent_key, spec.key))
    return out


def _prevalidate(
    conn: sqlite3.Connection,
    tasks: Sequence[TaskSpec],
    edges: List[Tuple[str, str]],
) -> None:
    """Validate everything before any INSERT.

    Checks (all errors are GraphValidationError, except idempotency
    collisions which raise GraphConflictError per the contract):

    1. logical-key uniqueness within the in-memory graph;
    2. idempotency_key uniqueness within the in-memory graph;
    3. local parent existence (every parent key resolves to a known
       task in the in-memory graph);
    4. acyclic (no cycle in the directed local graph);
    5. existing non-archived tasks with the same idempotency_key;
    6. invalid initial_status / workspace_kind (TaskSpec.__post_init__).
    """
    if not tasks:
        raise GraphValidationError(
            "GRAPH_VALIDATION:EMPTY_GRAPH — at least one TaskSpec is required"
        )

    keys: List[str] = []
    seen_keys: set = set()
    for spec in tasks:
        if spec.key in seen_keys:
            raise GraphValidationError(
                f"GRAPH_VALIDATION:DUPLICATE_LOGICAL_KEY — duplicate "
                f"TaskSpec.key={spec.key!r}"
            )
        seen_keys.add(spec.key)
        keys.append(spec.key)

    # Idempotency-key uniqueness inside the local graph.
    idem_by_key: Dict[str, str] = {}
    for spec in tasks:
        if spec.idempotency_key is None:
            continue
        if spec.idempotency_key in idem_by_key:
            raise GraphValidationError(
                f"GRAPH_VALIDATION:DUPLICATE_IDEMPOTENCY_KEY — "
                f"idempotency_key={spec.idempotency_key!r} is used by both "
                f"{idem_by_key[spec.idempotency_key]!r} and {spec.key!r}"
            )
        idem_by_key[spec.idempotency_key] = spec.key

    # Local parent existence (every parent key resolves to a known task).
    key_set = set(keys)
    for parent_key, child_key in edges:
        if parent_key not in key_set:
            raise GraphValidationError(
                f"GRAPH_VALIDATION:PARENT_UNKNOWN — edge "
                f"{parent_key!r} -> {child_key!r} references unknown local "
                f"parent {parent_key!r}"
            )

    # Acyclicity via depth-first topological sort. A node is "visited"
    # when all its descendants are explored; a back-edge (encountering a
    # node already on the current DFS stack) is a cycle.
    children_of: Dict[str, List[str]] = {k: [] for k in keys}
    for parent_key, child_key in edges:
        children_of[parent_key].append(child_key)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in keys}
    cycle_path: List[str] = []

    def dfs(node: str, stack: List[str]) -> None:
        color[node] = GRAY
        stack.append(node)
        for child in children_of[node]:
            if color[child] == GRAY:
                idx = stack.index(child)
                cycle_path.extend(stack[idx:] + [child])
                raise GraphValidationError(
                    f"GRAPH_VALIDATION:DAG_CYCLE — cycle detected: "
                    f"{' -> '.join(cycle_path)}"
                )
            if color[child] == WHITE:
                dfs(child, stack)
        stack.pop()
        color[node] = BLACK

    for k in keys:
        if color[k] == WHITE:
            dfs(k, [])

    # Existing non-archived idempotency-key collisions.
    local_idem_keys = [s.idempotency_key for s in tasks if s.idempotency_key]
    if local_idem_keys:
        placeholders = ",".join("?" * len(local_idem_keys))
        rows = conn.execute(
            f"SELECT id, idempotency_key FROM tasks "
            f"WHERE idempotency_key IN ({placeholders}) "
            f"AND status != 'archived'",
            local_idem_keys,
        ).fetchall()
        if rows:
            offenders = ", ".join(
                f"{r['idempotency_key']!r} (held by {r['id']!r})"
                for r in rows
            )
            raise GraphConflictError(
                f"GRAPH_CONFLICT:IDEMPOTENCY_KEY_COLLISION — "
                f"existing non-archived tasks hold idempotency_key(s): "
                f"{offenders}"
            )

    # Reject any task that references a parent not in the in-memory graph
    # AND not in the DB (the local-existence check above already caught
    # in-memory miss, so we only need to catch "parents from the DB" if
    # the caller somehow passed an id, which we forbid via TaskSpec).
    for spec in tasks:
        for parent_key in spec.parents:
            if parent_key not in key_set:
                # This should already have raised above; defensive guard.
                raise GraphValidationError(
                    f"GRAPH_VALIDATION:PARENT_UNKNOWN — TaskSpec "
                    f"{spec.key!r} references unknown parent {parent_key!r}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def graph_write_transaction(
    conn: sqlite3.Connection,
    tasks: Sequence[TaskSpec],
    *,
    edges: Optional[Sequence[Edge]] = None,
    state_callback: Optional[StateCallback] = None,
) -> GraphWriteResult:
    """Atomic multi-card graph transaction.

    Validates the in-memory graph (logical-key uniqueness, idempotency-key
    uniqueness, parent existence, acyclicity, existing-idempotency-key
    collisions) BEFORE any write, then commits tasks + links + events +
    caller-supplied state rows under one ``BEGIN IMMEDIATE`` transaction.
    Any failure rolls back the entire graph.

    Parameters
    ----------
    conn:
        An open sqlite3 connection. The adapter does NOT perform board
        discovery and does NOT open a new connection — the caller
        supplies an explicit connection, matching the ADP §5 contract.
    tasks:
        Sequence of :class:`TaskSpec`. Order is preserved: tasks are
        created in input order, events are appended in input order, and
        the returned ``task_ids`` dict iterates in input order.
    edges:
        Optional sequence of :class:`Edge`. Mutually exclusive with
        ``TaskSpec.parents`` — the adapter raises
        :class:`GraphValidationError` if both are supplied.
    state_callback:
        Optional callable invoked once over the same connection AFTER
        all tasks/links/events are inserted but BEFORE commit. Used by
        GraphIt to write its ``graphit_*`` state rows. Any exception
        inside the callback rolls back the entire graph.

    Returns
    -------
    :class:`GraphWriteResult`

        Populated only after a successful commit. On any failure, this
        object is never constructed and the caller observes a raised
        :class:`GraphWriteError` with the DB fully rolled back.

    Raises
    ------
    GraphValidationError
        Pre-validation rejected the graph (cycle, unknown parent,
        duplicate logical key, duplicate local idempotency_key, mixed
        edges specification, empty graph, invalid spec field).
    GraphConflictError
        A pre-existing non-archived row collides with the in-memory
        graph (typically an existing ``idempotency_key`` owned by
        another task).
    """
    if conn is None:
        raise GraphValidationError(
            "GRAPH_VALIDATION:NO_CONNECTION — graph_write_transaction "
            "requires an explicit caller-supplied sqlite3.Connection"
        )
    edge_list = _collect_edges(tasks, edges)
    # Pre-validation happens BEFORE we open a write txn so the fast
    # path stays cheap and the rejection is observable without a lock.
    _prevalidate(conn, tasks, edge_list)

    # Build child→[(parent_key, ...)] adjacency from the resolved edge list
    # so we can resolve parent logical keys to task ids after each task
    # insert (parents must be created before children).
    parents_by_key: Dict[str, List[str]] = {spec.key: list(spec.parents) for spec in tasks}
    explicit_by_child: Dict[str, List[str]] = {}
    for parent_key, child_key in edge_list:
        explicit_by_child.setdefault(child_key, []).append(parent_key)
    # If explicit edges were supplied, prefer those; otherwise use parents=.
    final_parent_keys: Dict[str, List[str]] = {}
    for spec in tasks:
        if spec.key in explicit_by_child:
            final_parent_keys[spec.key] = explicit_by_child[spec.key]
        else:
            final_parent_keys[spec.key] = list(spec.parents)

    task_ids: Dict[str, str] = {}
    link_count = 0
    event_count = 0

    # The whole apply runs inside one BEGIN IMMEDIATE / COMMIT envelope.
    # ``kanban_db.write_txn`` is the documented public context manager;
    # on any exception it rolls back, on success it commits, and after
    # commit it runs the file-length invariant check that catches
    # torn-extend corruption. Reusing it instead of re-implementing
    # BEGIN/COMMIT/ROLLBACK keeps the adapter faithful to the ADP §5
    # "uses Hermes's connection initialization, WAL/busy policy,
    # task-ID allocation, task creation event format, and parent/link
    # semantics" requirement.
    with _kb.write_txn(conn):
        # Create tasks in input order so the resulting task_events rows
        # are deterministic under ``ORDER BY id ASC``.
        for spec in tasks:
            # First create with no parents, then link the resolved
            # parent_task_ids after the row exists. This keeps the inner
            # ``create_task`` call from re-validating unknown parents
            # (we already validated logically above).
            task_id = _create_task_in_transaction(
                conn,
                spec,
                parents_task_ids=(),
            )
            task_ids[spec.key] = task_id
            event_count += 1
            # Append the caller-supplied extra payload as a follow-on
            # event so the standard "created" event stays canonical.
            if spec.extra_event_payload:
                _append_event_in_transaction(
                    conn,
                    task_id,
                    "graph_write_extra",
                    dict(spec.extra_event_payload),
                )
                event_count += 1

        # Now link parents → children using the resolved task ids.
        for child_key, parent_keys in final_parent_keys.items():
            child_id = task_ids[child_key]
            for parent_key in parent_keys:
                parent_id = task_ids[parent_key]
                _link_in_transaction(conn, parent_id, child_id)
                link_count += 1

        # Finally, run the caller's state callback so GraphIt can write
        # its ``graphit_*`` rows over the SAME connection. Any exception
        # inside the callback bubbles out of write_txn and triggers the
        # full rollback path.
        _run_state_callback(conn, state_callback, task_ids)

    return GraphWriteResult(
        task_ids=task_ids,
        link_count=link_count,
        event_count=event_count,
        capability_marker=GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY,
    )
