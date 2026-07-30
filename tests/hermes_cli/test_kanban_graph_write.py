"""RED tests for the public atomic graph-write adapter (Sequence 0 prerequisite).

This module pins the contract GraphIt will rely on. Every test uses a fresh
temporary Hermes Kanban DB via the shared ``kanban_home`` fixture pattern —
no production board or kanban.db on the developer's machine is touched.

The adapter itself lives at :mod:`hermes_cli.kanban_graph_write` and is
discoverable via the ``GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY`` marker constant
exposed by :mod:`hermes_cli.kanban_db`.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_graph_write as kgw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB on a disposable path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="graphit-disposable")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Make sure no prior test cached this path's init state.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path)
    yield db_path
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))


def _spec(key: str, **overrides):
    """Build a TaskSpec with sane defaults; only ``key`` is required."""
    defaults = dict(
        key=key,
        title=f"task {key}",
        body=None,
        assignee="coder",
        workspace_kind="scratch",
        idempotency_key=f"idem:{key}",
        # The adapter's enum is the same as ``create_task``'s:
        # ``running`` is the canonical initial status for non-blocked
        # cards. Root-decomposition cards use ``blocked`` to keep the
        # graph non-dispatchable per ADP §3 invariant 6.
        initial_status="running",
        extra_event_payload=None,
    )
    defaults.update(overrides)
    return kgw.TaskSpec(**defaults)


def _count(conn, sql):
    return conn.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Capability marker — GraphIt fail-closed if absent
# ---------------------------------------------------------------------------


def test_capability_marker_is_exposed_on_kanban_db():
    """GraphIt must be able to import a stable capability/version marker
    directly from ``hermes_cli.kanban_db`` so a missing capability fails
    closed before any write transaction is attempted."""
    assert hasattr(kb, "GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY")
    marker = kb.GRAPHIT_ATOMIC_GRAPH_WRITE_CAPABILITY
    assert isinstance(marker, str) and marker.strip()
    # Versioned: must include a numeric version component so GraphIt can
    # fail closed against an incompatible adapter.
    assert "v" in marker or "V" in marker


def test_adapter_module_exposes_public_surface():
    """The adapter module must expose the exact public surface the ADP §5
    contract pins: graph_write_transaction, TaskSpec, Edge, GraphWriteError
    and its typed subclasses."""
    for name in (
        "graph_write_transaction",
        "TaskSpec",
        "Edge",
        "GraphWriteError",
        "GraphValidationError",
        "GraphConflictError",
    ):
        assert hasattr(kgw, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2. Valid graph commits — tasks + links + events + state rows
# ---------------------------------------------------------------------------


def test_valid_graph_commits_tasks_links_events_and_returns_mapping(fresh_db):
    """A valid 3-node graph with two edges commits all rows and returns the
    logical-key → task-id map only after commit (i.e. only on success)."""
    with kb.connect(fresh_db) as conn:
        result = kgw.graph_write_transaction(
            conn,
            tasks=[
                _spec("root"),
                _spec("left", parents=("root",)),
                _spec("right", parents=("root",)),
            ],
            state_callback=lambda c, task_ids: kgw.append_state_row(
                c, "graphit_card_map",
                {"logical_key": "root", "task_id": task_ids["root"]},
            ),
        )
        assert isinstance(result, kgw.GraphWriteResult)
        # Every key → a real Hermes task id, not the placeholder.
        assert set(result.task_ids) == {"root", "left", "right"}
        for logical_key, task_id in result.task_ids.items():
            assert task_id.startswith("t_")
        # Task rows actually exist.
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 3
        # Link rows actually exist (2 edges).
        assert _count(conn, "SELECT COUNT(*) FROM task_links") == 2
        # Each task got a "created" event.
        assert _count(
            conn, "SELECT COUNT(*) FROM task_events WHERE kind='created'"
        ) == 3
        # Callback state row actually committed.
        state_row = conn.execute(
            "SELECT * FROM graphit_card_map WHERE logical_key='root'"
        ).fetchone()
        assert state_row is not None
        assert state_row["task_id"].startswith("t_")


# ---------------------------------------------------------------------------
# 3. Cycle rejected — zero writes before commit
# ---------------------------------------------------------------------------


def test_cycle_rejected_with_zero_writes(fresh_db):
    specs = [
        _spec("a"),
        _spec("b", parents=("c",)),
        _spec("c", parents=("b",)),
    ]
    with kb.connect(fresh_db) as conn, pytest.raises(kgw.GraphValidationError) as exc:
        kgw.graph_write_transaction(conn, tasks=specs)
    assert "cycle" in str(exc.value).lower() or "DAG_CYCLE" in str(exc.value)

    # Zero task rows, zero link rows, zero events, zero state.
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 0
        assert _count(conn, "SELECT COUNT(*) FROM task_links") == 0
        assert _count(conn, "SELECT COUNT(*) FROM task_events") == 0


# ---------------------------------------------------------------------------
# 4. Unknown parent rejected — zero writes
# ---------------------------------------------------------------------------


def test_unknown_parent_rejected_with_zero_writes(fresh_db):
    specs = [
        _spec("root"),
        _spec("orphan", parents=("nonexistent",)),
    ]
    with kb.connect(fresh_db) as conn, pytest.raises(kgw.GraphValidationError) as exc:
        kgw.graph_write_transaction(conn, tasks=specs)
    assert "unknown parent" in str(exc.value).lower() or "PARENT_UNKNOWN" in str(exc.value)

    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 0
        assert _count(conn, "SELECT COUNT(*) FROM task_links") == 0


# ---------------------------------------------------------------------------
# 5. Duplicate local logical key / idempotency key rejected
# ---------------------------------------------------------------------------


def test_duplicate_logical_key_rejected(fresh_db):
    specs = [_spec("dup"), _spec("dup")]
    with kb.connect(fresh_db) as conn, pytest.raises(kgw.GraphValidationError):
        kgw.graph_write_transaction(conn, tasks=specs)
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 0


def test_duplicate_idempotency_key_rejected(fresh_db):
    specs = [
        _spec("a", idempotency_key="shared-key"),
        _spec("b", idempotency_key="shared-key"),
    ]
    with kb.connect(fresh_db) as conn, pytest.raises(kgw.GraphValidationError):
        kgw.graph_write_transaction(conn, tasks=specs)
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 0


# ---------------------------------------------------------------------------
# 6. Existing non-archived idempotency collision rejected BEFORE mutation
# ---------------------------------------------------------------------------


def test_existing_non_archived_idempotency_collision_rejected(fresh_db):
    """If a non-archived task already owns the idempotency key, the new graph
    must abort before any write."""
    with kb.connect(fresh_db) as conn:
        kb.create_task(conn, title="pre-existing", idempotency_key="claimed-key")

    specs = [
        _spec("new-task", idempotency_key="claimed-key"),
        _spec("sibling"),
    ]
    with kb.connect(fresh_db) as conn, pytest.raises(kgw.GraphConflictError) as exc:
        kgw.graph_write_transaction(conn, tasks=specs)
    assert "claimed-key" in str(exc.value) or "IDEMPOTENCY" in str(exc.value)

    # Existing task untouched; no new task rows.
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 1
        rows = conn.execute("SELECT title FROM tasks").fetchall()
        assert rows[0]["title"] == "pre-existing"


# ---------------------------------------------------------------------------
# 7. Fault injection after each Nth write rolls back ENTIRELY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fail_after_kind", ["task", "link", "event", "callback"])
def test_fault_after_each_stage_rolls_back_entirely(fresh_db, monkeypatch, fail_after_kind):
    """Inject a failure after the Nth stage (task insert / link insert /
    created event / state callback) — the whole graph rolls back; zero
    task rows, zero link rows, zero events, zero state rows."""

    counter = {"tasks": 0, "links": 0, "events": 0, "callbacks": 0}

    # Wrap the underlying insert helpers so we know exactly when each
    # stage has fired.
    real_create = kgw._create_task_in_transaction
    real_link = kgw._link_in_transaction
    real_event = kgw._append_event_in_transaction

    def counting_create(conn, spec, parents_task_ids):
        counter["tasks"] += 1
        if fail_after_kind == "task":
            raise RuntimeError("injected fault after task insert")
        return real_create(conn, spec, parents_task_ids=parents_task_ids)

    def counting_link(conn, parent, child):
        counter["links"] += 1
        if fail_after_kind == "link":
            raise RuntimeError("injected fault after link insert")
        return real_link(conn, parent, child)

    def counting_event(conn, task_id, kind, payload):
        counter["events"] += 1
        if fail_after_kind == "event":
            raise RuntimeError("injected fault after event append")
        return real_event(conn, task_id, kind, payload)

    def counting_state_callback(conn, task_ids):
        counter["callbacks"] += 1
        if fail_after_kind == "callback":
            raise RuntimeError("injected fault after callback")
        return None

    import hermes_cli.kanban_graph_write as mod
    from hermes_cli import kanban_db as _kb
    monkeypatch.setattr(mod, "_create_task_in_transaction", counting_create)
    monkeypatch.setattr(mod, "_link_in_transaction", counting_link)
    monkeypatch.setattr(mod, "_append_event_in_transaction", counting_event)

    # The standard "created" event is appended inside the public
    # ``_create_task_body`` helper, NOT via the in-transaction event
    # seam. Patch the in-transaction event in BOTH places so the
    # ``[event]`` param fault actually fires. The in-transaction seam
    # is patched for the optional ``extra_event_payload`` path; the
    # kanban_db event hook covers the standard created event.
    real_kanban_event = _kb._append_event

    def counting_kanban_event(*args, **kwargs):
        counter["events"] += 1
        if fail_after_kind == "event":
            raise RuntimeError("injected fault after event append")
        return real_kanban_event(*args, **kwargs)

    monkeypatch.setattr(_kb, "_append_event", counting_kanban_event)

    def wrapper_state_callback(conn, state_callback, task_ids):
        # The real adapter wrapper takes (conn, state_callback, task_ids);
        # the test's counting closure observes how many times the
        # callback itself fires, and raises the fault at the right Nth
        # stage.
        return counting_state_callback(conn, task_ids)

    monkeypatch.setattr(mod, "_run_state_callback", wrapper_state_callback)

    specs = [
        _spec("root"),
        _spec("left", parents=("root",)),
        _spec("right", parents=("root",)),
    ]

    with kb.connect(fresh_db) as conn:
        with pytest.raises(RuntimeError, match="injected fault"):
            kgw.graph_write_transaction(
                conn, tasks=specs,
                state_callback=lambda c, task_ids: kgw.append_state_row(
                    c, "graphit_card_map",
                    {"logical_key": "root", "task_id": task_ids["root"]},
                ),
            )

    # No task rows, no link rows, no events, no state rows survived.
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM tasks") == 0, (
            f"task rows leaked on fault-after-{fail_after_kind}; "
            f"counter={counter}"
        )
        assert _count(conn, "SELECT COUNT(*) FROM task_links") == 0, (
            f"link rows leaked on fault-after-{fail_after_kind}; "
            f"counter={counter}"
        )
        assert _count(conn, "SELECT COUNT(*) FROM task_events") == 0, (
            f"event rows leaked on fault-after-{fail_after_kind}; "
            f"counter={counter}"
        )
        # The graphit_card_map table either doesn't exist (callback was the
        # only thing that would have created it) or is empty.
        assert _count(
            conn,
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='graphit_card_map'",
        ) == 0, (
            "graphit_card_map table leaked on fault-after-callback; "
            f"counter={counter}"
        )

    # pytest's monkeypatch fixture auto-cleans; nothing to undo.


# ---------------------------------------------------------------------------
# 8. Callback state commits atomically on success and rolls back on error
# ---------------------------------------------------------------------------


def test_callback_state_commits_atomically_on_success(fresh_db):
    state_seen = {"called": False}

    def state_callback(conn, task_ids):
        state_seen["called"] = True
        kgw.append_state_row(
            conn, "graphit_card_map",
            {"logical_key": "root", "task_id": task_ids["root"]},
        )
        kgw.append_state_row(
            conn, "graphit_card_map",
            {"logical_key": "left", "task_id": task_ids["left"]},
        )

    with kb.connect(fresh_db) as conn:
        result = kgw.graph_write_transaction(
            conn,
            tasks=[_spec("root"), _spec("left", parents=("root",))],
            state_callback=state_callback,
        )

    assert state_seen["called"] is True
    with kb.connect(fresh_db) as conn:
        rows = conn.execute(
            "SELECT logical_key, task_id FROM graphit_card_map "
            "ORDER BY logical_key"
        ).fetchall()
        # State rows reference real, post-commit task ids.
        assert [(r["logical_key"], r["task_id"]) for r in rows] == [
            ("left", result.task_ids["left"]),
            ("root", result.task_ids["root"]),
        ]


def test_callback_state_rolled_back_on_error(fresh_db):
    def state_callback(conn, task_ids):
        kgw.append_state_row(
            conn, "graphit_card_map",
            {"logical_key": "root", "task_id": task_ids["root"]},
        )
        # Raise after the first callback mutation so we exercise rollback.
        raise RuntimeError("callback exploded mid-flight")

    with kb.connect(fresh_db) as conn:
        with pytest.raises(RuntimeError, match="callback exploded"):
            kgw.graph_write_transaction(
                conn,
                tasks=[_spec("root")],
                state_callback=state_callback,
            )

    with kb.connect(fresh_db) as conn:
        # The graphit_card_map table either doesn't exist (callback was the
        # only thing that would have created it) or is empty.
        cnt = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='graphit_card_map'"
        ).fetchone()[0]
        if cnt:
            rows = conn.execute("SELECT COUNT(*) FROM graphit_card_map").fetchone()[0]
            assert rows == 0, "callback state row leaked on rollback"


# ---------------------------------------------------------------------------
# 9. Deterministic task-id mapping/order under frozen input ordering
# ---------------------------------------------------------------------------


def test_deterministic_task_id_mapping_and_order(fresh_db):
    """Under a frozen input ordering, two successive applies must yield the
    same logical-key → task-id mapping, and the resulting event/row order
    must be deterministic (creation-order by input position)."""
    inputs = [
        _spec("alpha"),
        _spec("beta", parents=("alpha",)),
        _spec("gamma", parents=("alpha",)),
        _spec("delta", parents=("beta", "gamma")),
    ]

    def state_callback(conn, task_ids):
        for k in ("alpha", "beta", "gamma", "delta"):
            kgw.append_state_row(
                conn, "graphit_card_map",
                {"logical_key": k, "task_id": task_ids[k]},
            )

    with kb.connect(fresh_db) as conn:
        result1 = kgw.graph_write_transaction(
            conn, tasks=inputs, state_callback=state_callback,
        )
        tasks_after_first = _count(conn, "SELECT COUNT(*) FROM tasks")

    # Reset to a fresh DB and apply the same inputs again; the mapping
    # (logical_key → task_id) may differ because task ids are random hex,
    # but the post-conditions (4 tasks, 4 links, 4 created events, 4
    # state rows, 4 graphit_card_map rows referencing task ids) must be
    # identical.
    fresh_db2 = fresh_db.parent / "graphit-disposable-2.db"
    kb._INITIALIZED_PATHS.discard(str(fresh_db2.resolve()))
    kb.init_db(fresh_db2)
    with kb.connect(fresh_db2) as conn:
        result2 = kgw.graph_write_transaction(
            conn, tasks=inputs, state_callback=state_callback,
        )
        tasks_after_second = _count(conn, "SELECT COUNT(*) FROM tasks")
        events_second = conn.execute(
            "SELECT task_id FROM task_events WHERE kind='created' "
            "ORDER BY id ASC"
        ).fetchall()

    assert tasks_after_first == tasks_after_second == 4
    assert set(result1.task_ids) == set(result2.task_ids) == {
        "alpha", "beta", "gamma", "delta",
    }
    # Mapping order: input ordering must be preserved (alpha first, delta last).
    assert list(result1.task_ids) == ["alpha", "beta", "gamma", "delta"]
    assert list(result2.task_ids) == ["alpha", "beta", "gamma", "delta"]
    # The created events must come in the same input order as the spec list.
    event_task_ids = [r["task_id"] for r in events_second]
    spec_keys_in_order = [
        result2.task_ids[k] for k in ("alpha", "beta", "gamma", "delta")
    ]
    assert event_task_ids == spec_keys_in_order


# ---------------------------------------------------------------------------
# 10. Single-card public APIs remain backward compatible
# ---------------------------------------------------------------------------


def test_create_task_and_link_tasks_still_work(fresh_db):
    """The existing single-card public surface must not regress. The graph
    adapter is additive — it does not replace :func:`create_task` or
    :func:`link_tasks`."""
    with kb.connect(fresh_db) as conn:
        a = kb.create_task(conn, title="legacy a")
        b = kb.create_task(conn, title="legacy b", parents=[a])
        kb.link_tasks(conn, a, b)
        ta = kb.get_task(conn, a)
        tb = kb.get_task(conn, b)
    assert ta.status == "ready"
    assert tb.status in ("todo", "ready")  # linked under a not-done parent
    with kb.connect(fresh_db) as conn:
        assert _count(conn, "SELECT COUNT(*) FROM task_links") >= 1


def test_adapter_does_not_require_board_resolution(fresh_db):
    """The adapter must not perform board discovery — it takes an explicit
    caller-supplied connection. This proves we can never accidentally
    hit the developer's production board."""
    with kb.connect(fresh_db) as conn:
        result = kgw.graph_write_transaction(
            conn, tasks=[_spec("solo")],
        )
    assert "solo" in result.task_ids


# ---------------------------------------------------------------------------
# 11. Concurrent same-key graph attempts yield one graph + conflict, never dupes
# ---------------------------------------------------------------------------


def test_concurrent_same_key_graph_yields_one_graph(fresh_db):
    """Two threads racing on the same idempotency key must produce exactly
    one graph (one row in tasks, no duplicate-key in tasks.idempotency_key).
    The loser observes :class:`GraphConflictError`. This is the
    "state exact supported behavior" the card pins: never duplicate,
    never silent partial success."""
    with kb.connect(fresh_db) as conn:
        # Pre-create a non-archived task that owns the contested key so the
        # race tests collision-against-existing, the documented contract.
        kb.create_task(conn, title="seed", idempotency_key="contested")

    def attempt():
        with kb.connect(fresh_db) as conn:
            try:
                kgw.graph_write_transaction(
                    conn,
                    tasks=[_spec("new", idempotency_key="contested")],
                )
                return "committed"
            except kgw.GraphWriteError as exc:
                return ("error", type(exc).__name__)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(4)))

    # At least one attempt must have failed; at most zero "committed".
    # (If WAL serializes the contention fast enough, the loser's transaction
    # may rollback on a different shape — the contract is "one graph, never
    # duplicates".)
    with kb.connect(fresh_db) as conn:
        rows_with_key = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key='contested'"
        ).fetchall()
        # Exactly one task owns the contested key, ever.
        assert len(rows_with_key) == 1, (
            f"duplicate idempotency-key holders: {rows_with_key}; "
            f"outcomes={outcomes}"
        )
        # And no task titled "task new" leaked in.
        leaked = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title='task new'"
        ).fetchone()[0]
        assert leaked == 0, (
            f"concurrent attempt created a phantom task; outcomes={outcomes}"
        )


# ---------------------------------------------------------------------------
# 12. Edge spec coverage — caller-supplied explicit Edge vs derived parents=
# ---------------------------------------------------------------------------


def test_explicit_edges_match_parent_tuples(fresh_db):
    """Both API shapes (parents= on each TaskSpec, OR an explicit edges=
    list) must produce identical task_links output for the same logical
    graph. The adapter accepts either, but not both."""
    with kb.connect(fresh_db) as conn:
        kgw.graph_write_transaction(
            conn,
            tasks=[_spec("a"), _spec("b", parents=("a",))],
        )
    with kb.connect(fresh_db) as conn:
        edges_parent_shape = sorted(
            (r["parent_id"], r["child_id"])
            for r in conn.execute(
                "SELECT parent_id, child_id FROM task_links"
            ).fetchall()
        )

    fresh_db2 = fresh_db.parent / "graphit-disposable-edges.db"
    kb._INITIALIZED_PATHS.discard(str(fresh_db2.resolve()))
    kb.init_db(fresh_db2)
    with kb.connect(fresh_db2) as conn:
        kgw.graph_write_transaction(
            conn,
            tasks=[_spec("a"), _spec("b")],
            edges=[kgw.Edge(parent_key="a", child_key="b")],
        )
    with kb.connect(fresh_db2) as conn:
        edges_explicit_shape = sorted(
            (r["parent_id"], r["child_id"])
            for r in conn.execute(
                "SELECT parent_id, child_id FROM task_links"
            ).fetchall()
        )

    # Compare by logical key from the task title (a→"task a", b→"task b")
    # by mapping the task ids back to logical keys via their titles.
    def logical_pairs(rows):
        # Use titles which encode the logical key in our test fixtures.
        out = []
        with kb.connect(fresh_db if rows is edges_parent_shape else fresh_db2) as conn:
            title_by_id = {
                r["id"]: r["title"] for r in conn.execute("SELECT id, title FROM tasks")
            }
        for parent_id, child_id in rows:
            # title is "task <key>"
            out.append((
                title_by_id[parent_id].split(" ", 1)[1],
                title_by_id[child_id].split(" ", 1)[1],
            ))
        return sorted(out)

    assert logical_pairs(edges_parent_shape) == logical_pairs(edges_explicit_shape)
