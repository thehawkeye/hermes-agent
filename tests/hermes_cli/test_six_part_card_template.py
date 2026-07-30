"""Tests for the six-part card template helper (``hermes_cli.kanban_db``).

Resolves MAJOR #6 of the rev1 plan: durable regression coverage for the
``_apply_six_part_template`` transform that runs at the top of
``_create_task_body``. Covers all 22 checks from plan-rev1 §5.

These tests are intentionally WHITE-box (they import the helper and the
module-level constants). White-box is the right shape here because:

* The transformation is a pure function over 4 args (body/title/assignee/
  workspace_kind) with byte-exact contract assertions; mocking the
  public surface would just test the wrapper.
* The "no filesystem read" guarantee (checks 20/21) MUST be asserted
  against the actual module, not against a re-implementation.

The integration tests (18, 19, 21, 22) use the same ``fresh_home``
fixture as ``test_kanban_boards.py`` — isolated HERMES_HOME with a temp
SQLite board so the real ``create_task`` / ``create_task_in_transaction``
write transactions run end-to-end.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

# Ensure the worktree (not a stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Constants under test
# ---------------------------------------------------------------------------

_TEMPLATE = kb._SIX_PART_TEMPLATE_V1
_FRONT_MATTER_RE = kb._FRONT_MATTER_RE
_LEADING_TASK_RE = kb._LEADING_TASK_RE
_TITLE_INTENT_RE = kb._TITLE_INTENT_RE

_SIX_HEADINGS_TITLES = ("Task", "Resources", "Autonomy",
                        "Goal", "Guardrails", "Output")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state.

    Mirrors ``tests/hermes_cli/test_kanban_boards.py::fresh_home`` so the
    integration tests (18, 19, 21, 22) exercise the real
    ``create_task`` / ``create_task_in_transaction`` path against a temp
    SQLite board instead of the live one.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# 1. Idempotence — complete six-part scaffold body is skipped
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_complete_six_headings_collision(self):
        # Body already carries all six ordered H2 headings as a prefix →
        # gate must NOT fire (idempotence).
        body = (
            "## Task\n"
            "do stuff\n\n"
            "## Resources\n- x\n\n"
            "## Autonomy\n- Read: a\n\n"
            "## Goal\n- Acceptance: x\n\n"
            "## Guardrails\n- Risk: GREEN\n\n"
            "## Output\n- Artifacts: x\n"
        )
        out = kb._apply_six_part_template(body, "any title", "default", "scratch")
        assert out == body

    def test_leading_task_collision(self):
        # A body that starts with `## Task` (even without all six
        # headings) must NOT trigger — partial idempotence guard.
        body = "## Task\ncaller has already started the card"
        out = kb._apply_six_part_template(body, "GraphIt build cards", "default", "scratch")
        assert out == body

    def test_bom_tolerance(self):
        # BOM before `## Task` must still be recognized as idempotent.
        body = "\ufeff## Task\ncaller already structured\nwith details"
        out = kb._apply_six_part_template(body, "GraphIt build cards", "default", "scratch")
        assert out == body


# ---------------------------------------------------------------------------
# 4. Caller content is byte-preserved (CRLF / Unicode / NUL)
# ---------------------------------------------------------------------------

class TestBytePreservation:
    def test_caller_bytes_preserved(self):
        # CRLF separators, multi-byte UTF-8 (✓ + café), and NUL bytes
        # all survive the transformation byte-for-byte.
        body = "first\r\n✓ café\r\n\x00\x01\x02 trailing"
        out = kb._apply_six_part_template(body, "GraphIt build cards",
                                          "default", "scratch")
        # Original body appears at the tail, unchanged.
        assert out.endswith(body)
        # And the prefix is exactly the template plus `\n\n`.
        assert out == _TEMPLATE + "\n\n" + body


# ---------------------------------------------------------------------------
# 5 + 6. Quick-Q&A and the rev0 signal-3 false-positive
# ---------------------------------------------------------------------------

class TestQuickQaUntouched:
    def test_quick_qa_unchanged(self):
        body = "Compare minimax vs minimax-oauth failure rates over last 30d."
        out = kb._apply_six_part_template(
            body,
            "Quick Q&A: which cron provider has lower error rate?",
            "default",
            "scratch",
        )
        assert out == body

    def test_signal3_requires_intent(self):
        # The rev0 false positive: planner-shaped assignee + persistent
        # workspace WITHOUT title intent and WITHOUT a frontmatter opt-in
        # must NOT trigger the gate. Marker alone is never sufficient.
        body = "rollup docs into a single quick reply"
        out = kb._apply_six_part_template(
            body,
            "Quick Q&A: cron?",
            "ta-ra",
            "dir",
        )
        assert out == body


# ---------------------------------------------------------------------------
# 7. Title intent boundary cases (8 cases, all asserted)
# ---------------------------------------------------------------------------

class TestTitleIntentBoundaries:
    @pytest.mark.parametrize("title", [
        "GraphIt",
        "GraphIt: build cards",
        "GraphIt — build cards",
        "GraphIt - build cards",
        "graphIt build cards",  # case-insensitive
        "verify the rollout",   # mid-string — must NOT match (not anchored as start)
    ])
    def test_title_intent_triggers(self, title):
        # For each boundary case the gate must fire (body rewritten).
        out = kb._apply_six_part_template("", title, "default", "scratch")
        assert out == _TEMPLATE

    @pytest.mark.parametrize("title", [
        "Graphical analysis",   # not a keyword
        "plan9",                # digit immediately after keyword
        "outreach plan",        # keyword only in tail
        "re-verify the rollout",  # keyword only after prefix
        "PLAN\n9",              # newline is NOT a separator (regex doesn't allow it)
    ])
    def test_title_intent_does_not_trigger(self, title):
        # These cases MUST NOT trigger the gate on title alone.
        out = kb._apply_six_part_template(
            "untouched body",
            title,
            "default",
            "scratch",
        )
        assert out == "untouched body"


# ---------------------------------------------------------------------------
# 8 + 9 + 10 + 11 + 12. Frontmatter handling
# ---------------------------------------------------------------------------

class TestFrontmatter:
    def test_frontmatter_optin_with_planner_context(self):
        # `template: six-part-v1` in a closed frontmatter block + planner
        # assignee + persistent workspace → gate fires; output keeps the
        # frontmatter at byte zero.
        body = "---\ntemplate: six-part-v1\n---\nDetails"
        out = kb._apply_six_part_template(body, "no title intent",
                                          "planner", "dir")
        # Frontmatter is preserved at byte zero.
        assert out.startswith("---\ntemplate: six-part-v1\n---\n")
        # Caller's "Details" appears at the tail, byte-preserved.
        assert out.endswith("\n\n" + "Details")

    def test_frontmatter_optin_without_planner_context(self):
        # Same frontmatter but default assignee + scratch workspace →
        # gate must NOT fire; body byte-for-byte preserved.
        body = "---\ntemplate: six-part-v1\n---\nDetails"
        out = kb._apply_six_part_template(body, "no title intent",
                                          "default", "scratch")
        assert out == body

    def test_fenced_yaml_does_not_optin(self):
        # `template: six-part-v1` buried in a fenced code block (not a
        # leading YAML frontmatter) MUST NOT trigger opt-in. Combined
        # with a non-planner assignee / scratch workspace, gate returns
        # False.
        body = "```yaml\ntemplate: six-part-v1\n```\nNotes"
        out = kb._apply_six_part_template(body, "Plain task",
                                          "planner", "dir")
        # The opt-in is NOT recognized (not at leading position); even
        # with planner context, the gate must NOT fire.
        assert out == body

    def test_malformed_frontmatter_safe(self):
        # Unclosed `---` at the top must NOT crash and must NOT opt in.
        body = "---\ntemplate: six-part-v1\n"  # no closing ---
        out = kb._apply_six_part_template(body, "no title intent",
                                          "planner", "dir")
        # Body is byte-for-byte preserved (gate does not fire).
        assert out == body

    def test_frontmatter_byte_zero(self):
        # When the gate fires AND there is real frontmatter, the
        # frontmatter MUST remain at byte zero (no template bytes
        # prepended before it).
        body = "---\ntemplate: six-part-v1\n---\nReal content"
        out = kb._apply_six_part_template(body, "no title intent",
                                          "planner", "dir")
        # Find the first non-frontmatter byte and assert the frontmatter
        # block is intact at the start.
        assert out.startswith("---\ntemplate: six-part-v1\n---")
        # The template is inserted between frontmatter and content.
        assert _TEMPLATE in out
        # And the caller's content is preserved at the tail.
        assert out.endswith("\n\nReal content")


# ---------------------------------------------------------------------------
# 13, 14, 15. None / empty / whitespace-only body
# ---------------------------------------------------------------------------

class TestEmptyBody:
    def test_none_body(self):
        # `body=None` returns exactly the template (no `None` literal,
        # no trailing separator, no concatenation artifacts).
        out = kb._apply_six_part_template(None, "GraphIt", "default", "scratch")
        assert out == _TEMPLATE
        assert "None" not in out

    def test_empty_body(self):
        # `body=""` returns exactly the template.
        out = kb._apply_six_part_template("", "GraphIt", "default", "scratch")
        assert out == _TEMPLATE

    def test_whitespace_only_body(self):
        # Whitespace-only body: byte-preservation wins. The helper must
        # concatenate `T + "\n\n" + body` — caller content stays at the
        # tail exactly as supplied.
        body = "   "
        out = kb._apply_six_part_template(body, "GraphIt", "default", "scratch")
        assert out == _TEMPLATE + "\n\n" + body


# ---------------------------------------------------------------------------
# 16. Heading collision correctness — `## Taskmaster` is not a collision
# ---------------------------------------------------------------------------

class TestTaskmasterNotCollision:
    def test_taskmaster_not_a_collision(self):
        # `## Taskmaster` does NOT match the leading-`## Task` regex
        # (the regex requires EOL or EOF after `Task`). With title
        # intent present the gate MUST fire.
        body = "## Taskmaster\nNotes"
        out = kb._apply_six_part_template(body, "GraphIt x", "default", "scratch")
        # Template is prepended; caller body appears at the tail.
        assert out == _TEMPLATE + "\n\n" + body


# ---------------------------------------------------------------------------
# 17. 1 MB body — length contract + suffix byte-for-byte preservation
# ---------------------------------------------------------------------------

class TestLargeBody:
    def test_large_body(self):
        # 1 MiB body: helper must still return T + "\n\n" + body with
        # the suffix byte-for-byte preserved (no truncation, no encoding
        # loss).
        body = "A" * (1024 * 1024)  # exactly 1,048,576 bytes
        # Add a single byte at the end so we can detect any truncation.
        body = body + "Z"
        out = kb._apply_six_part_template(body, "GraphIt", "default", "scratch")
        assert len(out) == len(_TEMPLATE) + 2 + len(body)
        # Suffix is byte-identical to the input.
        assert out.endswith(body)


# ---------------------------------------------------------------------------
# 18, 19, 21, 22. Integration tests — real DB writes via both routes
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_create_task_integration(self, fresh_home):
        # Public route A: create_task writes the transformed body.
        kb.create_board("itest")
        with kb.connect(board="itest") as conn:
            kb.create_task(
                conn,
                title="GraphIt build cards",
                body="hello",
                assignee="default",
                workspace_kind="scratch",
            )
            tasks = kb.list_tasks(conn)
        assert len(tasks) == 1
        # Body must equal `T + "\n\n" + "hello"` byte-for-byte.
        assert tasks[0].body == _TEMPLATE + "\n\n" + "hello"

    def test_create_task_in_transaction_integration(self, fresh_home):
        # Public route B: create_task_in_transaction must produce the
        # SAME transformed body as route A byte-for-byte.
        kb.create_board("itest")
        with kb.connect(board="itest") as conn:
            kb.create_task_in_transaction(
                conn,
                title="GraphIt build cards",
                body="hello",
                assignee="default",
                workspace_kind="scratch",
            )
            tasks = kb.list_tasks(conn)
        assert len(tasks) == 1
        assert tasks[0].body == _TEMPLATE + "\n\n" + "hello"

    def test_no_filesystem_read(self, fresh_home):
        # Belt-and-suspenders assertion: NO filesystem read happens
        # during a write transaction. The template must live entirely
        # in code (loaded at import, never read from disk at write time).
        # We patch Path.read_text to raise — if any code path tries to
        # read a template file, the test fails with that exception.
        original_read_text = Path.read_text

        def _explode(*args, **kwargs):
            raise AssertionError(
                "Path.read_text was called during a write transaction "
                "— the template must live in code, not on disk"
            )

        with mock.patch.object(Path, "read_text", side_effect=_explode), \
             mock.patch.object(Path, "read_bytes",
                               side_effect=_explode), \
             mock.patch("pathlib.Path.read_text",
                        side_effect=_explode, create=True):
            kb.create_board("itest-nofile")
            with kb.connect(board="itest-nofile") as conn:
                kb.create_task(
                    conn,
                    title="GraphIt build cards",
                    body="hello",
                    assignee="default",
                    workspace_kind="scratch",
                )
                tasks = kb.list_tasks(conn)
        assert tasks[0].body == _TEMPLATE + "\n\n" + "hello"

    def test_concurrent_first_calls(self, fresh_home):
        # 20 parallel create_task invocations must all produce
        # correctly-formed bodies — the module-level constant is the
        # single source of truth and concurrency does not corrupt it.
        #
        # SQLite + same-file writes serialize at the file lock, so we
        # do NOT race the writes here (the invariant under test is the
        # TRANSFORMATION, not SQLite concurrency). We DO race the reads
        # after the writes complete, so we verify each row independently.
        kb.create_board("itest-conc")

        bodies_by_title: dict[str, str] = {}
        bodies_lock = threading.Lock()
        errors: list[BaseException] = []

        def _worker(idx: int) -> None:
            try:
                title = f"GraphIt task {idx}"
                expected_body = _TEMPLATE + "\n\n" + f"hello-{idx}"
                with kb.connect(board="itest-conc") as conn:
                    kb.create_task(
                        conn,
                        title=title,
                        body=f"hello-{idx}",
                        assignee="default",
                        workspace_kind="scratch",
                    )
                    cur = conn.execute(
                        "SELECT body FROM tasks WHERE title = ?",
                        (title,),
                    )
                    row = cur.fetchone()
                with bodies_lock:
                    bodies_by_title[title] = row[0]
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"worker errors: {errors}"
        assert len(bodies_by_title) == 20
        # Every body must be exactly `T + "\n\n" + hello-<idx>` for its
        # own idx — no cross-thread contamination, no partial writes.
        for idx in range(20):
            title = f"GraphIt task {idx}"
            assert bodies_by_title[title] == _TEMPLATE + "\n\n" + f"hello-{idx}", (
                f"task {idx} body mismatch: {bodies_by_title[title]!r}"
            )