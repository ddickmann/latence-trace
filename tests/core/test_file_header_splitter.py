"""Unit tests for the multi-file ``raw_context`` splitter.

The splitter is the key piece that lets ``file_attribution.n_files`` and
``dead_weight_files`` populate correctly on multi-file code bundles
(Cursor / Aider / patch-style context blobs) and on enterprise RAG
contexts that label chunks by document path.
"""

from __future__ import annotations

import pytest

from latence_trace.core.groundedness import split_raw_context_by_file_headers


class TestMarkdownStyleHeaders:
    def test_hash_file_header_colon_lowercase(self) -> None:
        ctx = (
            "# file: src/auth.py\n"
            "def login(user):\n    return user.valid\n\n"
            "# file: src/db.py\n"
            "class Db:\n    pass\n\n"
            "# file: src/api.py\n"
            "def handle(req):\n    return Db.save(req)\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert len(blocks) == 3
        assert [b["path"] for b in blocks] == [
            "src/auth.py",
            "src/db.py",
            "src/api.py",
        ]
        assert "def login" in blocks[0]["text"]
        assert "class Db" in blocks[1]["text"]
        assert "def handle" in blocks[2]["text"]

    def test_capitalised_file_header(self) -> None:
        ctx = (
            "# File: a.py\nprint('a')\n\n"
            "# File: b.py\nprint('b')\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert [b["path"] for b in blocks] == ["a.py", "b.py"]

    def test_mixed_case_is_tolerated(self) -> None:
        ctx = "# FILE: x.py\nx=1\n# file: y.py\ny=2\n"
        blocks = split_raw_context_by_file_headers(ctx)
        assert [b["path"] for b in blocks] == ["x.py", "y.py"]


class TestVoyagerStyleHeaders:
    def test_triple_equals_wrapper(self) -> None:
        ctx = (
            "=== doc_001.pdf ===\n"
            "Q3 revenue was 48.2B.\n\n"
            "=== doc_002.pdf ===\n"
            "Operating margin held at 29.4%.\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert [b["path"] for b in blocks] == ["doc_001.pdf", "doc_002.pdf"]


class TestDiffStyleHeaders:
    def test_diff_a_only(self) -> None:
        ctx = (
            "--- a/src/auth.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-def login(user): pass\n"
            "+def login(user): return user.valid\n"
            "\n"
            "--- a/src/db.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-class Db: pass\n"
            "+class Db:\n    def save(self): ...\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert [b["path"] for b in blocks] == ["src/auth.py", "src/db.py"]

    def test_diff_a_and_b_same_path_dedupe_prefers_first(self) -> None:
        # A common diff emits both ``--- a/foo`` and ``+++ b/foo`` for
        # the same hunk on consecutive lines; we should still split
        # deterministically rather than producing zero-length blocks.
        ctx = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        # Two headers on consecutive lines produce two blocks; both
        # owned by foo.py which is correct for per-file attribution.
        assert {b["path"] for b in blocks} == {"foo.py"}
        # All characters accounted for.
        total = sum(b["offset_end"] - b["offset_start"] for b in blocks)
        assert total == len(ctx)


class TestFallbackAndEdgeCases:
    def test_no_headers_returns_single_block(self) -> None:
        ctx = (
            "This is a long enterprise passage. Apple Q3 revenue "
            "rose 11% to $48.2B, operating margin expanded to 29.4%, "
            "and EPS reached $2.10 vs $1.88 consensus."
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert len(blocks) == 1
        assert blocks[0]["path"] is None
        assert blocks[0]["offset_start"] == 0
        assert blocks[0]["offset_end"] == len(ctx)
        assert blocks[0]["text"] == ctx

    def test_empty_context(self) -> None:
        blocks = split_raw_context_by_file_headers("")
        assert len(blocks) == 1
        assert blocks[0]["path"] is None
        assert blocks[0]["offset_end"] == 0

    def test_preamble_before_first_header_is_preserved(self) -> None:
        ctx = (
            "Preamble notes.\nSee attached files.\n\n"
            "# file: a.py\nprint('a')\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert len(blocks) == 2
        assert blocks[0]["path"] is None
        assert "Preamble" in blocks[0]["text"]
        assert blocks[1]["path"] == "a.py"

    def test_prose_that_looks_like_diff_marker_inline_does_not_split(self) -> None:
        # Line must start with ``--- a/`` at column 0 for the diff
        # pattern to fire; prose like "see --- a/file" mid-paragraph
        # should NOT create a split.
        ctx = (
            "The proposal notes that `--- a/old_schema.sql` was "
            "deprecated last quarter and the migration path forward "
            "is documented here."
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert len(blocks) == 1
        assert blocks[0]["path"] is None


class TestOffsetBookkeeping:
    def test_offsets_cover_full_context_without_gaps(self) -> None:
        ctx = (
            "# file: a.py\nprint('a')\n\n"
            "# file: b.py\nprint('b')\n\n"
            "# file: c.py\nprint('c')\n"
        )
        blocks = split_raw_context_by_file_headers(ctx)
        assert len(blocks) == 3
        # Offsets must tile the context with no gaps.
        prev_end = 0
        for b in blocks:
            assert b["offset_start"] == prev_end
            prev_end = b["offset_end"]
        assert prev_end == len(ctx)

    def test_block_text_slice_matches_offsets(self) -> None:
        ctx = "# file: x.py\nX=1\n# file: y.py\nY=2\n"
        blocks = split_raw_context_by_file_headers(ctx)
        for b in blocks:
            assert ctx[b["offset_start"] : b["offset_end"]] == b["text"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
