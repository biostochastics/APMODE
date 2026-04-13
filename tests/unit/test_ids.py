# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for sparkid integration."""

from apmode.ids import generate_candidate_id, generate_gate_id, generate_run_id


class TestSparkidIntegration:
    def test_run_id_is_string(self) -> None:
        rid = generate_run_id()
        assert isinstance(rid, str)

    def test_run_id_length(self) -> None:
        rid = generate_run_id()
        assert len(rid) == 21

    def test_ids_are_unique(self) -> None:
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_ids_are_time_sortable(self) -> None:
        import time

        id1 = generate_run_id()
        time.sleep(0.01)
        id2 = generate_run_id()
        assert id1 < id2

    def test_candidate_id(self) -> None:
        cid = generate_candidate_id()
        assert isinstance(cid, str)
        assert len(cid) == 21

    def test_gate_id(self) -> None:
        gid = generate_gate_id()
        assert isinstance(gid, str)
        assert len(gid) == 21
