#!/usr/bin/env python3
"""Negative tests for check_traceability.py.

Each test builds a minimal valid tree in a temp dir, breaks exactly one
invariant (status enum / requirement ref / index counter / evidence receipt),
and asserts the gate fails — plus happy-path tests proving the fixtures
themselves are green.

Inherited from vector_algebra/sdd-template and extended here with the
evidence-receipt checks (7 and 8), which are this project's addition.

Run: python3 -m unittest discover -s scripts -p 'test_check_traceability_sdd.py'
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "check_traceability", HERE / "check_traceability.py"
)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def valid_receipt(**overrides) -> dict:
    """A receipt carrying every gate-required field."""
    doc = {
        "schema_version": "1.0",
        "spec_id": "SPEC-1",
        "run_id": "test-run-001",
        "created_at": "2026-09-18T00:00:00Z",
        "evidence_class": "REPLAY",
        "status": "PASS",
        "conclusion": "TEST_FIXTURE",
        "commands": ["true"],
        "acceptance": [{"id": "AC-01", "result": "PASS", "artifacts": []}],
        "limitations": [],
        "non_claims": ["proves nothing about real data"],
    }
    doc.update(overrides)
    return doc


def build_tree(root: pathlib.Path, *, status="planned", requirement="REQ-001",
               counter="SPEC-2", counter_line=True, receipts=()) -> None:
    specs = root / "specs"
    reqs = root / "requirements"
    specs.mkdir()
    reqs.mkdir()
    (specs / "SPEC-1-foo.md").write_text("# SPEC-1: foo\n\nStatus: planned\n")
    (reqs / "REQ-001.md").write_text("# REQ-001: pillar\n")
    index = "# Spec Index\n\n"
    if counter_line:
        index += f"**Next available number:** {counter}\n"
    (specs / "index.md").write_text(index)
    (root / "traceability.json").write_text(json.dumps({
        "specs": {
            "SPEC-1": {
                "title": "foo",
                "file": "specs/SPEC-1-foo.md",
                "requirement": requirement,
                "depends_on": [],
                "status": status,
            }
        },
        "legacy_untracked_specs": [],
        "requirements": {},
    }))
    for i, (spec_dir, payload) in enumerate(receipts):
        d = root / "evidence" / "receipts" / spec_dir
        d.mkdir(parents=True, exist_ok=True)
        body = payload if isinstance(payload, str) else json.dumps(payload)
        (d / f"receipt-{i}.json").write_text(body)


class SddGateTests(unittest.TestCase):
    def run_gate(self, **tree_kwargs) -> int:
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            build_tree(root, **tree_kwargs)
            old = (ct.ROOT, ct.SPECS_DIR, ct.REQS_DIR, ct.TRACEABILITY,
                   ct.INDEX, ct.RECEIPTS_DIR)
            try:
                ct.ROOT = root
                ct.SPECS_DIR = root / "specs"
                ct.REQS_DIR = root / "requirements"
                ct.TRACEABILITY = root / "traceability.json"
                ct.INDEX = root / "specs" / "index.md"
                ct.RECEIPTS_DIR = root / "evidence" / "receipts"
                return ct.check()
            finally:
                (ct.ROOT, ct.SPECS_DIR, ct.REQS_DIR, ct.TRACEABILITY,
                 ct.INDEX, ct.RECEIPTS_DIR) = old

    # --- inherited checks (1-6) ---

    def test_happy_path_passes(self):
        self.assertEqual(self.run_gate(), 0)

    def test_every_non_verified_status_passes(self):
        for status in sorted(ct.ALLOWED_STATUSES - {"verified"}):
            self.assertEqual(self.run_gate(status=status), 0, status)

    def test_status_outside_enum_fails(self):
        self.assertEqual(self.run_gate(status="draft"), 1)

    def test_dangling_requirement_ref_fails(self):
        self.assertEqual(self.run_gate(requirement="REQ-999"), 1)

    def test_stale_index_counter_fails(self):
        self.assertEqual(self.run_gate(counter="SPEC-1"), 1)

    def test_missing_counter_line_fails(self):
        self.assertEqual(self.run_gate(counter_line=False), 1)

    # --- evidence checks (7-8), this project's addition ---

    def test_verified_with_pass_receipt_passes(self):
        self.assertEqual(
            self.run_gate(status="verified",
                          receipts=[("SPEC-1", valid_receipt())]),
            0,
        )

    def test_verified_without_any_receipt_fails(self):
        self.assertEqual(self.run_gate(status="verified"), 1)

    def test_verified_with_only_failing_receipt_fails(self):
        for bad in ("FAIL", "INSUFFICIENT_EVIDENCE"):
            self.assertEqual(
                self.run_gate(status="verified",
                              receipts=[("SPEC-1", valid_receipt(status=bad))]),
                1,
                bad,
            )

    def test_non_verified_status_needs_no_receipt(self):
        self.assertEqual(self.run_gate(status="implemented"), 0)

    def test_receipt_missing_required_field_fails(self):
        r = valid_receipt()
        del r["non_claims"]
        self.assertEqual(self.run_gate(receipts=[("SPEC-1", r)]), 1)

    def test_receipt_spec_id_mismatching_directory_fails(self):
        self.assertEqual(
            self.run_gate(receipts=[("SPEC-1", valid_receipt(spec_id="SPEC-2"))]),
            1,
        )

    def test_receipt_status_outside_enum_fails(self):
        self.assertEqual(
            self.run_gate(receipts=[("SPEC-1", valid_receipt(status="ok"))]),
            1,
        )

    def test_invalid_json_receipt_fails(self):
        self.assertEqual(self.run_gate(receipts=[("SPEC-1", "{not json")]), 1)


if __name__ == "__main__":
    unittest.main()
