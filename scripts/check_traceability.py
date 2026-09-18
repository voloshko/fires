#!/usr/bin/env python3
"""SPEC-537: traceability coverage gate.

CLAUDE.md/AGENTS.md declare traceability mandatory, but at the time this gate
was written 246 of 520+ `specs/SPEC-*.md` files had no `traceability.json`
entry (a pre-existing gap, concentrated in early specs). Rather than either
(a) silently keep failing that promise, or (b) force a low-value mechanical
backfill of 246 historical entries, this gate freezes the existing gap as an
explicit, named allowlist (`legacy_untracked_specs` in traceability.json) and
enforces **zero new untracked specs** going forward.

Checks, all fail-closed:
  1. Every `specs/SPEC-*.md` file is either in `traceability.json["specs"]`
     or in `traceability.json["legacy_untracked_specs"]`. A spec file that is
     in neither is a NEW untracked spec — that's the regression this gate
     exists to catch.
  2. Every entry in `traceability.json["specs"]` has a `file` field pointing
     at a file that exists on disk (no "ghost" entries).
  3. Every id listed in an entry's `depends_on` resolves to a real spec —
     either tracked in `traceability.json["specs"]` or a legacy-grandfathered
     id with a real file on disk. (Many tracked specs legitimately depend on
     untracked legacy specs; this only catches a `depends_on` pointing at an
     id that doesn't exist anywhere.)

SPEC-546 extends the gate with three more fail-closed checks:
  4. Every entry's `status` is in the defined lifecycle enum (see
     ALLOWED_STATUSES below and CLAUDE.md "Status lifecycle").
  5. Every entry's `requirement` references an existing
     `requirements/REQ-*.md` file.
  6. `specs/index.md`'s "Next available number" counter is strictly greater
     than every spec number on disk (the counter used to rot silently).

Does NOT check: whether `legacy_untracked_specs` still needs every one of its
246 entries (some may since have gained real coverage — that's a cleanup
opportunity, not a gate failure; removing an id from that list because it
now has a real `specs` entry is always fine and encouraged).
"""
from __future__ import annotations
import re
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "specs"
REQS_DIR = ROOT / "requirements"
TRACEABILITY = ROOT / "traceability.json"
INDEX = SPECS_DIR / "index.md"
RECEIPTS_DIR = ROOT / "evidence" / "receipts"

SPEC_ID_RE = re.compile(r"^(SPEC-\d+(?:\.\d+)?)")
INDEX_COUNTER_RE = re.compile(r"\*\*Next available number:\*\*\s*SPEC-(\d+)")

# SPEC-546 status lifecycle (documented in CLAUDE.md):
#   planned -> analysis -> active -> implemented | partial -> verified
#   terminal at any point: deferred | rejected
ALLOWED_STATUSES = {
    "planned",      # spec written, work not started
    "analysis",     # investigation/measurement phase, no code landing yet
    "active",       # implementation in progress
    "partial",      # done with honest, documented scope reduction (see Resolution)
    "implemented",  # acceptance criteria met, verified by the implementer
    "verified",     # independently re-verified (receipts re-run by non-implementer)
    "deferred",     # deliberately parked; Resolution says why and what unblocks it
    "rejected",     # investigated and killed; Resolution records the kill signal
}

# Receipt status enum (evidence/README.md). A receipt records what a run
# actually produced; PASS is the only value that can support `verified`.
RECEIPT_STATUSES = {"PASS", "FAIL", "INSUFFICIENT_EVIDENCE"}

# Minimum shape every receipt must carry (evidence/templates/receipt.json).
RECEIPT_REQUIRED_FIELDS = (
    "schema_version", "spec_id", "run_id", "created_at",
    "status", "conclusion", "commands", "acceptance",
    "limitations", "non_claims",
)


def _receipts_for(spec_id: str) -> list[pathlib.Path]:
    d = RECEIPTS_DIR / spec_id
    return sorted(d.glob("*.json")) if d.is_dir() else []


def _on_disk_spec_ids() -> set[str]:
    ids = set()
    for p in SPECS_DIR.glob("SPEC-*.md"):
        m = SPEC_ID_RE.match(p.name)
        if m:
            ids.add(m.group(1))
    return ids


def check() -> int:
    if not TRACEABILITY.exists():
        print(f"traceability: {TRACEABILITY} is missing", file=sys.stderr)
        return 2

    d = json.loads(TRACEABILITY.read_text())
    specs = d.get("specs", {})
    legacy = set(d.get("legacy_untracked_specs", []))
    on_disk = _on_disk_spec_ids()
    tracked = set(specs.keys())

    problems: list[str] = []

    # 1. new untracked specs
    new_untracked = sorted(on_disk - tracked - legacy)
    if new_untracked:
        problems.append(
            "New untracked spec file(s) — add a traceability.json entry "
            f"(see CONTRIBUTING.md): {', '.join(new_untracked)}"
        )

    # legacy ids that no longer have a file at all (renamed/deleted without
    # updating the grandfather list) — informational, not fatal, since the
    # spec may have been deliberately removed; but surface it so it isn't
    # invisible.
    legacy_missing_file = sorted(legacy - on_disk - tracked)
    if legacy_missing_file:
        print(
            "traceability: note — legacy_untracked_specs entries with no "
            f"matching file on disk (informational, not failing): "
            f"{', '.join(legacy_missing_file)}",
            file=sys.stderr,
        )

    # 2. ghost entries: traceability entry whose `file` doesn't exist
    ghost = []
    for spec_id, entry in specs.items():
        f = entry.get("file")
        if not f or not (ROOT / f).exists():
            ghost.append(f"{spec_id} -> {f!r}")
    if ghost:
        problems.append(
            "traceability.json entr(y/ies) with a missing/nonexistent "
            f"'file': {'; '.join(ghost)}"
        )

    # 3. dangling depends_on references — resolvable against tracked specs,
    # legacy-grandfathered specs, or any spec file that exists on disk
    # (covers ids that are on disk but in neither set due to drift).
    resolvable = tracked | legacy | on_disk
    dangling = []
    for spec_id, entry in specs.items():
        for dep in entry.get("depends_on", []):
            if dep not in resolvable:
                dangling.append(f"{spec_id} depends_on {dep!r} (does not exist)")
    if dangling:
        problems.append(
            "Dangling depends_on reference(s): " + "; ".join(dangling)
        )

    # 4. status enum (SPEC-546)
    bad_status = [
        f"{spec_id} has status {entry.get('status')!r}"
        for spec_id, entry in specs.items()
        if entry.get("status") not in ALLOWED_STATUSES
    ]
    if bad_status:
        problems.append(
            "Status outside the lifecycle enum "
            f"({', '.join(sorted(ALLOWED_STATUSES))}): " + "; ".join(bad_status)
        )

    # 5. requirement references a real requirements/REQ-*.md (SPEC-546)
    bad_req = []
    for spec_id, entry in specs.items():
        req = entry.get("requirement")
        if req and not (REQS_DIR / f"{req}.md").exists():
            bad_req.append(f"{spec_id} -> {req!r}")
    if bad_req:
        problems.append(
            "requirement field(s) referencing no requirements/REQ-*.md file: "
            + "; ".join(bad_req)
        )

    # 6. specs/index.md counter freshness (SPEC-546)
    if INDEX.exists():
        m = INDEX_COUNTER_RE.search(INDEX.read_text())
        max_num = max(
            (int(n.group(1).split("-")[1].split(".")[0])
             for p in SPECS_DIR.glob("SPEC-*.md")
             if (n := SPEC_ID_RE.match(p.name))),
            default=0,
        )
        if m is None:
            problems.append(
                "specs/index.md has no '**Next available number:** SPEC-N' line"
            )
        elif int(m.group(1)) <= max_num:
            problems.append(
                f"specs/index.md counter is stale: says SPEC-{m.group(1)} but "
                f"SPEC-{max_num} already exists on disk (bump it, or use "
                "scripts/spec_tools.py add-spec which bumps it for you)"
            )

    # 7. `verified` requires a PASS receipt on disk
    unproven = []
    for spec_id, entry in specs.items():
        if entry.get("status") != "verified":
            continue
        receipts = _receipts_for(spec_id)
        if not receipts:
            unproven.append(f"{spec_id} (no evidence/receipts/{spec_id}/ at all)")
            continue
        if not any(
            json.loads(r.read_text()).get("status") == "PASS" for r in receipts
        ):
            unproven.append(f"{spec_id} ({len(receipts)} receipt(s), none PASS)")
    if unproven:
        problems.append(
            "Spec(s) marked 'verified' with no PASS receipt: " + "; ".join(unproven)
        )

    # 8. receipt structural validity
    bad_receipts = []
    if RECEIPTS_DIR.is_dir():
        for r in sorted(RECEIPTS_DIR.glob("*/*.json")):
            rel = r.relative_to(ROOT)
            try:
                doc = json.loads(r.read_text())
            except json.JSONDecodeError as exc:
                bad_receipts.append(f"{rel}: invalid JSON ({exc})")
                continue
            missing = [f for f in RECEIPT_REQUIRED_FIELDS if f not in doc]
            if missing:
                bad_receipts.append(f"{rel}: missing field(s) {', '.join(missing)}")
            if doc.get("spec_id") != r.parent.name:
                bad_receipts.append(
                    f"{rel}: spec_id {doc.get('spec_id')!r} != directory "
                    f"{r.parent.name!r}"
                )
            if doc.get("status") not in RECEIPT_STATUSES:
                bad_receipts.append(
                    f"{rel}: status {doc.get('status')!r} not in "
                    f"{sorted(RECEIPT_STATUSES)}"
                )
    if bad_receipts:
        problems.append("Malformed receipt(s): " + "; ".join(bad_receipts))

    if problems:
        print("TRACEABILITY CHECK: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(
        f"TRACEABILITY CHECK: PASS — {len(tracked)} tracked, "
        f"{len(legacy)} legacy-grandfathered, 0 new untracked, "
        f"0 ghost entries, 0 dangling depends_on, statuses valid, "
        f"requirement refs valid, index counter fresh, "
        f"{sum(len(_receipts_for(s)) for s in tracked)} receipt(s) valid"
    )
    return 0


def main(argv=None) -> int:
    return check()


if __name__ == "__main__":
    sys.exit(main())
