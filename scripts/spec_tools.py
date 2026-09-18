#!/usr/bin/env python3
"""SPEC-546: the real spec-driven-development CLI.

This replaces the fictional tool CLAUDE.md used to document. It automates
ONLY the mechanical, mistake-prone parts of the workflow; it deliberately
does NOT flip statuses or "propagate changes" — status transitions and
Resolution sections are deliberate acts by the orchestrating session (see
CLAUDE.md, "Status lifecycle" and "Resolution convention").

Commands:
  add-spec "Title" --requirement REQ-002 [--depends-on SPEC-1,SPEC-2] [--slug s]
      Compute the next spec number (max of disk + tracked + index counter),
      create specs/SPEC-NNN-slug.md from specs/TEMPLATE.md, add the
      traceability.json entry (status: planned), bump specs/index.md's
      "Next available number".
  add-req "Title" [--slug s]
      Create requirements/REQ-00N.md from requirements/TEMPLATE.md and
      register it in traceability.json's requirements section.
  status
      Authoritative status breakdown from traceability.json (specs/index.md
      statuses are at-a-glance only), plus the list of open specs.

Everything is plain-file manipulation; run `scripts/check_traceability.py`
(or `make traceability-check`) afterwards — the gate is the safety net.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "specs"
REQS_DIR = ROOT / "requirements"
TRACEABILITY = ROOT / "traceability.json"
INDEX = SPECS_DIR / "index.md"

SPEC_NUM_RE = re.compile(r"^SPEC-(\d+)")
REQ_NUM_RE = re.compile(r"^REQ-(\d+)")
INDEX_COUNTER_RE = re.compile(r"(\*\*Next available number:\*\*\s*)SPEC-(\d+)")

# Keep in sync with scripts/check_traceability.py::ALLOWED_STATUSES.
NEW_SPEC_STATUS = "planned"


def _load() -> dict:
    return json.loads(TRACEABILITY.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    TRACEABILITY.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:60]


def _spec_numbers_on_disk() -> list[int]:
    nums = []
    for p in SPECS_DIR.glob("SPEC-*.md"):
        m = SPEC_NUM_RE.match(p.name)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _next_spec_number(data: dict) -> int:
    nums = _spec_numbers_on_disk()
    for sid in list(data.get("specs", {})) + list(
        data.get("legacy_untracked_specs", [])
    ):
        m = SPEC_NUM_RE.match(sid)
        if m:
            nums.append(int(m.group(1)))
    m = INDEX_COUNTER_RE.search(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else None
    counter = int(m.group(2)) if m else 0
    return max(nums + [counter - 1], default=0) + 1


def _bump_index_counter(next_free: int) -> None:
    if not INDEX.exists():
        return
    text = INDEX.read_text(encoding="utf-8")
    new_text, n = INDEX_COUNTER_RE.subn(rf"\g<1>SPEC-{next_free}", text)
    if n:
        INDEX.write_text(new_text, encoding="utf-8")


def _render_template(path: pathlib.Path, mapping: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise SystemExit(f"template {path} has unfilled placeholders: {leftover}")
    return text


def cmd_add_spec(args: argparse.Namespace) -> int:
    data = _load()
    req = args.requirement
    if not (REQS_DIR / f"{req}.md").exists():
        raise SystemExit(f"requirement file requirements/{req}.md does not exist")

    deps = [d.strip() for d in (args.depends_on or "").split(",") if d.strip()]
    disk_id_re = re.compile(r"^(SPEC-\d+(?:\.\d+)?)")  # same as check_traceability.py
    resolvable = (
        set(data.get("specs", {}))
        | set(data.get("legacy_untracked_specs", []))
        | {m.group(1) for p in SPECS_DIR.glob("SPEC-*.md") if (m := disk_id_re.match(p.name))}
    )
    for d in deps:
        if d not in resolvable:
            raise SystemExit(f"depends_on {d!r} does not resolve to any known spec")

    num = _next_spec_number(data)
    spec_id = f"SPEC-{num}"
    slug = args.slug or _slugify(args.title)
    if not slug:
        raise SystemExit("could not derive a slug from the title; pass --slug")
    rel_file = f"specs/{spec_id}-{slug}.md"

    body = _render_template(
        SPECS_DIR / "TEMPLATE.md",
        {
            "ID": spec_id,
            "TITLE": args.title,
            "REQUIREMENT": req,
            "DEPENDS_ON": ", ".join(deps) if deps else "—",
        },
    )
    (ROOT / rel_file).write_text(body, encoding="utf-8")

    data["specs"][spec_id] = {
        "title": args.title,
        "file": rel_file,
        "requirement": req,
        "depends_on": deps,
        "status": NEW_SPEC_STATUS,
    }
    req_entry = data.get("requirements", {}).get(req)
    if req_entry is not None:
        req_entry.setdefault("specs", []).append(spec_id)
    _save(data)
    _bump_index_counter(num + 1)

    print(f"created {rel_file} (status: {NEW_SPEC_STATUS}, requirement: {req})")
    print(f"traceability.json entry added; index counter -> SPEC-{num + 1}")
    print("next: fill in Summary/Acceptance Criteria, then run the gate:")
    print("  python3 scripts/check_traceability.py")
    return 0


def cmd_add_req(args: argparse.Namespace) -> int:
    data = _load()
    nums = [
        int(m.group(1))
        for p in REQS_DIR.glob("REQ-*.md")
        if (m := REQ_NUM_RE.match(p.name))
    ]
    req_id = f"REQ-{max(nums, default=0) + 1:03d}"
    rel_file = f"requirements/{req_id}.md"
    body = _render_template(
        REQS_DIR / "TEMPLATE.md", {"ID": req_id, "TITLE": args.title}
    )
    (ROOT / rel_file).write_text(body, encoding="utf-8")
    data.setdefault("requirements", {})[req_id] = {
        "title": args.title,
        "file": rel_file,
        "specs": [],
    }
    _save(data)
    print(f"created {rel_file} and registered it in traceability.json")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    data = _load()
    specs = data.get("specs", {})
    by_status: dict[str, int] = {}
    for entry in specs.values():
        by_status[entry.get("status", "?")] = by_status.get(entry.get("status", "?"), 0) + 1
    print(f"tracked specs: {len(specs)}  "
          f"(+{len(data.get('legacy_untracked_specs', []))} legacy-grandfathered)")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {status:12s} {count}")
    open_specs = {
        sid: e for sid, e in specs.items()
        if e.get("status") in ("planned", "analysis", "active", "partial")
    }
    if open_specs:
        print("\nopen specs:")
        for sid in sorted(open_specs):
            e = open_specs[sid]
            deps = ",".join(e.get("depends_on", [])) or "-"
            print(f"  {sid}: [{e.get('status')}] {e.get('title', '?')}  (deps: {deps})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add-spec", help="scaffold a new spec + traceability entry")
    p.add_argument("title")
    p.add_argument("--requirement", required=True, help="e.g. REQ-002")
    p.add_argument("--depends-on", default="", help="comma-separated spec ids")
    p.add_argument("--slug", default="", help="override the filename slug")
    p.set_defaults(func=cmd_add_spec)

    p = sub.add_parser("add-req", help="scaffold a new requirement")
    p.add_argument("title")
    p.set_defaults(func=cmd_add_req)

    p = sub.add_parser("status", help="authoritative status breakdown")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
