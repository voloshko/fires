# {{ID}}: {{TITLE}}

Status: planned

Requirement: {{REQUIREMENT}}

Depends on: {{DEPENDS_ON}}

## Summary

What this spec delivers and why now. State the problem plainly, cite the
evidence that motivated it (a finding, a measurement, a review item — with
file/spec references), and say what "done" looks like in one paragraph.

If the work is a bet (performance, feasibility), state the honest
expectation AND the kill signal: what result means "stop, ship nothing but
the postmortem".

## Acceptance Criteria

- Specific, testable criterion — name the file/command/number that proves it.
- Prefer "gate X passes" / "test Y exists and is green" over prose claims.
- If a criterion may prove infeasible, say what an acceptable partial
  outcome is (this project's convention: an honest `partial` with a
  documented reason beats a forced `implemented`).

## Verification

- Exact commands a reviewer runs to independently confirm the criteria
  (build, test, gate sweep, benchmark) and what output counts as PASS.
- Anything the implementer's own run cannot prove (needs external infra,
  another machine, a human decision) — name it explicitly.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
