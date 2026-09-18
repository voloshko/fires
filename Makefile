.PHONY: gate traceability-check spec-status sdd-selftest

## Full fail-closed gate: traceability + evidence receipts
gate: traceability-check sdd-selftest

traceability-check:
	python3 scripts/check_traceability.py

## Authoritative status breakdown (index.md statuses are at-a-glance only)
spec-status:
	python3 scripts/spec_tools.py status

## The gate's own negative tests
sdd-selftest:
	python3 -m unittest discover -s scripts -p 'test_check_traceability_sdd.py'
