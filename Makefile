.PHONY: gate traceability-check spec-status sdd-selftest cache serve

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

## Пересобрать кеш за 60 суток (около 13 минут, нужен FIRMS_MAP_KEY)
cache:
	set -a && . ./.env && set +a && python3 -m src.api --config config/krasnoyarsk.toml \
		--days 60 --burns 6 --cache data/demo --rebuild --port 8077

## Сервер на готовом кеше: стартует за секунды и ключа не требует
serve:
	python3 -m src.api --config config/krasnoyarsk.toml --cache data/demo --port 8077
