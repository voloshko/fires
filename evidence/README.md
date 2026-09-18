# Evidence receipts

A receipt is an append-only record of an **executed** acceptance contract. It is
the only thing that can support a `verified` status — `scripts/check_traceability.py`
fails closed if a spec claims `verified` without a PASS receipt on disk.

## Layout

```text
evidence/receipts/<SPEC-ID>/<UTC-timestamp>-<short-run-id>.json
```

`evidence/templates/receipt.json` documents the minimum shape. Gate-enforced
fields: `schema_version`, `spec_id` (must equal the directory name), `run_id`,
`created_at`, `status`, `conclusion`, `commands`, `acceptance`, `limitations`,
`non_claims`.

## Receipt status enum

| Status | Meaning |
|---|---|
| `PASS` | every acceptance criterion ran and passed |
| `FAIL` | ran, criteria not met — **a valid deliverable, not a thing to hide** |
| `INSUFFICIENT_EVIDENCE` | could not be decided (external service down, no data in window, missing input) |

## Rules

1. **Never overwrite, amend or delete a cited receipt.** A correction is a
   successor receipt with `predecessor_run_id` and a stated reason.
2. **A failed or negative measurement is valid evidence.** Do not relabel it as
   a partial success and do not move a threshold after seeing the result.
3. **`commands` must be literally re-runnable** by a reviewer, copy-paste, with
   no hidden state.
4. **`non_claims` is mandatory and must be honest** — say exactly what the run
   does NOT prove. For this project that routinely includes: "does not prove the
   burn-area figure against ground truth", "does not prove the flare mask is
   complete", "covers only the tested MGRS tile".
5. **Never commit secrets** — `FIRMS_MAP_KEY`, Earthdata or CDSE credentials must
   never appear in a receipt, capture or output.

## Project-specific evidence classes

| Class | Meaning for this project |
|---|---|
| `LIVE` | hit a real external API (FIRMS, Planetary Computer) — record the UTC window and item ids, since the answer is not reproducible later |
| `REPLAY` | ran against a frozen fixture in `evidence/fixtures/` — fully reproducible |
| `SYNTHETIC` | ran against generated input — proves logic, proves nothing about real data |

Detection and burn-area numbers claimed against a real fire must cite a `LIVE`
or `REPLAY` receipt. A `SYNTHETIC` receipt can never support an accuracy claim.
