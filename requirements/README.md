# Requirements layer — how it works

- **Requirements are broad pillars, deliberately few.** One REQ per durable
  product/engineering pillar — something that would justify its own roadmap
  and its own definition of done. Not one per feature.

- **Fine-grained theming is NOT here.** It lives in `specs/index.md`'s
  number-range groups. Don't create a REQ for a feature cluster.

- **Every tracked spec carries a `requirement:` field** in
  `traceability.json`, and `scripts/check_traceability.py` fails CI if that
  field references a REQ file that doesn't exist. The field's job is "which
  pillar does this serve".

- **Creating a new REQ:** `python3 scripts/spec_tools.py add-req "Title"` —
  scaffolds `requirements/REQ-NNN.md` from TEMPLATE.md and registers it in
  `traceability.json`.

- **Заголовки на русском требуют явного `--slug`.** `spec_tools.py` вырезает не-ASCII при построении имени файла, и кириллический заголовок даёт бессмысленный слаг из случайных латинских обрывков. Передавайте `--slug` руками.
