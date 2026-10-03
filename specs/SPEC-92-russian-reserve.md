
## Error-report contract

Дополнение обязательного отчёта о пропусках/FP, не изменение acceptance.
`scripts/report_russian_errors.py` читает immutable итог и подтверждённые
probability caches, не запускает inference/training и не выбирает порог.
Публикует TP/FP/FN, число окон и групп с нулёмTP, долюFP среди известногофона,
masked_fraction и bins по area_ha исходного seed-контура. Если seed полностью
скрыт общеймаской, он unobservable, а не пропущенный пожар. Дополнительно
rasterize выбранных source-контура из исходногоZIP в gridокна: countвидимых
seedpixels и их TP на тех же масках/пороге0.5 у всехмоделей. Это потерянные
размеченные контуры/окна/кластеры, не доказанное число физических пожаров.
Россия grouping20км — пространственный bootstrapcluster, не fire_event_id.

Самые FP-трудные окна задаёт C1-MM (top25%,stableid tie), только diagnostics,
не новое tuning или held-out claim. EMS/FLOGA report восстанавливается из
уже подписанных cachearrays, обязательно той же counts_for логикой. Counts
России обязаны точно совпасть с parentresult; все cache/input/model hashes
проверяются. Отчёт append-only, связан SHA с parentresult и исходнымZIP.
