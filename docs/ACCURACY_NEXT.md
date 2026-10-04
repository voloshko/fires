# Кампания точности — состояние 2026-10-03

Цель: проверить обученную пару дат, российскую разметку и трудный фон,
L2A-дообучение и степень по паре дат; доказать превосходство на общей проверке.
Отрицательные эксперименты сами по себе цель не закрывают. Спеки перед кодом,
коммиты по спекам, неизменные критерии, append-only receipts.

## Размещение и ограничения

Worktree `/Users/mc/.codex/worktrees/fire-accuracy-next/fires`, ветка
`codex/fire-accuracy-next`, исходный main `1f0ea52`.
**Все вычисления и скачивания — ssh k8plus**, `/home/mc/fires-accuracy-next`.
На Mac только код и компактные доказательства. Данные в `data-store/`, веса
и результаты в `research/`. На k8plus свободно около16ГБ: новые данные ограничивать.
Старый `/home/mc/fires` и его данные читаются без изменений через symlinks.
`.runtime` — собственный venv; `.venv` ведёт в старую среду, ничего туда не ставить.
Свои задания omarchy остановлены; остаток прерванной копии сохраняется.

Один собственный GPU job, `flock research/accuracy-gpu.lock`, старт после
memory.free>=22000MiB. Не трогать соседние задания. Существующие модели и
receipts не переписывать. Native checkpoints: один tensor-файл и связанные
с ним JSON metadata, без дублирования весов.

## Завершённые измерения

**SPEC-91 rejected**, все6full models120epochs и scorer завершены.
141окно/125групп: pair IoU0.785478, post0.761802, C1-MM0.757951.
Pair−C1-MM=+0.027527, CI95[0.006894,0.055348]. Общий прирост реален,
но FLOGA lower=-0.053718 ниже-0.03 и потерянных событий2 против1 у C1-MM.
Предзаданные safeguards не пройдены. Receipt SPEC-91-REPLAY-001 FAIL,
`research/forest-pair-v1-result.json`. Не переключать продукт по pooled gain.

**SPEC-94 rejected** для inverse-sqrt весов, все6full models и scorer завершены.
172окна/172события. Balanced mIoU0.363802, hybrid0.308888:
+0.054914, CI95[0.021376,0.092000], все3класса лучше гибрида.
Equal weights0.367197; balanced−equal=-0.003395. Условие «не ниже контроля»
не выполнено. Receipt SPEC-94-REPLAY-001 FAIL,
`research/paired-severity-v1-result.json`. Оба парных варианта дают сигнал
улучшения степени; выбор equal после этого результата требует новой спеки
и свежих событий. Контур фиксирован: binaryIoU0.453463 у всех вариантов.
MTBS — спутниковая интерпретация, не наземная истина.

## Российское дообучение — SPEC-92 active

Источник PANGAEA.939429, CC-BY-4.0, визуальная Sentinel-2 разметка с GIS-проверкой;
авторы использовали FIRMS вспомогательно. Не field truth.
Исходник SHA256 `5f1cfb843d5579d800a1e8359a017f8cd33c5658f6e0c0a6eda7ac534a1c6b3c`.
196673контуров, целевой bbox80–115E/50–65N:29071. Источник содержит geometry/area,
не точные даты. Пары interval январь–15мая2020: postMay16–July15 2020,
pre тот же сезон2019, PC Sentinel-2 L2A, одна сетка/MGRS, строгий SCL.
Нельзя выдумывать точную ignition date.

Готовы447окон/127групп:309fit/61cal/77eval,24evalclusters.
Frozen reserve_plan/frame_selection, actual frames purged against prior2658
и между partitions до predictions. Geometry IDs из rawZIP, GPKG меняет ring order.
Данные `data-store/russia/manifest.json`.

Очередь `fires-accuracy-russian-transfer-v1` (MainPID742993) **обучает на GPU**.
Первый control-s1 дошёл до29/30эпох на последнем осмотре. Далее reserve-s1,
control/reserve-s2/s3, общий same-pixel scorer C1-MM+Prithvi.
Control продолжает SPEC91 pair на base374fit; reserve добавляет309Russianfit.
Одинаковый бюджет30epochs, batch8, шаги по base, AdamW3e-5/cosine/BF16.
Mining top25%FP только Russianfit, weight2, без удаления positives.
Initializer SHA/source/manifest/recipe сверяются с frozenSPEC91result.
Recipe `349b9c51f319432e707467c4ae25ce0d55db28050506e055c621346e391ca30f`.
Результат ожидается в `research/russian-transfer-v1-result.json`.

Только CI lower>0 **к обоим** C1-MM и Prithvi разрешает combined superiority.
Prithvi full512CPU preflight прошёл с официальными весами, accuracy не доказывает.
TerraTorch from_config принимает stringpaths; backbone_pretrained=False перед
strict полным checkpoint load. Targeted code review правок APPROVE;
общий независимый architecture verdict пока не закрыт.

## L2A — SPEC-93/96/97 active

Planetary Computer — основной источник. Отдельный ранний EarthSearch PREP
не смешивается с PC. PB<04 offset0; PB>=04 offset−0.1, scale0.0001;
неизвестный baseline fail-closed. Полный PC collector terminal exit2:
1008окон,889fit/119eval,861/886 известных событий сохранены.

Registry SPEC96:523/540 HLS matched MTBS,17unknown, prior2658frames.
SPEC97 защищает все GT-positive pixels неизвестных/остаточных частей;
если все сохранены, (R+U)/(K+U)>=R/K даёт честную нижнюю границу70%.
2784protected pixels потеряны в16окнах. Lower bound unavailable,
ready=false; **L2A fine-tuning не запускался**. Это нехватка data evidence,
не отрицательный результат модели. Receipt SPEC-97-BOUND001 INSUFFICIENT_EVIDENCE.

Причины в `evidence/experiments/accuracy-next-20261003/l2a-protected-pixel-causes.json`:
T11SKB1948pixels SCLcloud/shadow; T12SVC802nodata; остальные cloud/snow/missing.
PC same-day не даёт альтернативных продуктов; T10SGD20211102 отсутствует.
Публичный CDSE catalog показывает online N0500 counterparts для трёх окон,
но скачивания нет: credentials не обнаружены в проверенных конфигурациях.
Запрошены только наличие/путь конфигурации на k8plus, без секретов в чате.
Не снимать SCL, не выкидывать unknown из denominator, не снижать70%.

## Проверка и следующий шаг

SPEC90/95 implemented: isolation, provenance, grouped bootstrap, source-bound
caches, restricted legacy loading, tensor-bound metadata, exact recipe lineage.
`make gate` зелёный; это не доказательство accuracy acceptance.

```sh
ssh k8plus 'systemctl --user show fires-accuracy-russian-transfer-v1 -p ActiveState -p SubState -p MainPID -p ExecMainStatus'
ssh k8plus 'journalctl --user -u fires-accuracy-russian-transfer-v1 -n 12 --no-pager'
ssh k8plus 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Завершить Russian6runs и обе competitor comparisons. Подготовить независимое
подтверждение equal paired severity на ранее не использованных событиях.
Ремонтировать L2A admission реальными данными/доказательствами, затем matched
fine-tune. Сохранять отрицательные outcomes, per-window counts и ограничения.
Полного доказанного превосходства и завершения всех четырёх пунктов пока нет.

## Независимое подтверждение степени — новый запуск

SPEC98 deferred: после prior/event isolation осталось21western2024 событие,
при минимуме100. ДажеallCONUS2024 даёт57. No inference, admission receipt
INSUFFICIENT_EVIDENCE. SPEC99 deferred: PC hls2-s30 temporal extent начинается
2020-01-01, пары2016–2017 недоступны (QC0/2). Это ограничения данных,
не отрицательный результат модели; оба набора и receipts сохранены.

SPEC100 active, commit337ba1d (+последующие фиксы/документация): явный профиль
pc-covered-2020-24. Metadata-only240 independent candidates, cap240,
старые1300events и prior2658frames исключены. Frozen equal checkpoints
изSPEC94 и fixedC1-MM contour, без нового обучения/выбора по этому набору.
Минимум100admitted events/all3classes; delta>=.02 кhybrid, lowerCI>0,
perclassdrop<=.02. Это retrospectiveMTBS agreement, не prospectivefieldproof.

Все handles **на k8plus**:
- `fires-accuracy-covered-qc-v1` terminalexit0: QC2/4pairs, all3severityclasses represented.
  Это не fulladmission; QC завершён до последней scorer-mask правки.
- `fires-accuracy-covered-prepare-v1` PID831033: ждёт успешныйQC с>=1pair,
  затем собирает весь frozenнабор в data-store/severity-covered-v1.
- `fires-accuracy-covered-score-v1` PID831035: ждёт Russianqueue и fullcollection,
  проверяет ready/eventminimum/lineage, затем sealedGPUscore через общийflock.
  Результат research/fresh-covered-v1-result.json; пока нет qualityresult.
- Russianqueue:4/6fullmodels готовы, control-s3 15/30epochs на последнем срезе,
  далее reserve-s3 иcomparison. ActualGPUprocess828563,8802MiB.

Свободно около14GiB. Freshcollection проверяет>=8GiB перед каждым новымevent.
Код сcachedsource/grid/hash/model/count/Fmask checks:6regressions PASS наk8plus;
makegatePASS. Targeted code-review HIGH cache bypasses исправлены; повторная
проверка не нашлаHIGH. Общая accuracy цель и L2A admission ещё не закрыты.

SPEC100-PREP001 INSUFFICIENT_EVIDENCE: QC2pairs, classes[8838,4691,3478],
fullcollection running; actual frozen models inference pending. Последний
HIGH reviewfix26f50f8 independently checked: noHIGH/CRITICAL. 6remote regressions
PASS. Это scoped review, не подтверждение общей accuracy цели.

## Российский итог (последнее завершённое измерение)

SPEC92 rejected, recipe полностью выполнен. Reserve0.323049/control0.245309,
Prithvi0.206216/C1-MM0.326108. Significantgain toPrithvi only; FLOGA delta−0.020882
violates−0.01. FullreceiptFAIL, originalGPUqueue exit0terminal, не запускатьзаново.
Errorreport sourcebound:77observableseedcontours, misses10/7/14 (reserve/C1-MM/Prithvi),
не путать с missedwindows3/7/7 и spatialclusters1/2/3. Родительский SHA
242acac3cf2ccd30728e5919997895e7cf1d83de223a8802b06f3c4daaaa296d.
Новаяseveritycollection всёещёживая; SPEC100scorer ждётfulladmission.

## SPEC101: полный HLS replay + реальные L2A views

Старая SPEC93/97 dataadmission остаётсянепройденной. НовыйSPEC101 (68ead53,
implementation8f576e7) сохраняет fulloriginalfit929: HLS540/MTBS300/EMS49/FLOGA40.
Oddepochs — весь исходныйvalidHLSGT в обоих вариантах; evenepochs сtwin:
controlHLS/replayL2A сОДИНАКОВОЙcommonmask иunknownpixel fill. Fallbackодинаковый.
Каждыйsample посещаетсякаждуюэпоху, Med×2;30epochs/3seeds, batch8,lr3e-5,
normalizationlockedto C1-MMinitializer. GPUflock освобождаетсямеждуjobs.
Это прямыеL2Aweightupdates сполнымreplay, анеPASS L2A-onlyeventcoverage.

RealCPUprepare-v1 finished exit0 at23:51 Oct03:929originalfit/889fitPCtwins/
40fallback,119evaluationwindows (EMS29/FLOGA87groups),237735800originalvalid
pixels and229586644commonpixels. Existing immutable
`data-store/l2a-full-replay-v1/manifest.json` сохранён. Он созданДО исправления
полной5seedbaseline lineage; fulltraining его НЕ должениспользовать.
Currentlocalbuilder creates append-only `manifest-v2.json` (--manifest-name),
verifies/stores all5C1-MM hashes from exactSPEC92result; training initializers3.
Queues/trainer/scorer use v2. Allsource/identity/grid/band/date/SCL/hash checks.
Local5testsPASS by reviewer; earlier4testsPASS remotely before baselinefix.
Scopedcode-review afterrepair noHIGH/CRITICAL. FullGPUtraining НЕ стартовало.

**Operationalaccessblock:** sshk8plus asked TailscaleSSH reauthentication.
AwaitingSSHclient execsession64983 (it attempts to start smoke-v1 unit afterauth).
No serverjobs were stopped/restarted. Lastverified SPEC100CPUcollection alive;
currentstatus afterreauth must be read, not inferred. AuthURL onlyinchat/browser,
notinrepo. No computation/download onomarchy.

Nextactions afteraccess:
1. Resume session64983 andinspect fires-accuracy-full-replay-smoke-v1 actualstatus.
   Ifit ran, itused oldremotev1code/data; preserve itsartifacts, don't callit
   validationof latestlineagefix. Neveredit numericalcode during a livejob.
2. Beforeoverwriting remote prepare script, optionallyarchive itsv1text (observed
   SHA9cb144f8d9245544d82120c9db08e34a3aa13968d51fd6255446523c79e15d01)
   ascompactcode evidence. Sync only SPEC101files from thisworktree; don'ttouch
   sealedSPEC100 or oldmodels/data. Latestbaselinefix islocalonly untilthissync.
3. New CPUunit fires-accuracy-full-replay-prepare-v2 runs preparecommand inSPEC101
   with `--manifest-name manifest-v2.json`. Neveroverwrite v1. Verify929/889/
   119counts,ready,5baselinehashes; run5regressions onk8plus.
4. Use distinctsmoke-v2unit/outputdirectories ifsmoke-v1 artifacts exist.
   Smoke exercises actualodd/even/missingtwin andlastbatch1 paths, notaccuracy.
   Afterbothvariants terminalexit0 launch queue_l2a_full_replay.sh as own
   systemduserunit CPU400%/MemoryMax18G/OMP4. Perjobflock letsSPEC100score run.
5. Revalidatestatus of fires-accuracy-covered-prepare-v1 and-covered-score-v1.
   No restart fromobservationtimeout. Finishedresult cannot be assumed absent.

Completion stillrequires realaccuracy acceptance/fullgoal; source counts,
negativeoutcomes,code-review andgreen gate do not establish superiority.

## Возобновление2026-10-04

SSHreauth completed; pendingoldsmoke-v1unit notfound, noartifacts. SPEC100
collection173pairs andscore finishedexit0. Equal0.433595/hybrid0.407446,
delta+0.026149 CI[−0.007822,0.057260]: rejected/frozenFAIL, nopromotion.
Latest SPEC101code syncedtok8plus;5testsPASS remotely. Archivedactual
v1prep source9cb144… ascompactcodefixture beforeoverwrite. Newprepare-v2
finishedexit0:929/889/119,all5baselineSHAs,readytrue. Smoke-v2 active under
commonGPUflock, waits/usesmanifest-v2. Fulltrainingnotyetstarted.

## SPEC101 fulltrainingstarted (latest)

Prepare-v2 terminal0;5latestregressionsPASS onk8plus. Smoke-v2 terminal0,
bothvariants completed2epochs, matchedorder/commonmasks, actualL2Aexposures
[2,2,0] versuscontrol[0,0,0], alloriginalGTviews retained. Fullfallbackrecord
isMed×2, exposures[2,2,4]; genericlastbatch1 coveredbyregression, fullbatch8
lastbatch2 matches testedGPUshape. Noaccuracyclaim fromsmoke.
FullGPUqueue fires-accuracy-full-replay-train-v1 MainPID1106633, child1106704
control-s1 epoch17/30 confirmed;5536MiBGPU,~14.8GBMemoryCurrent under18Glimit.
Recipef78ae585bfa0748da0e8791d3dea90b6619443a251668b47376a2d60be20b7a5.
Do notedit sealednumericalsources/queue during6jobs. Expected result
research/l2a-full-replay-v1-result.json. Free disk8.6GiB; no newimagerydownload,
nativecheckpoints only. Goalactive; access/preparation/deployment resolved.

## Cheapfusion102/103 outcomes

Bothrejected by frozen lostgroupguard, Russianreservedcal61 remainsunused.
102mean improvesEMS+0.021/FLOGA+0.028 but loses1FLOGAgroup versusC1-MM0;
max lowersFLOGA−0.055. 103fixed64pixelcomponentrescue stillloses1FLOGAgroup;
Russia0.351100(+0.024992), EMS/FLOGA+0.021228/+0.026363, notadopted.
Code/tests/results/plans/FAILreceipts preserved. Do not tune cap postresult.
Only103confirmation scaffold exists, notexecuted/validated onGPU (noteligible).
MainSPEC101GPUqueue stilllive, control-s2 epoch9/30 onlastsnapshot,3rdof6runs.

## Corrected liveSPEC101 handle

Actualchildcommand confirms priorPID1127531 wasreplay-s2 (4thrun), notcontrol-s2.
Fourfullmodels nowcomplete. Queue1106633 active; child1134639 control-s3
(5thrun) startedafterreplay-s2 completed30epochs/exit0. Lastpair.source_recipe
unchangedf78ae585…; do notedit sealednumericalsources. About22seconds/epoch
measured, ~25minutes forlast2models plusactualscoring, not aqualityguarantee.
Russianconfirmation61 stillnotconsumed byrejected102/103.

## SPEC104 eligible and confirmationqueued

FLOGA786775:GT1382pixels, C1-MMTP5 maxp0.5806, pairmax0.3064, meanTP0.
The5TP liein106pixelcomponent with24meanpositivepixels outsideGT;
103small/fullyrejected-component condition neverapplies. No captuning.
104keepsALL C1-MM>=.5 andadds pair>=.95, commonvalid/watermask.
Developmenteligible:Russia0.352974(+.026865 CI[-.000609,.046231]), EMS+.008873,
FLOGA-.003932 within-.01, noaddedmiss. 2testsPASS; scopedreviewnoHIGH.
Notpromoted/PASS. Reservedcal61 nowfrozenfor oneconfirmation, no tuning.
Queue fires-accuracy-reference-additions-confirmation-v1 PID1144797 active,
waitingGPUflock. Main101queue1106633 child1141654 replay-s3 (6thrun),
epoch12/30 lastverified. ~7minutestraining thenboth actualmeasurements.
Do notedit frozen104plan/source or101numericcode whilelive. Results:
research/reference-additions-v1-confirmation.json and
research/l2a-full-replay-v1-result.json. Free6.3GiB; no imagerydownloads.

## SPEC104 confirmation finished

Terminalexit0, rejectedbyfullfrozencontract. Fusion0.231959 versusC1-MM0.205915
andPrithvi0.142659, positiveCIs toboth; reservealone0.304575, fusion−0.072616
violates−0.01. FAILreceipt; no promotion/thresholdchange. Cal61consumed,
never again callitunused/independentconfirmation afterthischoice.
All6SPEC101models complete,929fitIDs each,min15fulloriginalviews,matching
order/commonmasktraceforall3pairedseeds,14250actualL2Aviews perreplayseed.
GPUscorer child1149805 active underqueue1106633; resultstillpending.
Free5.4GiB; nonewimagerydownloads/duplicateweights.
