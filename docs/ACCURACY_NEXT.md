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

## All currentGPUjobs terminal; SPEC101 outcome

101sixruns+score terminalexit0, rejected:replay0.674091/continuedHLS0.671949/
originalC1-MM0.673324. Deltatooriginal+.000766 CI[−.003273,.004937], below.01.
Full929retained/min15originalexposures,14250L2Aviews perseed/matchedordering
and masks verified. Sourceadaption executed, no usefulaccuracygain established.
Allmodels/results/FAILreceipt preserved; do not rerun oldimmutablequeue.
104also rejecteddespite C1-MM/Prithviwins oncal61 becausepairedreservebetter.
Cal61nowconsumed;futureuse isdevelopment, notunusedconfirmation.
Originalusergoal stillnotproven across4directions, nevermarkcomplete onnegative
experiments orsinglecompetitorwins. Free5.4GiB: newdatacollection must have
a predeclared boundedstorage policy andkeep originals, not silentlydelete data.

## Next full-scope geographicvalidation105 (planned)

Architecturereview recommends fixedmodel/newdata, no morefusion/cap tuning.
Executedsecondary bootstrap ofpreregistered reserveonconsumedcal61:delta+.098660
toC1-MM CI[.033732,.156405],+.161916toPrithvi CI[.092576,.294923]. Not borrowed
fusionintervals, notfreshconfirmation afterselection. Existing FAILs preserved.
Metadata Western65–80E/54–65N:10143validcontours/7635>=1ha. Noimages downloaded,
no groups/admission verifiedyet. SPEC105 plannedbeforecode:wholegrouppriorpurge,
cap96frames/min50windows20groups,1.5GiBnewdatabudget/freefloor3GiB, unchanged
pairedweights againstC1-MM5/Prithvi, delta.01/positiveCI toboth/missguard.
No currentGPUjobsrunning; allsix101models+score and104confirm terminal0.
User4partgoal stillactive/unachieved; newregionalvalidationdoesnotclosegrade
orL2A usefulgain. No originaldata/modeldeletion orMac imagery/weight downloads.

## SPEC-105: collection and confirmation queue launched

At 2026-10-04 06:21 UTC the collector on k8plus was alive as PID 1185241,
with 55 completed frame metadata files and 539,735,684 bytes of new data.
Free disk was 5,092,061,184 bytes. Full admission is not established until
the collector writes its final manifest; no accuracy claim follows from this
snapshot. The scoring queue was alive as PID 1185243, waiting for collection.

Frozen selection-v2 contains 96 windows from 37 groups. Its SHA256 is
89bba735a19cc2aad34666f6e5e3e92720113a82b5f5e96b98d571d6be1cd116.
All three exclusion sources were checked for exact identity and population:
SPEC-96 2,658 footprints, SPEC-92 447 frames, SPEC-100 173 frames. Whole-group
purging uses the original EPSG:3576 / 20 km rule. Four regressions passed on
k8plus and locally; local Python lacks pytest, so local execution used unittest.
The traceability gate and its 14 negative tests passed.

Live units: fires-accuracy-western-collect-v1.service and
fires-accuracy-western-score-v1.service. Do not restart them because a polling
call times out. The queue requires final admission of at least 50 windows and
20 groups, then obtains research/accuracy-gpu.lock and checks free GPU memory.
Expected result: research/western-holdout-v1-result.json. Copy only compact
manifests, results and receipts to Mac; preserve all old remote images/weights.
SPEC-105 remains active. Regional model promotion needs its own spec even if
this comparison passes; the original four-direction objective remains open.

## SPEC-105 finished: negative confirmation

Both units terminated with exit 0. The collector admitted all 96 windows /
37 groups, zero failures, 949,699,940 bytes. Scoring passed source/grid/interval/
supervision checks and used recipe 0ffdcd295a60a5083b44a30886380f29f59d7eb578ffc2f42646e2bf1d28029c.
Fourteen numeric source files match the committed local worktree. Compact
manifests, recipe, result and reconstruction audit are saved locally; images,
weights and prediction arrays remain on k8plus. No assets were deleted.

Reserve IoU 0.220628; C1-MM 0.183899; Prithvi 0.084663. Reserve minus C1-MM
is +0.036730, CI [-0.088181, 0.121254]; minus Prithvi +0.135965,
CI [0.088455, 0.197404]. Missed positive groups: reserve 2, C1-MM 0,
Prithvi 2. SPEC-105 rejected: C1-MM superiority is inconclusive and the
missed-group safeguard fails. Do not promote regional routing from this result.
All 24 relevant tests plus 11 subtests passed on k8plus. Local reconstruction
exactly reproduced aggregate counts, masks summary and both bootstrap CIs.

The western population is now consumed, like original calibration61. Any
post-result analysis or changed recipe on these data is development. Next
action is an explicit spec for failure diagnosis before further training;
there is no current GPU job and free disk is about 4.1 GiB. Original four-part
goal remains active: one comparator win does not establish the requested end state.

## SPEC-106 diagnosis completed

CPU-only, 15.909 seconds, exit 0, 500.2 MiB peak. Five tests passed, all
96 parent confusions reconstructed from signed caches. Report and diagnostic
PASS receipt saved; this PASS establishes diagnosis execution, not accuracy.
One four-window group holds 52.88% of observed labelled pixels; removing it
would reverse the reserve-minus-C1-MM point delta to -0.053299. Do not remove it.
All four reserve-missed windows have no water-masked truth; their median
dNBR is negative, post NDVI 0.672–0.785. Thus water is not the explanation,
and temporal label/source semantics need checking before more training.
This is not proof of misannotation; all old FAILs and raw evidence remain.
No running GPU/collector jobs. Next: explicit temporal-source audit spec,
using publisher documentation, without changing old supervised datasets.

## SPEC-107 source audit completed

PANGAEA and author archive independently confirm the Jan1–May15 2020 fire
interval. Paper full text read at publisher-provided ResearchGate mirror,
sections 2.2–2.4/4: authors used multiple in-period observations, rather than
our fixed May–July scene pair. ZIP schema confirms area only, no fire dates.
Our pairing is an approximation, not proven wrong ground truth. Observed
negative dNBR/green post scenes can reflect recovery/phenology or an individual
label mismatch; neither is established causally. Raw HTML and hashes on
k8plus, compact schema/source audit and source-linked report in the worktree.
No old label/data/recipe/result altered. Next safe experiment: bounded temporal
visibility probe of the four missed windows, under a new spec, with strict SCL,
no inferred ignition day and no independent accuracy claim. No running GPU job.

## SPEC-108 temporal probe completed

Frozen sixteen bins / four missed windows, original MGRS and pixel grid.
First execution exited1 at JSON serialization of a NumPy boolean. Execution
FAIL, source snapshots and partial JSON preserved. Fixed serialization and
recovered against the exact original plan, with source-verified existing pixel
arrays; successor result-recovered-v2.json, exit0, no original overwrite.
Eight image/SCL pairs retained, seven usable observations, 43,469,115 total
bytes. Six regressions passed. No imagery or weights copied to Mac.

Three windows are temporally assessable (two dates / >=14 days); only one
shows required extra spectral separation, minimum two. SPEC-108 rejected,
scientific FAIL. Another window shows separation on May4 but has only one
usable date; April24 common coverage0.496784 does not meet .5. Do not tune
the rule or conclude all multi-date models are ineffective. It is not an
independent accuracy comparison and does not close the four-direction goal.
All probe units terminal; GPU unused, free disk about4.0 GiB. Next candidate
must retain the strong single-date signal that the unconstrained paired model
lost, rather than repeating date/threshold sweeps on these four examples.

## SPEC-109 anchored correction: preflight passed, full experiment pending

New spec committed before implementation. Frozen C1-MM with separately
encoded pre/post features, rank8 projections and zero-initialized correction
head. Pair versus capacity-matched post-only adapter; bounded logit delta
[-2,+4]. Base BN stays eval and all base parameters frozen. Scoring must anchor
original C1-MM5 probabilities, with mean3 learned corrections; do not substitute
the three-seed backbone mean for the deployed five-seed reference.

Five CPU regressions passed on k8plus. Real512 fit-window GPU preflight passed
both variants: exact initial FP32/BF16 logits, finite/nonzero adapter gradients,
frozen backbone/BN, native adapter-only round trip. Exit0, 5.288 seconds,
1.6 GiB host peak. 105,689 trainable parameters and 428,778 bytes per checkpoint;
no ~498MB backbone copy. Smoke result and recipe copied as compact JSON only.
Smoke receipt INSUFFICIENT_EVIDENCE for full SPEC-109, no accuracy promotion.

Remaining work: implement full trainer, six-run queue and scorer under the
frozen SPEC-109 recipe. Fit374+Russian309, all683 every epoch, repeated original
Med weights,30epochs/batch8/AdamW1e-3/cosine/BF16, locked initializer mean/std,
same batches/augmentations for pair/post seeds1/2/3. Preserve original bases.
Use disk floor3GiB and <=512MiB new outputs: only adapter weights and packed
binary predictions/counts, not duplicate backbones or full floating caches for
all old development populations. Full training has NOT started. All own units
terminal, GPU free. Existing populations are consumed development; eligibility
cannot become independent confirmation or a claim of achieving the full goal.

## SPEC-109 full experiment launched (latest)

Full trainer, six-run queue, packed-mask scorer and score queue implemented,
11 tests passed in the three relevant test files. Old reference caches checked
on all populations77/61/96/60/81; Russian per-window counts exactly reconstruct
each immutable parent. No reference re-inference, new data or backbone copy.
Training source seal8a9654584ff0d93f0256c18a0e8dba13c63916873186126a0610aa51252a7d3d
matches committed local numeric sources. Do not edit any sealed training source
while the queue is live; scoring code/queue are separate from the training seal.

At progress audit2026-10-04, pair-s1/post-s1/pair-s2 completed30epochs,
683 unique IDs each, minimum30 exposures, frozen backbone digest preserved.
The first seed's pair/post ordered batches and exposure counts match exactly.
Queue fires-accuracy-temporal-adapter-train-v1 PID1238180 remains active;
child1255988 was post-s2 epoch13/30 at the snapshot. GPU about4.8GiB,
host MemoryCurrent about12GiB within18G. About20seconds/epoch observed.
Score queue fires-accuracy-temporal-adapter-score-v1 PID1258419 waits for all
six complete summaries, then takes the same GPU lock and checks22GB free.

Expected result: research/temporal-adapter-v1/development.json. Both variants
anchor the original C1-MM5 mean probability and apply their mean3 corrections.
Lossless packed binary masks avoid full floating caches; <=512MiB new-output
budget and3GiB free floor retained. All existing populations are development.
Accuracy has NOT been measured yet. Goal active; SPEC-109 not implemented/
verified from training losses or the partial queue. Do not restart live handles.

## SPEC-109 completed: pooled win, full rejection

All6 full models and scorer terminalexit0; exact matched batches/exposure/
frozen base digest in all3seed pairs,683IDs each,30/60 exposures. Packed
predictions reconstruct every candidate count and all CI/eligibility values.
Total outputs28.5MB; no backbone copy or old-asset deletion, free~4.0GiB.
Independent review codeAPPROVE/architectureWATCH, finalCOMMENT, recorded in
docs/SPEC-109-REVIEW.md; no model-promotion approval from the review.

Pooled Russia234/90:pair0.249756,post0.233591,C1-MM0.219247,Prithvi0.116043.
PositiveCIs to all3:delta toC1+.030510 CI[.002114,.046720], tocontrol+.016165
CI[.009942,.026590]. But calibration/Western degrade from oldreserve-.042548/
-.022556, FLOGA-.018750 toC1-MM and1extra missedgroup. Full109rejected.
Do not waive guards, promote a regional subset or claim independent superiority.
Weights/caches/results/FAILreceipt preserved. All prior populations consumed.
SPEC110 CPU raw-MTBS attribution also finished, pending receipt/report closure.
No own GPU job remains; original four-direction goal active and unachieved.

## SPEC-110 MTBS attribution completed

Three tests, CPU exit0, exact reconstruction of345 old confusions. Raw0 is
outside MTBS assessment; raw5 is burned increased greenness, not no-fire.
Report docs/SPEC-110-MTBS-LEGEND.md; data audit/summary and PASS receipt for
decomposition only. Legacy full-frame metric, labels and FAILs unchanged.
69.10% /48.13% of legacy background grade FP lie on raw0 in dev/covered;
same fixed contour makes these counts identical across grade heads. On
actually reached raw2 pixels, neural equal head helps substantially; raw3
moderate remains worse than hybrid. Do not fix this by hiding unassessed
pixels or interpreting class5 as reviewed negatives. Separate contour coverage
from damage grading in further experiments. All jobs terminal, no GPU work
running; no model promoted. Four-direction user objective remains active.

## SPEC-111 learned monotonic expansion launched

Spec6abc229 before code54d3913. New probability p_new=p_C1+(1-p_C1)*q;
q is learned from paired/post features, bias-4 prior declared, no exact initial
identity claim. Weak C1 positives now preserved on identical valid/water pixels;
explicit scorer subset check. C1 false positives also preserved; IoU gain is
not guaranteed. Same frozen encoders/BN, fit mean/std,683ID full pool/762views,
30epochs/AdamW1e-3/cosine/BF16, three capacity-matched pair/post seed pairs.
Original109 files/weights/results/seals unchanged. All old guards reused.

Eleven tests and real512 GPU preflight passed both variants in FP32/BF16,
finite/nonzero gradient and frozen base/BN, small native state round trip.
Smoke exit0/5.522seconds/1.6GiB host peak;428,778-byte checkpoints.
Independent review codeAPPROVE/architectureWATCH; finalCOMMENT in
docs/SPEC-111-REVIEW.md. Structural invariant and theoretical ceiling are not
quality acceptance. Each adapter's training base differs from scoring C1-MM5;
post control uses the same declared construction.

Full train unit fires-accuracy-recall-expansion-train-v1 PID1297082 active,
child1297085 pair-s1 epoch2/30 at snapshot. Score unit
fires-accuracy-recall-expansion-score-v1 PID1297106 waiting for all6 summaries.
Training seal0f5769bd961f98a205947738aa650513d7c7ef24281ab614471902ad8ac9b6f5
matches local committed source. Do not edit its numeric sources/queue whilelive
or restart because observation times out. Expected result
research/recall-expansion-v1/development.json. Free disk~4.0GiB, no new data,
no original deletion, <=512MiB small weights/packed masks,3GiB free floor.
SPEC111 active; candidate accuracy NOT measured. Full four-part goal active.
