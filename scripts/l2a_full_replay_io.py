"""SPEC-101: independently gated full-pool manifest, not SPEC93 admission."""
import json
from src.comp.accuracy_contract import sha256_file,validate_manifest
from src.comp.l2a_full_replay import REFERENCE_RESULT_SHA256


def verify_original_population(fit,jobs):
    expected={r["sample_id"]:r for r in jobs if r["partition"]=="fit"}
    if len(expected)!=len(fit) or set(expected)!={r["sample_id"] for r in fit}:
        raise ValueError("original source population differs")
    for record in fit:
        job=expected[record["sample_id"]]
        for key in ("source","dataset","group_id","partition","sampling_weight"):
            if record[key]!=job[key]:raise ValueError("original source metadata differs")
        if record["files"]["hls"]!=job["source"]:
            raise ValueError("original source image was substituted")


def verify_inputs(root,record):
    for section in (record,record.get("twin")):
        if section is None:continue
        for key,relative in section["files"].items():
            if sha256_file(root/relative)!=section["hashes"][key]:
                raise ValueError("frozen replay input changed")


def verify_baseline(paths,expected):
    if len(paths)!=5 or len(expected)!=5:raise ValueError("incomplete five-seed baseline")
    if [sha256_file(path) for path in paths]!=expected:
        raise ValueError("five-seed C1-MM baseline differs from frozen reference")


def files_for(record):
    yield from record["files"].items()
    if record.get("twin") is not None:
        yield from record["twin"]["files"].items()


def load_manifest(root,path):
    manifest=json.loads(path.read_text())
    if manifest["spec_id"]!="SPEC-101" or not manifest["ready_for_replay"]:
        raise ValueError("no full-replay admission")
    if manifest["reference_result_sha256"]!=REFERENCE_RESULT_SHA256:
        raise ValueError("initializer source result differs")
    if len(manifest.get("baseline_model_hashes",[]))!=5:raise ValueError("baseline lineage is incomplete")
    records=manifest["records"];validate_manifest(records)
    fit=[r for r in records if r["partition"]=="fit"]
    if len(fit)!=929 or manifest["full_original_fit_count"]!=len(fit):
        raise ValueError("original population lost records")
    from scripts.prepare_l2a_finetuning import source_jobs
    registry=root/"hls-events.json"
    if sha256_file(registry)!=manifest["event_registry_sha256"]:raise ValueError("event registry changed")
    verify_original_population(fit,source_jobs(root,registry))
    evaluation=[r for r in records if r["partition"]=="evaluation"]
    if sum(r["twin"] is not None for r in fit)<300 or len(evaluation)<50:
        raise ValueError("replay/evaluation population below admission")
    if any(len({r["group_id"] for r in evaluation if r["dataset"]==d})<10 for d in ("ems","floga")):
        raise ValueError("evaluation lost required event groups")
    for record in records:
        verify_inputs(root,record)
    return manifest
