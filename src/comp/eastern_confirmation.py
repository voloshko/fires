"""SPEC-113: fail before data I/O without genuine candidate eligibility."""
from pathlib import Path
from threading import Lock
import shutil

SELECTION_SHA = "0ab2edea7db92eca8e27da5b179ed5b373314d44f025f3019444be162cb74587"
DATA_BUDGET = 768*1024**2
FREE_FLOOR = 3*1024**3
RESERVATION = 14*1024**2


def require_eligible(candidate):
    expected = {"pooled_gain", "regional_points", "mediterranean", "missed_groups"}
    if (candidate.get("spec_id") != "SPEC-111" or candidate.get("eligible") is not True
            or set(candidate.get("guards", {})) != expected
            or any(value is not True for value in candidate["guards"].values())):
        raise ValueError("complete eligible SPEC111 candidate required before external I/O")


def require_admission(manifest):
    if (manifest.get("spec_id") != "SPEC-113" or not manifest.get("ready_for_evaluation")
            or manifest.get("qc_probe") or manifest.get("selection_sha256") != SELECTION_SHA
            or manifest.get("counts", {}).get("samples", 0) < 50
            or manifest.get("counts", {}).get("groups", 0) < 20):
        raise ValueError("actual50windows/20groups source admission required")


def storage_guard(used, reserved, free):
    if min(used, reserved, free) < 0 or used+reserved > DATA_BUDGET:
        raise ValueError("eastern confirmation imagery budget unavailable")
    if free-reserved < FREE_FLOOR:
        raise ValueError("eastern confirmation free-space floor unavailable")


class StorageBudget:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock, self.reserved = Lock(), 0

    def used(self):
        files = list(self.directory.rglob("*"))
        if any(p.is_symlink() for p in files):
            raise ValueError("owned output cannot contain symlinks")
        return sum(p.stat().st_size for p in files if p.is_file())

    def reserve(self):
        with self.lock:
            storage_guard(self.used(), self.reserved+RESERVATION, shutil.disk_usage(self.directory).free)
            self.reserved += RESERVATION

    def release(self):
        with self.lock:
            self.reserved -= RESERVATION

    def check_write(self):
        storage_guard(self.used(), self.reserved, shutil.disk_usage(self.directory).free)
