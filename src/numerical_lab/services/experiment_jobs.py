from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExperimentJob:
    job_id: str
    job_type: str
    status: str = "queued"   # queued, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_JOBS: Dict[str, ExperimentJob] = {}
_LOCK = threading.Lock()


def create_job(job_type: str, message: str = "") -> ExperimentJob:
    job = ExperimentJob(
        job_id=uuid.uuid4().hex[:12],
        job_type=job_type,
        message=message,
    )
    with _LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[ExperimentJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def update_job(job_id: str, **kwargs) -> Optional[ExperimentJob]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            setattr(job, k, v)
        return job


def list_jobs() -> list[ExperimentJob]:
    with _LOCK:
        return list(_JOBS.values())