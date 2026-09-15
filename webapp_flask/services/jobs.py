"""In-process background-job registry for long-running library calls
(building a reference from a corpus, adding texts, profiling, downloading
a spaCy model).

Each job runs in its own daemon thread; `progress_callback(current, total,
message)` (the exact contract lexical_profiler already uses) just writes
into the shared Job object under a lock, and a client polls
GET /jobs/<id>/progress to render a progress bar.

This is a single-process, in-memory registry -- correct only when the app
runs as one process (`flask run --no-reload`, or a single gunicorn
worker). Swapping this for a Redis/RQ-backed one later shouldn't need to
change the JobManager.start()/get() interface, just its internals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Job:
    id: str
    session_id: str
    status: str = "running"  # "running" | "done" | "error"
    current: int = 0
    total: int = 0
    message: str = "Starting..."
    error: str | None = None
    redirect_url: str | None = None
    result: object | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobManager:
    def __init__(self, job_ttl: int = 30 * 60, run_inline: bool = False):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.job_ttl = job_ttl
        # Set True under TESTING so callers don't have to poll/sleep.
        self.run_inline = run_inline

    def start(self, session_id: str, target_fn, redirect_url: str) -> str:
        job = Job(id=uuid4().hex, session_id=session_id, redirect_url=redirect_url)
        with self._lock:
            self._jobs[job.id] = job

        def progress_callback(current: int, total: int, message: str) -> None:
            with self._lock:
                job.current, job.total, job.message = current, total, message

        def run() -> None:
            try:
                result = target_fn(progress_callback)
                with self._lock:
                    job.result, job.status, job.finished_at = result, "done", time.time()
            except ValueError as e:
                with self._lock:
                    job.error, job.status, job.finished_at = str(e), "error", time.time()
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Background job %s failed", job.id)
                with self._lock:
                    job.error = "An unexpected error occurred."
                    job.status, job.finished_at = "error", time.time()

        if self.run_inline:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()

        self._reap()
        return job.id

    def configure(self, job_ttl: int | None = None, run_inline: bool | None = None) -> None:
        if job_ttl is not None:
            self.job_ttl = job_ttl
        if run_inline is not None:
            self.run_inline = run_inline

    def get(self, job_id: str, session_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        return job if job and job.session_id == session_id else None

    def _reap(self) -> None:
        cutoff = time.time() - self.job_ttl
        with self._lock:
            stale = [jid for jid, job in self._jobs.items()
                     if job.finished_at and job.finished_at < cutoff]
            for jid in stale:
                del self._jobs[jid]
