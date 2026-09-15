"""Module-level singletons shared across blueprints."""

from __future__ import annotations

from .services.jobs import JobManager

job_manager = JobManager()
