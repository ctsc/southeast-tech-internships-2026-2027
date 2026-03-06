"""Shared database I/O utilities for loading and saving JobsDatabase files.

Provides load_database() and save_database() used across the pipeline:
deduplicate, check_links, validate, el_validate, and archive_stale.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils.models import JobsDatabase

logger = logging.getLogger(__name__)


def load_database(path: Path) -> JobsDatabase:
    """Load a jobs JSON file into a JobsDatabase model.

    Args:
        path: Path to the jobs JSON file.

    Returns:
        The current jobs database, or an empty one if the file
        is missing, empty, or cannot be parsed.
    """
    if not path.exists():
        logger.warning("%s not found, starting with empty database", path.name)
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path.name, exc)
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    if not raw or not raw.get("listings"):
        logger.info("%s is empty, starting with empty database", path.name)
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    try:
        return JobsDatabase.model_validate(raw)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))


def save_database(db: JobsDatabase, path: Path) -> None:
    """Update stats, set last_updated, and save to a jobs JSON file atomically.

    Writes to a .tmp file first, then renames for atomic replacement.

    Args:
        db: The JobsDatabase to persist.
        path: Path to the jobs JSON file.
    """
    db.last_updated = datetime.now(timezone.utc)
    db.compute_stats()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(mode="json"), f, indent=2, default=str)
    tmp_path.replace(path)

    logger.info(
        "Saved %s: %d listings, %d open", path.name, len(db.listings), db.total_open
    )
