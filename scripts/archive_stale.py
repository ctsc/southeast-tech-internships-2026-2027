"""Moves closed and stale listings from jobs.json to archived.json.

Archival criteria:
- CLOSED listings whose date_last_verified is older than 7 days.
- Any listing whose date_added is older than 120 days, regardless of status.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.utils.config import PROJECT_ROOT
from scripts.utils.models import JobListing, JobsDatabase, ListingStatus

logger = logging.getLogger(__name__)

JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
ARCHIVED_PATH = PROJECT_ROOT / "data" / "archived.json"

CLOSED_ARCHIVE_DAYS = 7
STALE_ARCHIVE_DAYS = 120


def _load_database(path: Path) -> JobsDatabase:
    """Load a JobsDatabase from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JobsDatabase.model_validate(data)


def _save_database(db: JobsDatabase, path: Path) -> None:
    """Save a JobsDatabase to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(mode="json"), f, indent=2, default=str)


def _should_archive(listing: JobListing, today: date) -> str | None:
    """Determine whether a listing should be archived.

    Returns a reason string if it should be archived, or None if it should stay.
    """
    # Closed listings older than 7 days (using date_last_verified as proxy)
    if listing.status == ListingStatus.CLOSED:
        days_since_verified = (today - listing.date_last_verified).days
        if days_since_verified > CLOSED_ARCHIVE_DAYS:
            return f"closed, last verified {days_since_verified} days ago (>{CLOSED_ARCHIVE_DAYS})"

    # Stale listings older than 120 days regardless of status
    days_since_added = (today - listing.date_added).days
    if days_since_added > STALE_ARCHIVE_DAYS:
        return f"stale, added {days_since_added} days ago (>{STALE_ARCHIVE_DAYS})"

    return None


def archive_stale(
    jobs_path: Path = JOBS_PATH,
    archived_path: Path = ARCHIVED_PATH,
    today: date | None = None,
) -> int:
    """Move closed and stale listings from jobs.json to archived.json.

    Args:
        jobs_path: Path to the active jobs database file.
        archived_path: Path to the archived listings file.
        today: Override for the current date (useful for testing).

    Returns:
        The number of listings that were archived.
    """
    if today is None:
        today = date.today()

    # Load jobs database
    jobs_db = _load_database(jobs_path)

    # Load or create archived database
    if archived_path.exists():
        archived_db = _load_database(archived_path)
    else:
        logger.info("archived.json not found, creating new archive")
        archived_db = JobsDatabase(
            listings=[],
            last_updated=datetime.now(timezone.utc),
            total_open=0,
        )

    keep: list[JobListing] = []
    archived_count = 0

    for listing in jobs_db.listings:
        reason = _should_archive(listing, today)
        if reason is not None:
            logger.info(
                "Archiving: %s — %s (reason: %s)",
                listing.company,
                listing.role,
                reason,
            )
            archived_db.listings.append(listing)
            archived_count += 1
        else:
            keep.append(listing)

    if archived_count == 0:
        logger.info("No listings to archive")
        return 0

    # Update jobs database
    jobs_db.listings = keep
    jobs_db.last_updated = datetime.now(timezone.utc)
    jobs_db.compute_stats()

    # Update archived database
    archived_db.last_updated = datetime.now(timezone.utc)
    archived_db.compute_stats()

    # Save both
    _save_database(jobs_db, jobs_path)
    _save_database(archived_db, archived_path)

    logger.info(
        "Archived %d listings. Jobs: %d remaining, Archive: %d total",
        archived_count,
        len(jobs_db.listings),
        len(archived_db.listings),
    )

    return archived_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    count = archive_stale()
    logger.info("Done. Archived %d listings.", count)
