"""Async link health checker that verifies all application URLs are still active.

Uses httpx.AsyncClient for async HEAD requests with a concurrency limiter.
Tracks consecutive failures in data/link_health.json; a listing must fail
2 consecutive checks across 2 runs before being marked CLOSED.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

import httpx

from scripts.utils.config import PROJECT_ROOT
from scripts.utils.models import JobsDatabase, ListingStatus

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"
LINK_HEALTH_PATH = DATA_DIR / "link_health.json"

MAX_CONCURRENT = 10
REQUEST_TIMEOUT = 10.0
USER_AGENT = "InternshipTracker/1.0 (github.com/ctsc/atlanta-tech-internships-2026)"

# Status code classifications
DEAD_STATUSES = {404, 410, 403}
TRANSIENT_STATUSES = {429, 500, 502, 503}


def _load_database() -> JobsDatabase:
    """Load data/jobs.json into a JobsDatabase model.

    Returns:
        The current jobs database, or an empty one if missing/invalid.
    """
    if not JOBS_PATH.exists():
        logger.warning("jobs.json not found, returning empty database")
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    try:
        with open(JOBS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return JobsDatabase.model_validate(raw)
    except Exception as exc:
        logger.error("Failed to parse jobs.json: %s", exc)
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))


def _save_database(db: JobsDatabase) -> None:
    """Save the jobs database to data/jobs.json.

    Args:
        db: The jobs database to persist.
    """
    db.last_updated = datetime.now(timezone.utc)
    db.compute_stats()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(
        "Saved jobs.json: %d listings, %d open", len(db.listings), db.total_open
    )


def _load_link_health() -> dict[str, Any]:
    """Load the link health tracking data.

    Returns:
        Dict mapping listing IDs to their health records, e.g.
        {"id": {"consecutive_failures": 1, "last_checked": "2026-02-28"}}
    """
    if not LINK_HEALTH_PATH.exists():
        logger.info("link_health.json not found, starting fresh")
        return {}

    try:
        with open(LINK_HEALTH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to parse link_health.json: %s", exc)
        return {}


def _save_link_health(health: dict[str, Any]) -> None:
    """Save the link health tracking data.

    Args:
        health: Dict mapping listing IDs to health records.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LINK_HEALTH_PATH, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, default=str)
    logger.info("Saved link_health.json with %d entries", len(health))


async def _check_single_link(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    listing_id: str,
    url: str,
    company: str,
    role: str,
) -> tuple[str, str, int | None, str | None]:
    """Check a single listing URL via HEAD request.

    Args:
        client: The shared httpx async client.
        semaphore: Concurrency limiter.
        listing_id: The listing's unique ID.
        url: The apply URL to check.
        company: Company name for logging.
        role: Role title for logging.

    Returns:
        Tuple of (listing_id, result_type, status_code, error_message).
        result_type is one of: "healthy", "dead", "transient", "unknown", "error".
    """
    async with semaphore:
        try:
            response = await client.head(
                str(url),
                follow_redirects=True,
                timeout=REQUEST_TIMEOUT,
            )
            status = response.status_code

            if status == 200:
                logger.info(
                    "Healthy (%d): %s — %s", status, company, role
                )
                return (listing_id, "healthy", status, None)
            elif status in DEAD_STATUSES:
                logger.warning(
                    "Dead (%d): %s — %s", status, company, role
                )
                return (listing_id, "dead", status, None)
            elif status in TRANSIENT_STATUSES:
                logger.warning(
                    "Transient error (%d): %s — %s", status, company, role
                )
                return (listing_id, "transient", status, None)
            else:
                logger.warning(
                    "Unknown status (%d): %s — %s", status, company, role
                )
                return (listing_id, "unknown", status, None)

        except httpx.TimeoutException:
            logger.warning(
                "Timeout: %s — %s (%s)", company, role, url
            )
            return (listing_id, "error", None, "timeout")
        except httpx.RequestError as exc:
            logger.warning(
                "Request error: %s — %s: %s", company, role, exc
            )
            return (listing_id, "error", None, str(exc))
        except Exception as exc:
            logger.error(
                "Unexpected error: %s — %s: %s", company, role, exc
            )
            return (listing_id, "error", None, str(exc))


async def check_all_links() -> dict[str, int]:
    """Check all open listing URLs and update their status.

    Loads jobs.json and link_health.json, checks every OPEN listing
    concurrently (max 10 at a time), tracks consecutive failures,
    and marks listings as CLOSED after 2 consecutive failures.

    Returns:
        Stats dict with keys: checked, healthy, closed, transient_errors, unknown.
    """
    db = _load_database()
    health = _load_link_health()
    today_str = date.today().isoformat()

    # Filter to only OPEN listings
    open_listings = [
        listing for listing in db.listings
        if listing.status == ListingStatus.OPEN
    ]

    stats = {
        "checked": 0,
        "healthy": 0,
        "closed": 0,
        "transient_errors": 0,
        "unknown": 0,
    }

    if not open_listings:
        logger.info("No open listings to check")
        return stats

    logger.info("Checking %d open listing links", len(open_listings))

    # Build a lookup by ID for quick access
    listing_by_id = {listing.id: listing for listing in db.listings}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        tasks = [
            _check_single_link(
                client,
                semaphore,
                listing.id,
                str(listing.apply_url),
                listing.company,
                listing.role,
            )
            for listing in open_listings
        ]
        results = await asyncio.gather(*tasks)

    stats["checked"] = len(results)

    for listing_id, result_type, status_code, error_msg in results:
        listing = listing_by_id.get(listing_id)
        if listing is None:
            continue

        if result_type == "healthy":
            # Success: update verification date and reset failure counter
            listing.date_last_verified = date.today()
            if listing_id in health:
                health[listing_id]["consecutive_failures"] = 0
                health[listing_id]["last_checked"] = today_str
            stats["healthy"] += 1

        elif result_type == "dead":
            # Potential dead link: check consecutive failure count
            if listing_id not in health:
                health[listing_id] = {
                    "consecutive_failures": 0,
                    "last_checked": today_str,
                }

            health[listing_id]["consecutive_failures"] += 1
            health[listing_id]["last_checked"] = today_str

            if health[listing_id]["consecutive_failures"] >= 2:
                listing.status = ListingStatus.CLOSED
                logger.warning(
                    "CLOSED (2+ consecutive failures): %s — %s",
                    listing.company,
                    listing.role,
                )
                stats["closed"] += 1
            else:
                logger.info(
                    "First failure for %s — %s (will close on next failure)",
                    listing.company,
                    listing.role,
                )

        elif result_type == "transient":
            stats["transient_errors"] += 1

        elif result_type == "unknown":
            stats["unknown"] += 1

        elif result_type == "error":
            # Network errors count as transient — don't increment failure counter
            stats["transient_errors"] += 1

    # Save updated data
    _save_database(db)
    _save_link_health(health)

    logger.info(
        "Link check complete: %d checked, %d healthy, %d closed, "
        "%d transient errors, %d unknown",
        stats["checked"],
        stats["healthy"],
        stats["closed"],
        stats["transient_errors"],
        stats["unknown"],
    )

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    result_stats = asyncio.run(check_all_links())
    logger.info("Final stats: %s", result_stats)
