"""Deduplication engine for internship listings.

Applies three dedup strategies: content hash, URL, and fuzzy matching.
Keeps the newest listing when duplicates are found.
"""

import json
import logging
from datetime import datetime, timezone

from thefuzz import fuzz

from scripts.utils.config import PROJECT_ROOT
from scripts.utils.models import JobListing, JobsDatabase

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"
ARCHIVED_PATH = DATA_DIR / "archived.json"


def _load_database() -> JobsDatabase:
    """Load data/jobs.json into a JobsDatabase model.

    Returns:
        The current jobs database, or an empty one if the file
        is missing or contains no listings.
    """
    if not JOBS_PATH.exists():
        logger.warning("jobs.json not found, starting with empty database")
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw or not raw.get("listings"):
        logger.info("jobs.json is empty, starting with empty database")
        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    return JobsDatabase.model_validate(raw)


def _load_archived_hashes() -> set[str]:
    """Load archived listing IDs (content hashes) for repost detection.

    Returns:
        A set of all listing IDs in archived.json, or an empty set
        if the file is missing or empty.
    """
    if not ARCHIVED_PATH.exists():
        logger.debug("archived.json not found, no archived hashes")
        return set()

    with open(ARCHIVED_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw or not raw.get("listings"):
        return set()

    return {listing["id"] for listing in raw["listings"] if "id" in listing}


def _save_database(db: JobsDatabase) -> None:
    """Update stats, set last_updated, and save to data/jobs.json.

    Args:
        db: The JobsDatabase to persist.
    """
    db.last_updated = datetime.now(timezone.utc)
    db.compute_stats()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(
        "Saved jobs.json: %d listings, %d open", len(db.listings), db.total_open
    )


def _dedup_by_content_hash(
    listings: list[JobListing],
) -> tuple[list[JobListing], int]:
    """Remove listings that share the same content hash (id).

    When duplicates are found, the listing with the most recent
    date_added is kept.

    Args:
        listings: The list of listings to deduplicate.

    Returns:
        A tuple of (deduplicated listings, count removed).
    """
    seen: dict[str, JobListing] = {}
    removed = 0

    for listing in listings:
        if listing.id in seen:
            existing = seen[listing.id]
            # Keep the newer one
            if listing.date_added > existing.date_added:
                logger.info(
                    "Dedup (hash): removed duplicate of %s - %s (keeping newer)",
                    existing.company,
                    existing.role,
                )
                seen[listing.id] = listing
            else:
                logger.info(
                    "Dedup (hash): removed duplicate of %s - %s (keeping newer)",
                    listing.company,
                    listing.role,
                )
            removed += 1
        else:
            seen[listing.id] = listing

    return list(seen.values()), removed


def _dedup_by_url(
    listings: list[JobListing],
) -> tuple[list[JobListing], int]:
    """Remove listings that share the same apply URL.

    When duplicates are found, the listing with the most recent
    date_added is kept.

    Args:
        listings: The list of listings to deduplicate.

    Returns:
        A tuple of (deduplicated listings, count removed).
    """
    seen: dict[str, JobListing] = {}
    removed = 0

    for listing in listings:
        url_key = str(listing.apply_url)
        if url_key in seen:
            existing = seen[url_key]
            if listing.date_added > existing.date_added:
                logger.info(
                    "Dedup (url): removed duplicate of %s - %s (same URL as %s - %s, keeping newer)",
                    existing.company,
                    existing.role,
                    listing.company,
                    listing.role,
                )
                seen[url_key] = listing
            else:
                logger.info(
                    "Dedup (url): removed duplicate of %s - %s (same URL as %s - %s, keeping newer)",
                    listing.company,
                    listing.role,
                    existing.company,
                    existing.role,
                )
            removed += 1
        else:
            seen[url_key] = listing

    return list(seen.values()), removed


def _compute_token_overlap(title_a: str, title_b: str) -> float:
    """Compute Jaccard similarity between tokenized titles.

    Args:
        title_a: First title string.
        title_b: Second title string.

    Returns:
        Jaccard similarity (0.0 to 1.0). Returns 0.0 if both are empty.
    """
    tokens_a = set(title_a.lower().split())
    tokens_b = set(title_b.lower().split())

    if not tokens_a and not tokens_b:
        return 0.0

    union = tokens_a | tokens_b
    if not union:
        return 0.0

    intersection = tokens_a & tokens_b
    return len(intersection) / len(union)


def _dedup_fuzzy(
    listings: list[JobListing], archived_hashes: set[str]
) -> tuple[list[JobListing], int]:
    """Remove listings with similar company names and overlapping role titles.

    Uses thefuzz.fuzz.ratio for company name comparison (> 90 threshold)
    and Jaccard token overlap for role titles (> 0.8 threshold).

    Listings whose IDs appear in archived_hashes are treated as reposts
    and are not deduplicated away.

    Args:
        listings: The list of listings to deduplicate.
        archived_hashes: Set of IDs from archived.json for repost detection.

    Returns:
        A tuple of (deduplicated listings, count removed).
    """
    if len(listings) <= 1:
        return listings, 0

    removed_indices: set[int] = set()
    removed = 0

    for i in range(len(listings)):
        if i in removed_indices:
            continue

        # Skip repost detection for archived listings
        if listings[i].id in archived_hashes:
            continue

        for j in range(i + 1, len(listings)):
            if j in removed_indices:
                continue

            if listings[j].id in archived_hashes:
                continue

            # Compare company names using fuzzy ratio
            company_similarity = fuzz.ratio(
                listings[i].company.lower(), listings[j].company.lower()
            )

            if company_similarity <= 90:
                continue

            # Compare role titles using Jaccard token overlap
            token_overlap = _compute_token_overlap(
                listings[i].role, listings[j].role
            )

            if token_overlap <= 0.8:
                continue

            # These are fuzzy duplicates — keep the newer one
            if listings[j].date_added >= listings[i].date_added:
                # j is newer or same age, remove i
                logger.info(
                    "Dedup (fuzzy): %s '%s' ~ %s '%s' — keeping newer",
                    listings[i].company,
                    listings[i].role,
                    listings[j].company,
                    listings[j].role,
                )
                removed_indices.add(i)
                removed += 1
                break  # i is removed, move to next i
            else:
                # i is newer, remove j
                logger.info(
                    "Dedup (fuzzy): %s '%s' ~ %s '%s' — keeping newer",
                    listings[j].company,
                    listings[j].role,
                    listings[i].company,
                    listings[i].role,
                )
                removed_indices.add(j)
                removed += 1

    result = [
        listing for idx, listing in enumerate(listings) if idx not in removed_indices
    ]
    return result, removed


def deduplicate_all() -> int:
    """Run all deduplication strategies on the jobs database.

    Executes dedup in order: content hash, URL, then fuzzy matching.
    Saves the updated database after processing.

    Returns:
        Total number of duplicates removed.
    """
    db = _load_database()

    if not db.listings:
        logger.info("No listings to deduplicate")
        return 0

    logger.info("Starting deduplication on %d listings", len(db.listings))

    archived_hashes = _load_archived_hashes()

    # Stage 1: Content hash dedup
    listings, hash_removed = _dedup_by_content_hash(db.listings)

    # Stage 2: URL dedup
    listings, url_removed = _dedup_by_url(listings)

    # Stage 3: Fuzzy dedup
    listings, fuzzy_removed = _dedup_fuzzy(listings, archived_hashes)

    total_removed = hash_removed + url_removed + fuzzy_removed

    logger.info(
        "Deduplication complete: %d hash, %d URL, %d fuzzy — %d total duplicates removed",
        hash_removed,
        url_removed,
        fuzzy_removed,
        total_removed,
    )

    db.listings = listings
    _save_database(db)

    return total_removed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    total = deduplicate_all()
    logger.info("Deduplication finished: %d duplicates removed", total)
