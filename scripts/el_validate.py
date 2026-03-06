"""Validation pipeline for entry-level/new-grad job listings.

Loads the most recent raw entry-level discovery file, enriches each listing
via Gemini AI using the entry-level prompt, filters by validation criteria,
and appends valid listings to data/entry_level_jobs.json.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from scripts.utils.ai_enrichment import enrich_listing, reset_budget
from scripts.utils.config import PROJECT_ROOT, get_config, is_big_tech
from scripts.utils.db_io import load_database, save_database
from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    ListingType,
    RawListing,
)
from scripts.validate import (
    _generate_listing_id,
    _infer_category_from_title,
    _map_category,
    _map_industry,
    _map_sponsorship,
    _parse_locations,
    _slugify,
)

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
EL_JOBS_PATH = DATA_DIR / "entry_level_jobs.json"


def _find_latest_raw_el_discovery() -> Optional[Path]:
    """Find the most recent raw entry-level discovery JSON file."""
    raw_files = sorted(DATA_DIR.glob("raw_el_discovery_*.json"))
    if not raw_files:
        logger.warning("No raw entry-level discovery files found in %s", DATA_DIR)
        return None
    latest = raw_files[-1]
    logger.info("Found latest raw entry-level discovery file: %s", latest.name)
    return latest


def _load_raw_listings(path: Path) -> list[RawListing]:
    """Load raw listings from a discovery JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data.get("listings", [])
    listings: list[RawListing] = []
    for item in raw_items:
        try:
            listings.append(RawListing.model_validate(item))
        except Exception as exc:
            logger.warning("Skipping malformed raw listing: %s", exc)
    logger.info("Loaded %d raw entry-level listings from %s", len(listings), path.name)
    return listings


def _load_existing_database() -> JobsDatabase:
    """Load the existing entry-level jobs database."""
    return load_database(EL_JOBS_PATH)


def _get_existing_hashes(db: JobsDatabase) -> set[str]:
    """Extract all listing IDs from the database."""
    return {listing.id for listing in db.listings}


def _build_entry_level_listing(
    raw: RawListing, metadata: dict, config_industries: Optional[dict[str, str]] = None
) -> JobListing:
    """Build a JobListing from raw listing data and AI-enriched metadata."""
    locations = _parse_locations(raw.location, metadata.get("locations"))
    listing_id = _generate_listing_id(raw.company, raw.title, locations)
    today = date.today()

    industry = _map_industry(
        metadata.get("industry", "other"),
        raw.company,
        config_industries or {},
    )

    return JobListing(
        id=listing_id,
        company=raw.company,
        company_slug=_slugify(raw.company),
        role=raw.title,
        category=_map_category(metadata.get("category", "other")),
        locations=locations,
        apply_url=raw.url,
        sponsorship=_map_sponsorship(metadata.get("sponsorship", "unknown")),
        requires_us_citizenship=metadata.get("sponsorship", "").lower() == "us_citizenship",
        is_faang_plus=raw.is_faang_plus or is_big_tech(raw.company),
        requires_advanced_degree=metadata.get("requires_advanced_degree", False),
        remote_friendly=metadata.get("remote_friendly", False),
        open_to_international=metadata.get("open_to_international", False),
        date_added=today,
        date_last_verified=today,
        source=raw.source,
        status=ListingStatus.OPEN,
        tech_stack=metadata.get("tech_stack", []),
        season="n/a",
        industry=industry,
        listing_type=ListingType.ENTRY_LEVEL,
    )


def _save_database(db: JobsDatabase) -> None:
    """Save the entry-level jobs database."""
    save_database(db, EL_JOBS_PATH)


def validate_entry_level() -> list[JobListing]:
    """Main entry-level validation entry point.

    Finds the latest raw entry-level discovery file, enriches each listing
    via AI, filters by validation criteria, and appends valid listings to
    data/entry_level_jobs.json.

    Returns:
        List of newly validated JobListing objects.
    """
    raw_path = _find_latest_raw_el_discovery()
    if raw_path is None:
        logger.warning("No raw entry-level discovery files found — nothing to validate")
        return []

    raw_listings = _load_raw_listings(raw_path)
    if not raw_listings:
        logger.warning("No raw listings in %s — nothing to validate", raw_path.name)
        return []

    db = _load_existing_database()
    existing_hashes = _get_existing_hashes(db)

    try:
        config = get_config()
    except Exception as exc:
        logger.warning("Could not load config: %s", exc)
        config = None

    logger.info(
        "Validating %d raw entry-level listings (%d already in database)",
        len(raw_listings), len(existing_hashes),
    )

    reset_budget()

    # Get entry-level enrichment prompt
    el_prompt = None
    try:
        if config and config.ai.entry_level_enrichment_prompt:
            el_prompt = config.ai.entry_level_enrichment_prompt
    except Exception:
        pass

    validated: list[JobListing] = []
    skipped_existing = 0
    rejected_not_entry_level = 0
    rejected_is_internship = 0
    rejected_low_confidence = 0
    errors = 0

    try:
        role_categories_map = config.filters.role_categories if config else {}
    except Exception:
        role_categories_map = {}

    try:
        config_industries = config.company_industries if config else {}
    except Exception:
        config_industries = {}

    for raw in raw_listings:
        if raw.content_hash in existing_hashes:
            skipped_existing += 1
            continue

        try:
            metadata = enrich_listing(raw, config=config, prompt_override=el_prompt)

            if metadata is None:
                logger.warning(
                    "AI enrichment returned None for %s — %s (skipping)",
                    raw.company, raw.title,
                )
                errors += 1
                continue

            # Detect DEFAULT_METADATA (Gemini unavailable / budget exceeded)
            is_default_metadata = (
                metadata.get("confidence", 0.0) == 0.0
                and metadata.get("season", "none") == "none"
            )

            if is_default_metadata:
                metadata["is_entry_level"] = True
                metadata["confidence"] = 0.7
                metadata["category"] = _infer_category_from_title(
                    raw.title, role_categories_map
                )
                logger.info(
                    "Accepted without AI validation (Gemini unavailable): %s — %s",
                    raw.company, raw.title,
                )

            # Reject if it's actually an internship
            if metadata.get("is_internship", False):
                logger.info(
                    "Rejected (is internship): %s — %s",
                    raw.company, raw.title,
                )
                rejected_is_internship += 1
                continue

            # Check if it's entry-level (if AI provided this field)
            if not metadata.get("is_entry_level", True):
                logger.info(
                    "Rejected (not entry-level): %s — %s",
                    raw.company, raw.title,
                )
                rejected_not_entry_level += 1
                continue

            confidence = metadata.get("confidence", 0.0)
            if confidence < 0.7:
                logger.info(
                    "Rejected (low confidence %.2f): %s — %s",
                    confidence, raw.company, raw.title,
                )
                rejected_low_confidence += 1
                continue

            job = _build_entry_level_listing(raw, metadata, config_industries)
            validated.append(job)

            logger.info(
                "Validated entry-level: %s — %s [%s] (confidence: %.2f)",
                job.company, job.role, ", ".join(job.locations), confidence,
            )

        except Exception as exc:
            logger.error(
                "Error processing listing %s — %s: %s",
                raw.company, raw.title, exc,
            )
            errors += 1

    if validated:
        new_ids: set[str] = set()
        unique_validated: list[JobListing] = []
        for job in validated:
            if job.id not in existing_hashes and job.id not in new_ids:
                unique_validated.append(job)
                new_ids.add(job.id)
            else:
                logger.info(
                    "Skipping duplicate (same content hash): %s — %s",
                    job.company, job.role,
                )

        db.listings.extend(unique_validated)
        _save_database(db)
        validated = unique_validated

    logger.info(
        "Entry-level validation complete: %d validated, %d skipped (existing), "
        "%d rejected (internship), %d rejected (not entry-level), "
        "%d rejected (low confidence), %d errors",
        len(validated), skipped_existing,
        rejected_is_internship, rejected_not_entry_level,
        rejected_low_confidence, errors,
    )

    return validated


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    results = validate_entry_level()
    logger.info("Validated %d new entry-level listings", len(results))
