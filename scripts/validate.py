"""Validation pipeline: AI-validates raw discovery results and produces JobListing objects.

Loads the most recent raw discovery file, enriches each listing via Gemini AI,
filters by validation criteria, and appends valid listings to jobs.json.
"""

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.utils.ai_enrichment import enrich_listing, reset_budget
from scripts.utils.config import PROJECT_ROOT, get_config
from scripts.utils.models import (
    IndustrySector,
    JobListing,
    JobsDatabase,
    ListingStatus,
    RawListing,
    RoleCategory,
    SponsorshipStatus,
)

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"


def _find_latest_raw_discovery() -> Optional[Path]:
    """Find the most recent raw discovery JSON file in the data directory.

    Looks for files matching the pattern raw_discovery_*.json and returns
    the one with the lexicographically latest filename (which corresponds
    to the most recent timestamp).

    Returns:
        Path to the latest raw discovery file, or None if none exist.
    """
    raw_files = sorted(DATA_DIR.glob("raw_discovery_*.json"))
    if not raw_files:
        logger.warning("No raw discovery files found in %s", DATA_DIR)
        return None
    latest = raw_files[-1]
    logger.info("Found latest raw discovery file: %s", latest.name)
    return latest


def _load_raw_listings(path: Path) -> list[RawListing]:
    """Load raw listings from a discovery JSON file.

    Args:
        path: Path to the raw discovery JSON file.

    Returns:
        List of parsed RawListing objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data.get("listings", [])
    listings: list[RawListing] = []
    for item in raw_items:
        try:
            listings.append(RawListing.model_validate(item))
        except Exception as exc:
            logger.warning("Skipping malformed raw listing: %s", exc)
    logger.info("Loaded %d raw listings from %s", len(listings), path.name)
    return listings


def _load_existing_database() -> JobsDatabase:
    """Load the existing jobs database from data/jobs.json.

    Returns:
        The current JobsDatabase, or an empty one if the file
        doesn't exist or cannot be parsed.
    """
    if not JOBS_PATH.exists():
        logger.info("jobs.json not found, starting with empty database")
        return JobsDatabase(
            listings=[], last_updated=datetime.now(timezone.utc), total_open=0
        )

    try:
        with open(JOBS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = JobsDatabase.model_validate(data)
        logger.info("Loaded existing database with %d listings", len(db.listings))
        return db
    except Exception as exc:
        logger.error("Failed to parse jobs.json, starting fresh: %s", exc)
        return JobsDatabase(
            listings=[], last_updated=datetime.now(timezone.utc), total_open=0
        )


def _get_existing_hashes(db: JobsDatabase) -> set[str]:
    """Extract all listing IDs (content hashes) from the database.

    Args:
        db: The current jobs database.

    Returns:
        Set of listing ID strings.
    """
    return {listing.id for listing in db.listings}


def _generate_listing_id(company: str, role: str, locations: list[str]) -> str:
    """Generate a deterministic content hash for a listing.

    Creates a SHA-256 hash from the normalized combination of company name,
    role title, and sorted locations.

    Args:
        company: Company name.
        role: Role/job title.
        locations: List of location strings.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    normalized_locations = ",".join(sorted(loc.lower().strip() for loc in locations))
    raw = f"{company.lower().strip()}|{role.lower().strip()}|{normalized_locations}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _map_category(category_str: str) -> RoleCategory:
    """Map an AI-returned category string to a RoleCategory enum value.

    Args:
        category_str: Category string from AI enrichment (e.g. "swe", "ml_ai").

    Returns:
        The corresponding RoleCategory, defaulting to OTHER for unknown values.
    """
    mapping = {
        "swe": RoleCategory.SWE,
        "ml_ai": RoleCategory.ML_AI,
        "data_science": RoleCategory.DATA_SCIENCE,
        "quant": RoleCategory.QUANT,
        "pm": RoleCategory.PM,
        "hardware": RoleCategory.HARDWARE,
        "other": RoleCategory.OTHER,
    }
    return mapping.get(category_str.lower().strip(), RoleCategory.OTHER)


def _map_sponsorship(sponsorship_str: str) -> SponsorshipStatus:
    """Map an AI-returned sponsorship string to a SponsorshipStatus enum value.

    Args:
        sponsorship_str: Sponsorship string from AI enrichment.

    Returns:
        The corresponding SponsorshipStatus, defaulting to UNKNOWN.
    """
    mapping = {
        "sponsors": SponsorshipStatus.SPONSORS,
        "no_sponsorship": SponsorshipStatus.NO_SPONSORSHIP,
        "us_citizenship": SponsorshipStatus.US_CITIZENSHIP,
        "unknown": SponsorshipStatus.UNKNOWN,
    }
    return mapping.get(sponsorship_str.lower().strip(), SponsorshipStatus.UNKNOWN)


def _slugify(name: str) -> str:
    """Convert a company name to a kebab-case slug.

    Args:
        name: Company name.

    Returns:
        Kebab-case slug string.
    """
    slug = name.lower().strip()
    slug = slug.replace(" ", "-").replace(".", "").replace("'", "")
    # Remove consecutive hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _parse_locations(raw_location: str, ai_locations: Optional[list[str]] = None) -> list[str]:
    """Parse location strings into a list of individual locations.

    Prefers AI-provided locations if available; otherwise splits the raw
    location string on common delimiters.

    Args:
        raw_location: The raw location string from the listing.
        ai_locations: Optional list of locations from AI enrichment.

    Returns:
        List of individual location strings.
    """
    if ai_locations and len(ai_locations) > 0:
        return ai_locations

    # Split on common delimiters
    for delimiter in [" / ", "/", " | ", "|", " ; ", ";"]:
        if delimiter in raw_location:
            parts = [loc.strip() for loc in raw_location.split(delimiter) if loc.strip()]
            if parts:
                return parts

    # Try comma, but be careful with "City, State" patterns
    if ", " in raw_location:
        parts = raw_location.split(", ")
        # If we have exactly 2 parts and second is 2 chars, it's likely "City, ST"
        if len(parts) == 2 and len(parts[1].strip()) <= 3:
            return [raw_location.strip()]
        # Otherwise check if it looks like multiple locations
        # "NYC, SF, Remote" vs "San Francisco, CA"
        if len(parts) > 2:
            return [loc.strip() for loc in parts if loc.strip()]

    return [raw_location.strip()] if raw_location.strip() else ["Unknown"]


def _map_industry(industry_str: str, company: str, config_industries: dict[str, str]) -> IndustrySector:
    """Map an industry string to an IndustrySector enum value.

    Uses AI-detected industry first, then falls back to the config mapping,
    then defaults to OTHER.

    Args:
        industry_str: Industry string from AI enrichment.
        company: Company name for config-based lookup.
        config_industries: Mapping of company name -> industry from config.yaml.

    Returns:
        The corresponding IndustrySector enum value.
    """
    # Try AI-provided industry first
    if industry_str and industry_str != "other":
        try:
            return IndustrySector(industry_str.lower().strip())
        except ValueError:
            pass

    # Fall back to config mapping
    config_industry = config_industries.get(company)
    if config_industry:
        try:
            return IndustrySector(config_industry.lower().strip())
        except ValueError:
            pass

    return IndustrySector.OTHER


def _build_job_listing(
    raw: RawListing, metadata: dict, config_industries: Optional[dict[str, str]] = None
) -> JobListing:
    """Build a JobListing from raw listing data and AI-enriched metadata.

    Args:
        raw: The raw listing from discovery.
        metadata: Enriched metadata dict from AI validation.
        config_industries: Company->industry mapping from config.yaml.

    Returns:
        A fully populated JobListing object.
    """
    locations = _parse_locations(
        raw.location, metadata.get("locations")
    )
    listing_id = _generate_listing_id(raw.company, raw.title, locations)
    today = date.today()

    # Determine season from AI metadata (with backward compat for cached responses)
    season = metadata.get("season", "none")
    if season == "none" and metadata.get("is_summer_2026"):
        season = "summer_2026"

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
        is_faang_plus=raw.is_faang_plus,
        requires_advanced_degree=metadata.get("requires_advanced_degree", False),
        remote_friendly=metadata.get("remote_friendly", False),
        date_added=today,
        date_last_verified=today,
        source=raw.source,
        status=ListingStatus.OPEN,
        tech_stack=metadata.get("tech_stack", []),
        season=season,
        industry=industry,
    )


def _save_database(db: JobsDatabase) -> None:
    """Save the jobs database to data/jobs.json.

    Updates the last_updated timestamp and recomputes stats before writing.

    Args:
        db: The jobs database to save.
    """
    db.last_updated = datetime.now(timezone.utc)
    db.compute_stats()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = db.model_dump(mode="json")
    tmp_path = JOBS_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp_path.replace(JOBS_PATH)

    logger.info(
        "Saved database: %d total listings, %d open",
        len(db.listings),
        db.total_open,
    )


def _infer_category_from_title(
    title: str, role_categories: dict[str, list[str]]
) -> str:
    """Infer a role category from the job title using keyword matching.

    Checks the title against each category's keyword list from config.yaml.
    Returns the first matching category key, or "other" if none match.

    Args:
        title: The job title string.
        role_categories: Mapping of category key -> list of keyword phrases.

    Returns:
        Category string (e.g. "swe", "ml_ai", "other").
    """
    title_lower = title.lower()
    for category, keywords in role_categories.items():
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    return "other"


def validate_all() -> list[JobListing]:
    """Main validation entry point.

    Finds the latest raw discovery file, enriches each listing via AI,
    filters by validation criteria, and appends valid listings to jobs.json.

    Returns:
        List of newly validated JobListing objects.
    """
    # Find latest raw discovery file
    raw_path = _find_latest_raw_discovery()
    if raw_path is None:
        logger.warning("No raw discovery files found — nothing to validate")
        return []

    # Load raw listings and existing database
    raw_listings = _load_raw_listings(raw_path)
    if not raw_listings:
        logger.warning("No raw listings in %s — nothing to validate", raw_path.name)
        return []

    db = _load_existing_database()
    existing_hashes = _get_existing_hashes(db)

    # Load active seasons from config
    try:
        config = get_config()
        active_seasons = set(config.project.active_seasons)
    except Exception as exc:
        logger.warning("Could not load active_seasons from config: %s", exc)
        active_seasons = {"summer_2026"}

    logger.info(
        "Validating %d raw listings (%d already in database), active seasons: %s",
        len(raw_listings),
        len(existing_hashes),
        ", ".join(sorted(active_seasons)),
    )

    # Reset AI budget for this run
    reset_budget()

    validated: list[JobListing] = []
    skipped_existing = 0
    rejected_not_internship = 0
    rejected_wrong_season = 0
    rejected_low_confidence = 0
    errors = 0

    # Load role category keywords for fallback classification
    try:
        role_categories_map = config.filters.role_categories
    except Exception:
        role_categories_map = {}

    # Load company -> industry mapping from config
    try:
        config_industries = config.company_industries
    except Exception:
        config_industries = {}

    for i, raw in enumerate(raw_listings):
        # Skip listings already in the database
        if raw.content_hash in existing_hashes:
            skipped_existing += 1
            continue

        try:
            # Enrich via AI (sync call — Gemini client is synchronous)
            metadata = enrich_listing(raw)

            if metadata is None:
                logger.warning(
                    "AI enrichment returned None for %s — %s (skipping)",
                    raw.company,
                    raw.title,
                )
                errors += 1
                continue

            # Detect DEFAULT_METADATA (Gemini unavailable / budget exceeded):
            # confidence == 0.0 AND season == "none" means AI didn't run.
            # Since discovery already filtered for intern keywords, accept
            # these listings with reasonable defaults instead of rejecting.
            is_default_metadata = (
                metadata.get("confidence", 0.0) == 0.0
                and metadata.get("season", "none") == "none"
            )

            if is_default_metadata:
                default_season = sorted(active_seasons)[0] if active_seasons else "summer_2026"
                metadata["season"] = default_season
                metadata["confidence"] = 0.7
                metadata["category"] = _infer_category_from_title(
                    raw.title, role_categories_map
                )
                logger.info(
                    "Accepted without AI validation (Gemini unavailable): %s — %s "
                    "(default season=%s, category=%s)",
                    raw.company,
                    raw.title,
                    default_season,
                    metadata["category"],
                )

            # Validation checks
            if not metadata.get("is_internship", False):
                logger.info(
                    "Rejected (not internship): %s — %s",
                    raw.company,
                    raw.title,
                )
                rejected_not_internship += 1
                continue

            # Season check: support both new "season" key and legacy "is_summer_2026"
            season = metadata.get("season", "none")
            if season == "none" and metadata.get("is_summer_2026"):
                season = "summer_2026"
            if season not in active_seasons:
                logger.info(
                    "Rejected (season %s not active): %s — %s",
                    season,
                    raw.company,
                    raw.title,
                )
                rejected_wrong_season += 1
                continue

            confidence = metadata.get("confidence", 0.0)
            if confidence < 0.7:
                logger.info(
                    "Rejected (low confidence %.2f): %s — %s",
                    confidence,
                    raw.company,
                    raw.title,
                )
                rejected_low_confidence += 1
                continue

            # Build validated listing
            job = _build_job_listing(raw, metadata, config_industries)
            validated.append(job)

            logger.info(
                "Validated: %s — %s [%s] (confidence: %.2f)",
                job.company,
                job.role,
                ", ".join(job.locations),
                confidence,
            )

        except Exception as exc:
            logger.error(
                "Error processing listing %s — %s: %s",
                raw.company,
                raw.title,
                exc,
            )
            errors += 1

    # Append validated listings to database
    if validated:
        # Check for ID collisions with newly validated listings
        new_ids: set[str] = set()
        unique_validated: list[JobListing] = []
        for job in validated:
            if job.id not in existing_hashes and job.id not in new_ids:
                unique_validated.append(job)
                new_ids.add(job.id)
            else:
                logger.info(
                    "Skipping duplicate (same content hash): %s — %s",
                    job.company,
                    job.role,
                )

        db.listings.extend(unique_validated)
        _save_database(db)
        validated = unique_validated

    logger.info(
        "Validation complete: %d validated, %d skipped (existing), "
        "%d rejected (not internship), %d rejected (wrong season), "
        "%d rejected (low confidence), %d errors",
        len(validated),
        skipped_existing,
        rejected_not_internship,
        rejected_wrong_season,
        rejected_low_confidence,
        errors,
    )

    return validated


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    results = validate_all()
    logger.info("Validated %d new listings", len(results))
