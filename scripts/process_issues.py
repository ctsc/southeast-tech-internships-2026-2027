"""Processes community-submitted GitHub issues and adds valid listings to the database.

Fetches open issues labeled 'new-internship', parses the structured issue form
data, validates submissions, and adds valid listings to jobs.json.
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Optional

from scripts.utils.config import PROJECT_ROOT, get_config, get_secret
from scripts.utils.github_utils import close_issue, comment_on_issue, fetch_issues
from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
)

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"

# Mapping from issue form category strings to RoleCategory enum
CATEGORY_MAP: dict[str, RoleCategory] = {
    "software engineering": RoleCategory.SWE,
    "ml / ai / data science": RoleCategory.ML_AI,
    "quantitative finance": RoleCategory.QUANT,
    "product management": RoleCategory.PM,
    "hardware engineering": RoleCategory.HARDWARE,
    "other": RoleCategory.OTHER,
}


def _parse_issue_body(body: str) -> Optional[dict]:
    """Parse a structured GitHub issue form body into a dict.

    The issue form produces a body with ``### `` headers followed by
    values on subsequent lines. Checkbox sections use ``[X]`` / ``[ ]``
    notation.

    Args:
        body: Raw issue body text.

    Returns:
        Dict with keys: company, role, url, location, category, flags.
        Returns None if the body cannot be parsed.
    """
    if not body or not body.strip():
        logger.warning("Empty issue body")
        return None

    sections: dict[str, str] = {}
    current_header: Optional[str] = None
    current_lines: list[str] = []

    for line in body.split("\n"):
        if line.startswith("### "):
            if current_header is not None:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = line[4:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header is not None:
        sections[current_header] = "\n".join(current_lines).strip()

    company = sections.get("company name", "").strip()
    role = sections.get("role title", "").strip()
    url = sections.get("application url", "").strip()
    location = sections.get("location(s)", "").strip()
    category = sections.get("role category", "").strip()
    flags_raw = sections.get("additional info", "")

    if not company or not role or not url or not location:
        logger.info(
            "Issue body missing required fields — company=%r, role=%r, url=%r, location=%r",
            company, role, url, location,
        )
        return None

    # Parse checkbox flags
    flags = {
        "sponsors": bool(re.search(r"\[(?:x|X)\]\s*Offers visa sponsorship", flags_raw)),
        "us_citizenship": bool(re.search(r"\[(?:x|X)\]\s*Requires U\.S\. citizenship", flags_raw)),
        "remote_friendly": bool(re.search(r"\[(?:x|X)\]\s*Remote friendly", flags_raw)),
        "advanced_degree": bool(re.search(r"\[(?:x|X)\]\s*Requires advanced degree", flags_raw)),
        "open_to_international": bool(re.search(r"\[(?:x|X)\]\s*Open to international students", flags_raw)),
    }

    return {
        "company": _sanitize_field(company),
        "role": _sanitize_field(role),
        "url": url,
        "location": _sanitize_field(location),
        "category": category,
        "flags": flags,
    }


def _validate_url(url: str) -> bool:
    """Check if a URL is a valid HTTP(S) URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is a well-formed http/https URL.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _sanitize_field(text: str, max_length: int = 500) -> str:
    """Sanitize a user-submitted field for safe markdown rendering.

    Strips pipe characters (which break tables), control characters,
    and enforces a maximum length.

    Args:
        text: The raw user input.
        max_length: Maximum allowed length.

    Returns:
        Sanitized string safe for markdown table cells.
    """
    # Strip leading/trailing whitespace
    text = text.strip()
    # Remove pipe characters that break markdown tables
    text = text.replace("|", "-")
    # Remove markdown link injection attempts
    text = text.replace("[", "").replace("]", "")
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]
    return text


def _map_category(category_str: str) -> RoleCategory:
    """Map an issue form category string to RoleCategory.

    Args:
        category_str: Category string from the issue form.

    Returns:
        The matching RoleCategory, or OTHER if unknown.
    """
    return CATEGORY_MAP.get(category_str.lower().strip(), RoleCategory.OTHER)


def _parse_locations(raw_location: str) -> list[str]:
    """Parse a location string into a list of individual locations.

    Args:
        raw_location: Raw location string from the issue form.

    Returns:
        List of individual location strings.
    """
    for delimiter in [" / ", "/", " | ", "|", " ; ", ";"]:
        if delimiter in raw_location:
            parts = [loc.strip() for loc in raw_location.split(delimiter) if loc.strip()]
            if parts:
                return parts

    if ", " in raw_location:
        parts = raw_location.split(", ")
        if len(parts) == 2 and len(parts[1].strip()) <= 3:
            return [raw_location.strip()]
        if len(parts) > 2:
            return [loc.strip() for loc in parts if loc.strip()]

    return [raw_location.strip()] if raw_location.strip() else ["Unknown"]


def _slugify(name: str) -> str:
    """Convert a company name to a kebab-case slug.

    Args:
        name: Company name.

    Returns:
        Kebab-case slug string.
    """
    slug = name.lower().strip()
    slug = slug.replace(" ", "-").replace(".", "").replace("'", "")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _generate_listing_id(company: str, role: str, locations: list[str]) -> str:
    """Generate a SHA-256 content hash for a listing.

    Args:
        company: Company name.
        role: Role title.
        locations: List of location strings.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    normalized_locations = ",".join(sorted(loc.lower().strip() for loc in locations))
    raw = f"{company.lower().strip()}|{role.lower().strip()}|{normalized_locations}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_database() -> JobsDatabase:
    """Load the existing jobs database from data/jobs.json.

    Returns:
        The current JobsDatabase, or an empty one if the file is missing.
    """
    if not JOBS_PATH.exists():
        return JobsDatabase(
            listings=[], last_updated=datetime.now(timezone.utc), total_open=0
        )

    try:
        with open(JOBS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JobsDatabase.model_validate(data)
    except Exception as exc:
        logger.error("Failed to parse jobs.json: %s", exc)
        return JobsDatabase(
            listings=[], last_updated=datetime.now(timezone.utc), total_open=0
        )


def _save_database(db: JobsDatabase) -> None:
    """Save the jobs database to data/jobs.json.

    Args:
        db: The jobs database to save.
    """
    db.last_updated = datetime.now(timezone.utc)
    db.compute_stats()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = db.model_dump(mode="json")
    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(
        "Saved database: %d total listings, %d open",
        len(db.listings),
        db.total_open,
    )


def _build_job_listing(parsed: dict) -> JobListing:
    """Build a JobListing from parsed issue data.

    Args:
        parsed: Dict from _parse_issue_body with company, role, url, etc.

    Returns:
        A fully populated JobListing.
    """
    locations = _parse_locations(parsed["location"])
    listing_id = _generate_listing_id(parsed["company"], parsed["role"], locations)
    today = date.today()
    flags = parsed["flags"]

    sponsorship = SponsorshipStatus.UNKNOWN
    if flags.get("us_citizenship"):
        sponsorship = SponsorshipStatus.US_CITIZENSHIP
    elif flags.get("sponsors"):
        sponsorship = SponsorshipStatus.SPONSORS

    config = get_config()

    return JobListing(
        id=listing_id,
        company=parsed["company"],
        company_slug=_slugify(parsed["company"]),
        role=parsed["role"],
        category=_map_category(parsed["category"]),
        locations=locations,
        apply_url=parsed["url"],
        sponsorship=sponsorship,
        requires_us_citizenship=flags.get("us_citizenship", False),
        is_faang_plus=False,
        requires_advanced_degree=flags.get("advanced_degree", False),
        remote_friendly=flags.get("remote_friendly", False),
        open_to_international=flags.get("open_to_international", False),
        date_added=today,
        date_last_verified=today,
        source="community",
        status=ListingStatus.OPEN,
        tech_stack=[],
        season=config.project.season,
    )


def _get_missing_fields(parsed: Optional[dict]) -> list[str]:
    """Check which required fields are missing from parsed issue data.

    Args:
        parsed: Parsed issue data dict, or None if parsing failed.

    Returns:
        List of missing field names.
    """
    if parsed is None:
        return ["company", "role", "url", "location"]

    missing = []
    if not parsed.get("company"):
        missing.append("company")
    if not parsed.get("role"):
        missing.append("role")
    if not parsed.get("url"):
        missing.append("url")
    if not parsed.get("location"):
        missing.append("location")
    return missing


async def process_issues() -> int:
    """Process all open 'new-internship' issues.

    Fetches issues, parses form data, validates, adds to jobs.json,
    and comments/closes issues.

    Returns:
        Count of accepted listings.
    """
    config = get_config()
    repo = config.project.github_repo
    token = get_secret("GITHUB_TOKEN")

    if not token:
        logger.error("GITHUB_TOKEN not set — cannot process issues")
        return 0

    issues = await fetch_issues(repo, label="new-internship", token=token)
    if not issues:
        logger.info("No open issues with 'new-internship' label")
        return 0

    logger.info("Processing %d issues from %s", len(issues), repo)

    db = _load_database()
    existing_ids = {listing.id for listing in db.listings}
    accepted = 0

    for issue in issues:
        issue_number = issue.get("number")
        issue_title = issue.get("title", "Unknown")
        issue_body = issue.get("body", "")

        try:
            logger.info("Processing issue #%d: %s", issue_number, issue_title)

            parsed = _parse_issue_body(issue_body)

            if parsed is None:
                reason = (
                    "Could not parse the issue body. Please make sure you used "
                    "the issue template and filled in all required fields "
                    "(Company Name, Role Title, Application URL, Location)."
                )
                logger.info("Rejected issue #%d: unparseable body", issue_number)
                await comment_on_issue(repo, issue_number, reason, token=token)
                await close_issue(repo, issue_number, token=token)
                continue

            if not _validate_url(parsed["url"]):
                reason = (
                    f"The application URL `{parsed['url']}` does not appear to be "
                    "a valid URL. Please resubmit with a URL starting with "
                    "`https://`."
                )
                logger.info(
                    "Rejected issue #%d: invalid URL %s",
                    issue_number,
                    parsed["url"],
                )
                await comment_on_issue(repo, issue_number, reason, token=token)
                await close_issue(repo, issue_number, token=token)
                continue

            job = _build_job_listing(parsed)

            if job.id in existing_ids:
                reason = (
                    "This listing appears to already exist in our database. "
                    "Thanks for checking though!"
                )
                logger.info(
                    "Rejected issue #%d: duplicate listing %s",
                    issue_number,
                    job.id[:12],
                )
                await comment_on_issue(repo, issue_number, reason, token=token)
                await close_issue(repo, issue_number, token=token)
                continue

            db.listings.append(job)
            existing_ids.add(job.id)
            accepted += 1

            await comment_on_issue(
                repo, issue_number,
                "Added! Thanks for contributing.",
                token=token,
            )
            await close_issue(repo, issue_number, token=token)

            logger.info(
                "Accepted issue #%d: %s — %s",
                issue_number,
                job.company,
                job.role,
            )

        except Exception as exc:
            logger.error(
                "Error processing issue #%d: %s",
                issue_number,
                exc,
            )

    if accepted > 0:
        _save_database(db)

    logger.info(
        "Issue processing complete: %d accepted out of %d issues",
        accepted,
        len(issues),
    )
    return accepted


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    count = asyncio.run(process_issues())
    logger.info("Processed %d issues", count)
