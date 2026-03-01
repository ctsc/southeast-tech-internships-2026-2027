"""Markdown table generation logic for rendering the README from job data.

Transforms a JobsDatabase into a beautifully formatted README.md with
categorized internship tables, stats, and legend. Only includes Southeast
region listings (GA, FL, AL, TX, SC, NC, TN).
"""

import logging
from datetime import date

from scripts.utils.config import get_config
from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
)

logger = logging.getLogger(__name__)

# Category display order and metadata
CATEGORY_INFO: list[tuple[RoleCategory, str, str, str]] = [
    (RoleCategory.SWE, "Software Engineering", "-software-engineering", "💻"),
    (RoleCategory.ML_AI, "ML / AI / Data Science", "-ml--ai--data-science", "🤖"),
    (RoleCategory.DATA_SCIENCE, "Data Science & Analytics", "-data-science--analytics", "📊"),
    (RoleCategory.QUANT, "Quantitative Finance", "-quantitative-finance", "📈"),
    (RoleCategory.PM, "Product Management", "-product-management", "📱"),
    (RoleCategory.HARDWARE, "Hardware Engineering", "-hardware-engineering", "🔧"),
    (RoleCategory.OTHER, "Other", "-other", "🔹"),
]


def _format_locations(locations: list[str], max_display: int = 3) -> str:
    """Format a list of locations, truncating if needed."""
    if not locations:
        return "Unknown"
    if len(locations) <= max_display:
        return ", ".join(locations)
    displayed = ", ".join(locations[:max_display])
    remaining = len(locations) - max_display
    return f"{displayed} and {remaining} more"


SEASON_BADGES: dict[str, str] = {
    "summer_2026": "S26",
    "fall_2026": "F26",
    "spring_2027": "Sp27",
    "summer_2027": "S27",
}


def _format_season(season: str) -> str:
    """Format a season string as a short badge."""
    return SEASON_BADGES.get(season, season)


def _format_relative_date(d: date) -> str:
    """Format a date as relative time (e.g., 'today', '2d ago', '3w ago')."""
    delta = (date.today() - d).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "1d ago"
    if delta < 7:
        return f"{delta}d ago"
    if delta < 30:
        weeks = delta // 7
        return f"{weeks}w ago"
    if delta < 365:
        months = delta // 30
        return f"{months}mo ago"
    years = delta // 365
    return f"{years}y ago"


def _escape_markdown_cell(text: str) -> str:
    """Escape pipe characters in text destined for a markdown table cell."""
    return text.replace("|", "\\|")


def _format_listing_row(listing: JobListing) -> str:
    """Format a single listing as a markdown table row."""
    company = f"**{_escape_markdown_cell(listing.company)}**"
    if listing.is_faang_plus:
        company = f"🔥 {company}"

    # Role with status/flag indicators
    role = _escape_markdown_cell(listing.role)
    flags = []
    if listing.status == ListingStatus.CLOSED:
        flags.append("🔒")
    if listing.open_to_international:
        flags.append("🌍")
    if listing.remote_friendly:
        flags.append("🏠")
    if flags:
        role = f"{role} {''.join(flags)}"

    locations = _format_locations(listing.locations)
    season_badge = _format_season(listing.season)
    date_str = _format_relative_date(listing.date_added)
    apply_url = str(listing.apply_url)

    if listing.status == ListingStatus.CLOSED:
        apply_link = "🔒 Closed"
    else:
        apply_link = f"[Apply]({apply_url})"

    return f"| {company} | {role} | {locations} | {season_badge} | {apply_link} | {date_str} |"


def _render_category_section(
    category: RoleCategory,
    emoji: str,
    title: str,
    listings: list[JobListing],
) -> str:
    """Render a single category section with its table."""
    lines = [
        f"## {emoji} {title}",
        "",
    ]

    open_listings = [x for x in listings if x.status == ListingStatus.OPEN]
    closed_listings = [x for x in listings if x.status == ListingStatus.CLOSED]

    # Sort by date_added descending (newest first)
    sorted_listings = sorted(
        open_listings + closed_listings,
        key=lambda x: x.date_added,
        reverse=True,
    )

    if not sorted_listings:
        lines.append("No listings yet. Check back soon!")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Company | Role | Location | Season | Apply | Posted |")
    lines.append("|---------|------|----------|--------|-------|------------|")
    for listing in sorted_listings:
        lines.append(_format_listing_row(listing))
    lines.append("")
    return "\n".join(lines)


SOUTHEAST_PATTERNS: dict[str, list[str]] = {
    "states": [
        ", ga", ", fl", ", al", ", tx", ", sc", ", nc", ", tn",
        "georgia", "florida", "alabama", "texas",
        "south carolina", "north carolina", "tennessee",
    ],
    "cities": [
        "atlanta", "alpharetta", "marietta", "savannah", "augusta",
        "miami", "orlando", "tampa", "jacksonville",
        "birmingham", "huntsville",
        "dallas", "austin", "houston", "san antonio",
        "charlotte", "raleigh", "durham", "research triangle",
        "charleston", "greenville", "columbia",
        "nashville", "knoxville", "memphis", "chattanooga",
    ],
}


def _is_southeast_listing(listing: JobListing) -> bool:
    """Check if a listing is in the Southeast region (GA, FL, AL, TX, SC, NC, TN)."""
    for loc in listing.locations:
        loc_lower = loc.lower()
        for pattern in SOUTHEAST_PATTERNS["states"]:
            if pattern in loc_lower:
                return True
        for city in SOUTHEAST_PATTERNS["cities"]:
            if city in loc_lower:
                return True
    return False


def _count_open(listings: list[JobListing], category: RoleCategory) -> int:
    """Count open SE listings for a given category."""
    return len([
        x for x in listings
        if x.category == category
        and x.status == ListingStatus.OPEN
        and _is_southeast_listing(x)
    ])


def render_readme(jobs_db: JobsDatabase) -> str:
    """Render a complete README.md from a JobsDatabase.

    Only includes listings in the Southeast region (GA, FL, AL, TX, SC, NC, TN).

    Args:
        jobs_db: The jobs database to render.

    Returns:
        A string containing the full README markdown.
    """
    try:
        config = get_config()
        repo = config.project.github_repo
    except Exception:
        logger.warning("Could not load config, using defaults for README rendering")
        repo = "ctsc/atlanta-tech-internships-2026"

    jobs_db.compute_stats()
    listings = jobs_db.listings
    timestamp = jobs_db.last_updated.strftime("%B %d, %Y at %H:%M UTC")

    # Compute category counts (SE-only)
    category_counts: dict[RoleCategory, int] = {}
    for cat, _, _, _ in CATEGORY_INFO:
        category_counts[cat] = _count_open(listings, cat)
    total_open = sum(category_counts.values())

    # Build the issue URL
    issue_url = f"https://github.com/{repo}/issues/new?template=new-internship.yml"

    # --- Header ---
    parts: list[str] = []
    parts.append("# Atlanta Tech Internships 🚀")
    parts.append("")
    parts.append(f"> 🤖 **Auto-updated every 6 hours** | Last updated: {timestamp}")
    parts.append(">")
    parts.append("> Catered to Georgia / Southeast ⭐ Leave a star on the repo if you enjoy this project :)")
    parts.append(">")
    parts.append("> Built and maintained by [Carter](https://github.com/ctsc) | President, IEEE @ Georgia State")
    parts.append("")
    parts.append(
        "Use this repo to discover and track **tech internships** "
        "across software engineering, ML/AI, data science, quant, and more."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # --- Stats Table ---
    parts.append("### 📊 Stats")
    parts.append("")
    parts.append("| Category | Open Roles |")
    parts.append("|----------|-----------|")
    for cat, title, anchor, emoji in CATEGORY_INFO:
        count = category_counts[cat]
        if cat == RoleCategory.OTHER and count == 0:
            continue
        parts.append(f"| {emoji} [{title}](#{anchor}) | {count} |")

    parts.append(f"| **Total** | **{total_open}** |")
    parts.append("")
    parts.append("---")
    parts.append("")

    # --- Legend ---
    parts.append("### Legend")
    parts.append("")
    parts.append("| Symbol | Meaning |")
    parts.append("|--------|---------|")
    parts.append("| 🔥 | Major tech company |")
    parts.append("| 🔒 | Application closed |")
    parts.append("| 🌍 | Open to international students |")
    parts.append("| 🏠 | Remote friendly |")
    parts.append("| S26 | Summer 2026 |")
    parts.append("| F26 | Fall 2026 |")
    parts.append("| Sp27 | Spring 2027 |")
    parts.append("| S27 | Summer 2027 |")
    parts.append("")
    parts.append("---")
    parts.append("")

    # --- Category Sections (SE-only) ---
    for cat, title, anchor, emoji in CATEGORY_INFO:
        cat_listings = [
            x for x in listings
            if x.category == cat and _is_southeast_listing(x)
        ]
        # Skip OTHER section if empty
        if cat == RoleCategory.OTHER and not cat_listings:
            continue
        section = _render_category_section(cat, emoji, title, cat_listings)
        parts.append(section)
        parts.append("---")
        parts.append("")

    # --- How This Works ---
    parts.append("## How This Works")
    parts.append("")
    parts.append("This repo is **automatically maintained by AI**. Every 6 hours:")
    parts.append("1. Scripts scan 100+ company career pages and job board APIs")
    parts.append("2. Gemini AI validates each listing is a real tech internship")
    parts.append("3. Dead links are detected and removed")
    parts.append("4. The README is regenerated with fresh data")
    parts.append("")

    # --- Contributing ---
    parts.append("## Contributing")
    parts.append("")
    parts.append(f"Found a listing we missed? [Submit an issue]({issue_url})!")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("⭐ **Star this repo** to stay updated!")
    parts.append("")

    readme = "\n".join(parts)
    logger.info(
        "README rendered: %d total listings, %d open, %d categories",
        len(listings),
        total_open,
        len([c for c, cnt in category_counts.items() if cnt > 0]),
    )
    return readme
