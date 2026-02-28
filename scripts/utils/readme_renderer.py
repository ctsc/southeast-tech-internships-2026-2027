"""Markdown table generation logic for rendering the README from job data.

Transforms a JobsDatabase into a beautifully formatted README.md with
categorized internship tables, stats, legend, and Georgia focus section.
"""

import logging

from scripts.utils.config import get_config
from scripts.utils.models import (
    IndustrySector,
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
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


INDUSTRY_EMOJI: dict[str, str] = {
    IndustrySector.FINTECH: "💳",
    IndustrySector.HEALTHCARE: "🏥",
    IndustrySector.ENERGY: "⚡",
    IndustrySector.ECOMMERCE: "🛒",
    IndustrySector.BANKING: "🏦",
    IndustrySector.AUTOMOTIVE: "🚗",
    IndustrySector.GAMING: "🎮",
    IndustrySector.SOCIAL_MEDIA: "💬",
    IndustrySector.CYBERSECURITY: "🔐",
    IndustrySector.CLOUD: "☁️",
    IndustrySector.ENTERPRISE: "🏢",
    IndustrySector.AI_ML: "🧠",
    IndustrySector.AEROSPACE: "🚀",
    IndustrySector.TELECOM: "📡",
    IndustrySector.MEDIA: "🎬",
    IndustrySector.FOOD: "🍔",
    IndustrySector.LOGISTICS: "📦",
    IndustrySector.SEMICONDUCTOR: "🔬",
    IndustrySector.OTHER: "🏷️",
}


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


def _escape_markdown_cell(text: str) -> str:
    """Escape pipe characters in text destined for a markdown table cell."""
    return text.replace("|", "\\|")


def _format_listing_row(listing: JobListing) -> str:
    """Format a single listing as a markdown table row."""
    # Company name with industry emoji and FAANG+ indicator
    industry_emoji = INDUSTRY_EMOJI.get(listing.industry, "🏷️")
    company = f"**{_escape_markdown_cell(listing.company)}**"
    if listing.is_faang_plus:
        company = f"{industry_emoji} 🔥 {company}"
    else:
        company = f"{industry_emoji} {company}"

    # Role with status/flag indicators
    role = _escape_markdown_cell(listing.role)
    flags = []
    if listing.status == ListingStatus.CLOSED:
        flags.append("🔒")
    if listing.sponsorship == SponsorshipStatus.NO_SPONSORSHIP:
        flags.append("🛂")
    if listing.requires_us_citizenship:
        flags.append("🇺🇸")
    if listing.requires_advanced_degree:
        flags.append("🎓")
    if listing.remote_friendly:
        flags.append("🏠")
    if flags:
        role = f"{role} {''.join(flags)}"

    locations = _format_locations(listing.locations)
    season_badge = _format_season(listing.season)
    date_str = listing.date_added.strftime("%b %d")
    apply_url = str(listing.apply_url)

    if listing.status == ListingStatus.CLOSED:
        apply_link = "🔒 Closed"
    else:
        apply_link = f"[Apply]({apply_url})"

    return f"| {company} | {role} | {locations} | {season_badge} | {apply_link} | {date_str} |"


def _is_georgia_listing(listing: JobListing, ga_locations: list[str]) -> bool:
    """Check if a listing has any Georgia location."""
    for loc in listing.locations:
        loc_lower = loc.lower()
        # Check for explicit GA mentions
        if ", ga" in loc_lower or "georgia" in loc_lower:
            return True
        # Check against configured priority locations
        for ga_loc in ga_locations:
            if ga_loc.lower() in loc_lower or loc_lower in ga_loc.lower():
                return True
    return False


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

    lines.append("| Company | Role | Location | Season | Apply | Date Added |")
    lines.append("|---------|------|----------|--------|-------|------------|")
    for listing in sorted_listings:
        lines.append(_format_listing_row(listing))
    lines.append("")
    return "\n".join(lines)


def _render_georgia_section(listings: list[JobListing], ga_locations: list[str]) -> str:
    """Render the Georgia-focused section."""
    ga_listings = [
        x for x in listings
        if x.status == ListingStatus.OPEN and _is_georgia_listing(x, ga_locations)
    ]

    lines = [
        "## 🍑 Georgia Internships",
        "",
        "> Internships located in Georgia — Atlanta, Alpharetta, and across the state.",
        "",
    ]

    if not ga_listings:
        lines.append("No Georgia-based listings yet. Check back soon!")
        lines.append("")
        return "\n".join(lines)

    sorted_listings = sorted(ga_listings, key=lambda x: x.date_added, reverse=True)

    lines.append("| Company | Role | Location | Season | Apply | Date Added |")
    lines.append("|---------|------|----------|--------|-------|------------|")
    for listing in sorted_listings:
        lines.append(_format_listing_row(listing))
    lines.append("")
    return "\n".join(lines)


def _count_open(listings: list[JobListing], category: RoleCategory) -> int:
    """Count open listings for a given category."""
    return len([
        x for x in listings
        if x.category == category and x.status == ListingStatus.OPEN
    ])


def render_readme(jobs_db: JobsDatabase) -> str:
    """Render a complete README.md from a JobsDatabase.

    Args:
        jobs_db: The jobs database to render.

    Returns:
        A string containing the full README markdown.
    """
    try:
        config = get_config()
        georgia_focus = config.georgia_focus
        repo = config.project.github_repo
    except Exception:
        logger.warning("Could not load config, using defaults for README rendering")
        georgia_focus = None
        repo = "ctsc/atlanta-tech-internships-2026"

    jobs_db.compute_stats()
    listings = jobs_db.listings
    timestamp = jobs_db.last_updated.strftime("%B %d, %Y at %H:%M UTC")

    # Compute category counts
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
    parts.append("| 🔥 | FAANG+ company |")
    parts.append("| 🛂 | Does NOT offer sponsorship |")
    parts.append("| 🇺🇸 | Requires U.S. Citizenship |")
    parts.append("| 🔒 | Application closed |")
    parts.append("| 🎓 | Advanced degree required |")
    parts.append("| 🏠 | Remote friendly |")
    parts.append("| S26 | Summer 2026 |")
    parts.append("| F26 | Fall 2026 |")
    parts.append("| Sp27 | Spring 2027 |")
    parts.append("| S27 | Summer 2027 |")
    parts.append("")
    parts.append("**Industry**")
    parts.append("")
    parts.append("| Symbol | Industry |")
    parts.append("|--------|----------|")
    parts.append("| 💳 | Fintech |")
    parts.append("| 🏦 | Banking / Finance |")
    parts.append("| 🧠 | AI / ML |")
    parts.append("| ☁️ | Cloud / Infrastructure |")
    parts.append("| 🔐 | Cybersecurity |")
    parts.append("| 🏢 | Enterprise / SaaS |")
    parts.append("| 💬 | Social Media |")
    parts.append("| 🎬 | Media / Entertainment |")
    parts.append("| 🛒 | Ecommerce |")
    parts.append("| 🚗 | Automotive |")
    parts.append("| 🔬 | Semiconductor |")
    parts.append("| 📦 | Logistics |")
    parts.append("| 🏥 | Healthcare |")
    parts.append("| ⚡ | Energy |")
    parts.append("| 🍔 | Food / Beverage |")
    parts.append("| 🏷️ | Other |")
    parts.append("")
    parts.append("---")
    parts.append("")

    # --- Georgia Section (if enabled) ---
    if georgia_focus and georgia_focus.georgia_section_in_readme:
        ga_locations = georgia_focus.priority_locations
        parts.append(_render_georgia_section(listings, ga_locations))
        parts.append("---")
        parts.append("")

    # --- Category Sections ---
    for cat, title, anchor, emoji in CATEGORY_INFO:
        cat_listings = [x for x in listings if x.category == cat]
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
