"""Discovery engine for entry-level/new-grad tech jobs in Georgia.

Mirrors the internship discovery pipeline but uses entry_level_filters from
config.yaml to find full-time entry-level positions instead of internships.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.discover import gather_ats_results
from scripts.utils.ats_clients import (
    AshbyClient,
    GreenhouseClient,
    LeverClient,
    SmartRecruitersClient,
    WorkdayClient,
)
from scripts.utils.config import AppConfig, load_config, PROJECT_ROOT
from scripts.utils.models import RawListing
from scripts.utils.scraper import GenericScraper, monitor_github_repo

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"


class _EntryLevelFilters:
    """Adapter that wraps EntryLevelFiltersConfig to look like FiltersConfig
    for ATS clients that expect keywords_include / keywords_exclude."""

    def __init__(self, config: AppConfig):
        el = config.entry_level_filters
        self.keywords_include = el.keywords_include
        self.keywords_exclude = el.keywords_exclude
        # Reuse role_categories and exclude_companies from main filters
        self.role_categories = config.filters.role_categories
        self.exclude_companies = config.filters.exclude_companies


async def _run_greenhouse(config: AppConfig, filters: object) -> list[RawListing]:
    """Fetch entry-level listings from all configured Greenhouse boards."""
    return await gather_ats_results(
        GreenhouseClient(filters), config.greenhouse_boards, "Greenhouse (entry-level)",
    )


async def _run_lever(config: AppConfig, filters: object) -> list[RawListing]:
    """Fetch entry-level listings from all configured Lever boards."""
    return await gather_ats_results(
        LeverClient(filters), config.lever_boards, "Lever (entry-level)",
    )


async def _run_ashby(config: AppConfig, filters: object) -> list[RawListing]:
    """Fetch entry-level listings from all configured Ashby boards."""
    return await gather_ats_results(
        AshbyClient(filters), config.ashby_boards, "Ashby (entry-level)",
    )


async def _run_workday(config: AppConfig, filters: object) -> list[RawListing]:
    """Fetch entry-level listings from all configured Workday boards."""
    return await gather_ats_results(
        WorkdayClient(filters), config.workday_boards, "Workday (entry-level)",
    )


async def _run_smartrecruiters(config: AppConfig, filters: object) -> list[RawListing]:
    """Fetch entry-level listings from all configured SmartRecruiters boards."""
    return await gather_ats_results(
        SmartRecruitersClient(filters), config.smartrecruiters_boards, "SmartRecruiters (entry-level)",
    )


async def _run_scraping(config: AppConfig) -> list[RawListing]:
    """Scrape all configured career pages for entry-level listings."""
    if not config.scrape_sources:
        return []

    scraper = GenericScraper()
    tasks = [scraper.scrape_career_page(source) for source in config.scrape_sources]

    results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

    listings: list[RawListing] = []
    succeeded = 0
    failed = 0
    for i, result in enumerate(results_or_errors):
        source = config.scrape_sources[i]
        if isinstance(result, BaseException):
            logger.error("Scrape %s failed: %s", source.company, result)
            failed += 1
        else:
            listings.extend(result)
            succeeded += 1

    logger.info(
        "Scraping (entry-level): %d/%d sources succeeded, %d listings found",
        succeeded, succeeded + failed, len(listings),
    )
    return listings


async def _run_github_monitors(config: AppConfig) -> list[RawListing]:
    """Monitor all configured GitHub repos for entry-level listings."""
    if not config.github_monitors:
        return []

    tasks = [monitor_github_repo(monitor) for monitor in config.github_monitors]
    results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

    listings: list[RawListing] = []
    succeeded = 0
    failed = 0
    for i, result in enumerate(results_or_errors):
        monitor = config.github_monitors[i]
        if isinstance(result, BaseException):
            logger.error("GitHub monitor %s failed: %s", monitor.repo, result)
            failed += 1
        else:
            listings.extend(result)
            succeeded += 1

    logger.info(
        "GitHub monitors (entry-level): %d/%d succeeded, %d listings found",
        succeeded, succeeded + failed, len(listings),
    )
    return listings


def _save_raw_results(listings: list[RawListing]) -> Path:
    """Save raw entry-level discovery results to a timestamped JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = DATA_DIR / f"raw_el_discovery_{timestamp}.json"

    serialized: list[dict[str, Any]] = []
    for listing in listings:
        data = listing.model_dump(mode="json")
        serialized.append(data)

    payload = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "total_count": len(serialized),
        "listings": serialized,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Saved %d raw entry-level listings to %s", len(serialized), output_path)
    return output_path


async def discover_entry_level() -> list[RawListing]:
    """Run all discovery sources for entry-level jobs and return combined results.

    Uses entry_level_filters from config.yaml instead of internship filters.
    Tags all discovered listings with listing_type="entry_level".

    Returns:
        Combined list of all discovered RawListing objects.
    """
    config = load_config()
    filters = _EntryLevelFilters(config)

    logger.info(
        "Starting entry-level discovery across %d configured sources",
        config.total_sources,
    )

    source_tasks = [
        _run_greenhouse(config, filters),
        _run_lever(config, filters),
        _run_ashby(config, filters),
        _run_workday(config, filters),
        _run_smartrecruiters(config, filters),
        _run_scraping(config),
        _run_github_monitors(config),
    ]

    results = await asyncio.gather(*source_tasks, return_exceptions=True)

    all_listings: list[RawListing] = []
    source_names = [
        "Greenhouse", "Lever", "Ashby", "Workday",
        "SmartRecruiters", "Scraping", "GitHub Monitors",
    ]

    sources_succeeded = 0
    sources_failed = 0

    for name, result in zip(source_names, results):
        if isinstance(result, BaseException):
            logger.error("Source category %s failed entirely: %s", name, result)
            sources_failed += 1
        else:
            all_listings.extend(result)
            sources_succeeded += 1

    # Tag all listings as entry-level
    for listing in all_listings:
        listing.listing_type = "entry_level"

    logger.info(
        "Entry-level discovery complete: %d total listings from %d/%d source categories",
        len(all_listings), sources_succeeded, sources_succeeded + sources_failed,
    )

    if all_listings:
        _save_raw_results(all_listings)
    else:
        logger.warning("No entry-level listings discovered from any source")

    return all_listings


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    results = asyncio.run(discover_entry_level())
    logging.getLogger(__name__).info("Discovered %d raw entry-level listings", len(results))
