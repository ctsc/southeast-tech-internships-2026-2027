"""Discovery engine that scans ATS APIs and career pages to find new internship listings.

Orchestrates all configured sources (Greenhouse, Lever, Ashby, web scraping,
GitHub monitors) in parallel using asyncio, isolates errors per source, and
saves raw results to data/raw_discovery_{timestamp}.json.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


async def gather_ats_results(
    client: object, boards: list, source_name: str,
) -> list[RawListing]:
    """Gather listings from an ATS client across multiple boards.

    Args:
        client: An ATS client instance with a fetch_listings method.
        boards: List of board config objects (each must have .company).
        source_name: Human-readable source name for logging.

    Returns:
        Combined list of all discovered RawListing objects.
    """
    tasks = [client.fetch_listings(board) for board in boards]
    if not tasks:
        return []

    results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

    listings: list[RawListing] = []
    succeeded = 0
    failed = 0
    for i, result in enumerate(results_or_errors):
        board = boards[i]
        if isinstance(result, BaseException):
            logger.error("%s %s failed: %s", source_name, board.company, result)
            failed += 1
        else:
            listings.extend(result)
            succeeded += 1

    logger.info(
        "%s: %d/%d boards succeeded, %d listings found",
        source_name, succeeded, succeeded + failed, len(listings),
    )
    return listings


async def _run_greenhouse(config: AppConfig) -> list[RawListing]:
    """Fetch listings from all configured Greenhouse boards."""
    return await gather_ats_results(
        GreenhouseClient(config.filters), config.greenhouse_boards, "Greenhouse",
    )


async def _run_lever(config: AppConfig) -> list[RawListing]:
    """Fetch listings from all configured Lever boards."""
    return await gather_ats_results(
        LeverClient(config.filters), config.lever_boards, "Lever",
    )


async def _run_ashby(config: AppConfig) -> list[RawListing]:
    """Fetch listings from all configured Ashby boards."""
    return await gather_ats_results(
        AshbyClient(config.filters), config.ashby_boards, "Ashby",
    )


async def _run_workday(config: AppConfig) -> list[RawListing]:
    """Fetch listings from all configured Workday boards."""
    return await gather_ats_results(
        WorkdayClient(config.filters), config.workday_boards, "Workday",
    )


async def _run_smartrecruiters(config: AppConfig) -> list[RawListing]:
    """Fetch listings from all configured SmartRecruiters boards."""
    return await gather_ats_results(
        SmartRecruitersClient(config.filters), config.smartrecruiters_boards, "SmartRecruiters",
    )


async def _run_scraping(config: AppConfig) -> list[RawListing]:
    """Scrape all configured career pages."""
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
        "Scraping: %d/%d sources succeeded, %d listings found",
        succeeded,
        succeeded + failed,
        len(listings),
    )
    return listings


async def _run_github_monitors(config: AppConfig) -> list[RawListing]:
    """Monitor all configured GitHub repositories for new listings."""
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
            logger.error(
                "GitHub monitor %s failed: %s", monitor.repo, result
            )
            failed += 1
        else:
            listings.extend(result)
            succeeded += 1

    logger.info(
        "GitHub monitors: %d/%d succeeded, %d listings found",
        succeeded,
        succeeded + failed,
        len(listings),
    )
    return listings


def _save_raw_results(listings: list[RawListing]) -> Path:
    """Save raw discovery results to a timestamped JSON file.

    Args:
        listings: All discovered RawListing objects.

    Returns:
        Path to the saved JSON file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = DATA_DIR / f"raw_discovery_{timestamp}.json"

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

    logger.info("Saved %d raw listings to %s", len(serialized), output_path)
    return output_path


async def discover_all() -> list[RawListing]:
    """Run all discovery sources and return combined results.

    Orchestrates Greenhouse, Lever, Ashby, web scraping, and GitHub monitors
    in parallel. Errors from individual sources are isolated and logged.

    Returns:
        Combined list of all discovered RawListing objects.
    """
    config = load_config()

    logger.info(
        "Starting discovery across %d configured sources",
        config.total_sources,
    )

    # Run all source categories in parallel
    source_tasks = [
        _run_greenhouse(config),
        _run_lever(config),
        _run_ashby(config),
        _run_workday(config),
        _run_smartrecruiters(config),
        _run_scraping(config),
        _run_github_monitors(config),
    ]

    results = await asyncio.gather(*source_tasks, return_exceptions=True)

    # Collect all listings, isolating any top-level failures
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

    logger.info(
        "Discovery complete: %d total listings from %d/%d source categories",
        len(all_listings),
        sources_succeeded,
        sources_succeeded + sources_failed,
    )

    # Save raw results for debugging and downstream processing
    if all_listings:
        _save_raw_results(all_listings)
    else:
        logger.warning("No listings discovered from any source")

    return all_listings


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    results = asyncio.run(discover_all())
    logging.getLogger(__name__).info("Discovered %d raw listings", len(results))
