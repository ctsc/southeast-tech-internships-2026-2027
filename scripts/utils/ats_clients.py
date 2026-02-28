"""API clients for Greenhouse, Lever, and Ashby applicant tracking systems.

Each client is async, rate-limited (2 req/sec per domain), and uses tenacity
for retry with exponential backoff. All clients return list[RawListing].
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scripts.utils.config import (
    AshbyBoard,
    FiltersConfig,
    GreenhouseBoard,
    LeverBoard,
)
from scripts.utils.models import RawListing

logger = logging.getLogger(__name__)

USER_AGENT = (
    "InternshipTracker/1.0 (github.com/ctsc/atlanta-tech-internships-2026)"
)
DEFAULT_TIMEOUT = 15.0


def _slugify(name: str) -> str:
    """Convert a company name to a kebab-case slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _title_matches_include(title: str, keywords: list[str]) -> bool:
    """Return True if the title contains at least one include keyword (word-boundary match)."""
    title_lower = title.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
            return True
    return False


def _title_matches_exclude(title: str, keywords: list[str]) -> bool:
    """Return True if the title contains any exclude keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


class BaseATSClient(ABC):
    """Base class for ATS API clients with shared rate limiting and HTTP setup."""

    # Class-level semaphores keyed by domain for rate limiting (2 req/sec).
    _semaphores: dict[str, asyncio.Semaphore] = {}
    _domain: str = ""

    def __init__(self, filters: FiltersConfig) -> None:
        self.filters = filters
        self._include_keywords = [
            kw.lower() for kw in filters.keywords_include
        ]
        self._exclude_keywords = [
            kw.lower() for kw in filters.keywords_exclude
        ]

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create a per-domain semaphore for rate limiting."""
        if self._domain not in BaseATSClient._semaphores:
            BaseATSClient._semaphores[self._domain] = asyncio.Semaphore(2)
        return BaseATSClient._semaphores[self._domain]

    def _build_client(self) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with standard headers and timeout."""
        return httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        )

    def _should_include(self, title: str) -> bool:
        """Check if a job title passes include/exclude keyword filters."""
        if not _title_matches_include(title, self._include_keywords):
            return False
        if _title_matches_exclude(title, self._exclude_keywords):
            return False
        return True

    @abstractmethod
    async def fetch_listings(self, board: object) -> list[RawListing]:
        """Fetch listings from an ATS board. Subclasses must implement."""
        ...


class GreenhouseClient(BaseATSClient):
    """Client for the Greenhouse public job board API.

    Endpoint: GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs
    """

    _domain = "boards-api.greenhouse.io"

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self, client: httpx.AsyncClient, url: str
    ) -> httpx.Response:
        sem = self._get_semaphore()
        async with sem:
            response = await client.get(url)
            response.raise_for_status()
            return response

    async def fetch_listings(
        self, board: GreenhouseBoard
    ) -> list[RawListing]:
        """Fetch internship listings from a Greenhouse board.

        Args:
            board: GreenhouseBoard config with token, company, is_faang_plus.

        Returns:
            Filtered list of RawListing objects.
        """
        url = f"https://boards-api.greenhouse.io/v1/boards/{board.token}/jobs"
        results: list[RawListing] = []

        async with self._build_client() as client:
            try:
                response = await self._request(client, url)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.warning(
                    "Greenhouse %s returned HTTP %d — skipping",
                    board.company,
                    status,
                )
                return []
            except httpx.TransportError as exc:
                logger.warning(
                    "Greenhouse %s transport error: %s — skipping",
                    board.company,
                    exc,
                )
                return []

        data = response.json()
        jobs = data.get("jobs", [])
        logger.info(
            "Greenhouse %s: fetched %d total jobs", board.company, len(jobs)
        )

        for job in jobs:
            title = job.get("title", "")
            if not self._should_include(title):
                continue

            location_obj = job.get("location", {})
            location = location_obj.get("name", "Unknown") if isinstance(location_obj, dict) else "Unknown"
            apply_url = job.get("absolute_url", "")

            if not apply_url:
                continue

            listing = RawListing(
                company=board.company,
                company_slug=_slugify(board.company),
                title=title,
                location=location,
                url=apply_url,
                source="greenhouse_api",
                is_faang_plus=board.is_faang_plus,
                raw_data=job,
                discovered_at=datetime.now(timezone.utc),
            )
            results.append(listing)

        logger.info(
            "Greenhouse %s: %d listings after filtering",
            board.company,
            len(results),
        )
        return results


class LeverClient(BaseATSClient):
    """Client for the Lever public postings API.

    Endpoint: GET https://api.lever.co/v0/postings/{company_slug}
    """

    _domain = "api.lever.co"

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self, client: httpx.AsyncClient, url: str
    ) -> httpx.Response:
        sem = self._get_semaphore()
        async with sem:
            response = await client.get(url)
            response.raise_for_status()
            return response

    async def fetch_listings(self, board: LeverBoard) -> list[RawListing]:
        """Fetch internship listings from a Lever board.

        Args:
            board: LeverBoard config with company_slug, company, is_faang_plus.

        Returns:
            Filtered list of RawListing objects.
        """
        url = f"https://api.lever.co/v0/postings/{board.company_slug}"
        results: list[RawListing] = []

        async with self._build_client() as client:
            try:
                response = await self._request(client, url)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.warning(
                    "Lever %s returned HTTP %d — skipping",
                    board.company,
                    status,
                )
                return []
            except httpx.TransportError as exc:
                logger.warning(
                    "Lever %s transport error: %s — skipping",
                    board.company,
                    exc,
                )
                return []

        data = response.json()
        if not isinstance(data, list):
            logger.warning(
                "Lever %s: unexpected response format — skipping",
                board.company,
            )
            return []

        logger.info(
            "Lever %s: fetched %d total postings", board.company, len(data)
        )

        for posting in data:
            title = posting.get("text", "")
            if not self._should_include(title):
                continue

            categories = posting.get("categories", {})
            location = categories.get("location", "Unknown") if isinstance(categories, dict) else "Unknown"
            hosted_url = posting.get("hostedUrl", "")

            if not hosted_url:
                continue

            listing = RawListing(
                company=board.company,
                company_slug=_slugify(board.company),
                title=title,
                location=location,
                url=hosted_url,
                source="lever_api",
                is_faang_plus=board.is_faang_plus,
                raw_data=posting,
                discovered_at=datetime.now(timezone.utc),
            )
            results.append(listing)

        logger.info(
            "Lever %s: %d listings after filtering",
            board.company,
            len(results),
        )
        return results


class AshbyClient(BaseATSClient):
    """Client for the Ashby public job board API.

    Endpoint: POST https://jobs.ashbyhq.com/api/non-user-graphql
    Uses a GraphQL query to retrieve job board listings.
    """

    _domain = "jobs.ashbyhq.com"

    # Ashby's public API uses a specific operationName and query structure.
    _GRAPHQL_QUERY = """
query ApiJobBoardWithTeams($organizationHostedJobsPageName: String!) {
  jobBoard: jobBoardWithTeams(
    organizationHostedJobsPageName: $organizationHostedJobsPageName
  ) {
    teams {
      ... on JobBoardTeam {
        jobs {
          id
          title
          locationName
          employmentType
          externalLink
        }
      }
    }
  }
}
"""

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self, client: httpx.AsyncClient, payload: dict
    ) -> httpx.Response:
        sem = self._get_semaphore()
        async with sem:
            response = await client.post(
                "https://jobs.ashbyhq.com/api/non-user-graphql",
                json=payload,
            )
            response.raise_for_status()
            return response

    async def fetch_listings(self, board: AshbyBoard) -> list[RawListing]:
        """Fetch internship listings from an Ashby job board.

        Args:
            board: AshbyBoard config with company_slug, company, is_faang_plus.

        Returns:
            Filtered list of RawListing objects.
        """
        payload = {
            "operationName": "ApiJobBoardWithTeams",
            "variables": {
                "organizationHostedJobsPageName": board.company_slug,
            },
            "query": self._GRAPHQL_QUERY,
        }

        results: list[RawListing] = []

        async with self._build_client() as client:
            try:
                response = await self._request(client, payload)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.warning(
                    "Ashby %s returned HTTP %d — skipping",
                    board.company,
                    status,
                )
                return []
            except httpx.TransportError as exc:
                logger.warning(
                    "Ashby %s transport error: %s — skipping",
                    board.company,
                    exc,
                )
                return []

        data = response.json()

        # Navigate the GraphQL response structure
        job_board = data.get("data", {}).get("jobBoard", {})
        teams = job_board.get("teams", [])

        total_jobs = 0
        for team in teams:
            jobs = team.get("jobs", [])
            total_jobs += len(jobs)

            for job in jobs:
                title = job.get("title", "")
                if not self._should_include(title):
                    continue

                location = job.get("locationName", "Unknown") or "Unknown"
                job_id = job.get("id", "")
                external_link = job.get("externalLink")

                # Build the application URL
                if external_link:
                    apply_url = external_link
                elif job_id:
                    apply_url = f"https://jobs.ashbyhq.com/{board.company_slug}/{job_id}"
                else:
                    continue

                listing = RawListing(
                    company=board.company,
                    company_slug=_slugify(board.company),
                    title=title,
                    location=location,
                    url=apply_url,
                    source="ashby_api",
                    is_faang_plus=board.is_faang_plus,
                    raw_data=job,
                    discovered_at=datetime.now(timezone.utc),
                )
                results.append(listing)

        logger.info(
            "Ashby %s: fetched %d total jobs, %d after filtering",
            board.company,
            total_jobs,
            len(results),
        )
        return results
