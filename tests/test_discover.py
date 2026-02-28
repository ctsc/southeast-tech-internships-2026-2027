"""Comprehensive tests for the discovery engine.

Tests cover:
- GreenhouseClient: HTTP 200/404/429/500, keyword filtering, output shape
- LeverClient: HTTP 200/404/429/500, keyword filtering, output shape
- AshbyClient: GraphQL response, error handling
- GenericScraper: HTML parsing, keyword filtering, robots.txt
- GitHub monitor: markdown table parsing, diff logic, new entries only
- discover_all(): aggregation, error isolation, JSON output saved
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scripts.utils.ats_clients import (
    AshbyClient,
    GreenhouseClient,
    LeverClient,
    _slugify,
    _title_matches_exclude,
    _title_matches_include,
)
from scripts.utils.config import (
    AshbyBoard,
    FiltersConfig,
    GitHubMonitor,
    GreenhouseBoard,
    LeverBoard,
    ScrapeSource,
)
from scripts.utils.models import RawListing
from scripts.utils.scraper import (
    GenericScraper,
    _parse_html_table,
    _parse_readme_table,
    _strip_markup,
    monitor_github_repo,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def filters():
    """Standard filters config for testing."""
    return FiltersConfig(
        keywords_include=["intern", "internship", "co-op"],
        keywords_exclude=["senior", "staff", "principal", "director"],
        role_categories={"swe": ["software engineer", "backend"]},
        exclude_companies=["Revature"],
    )


@pytest.fixture
def greenhouse_board():
    return GreenhouseBoard(token="testco", company="TestCo", is_faang_plus=False)


@pytest.fixture
def greenhouse_board_faang():
    return GreenhouseBoard(token="bigco", company="BigCo", is_faang_plus=True)


@pytest.fixture
def lever_board():
    return LeverBoard(company_slug="testco", company="TestCo", is_faang_plus=False)


@pytest.fixture
def ashby_board():
    return AshbyBoard(company_slug="testco", company="TestCo", is_faang_plus=False)


@pytest.fixture
def scrape_source():
    return ScrapeSource(
        company="ScrapeInc", url="https://scrapeinc.com/careers", is_faang_plus=False
    )


@pytest.fixture
def github_monitor():
    return GitHubMonitor(
        repo="SimplifyJobs/Summer2026-Internships", branch="dev", file="README.md"
    )


# ======================================================================
# Helper: utility function tests
# ======================================================================


class TestUtilityFunctions:
    """Tests for shared helper functions in ats_clients."""

    def test_slugify_basic(self):
        assert _slugify("Anthropic") == "anthropic"

    def test_slugify_spaces(self):
        assert _slugify("Scale AI") == "scale-ai"

    def test_slugify_special_chars(self):
        assert _slugify("Stripe, Inc.") == "stripe-inc"

    def test_slugify_leading_trailing(self):
        assert _slugify("  --Test Co-- ") == "test-co"

    def test_title_matches_include_positive(self):
        assert _title_matches_include("Software Engineer Intern", ["intern"])

    def test_title_matches_include_negative(self):
        assert not _title_matches_include("Software Engineer", ["intern"])

    def test_title_matches_include_case_insensitive(self):
        assert _title_matches_include("INTERNSHIP Program", ["internship"])

    def test_title_matches_include_word_boundary_rejects_internal(self):
        """'intern' should NOT match 'internal' or 'international'."""
        assert not _title_matches_include("Internal Revenue Accountant", ["intern"])

    def test_title_matches_include_word_boundary_rejects_international(self):
        assert not _title_matches_include("International Marketing Manager", ["intern"])

    def test_title_matches_include_word_boundary_accepts_intern(self):
        assert _title_matches_include("Software Intern", ["intern"])

    def test_title_matches_include_word_boundary_accepts_internship(self):
        assert _title_matches_include("Summer Internship Program", ["internship"])

    def test_title_matches_include_word_boundary_coop(self):
        assert _title_matches_include("Engineering Co-op Position", ["co-op"])

    def test_title_matches_include_rejects_no_word_boundary(self):
        """'intern' within 'internalize' should not match."""
        assert not _title_matches_include("Internalize Process Lead", ["intern"])

    def test_title_matches_exclude_positive(self):
        assert _title_matches_exclude("Senior Software Intern", ["senior"])

    def test_title_matches_exclude_negative(self):
        assert not _title_matches_exclude("Software Intern", ["senior", "staff"])


# ======================================================================
# GreenhouseClient
# ======================================================================


class TestGreenhouseClient:
    """Tests for the Greenhouse ATS client."""

    @pytest.mark.asyncio
    async def test_fetch_200_with_matching_jobs(self, filters, greenhouse_board):
        """Verify that matching intern jobs are returned as RawListings."""
        mock_response = httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Software Engineer Intern",
                        "location": {"name": "San Francisco, CA"},
                        "absolute_url": "https://boards.greenhouse.io/testco/jobs/1",
                        "id": 1,
                    },
                    {
                        "title": "Product Manager",
                        "location": {"name": "NYC"},
                        "absolute_url": "https://boards.greenhouse.io/testco/jobs/2",
                        "id": 2,
                    },
                ]
            },
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board)

        assert len(results) == 1
        assert results[0].title == "Software Engineer Intern"
        assert results[0].company == "TestCo"
        assert results[0].source == "greenhouse_api"
        assert results[0].location == "San Francisco, CA"
        assert isinstance(results[0], RawListing)

    @pytest.mark.asyncio
    async def test_fetch_200_no_matching_jobs(self, filters, greenhouse_board):
        """No listings returned when no titles match keywords."""
        mock_response = httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Senior Software Engineer",
                        "location": {"name": "NYC"},
                        "absolute_url": "https://boards.greenhouse.io/testco/jobs/1",
                        "id": 1,
                    },
                ]
            },
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fetch_excludes_senior_intern(self, filters, greenhouse_board):
        """Exclude keyword 'senior' filters out 'Senior Intern' titles."""
        mock_response = httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Senior Software Intern",
                        "location": {"name": "NYC"},
                        "absolute_url": "https://boards.greenhouse.io/testco/jobs/1",
                        "id": 1,
                    },
                ]
            },
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fetch_404_returns_empty(self, filters, greenhouse_board):
        """HTTP 404 should return empty list, not raise."""
        client = GreenhouseClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
                response=httpx.Response(404),
            ),
        ):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_429_returns_empty(self, filters, greenhouse_board):
        """HTTP 429 rate limited should return empty list."""
        client = GreenhouseClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Rate Limited",
                request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
                response=httpx.Response(429),
            ),
        ):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_500_returns_empty(self, filters, greenhouse_board):
        """HTTP 500 server error should return empty list."""
        client = GreenhouseClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
                response=httpx.Response(500),
            ),
        ):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_transport_error_returns_empty(self, filters, greenhouse_board):
        """Network transport errors should return empty list."""
        client = GreenhouseClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.TransportError("Connection refused"),
        ):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_faang_plus_flag_propagated(self, filters, greenhouse_board_faang):
        """is_faang_plus from board config should propagate to RawListing."""
        mock_response = httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "ML Intern",
                        "location": {"name": "Remote"},
                        "absolute_url": "https://boards.greenhouse.io/bigco/jobs/1",
                        "id": 1,
                    },
                ]
            },
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/bigco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board_faang)

        # "ML Intern" contains "intern" so it passes
        assert len(results) == 1
        assert results[0].is_faang_plus is True

    @pytest.mark.asyncio
    async def test_empty_jobs_response(self, filters, greenhouse_board):
        """Empty jobs array should return empty list."""
        mock_response = httpx.Response(
            200,
            json={"jobs": []},
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_missing_url_skipped(self, filters, greenhouse_board):
        """Jobs without absolute_url should be skipped."""
        mock_response = httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Software Intern",
                        "location": {"name": "NYC"},
                        "absolute_url": "",
                        "id": 1,
                    },
                ]
            },
            request=httpx.Request("GET", "https://boards-api.greenhouse.io/v1/boards/testco/jobs"),
        )

        client = GreenhouseClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(greenhouse_board)

        assert results == []


# ======================================================================
# LeverClient
# ======================================================================


class TestLeverClient:
    """Tests for the Lever ATS client."""

    @pytest.mark.asyncio
    async def test_fetch_200_with_matching_postings(self, filters, lever_board):
        """Matching intern postings are returned."""
        mock_response = httpx.Response(
            200,
            json=[
                {
                    "text": "Software Engineering Intern",
                    "categories": {"location": "San Francisco"},
                    "hostedUrl": "https://jobs.lever.co/testco/abc123",
                },
                {
                    "text": "Staff Engineer",
                    "categories": {"location": "NYC"},
                    "hostedUrl": "https://jobs.lever.co/testco/def456",
                },
            ],
            request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
        )

        client = LeverClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(lever_board)

        assert len(results) == 1
        assert results[0].title == "Software Engineering Intern"
        assert results[0].source == "lever_api"

    @pytest.mark.asyncio
    async def test_fetch_404_returns_empty(self, filters, lever_board):
        """HTTP 404 returns empty list."""
        client = LeverClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
                response=httpx.Response(404),
            ),
        ):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_429_returns_empty(self, filters, lever_board):
        """HTTP 429 rate limited returns empty list."""
        client = LeverClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Rate Limited",
                request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
                response=httpx.Response(429),
            ),
        ):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_500_returns_empty(self, filters, lever_board):
        """HTTP 500 server error returns empty list."""
        client = LeverClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
                response=httpx.Response(500),
            ),
        ):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_non_list_response_returns_empty(self, filters, lever_board):
        """Non-list response body should return empty list."""
        mock_response = httpx.Response(
            200,
            json={"error": "unexpected"},
            request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
        )

        client = LeverClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_transport_error_returns_empty(self, filters, lever_board):
        """Transport errors return empty list."""
        client = LeverClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.TransportError("Timeout"),
        ):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_missing_hosted_url_skipped(self, filters, lever_board):
        """Postings without hostedUrl are skipped."""
        mock_response = httpx.Response(
            200,
            json=[
                {
                    "text": "Software Intern",
                    "categories": {"location": "SF"},
                    "hostedUrl": "",
                },
            ],
            request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
        )

        client = LeverClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(lever_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_co_op_keyword_matches(self, filters, lever_board):
        """co-op keyword in title should match."""
        mock_response = httpx.Response(
            200,
            json=[
                {
                    "text": "Software Engineering Co-Op",
                    "categories": {"location": "Boston"},
                    "hostedUrl": "https://jobs.lever.co/testco/xyz",
                },
            ],
            request=httpx.Request("GET", "https://api.lever.co/v0/postings/testco"),
        )

        client = LeverClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(lever_board)

        assert len(results) == 1
        assert results[0].title == "Software Engineering Co-Op"


# ======================================================================
# AshbyClient
# ======================================================================


class TestAshbyClient:
    """Tests for the Ashby GraphQL client."""

    @pytest.mark.asyncio
    async def test_fetch_200_with_matching_jobs(self, filters, ashby_board):
        """Matching intern jobs from GraphQL response are returned."""
        mock_response = httpx.Response(
            200,
            json={
                "data": {
                    "jobBoard": {
                        "teams": [
                            {
                                "jobs": [
                                    {
                                        "id": "job-1",
                                        "title": "Software Engineering Intern",
                                        "locationName": "New York, NY",
                                        "employmentType": "Intern",
                                        "externalLink": None,
                                    },
                                    {
                                        "id": "job-2",
                                        "title": "Senior Backend Engineer",
                                        "locationName": "Remote",
                                        "employmentType": "FullTime",
                                        "externalLink": None,
                                    },
                                ]
                            }
                        ]
                    }
                }
            },
            request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
        )

        client = AshbyClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(ashby_board)

        assert len(results) == 1
        assert results[0].title == "Software Engineering Intern"
        assert results[0].source == "ashby_api"
        # Should construct URL from board slug + job id
        assert "testco" in results[0].url
        assert "job-1" in results[0].url

    @pytest.mark.asyncio
    async def test_fetch_with_external_link(self, filters, ashby_board):
        """Jobs with externalLink should use that URL."""
        mock_response = httpx.Response(
            200,
            json={
                "data": {
                    "jobBoard": {
                        "teams": [
                            {
                                "jobs": [
                                    {
                                        "id": "job-1",
                                        "title": "Data Science Intern",
                                        "locationName": "Remote",
                                        "employmentType": "Intern",
                                        "externalLink": "https://external.com/apply/123",
                                    },
                                ]
                            }
                        ]
                    }
                }
            },
            request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
        )

        client = AshbyClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(ashby_board)

        assert len(results) == 1
        assert results[0].url == "https://external.com/apply/123"

    @pytest.mark.asyncio
    async def test_fetch_404_returns_empty(self, filters, ashby_board):
        """HTTP 404 returns empty list."""
        client = AshbyClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
                response=httpx.Response(404),
            ),
        ):
            results = await client.fetch_listings(ashby_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_500_returns_empty(self, filters, ashby_board):
        """HTTP 500 returns empty list."""
        client = AshbyClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
                response=httpx.Response(500),
            ),
        ):
            results = await client.fetch_listings(ashby_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_teams(self, filters, ashby_board):
        """Jobs from multiple teams are aggregated."""
        mock_response = httpx.Response(
            200,
            json={
                "data": {
                    "jobBoard": {
                        "teams": [
                            {
                                "jobs": [
                                    {
                                        "id": "j1",
                                        "title": "Backend Intern",
                                        "locationName": "SF",
                                        "employmentType": "Intern",
                                        "externalLink": None,
                                    },
                                ]
                            },
                            {
                                "jobs": [
                                    {
                                        "id": "j2",
                                        "title": "ML Internship",
                                        "locationName": "NYC",
                                        "employmentType": "Intern",
                                        "externalLink": None,
                                    },
                                ]
                            },
                        ]
                    }
                }
            },
            request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
        )

        client = AshbyClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(ashby_board)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_teams(self, filters, ashby_board):
        """Empty teams array returns empty list."""
        mock_response = httpx.Response(
            200,
            json={"data": {"jobBoard": {"teams": []}}},
            request=httpx.Request("POST", "https://jobs.ashbyhq.com/api/non-user-graphql"),
        )

        client = AshbyClient(filters)
        with patch.object(client, "_request", new_callable=AsyncMock, return_value=mock_response):
            results = await client.fetch_listings(ashby_board)

        assert results == []

    @pytest.mark.asyncio
    async def test_transport_error_returns_empty(self, filters, ashby_board):
        """Transport errors return empty list."""
        client = AshbyClient(filters)
        with patch.object(
            client,
            "_request",
            new_callable=AsyncMock,
            side_effect=httpx.TransportError("DNS failure"),
        ):
            results = await client.fetch_listings(ashby_board)

        assert results == []


# ======================================================================
# GenericScraper
# ======================================================================


class TestGenericScraper:
    """Tests for the generic career page scraper."""

    @pytest.mark.asyncio
    async def test_scrape_finds_intern_links(self, scrape_source):
        """Scraper should find anchor tags with intern keywords."""
        html = """
        <html><body>
            <a href="/jobs/123">Software Engineering Intern</a>
            <a href="/jobs/456">Senior Staff Engineer</a>
            <a href="/jobs/789">ML Internship - Summer 2026</a>
        </body></html>
        """

        with patch.object(GenericScraper, "__init__", lambda self: None):
            scraper = GenericScraper()
            scraper._rate_limiter = MagicMock()
            scraper._intern_keywords = ["intern", "internship"]
            scraper._exclude_keywords = ["senior", "staff"]
            scraper._config = MagicMock()

        with patch.object(scraper, "check_robots_txt", new_callable=AsyncMock, return_value=True), \
             patch.object(scraper, "_fetch_page", new_callable=AsyncMock, return_value=html), \
             patch("scripts.utils.scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.scrape_career_page(scrape_source)

        # Should find "Software Engineering Intern" and "ML Internship"
        # but not "Senior Staff Engineer"
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any("Intern" in t for t in titles)
        assert all("Senior" not in t for t in titles)

    @pytest.mark.asyncio
    async def test_scrape_robots_blocked(self, scrape_source):
        """Scraper should respect robots.txt denial."""
        with patch.object(GenericScraper, "__init__", lambda self: None):
            scraper = GenericScraper()
            scraper._rate_limiter = MagicMock()
            scraper._config = MagicMock()

        with patch.object(scraper, "check_robots_txt", new_callable=AsyncMock, return_value=False):
            results = await scraper.scrape_career_page(scrape_source)

        assert results == []

    @pytest.mark.asyncio
    async def test_scrape_fetch_failure(self, scrape_source):
        """Scraper should return empty list on fetch failure."""
        with patch.object(GenericScraper, "__init__", lambda self: None):
            scraper = GenericScraper()
            scraper._rate_limiter = MagicMock()
            scraper._config = MagicMock()

        with patch.object(scraper, "check_robots_txt", new_callable=AsyncMock, return_value=True), \
             patch.object(
                 scraper, "_fetch_page", new_callable=AsyncMock,
                 side_effect=httpx.HTTPError("Connection failed"),
             ):
            results = await scraper.scrape_career_page(scrape_source)

        assert results == []

    @pytest.mark.asyncio
    async def test_scrape_empty_page(self, scrape_source):
        """Empty page returns empty list."""
        with patch.object(GenericScraper, "__init__", lambda self: None):
            scraper = GenericScraper()
            scraper._rate_limiter = MagicMock()
            scraper._config = MagicMock()

        with patch.object(scraper, "check_robots_txt", new_callable=AsyncMock, return_value=True), \
             patch.object(scraper, "_fetch_page", new_callable=AsyncMock, return_value=""):
            results = await scraper.scrape_career_page(scrape_source)

        assert results == []


# ======================================================================
# Markdown table parsing (used by GitHub monitor)
# ======================================================================


class TestMarkdownParsing:
    """Tests for README markdown table parsing."""

    def test_parse_standard_table(self):
        """Parse a standard SimplifyJobs-style markdown table."""
        content = """
# Summer 2026 Internships

| Company | Role | Location | Application/Link | Date Posted |
|---------|------|----------|------------------|-------------|
| **Stripe** | Software Engineer Intern | San Francisco, CA | [Apply](https://stripe.com/jobs/1) | Jan 15 |
| **Anthropic** | ML Research Intern | San Francisco, CA | [Apply](https://anthropic.com/jobs/2) | Jan 20 |
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 2
        assert rows[0]["company"] == "Stripe"
        assert rows[0]["role"] == "Software Engineer Intern"
        assert rows[0]["url"] == "https://stripe.com/jobs/1"
        assert rows[1]["company"] == "Anthropic"

    def test_parse_empty_content(self):
        """Empty content returns empty list."""
        rows = _parse_readme_table("", "test/repo")
        assert rows == []

    def test_parse_no_tables(self):
        """Content without tables returns empty list."""
        content = "# Hello\n\nThis is just text without any tables."
        rows = _parse_readme_table(content, "test/repo")
        assert rows == []

    def test_parse_skips_header_row(self):
        """Header rows like 'Company' are skipped."""
        content = """
| Company | Role | Location | Link |
|---------|------|----------|------|
| **Google** | SWE Intern | MTV | [Apply](https://google.com/1) |
"""
        rows = _parse_readme_table(content, "test/repo")
        # Should only have Google, not the header
        assert all(r["company"] != "Company" for r in rows)

    def test_parse_handles_links_in_cells(self):
        """URLs are correctly extracted from markdown links in cells."""
        content = """
| Company | Role | Location | Link |
|---------|------|----------|------|
| **Meta** | Intern | Menlo Park | [Apply](https://metacareers.com/job/1) |
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) >= 1
        assert rows[0]["url"] == "https://metacareers.com/job/1"

    def test_strip_markup_bold(self):
        """Bold markdown is stripped."""
        assert _strip_markup("**Google**") == "Google"

    def test_strip_markup_link(self):
        """Markdown links keep text, remove URL."""
        assert _strip_markup("[Apply](https://example.com)") == "Apply"

    def test_strip_markup_combined(self):
        """Combined markdown formatting is stripped."""
        assert _strip_markup("**[Google](https://google.com)**") == "Google"


# ======================================================================
# HTML table parsing (Simplify format)
# ======================================================================


class TestHTMLTableParsing:
    """Tests for HTML table parsing used by Simplify and similar repos."""

    def test_parse_basic_html_table(self):
        """Parse a simple HTML table with company, role, location, apply link."""
        content = """
<table>
<thead><tr><th>Company</th><th>Role</th><th>Location</th><th>Application</th><th>Date</th></tr></thead>
<tbody>
<tr>
  <td><strong><a href="https://stripe.com">Stripe</a></strong></td>
  <td>Software Engineer Intern</td>
  <td>San Francisco, CA</td>
  <td><a href="https://stripe.com/jobs/123">Apply</a></td>
  <td>Jan 15</td>
</tr>
<tr>
  <td><strong><a href="https://anthropic.com">Anthropic</a></strong></td>
  <td>ML Research Intern</td>
  <td>San Francisco, CA</td>
  <td><a href="https://anthropic.com/jobs/456">Apply</a></td>
  <td>Jan 20</td>
</tr>
</tbody>
</table>
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 2
        assert rows[0]["company"] == "Stripe"
        assert rows[0]["role"] == "Software Engineer Intern"
        assert rows[0]["location"] == "San Francisco, CA"
        assert rows[0]["url"] == "https://stripe.com/jobs/123"
        assert rows[1]["company"] == "Anthropic"
        assert rows[1]["url"] == "https://anthropic.com/jobs/456"

    def test_parse_continuation_rows(self):
        """↳ continuation rows carry forward the previous company."""
        content = """
<table>
<tbody>
<tr>
  <td><strong><a href="https://google.com">Google</a></strong></td>
  <td>SWE Intern</td>
  <td>Mountain View, CA</td>
  <td><a href="https://google.com/jobs/1">Apply</a></td>
</tr>
<tr>
  <td>↳</td>
  <td>ML Intern</td>
  <td>New York, NY</td>
  <td><a href="https://google.com/jobs/2">Apply</a></td>
</tr>
<tr>
  <td>↳</td>
  <td>Data Science Intern</td>
  <td>Remote</td>
  <td><a href="https://google.com/jobs/3">Apply</a></td>
</tr>
</tbody>
</table>
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 3
        assert all(r["company"] == "Google" for r in rows)
        assert rows[0]["role"] == "SWE Intern"
        assert rows[1]["role"] == "ML Intern"
        assert rows[2]["role"] == "Data Science Intern"

    def test_parse_details_location(self):
        """Locations inside <details> tags are expanded and included."""
        content = """
<table>
<tbody>
<tr>
  <td><strong>Meta</strong></td>
  <td>SWE Intern</td>
  <td>Menlo Park, CA<br><details><summary>3 more</summary>New York, NY<br>Seattle, WA<br>Austin, TX</details></td>
  <td><a href="https://metacareers.com/jobs/1">Apply</a></td>
</tr>
</tbody>
</table>
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 1
        loc = rows[0]["location"]
        assert "Menlo Park" in loc
        assert "New York" in loc
        assert "Seattle" in loc
        assert "Austin" in loc

    def test_parse_html_table_no_apply_link_skipped(self):
        """Rows without an apply URL are skipped."""
        content = """
<table>
<tbody>
<tr>
  <td><strong>NoLink Corp</strong></td>
  <td>Intern</td>
  <td>Remote</td>
  <td>Closed</td>
</tr>
</tbody>
</table>
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 0

    def test_parse_html_table_emoji_stripped_from_role(self):
        """Emoji in role cells is stripped by _strip_markup."""
        content = """
<table>
<tbody>
<tr>
  <td><strong>Stripe</strong></td>
  <td>🔒 SWE Intern</td>
  <td>SF</td>
  <td><a href="https://stripe.com/jobs/99">Apply</a></td>
</tr>
</tbody>
</table>
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) == 1
        assert "SWE Intern" in rows[0]["role"]
        # Emoji should be stripped
        assert "\U0001f512" not in rows[0]["role"]

    def test_parse_html_falls_back_to_markdown(self):
        """If content has no <table> tag, falls back to markdown pipe parsing."""
        content = """
| Company | Role | Location | Link |
|---------|------|----------|------|
| **Google** | SWE Intern | MTV | [Apply](https://google.com/1) |
"""
        rows = _parse_readme_table(content, "test/repo")
        assert len(rows) >= 1
        assert rows[0]["company"] == "Google"

    def test_parse_html_table_directly(self):
        """_parse_html_table can be called directly."""
        content = """
<table><tbody>
<tr>
  <td><strong>Ramp</strong></td>
  <td>Backend Intern</td>
  <td>NYC</td>
  <td><a href="https://ramp.com/jobs/5">Apply</a></td>
</tr>
</tbody></table>
"""
        rows = _parse_html_table(content)
        assert len(rows) == 1
        assert rows[0]["company"] == "Ramp"
        assert rows[0]["url"] == "https://ramp.com/jobs/5"


# ======================================================================
# GitHub Monitor (monitor_github_repo)
# ======================================================================


class TestGitHubMonitor:
    """Tests for the GitHub repo monitor."""

    @pytest.mark.asyncio
    async def test_monitor_new_entries(self, github_monitor):
        """New entries from a monitored repo are returned."""
        readme_content = """
| Company | Role | Location | Application/Link | Date |
|---------|------|----------|------------------|------|
| **Stripe** | SWE Intern | SF | [Apply](https://stripe.com/jobs/1) | Jan 15 |
| **Ramp** | Data Intern | NYC | [Apply](https://ramp.com/jobs/2) | Jan 20 |
"""
        mock_response = httpx.Response(
            200,
            text=readme_content,
            request=httpx.Request("GET", "https://raw.githubusercontent.com/test/test"),
        )

        # Patch state to have no previous entries
        with patch("scripts.utils.scraper.httpx.AsyncClient") as mock_client_cls, \
             patch("scripts.utils.scraper._load_monitor_state", return_value=set()), \
             patch("scripts.utils.scraper._save_monitor_state"):

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            results = await monitor_github_repo(github_monitor)

        assert len(results) == 2
        companies = {r.company for r in results}
        assert "Stripe" in companies
        assert "Ramp" in companies
        assert all(r.source == "github_monitor" for r in results)

    @pytest.mark.asyncio
    async def test_monitor_only_new_entries(self, github_monitor):
        """Only entries not previously seen are returned."""
        readme_content = """
| Company | Role | Location | Application/Link | Date |
|---------|------|----------|------------------|------|
| **Stripe** | SWE Intern | SF | [Apply](https://stripe.com/jobs/1) | Jan 15 |
| **Ramp** | Data Intern | NYC | [Apply](https://ramp.com/jobs/2) | Jan 20 |
"""
        mock_response = httpx.Response(
            200,
            text=readme_content,
            request=httpx.Request("GET", "https://raw.githubusercontent.com/test/test"),
        )

        # Previous state already has Stripe
        with patch("scripts.utils.scraper.httpx.AsyncClient") as mock_client_cls, \
             patch("scripts.utils.scraper._load_monitor_state", return_value={"https://stripe.com/jobs/1"}), \
             patch("scripts.utils.scraper._save_monitor_state"):

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            results = await monitor_github_repo(github_monitor)

        # Only Ramp should be new
        assert len(results) == 1
        assert results[0].company == "Ramp"

    @pytest.mark.asyncio
    async def test_monitor_http_error(self, github_monitor):
        """HTTP errors return empty list."""
        mock_response = httpx.Response(
            404,
            request=httpx.Request("GET", "https://raw.githubusercontent.com/test/test"),
        )

        with patch("scripts.utils.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=mock_response.request,
                    response=mock_response,
                )
            )
            mock_client_cls.return_value = mock_client

            results = await monitor_github_repo(github_monitor)

        assert results == []

    @pytest.mark.asyncio
    async def test_monitor_network_error(self, github_monitor):
        """Network errors return empty list."""
        with patch("scripts.utils.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_cls.return_value = mock_client

            results = await monitor_github_repo(github_monitor)

        assert results == []


# ======================================================================
# discover_all() orchestrator
# ======================================================================


class TestDiscoverAll:
    """Tests for the main discover_all orchestrator."""

    @pytest.mark.asyncio
    async def test_aggregates_results_from_all_sources(self):
        """discover_all combines results from all source types."""
        greenhouse_listings = [
            RawListing(
                company="Stripe",
                company_slug="stripe",
                title="SWE Intern",
                location="SF",
                url="https://stripe.com/jobs/1",
                source="greenhouse_api",
            ),
        ]
        lever_listings = [
            RawListing(
                company="Netflix",
                company_slug="netflix",
                title="Backend Intern",
                location="LA",
                url="https://netflix.com/jobs/2",
                source="lever_api",
            ),
        ]

        with patch("scripts.discover.load_config") as mock_config, \
             patch("scripts.discover._run_greenhouse", new_callable=AsyncMock, return_value=greenhouse_listings), \
             patch("scripts.discover._run_lever", new_callable=AsyncMock, return_value=lever_listings), \
             patch("scripts.discover._run_ashby", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_scraping", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_github_monitors", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._save_raw_results"):

            mock_config.return_value = MagicMock(total_sources=5)

            from scripts.discover import discover_all
            results = await discover_all()

        assert len(results) == 2
        companies = {r.company for r in results}
        assert "Stripe" in companies
        assert "Netflix" in companies

    @pytest.mark.asyncio
    async def test_isolates_source_failures(self):
        """Failure in one source category does not affect others."""
        good_listings = [
            RawListing(
                company="OK Corp",
                company_slug="ok-corp",
                title="Intern",
                location="NYC",
                url="https://okcorp.com/1",
                source="lever_api",
            ),
        ]

        with patch("scripts.discover.load_config") as mock_config, \
             patch("scripts.discover._run_greenhouse", new_callable=AsyncMock, side_effect=Exception("Greenhouse crashed")), \
             patch("scripts.discover._run_lever", new_callable=AsyncMock, return_value=good_listings), \
             patch("scripts.discover._run_ashby", new_callable=AsyncMock, side_effect=Exception("Ashby crashed")), \
             patch("scripts.discover._run_scraping", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_github_monitors", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._save_raw_results"):

            mock_config.return_value = MagicMock(total_sources=5)

            from scripts.discover import discover_all
            results = await discover_all()

        # Should still get Lever results despite Greenhouse + Ashby failures
        assert len(results) == 1
        assert results[0].company == "OK Corp"

    @pytest.mark.asyncio
    async def test_no_listings_discovered(self):
        """When no sources return results, returns empty list."""
        with patch("scripts.discover.load_config") as mock_config, \
             patch("scripts.discover._run_greenhouse", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_lever", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_ashby", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_scraping", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_github_monitors", new_callable=AsyncMock, return_value=[]):

            mock_config.return_value = MagicMock(total_sources=5)

            from scripts.discover import discover_all
            results = await discover_all()

        assert results == []

    @pytest.mark.asyncio
    async def test_save_raw_results_called(self):
        """_save_raw_results is called when listings are found."""
        listings = [
            RawListing(
                company="Test",
                company_slug="test",
                title="Intern",
                location="Remote",
                url="https://test.com/1",
                source="greenhouse_api",
            ),
        ]

        with patch("scripts.discover.load_config") as mock_config, \
             patch("scripts.discover._run_greenhouse", new_callable=AsyncMock, return_value=listings), \
             patch("scripts.discover._run_lever", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_ashby", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_scraping", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._run_github_monitors", new_callable=AsyncMock, return_value=[]), \
             patch("scripts.discover._save_raw_results") as mock_save:

            mock_config.return_value = MagicMock(total_sources=5)

            from scripts.discover import discover_all
            await discover_all()

        mock_save.assert_called_once()
        saved_listings = mock_save.call_args[0][0]
        assert len(saved_listings) == 1


class TestSaveRawResults:
    """Tests for the _save_raw_results helper."""

    def test_saves_json_file(self, tmp_path):
        """Raw results are saved as valid JSON."""
        from scripts.discover import _save_raw_results

        listings = [
            RawListing(
                company="Test",
                company_slug="test",
                title="Intern",
                location="Remote",
                url="https://test.com/1",
                source="greenhouse_api",
            ),
        ]

        with patch("scripts.discover.DATA_DIR", tmp_path):
            output_path = _save_raw_results(listings)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["total_count"] == 1
        assert len(data["listings"]) == 1
        assert data["listings"][0]["company"] == "Test"

    def test_saves_empty_list(self, tmp_path):
        """Empty listings save as valid JSON with 0 count."""
        from scripts.discover import _save_raw_results

        with patch("scripts.discover.DATA_DIR", tmp_path):
            output_path = _save_raw_results([])

        data = json.loads(output_path.read_text())
        assert data["total_count"] == 0
        assert data["listings"] == []


# ======================================================================
# RawListing model
# ======================================================================


class TestRawListingModel:
    """Tests for the RawListing Pydantic model."""

    def test_create_valid(self):
        listing = RawListing(
            company="Anthropic",
            company_slug="anthropic",
            title="Software Intern",
            location="San Francisco",
            url="https://anthropic.com/jobs/1",
            source="greenhouse_api",
        )
        assert listing.company == "Anthropic"
        assert listing.is_faang_plus is False

    def test_content_hash_deterministic(self):
        """Same inputs produce same content hash."""
        a = RawListing(
            company="Stripe",
            company_slug="stripe",
            title="SWE Intern",
            location="SF",
            url="https://stripe.com/1",
            source="greenhouse_api",
        )
        b = RawListing(
            company="Stripe",
            company_slug="stripe",
            title="SWE Intern",
            location="SF",
            url="https://stripe.com/1",
            source="greenhouse_api",
        )
        assert a.content_hash == b.content_hash

    def test_content_hash_different_for_different_roles(self):
        """Different roles produce different hashes."""
        a = RawListing(
            company="Stripe",
            company_slug="stripe",
            title="SWE Intern",
            location="SF",
            url="https://stripe.com/1",
            source="greenhouse_api",
        )
        b = RawListing(
            company="Stripe",
            company_slug="stripe",
            title="ML Intern",
            location="SF",
            url="https://stripe.com/2",
            source="greenhouse_api",
        )
        assert a.content_hash != b.content_hash

    def test_content_hash_case_insensitive(self):
        """Content hash is case-insensitive."""
        a = RawListing(
            company="Stripe",
            company_slug="stripe",
            title="SWE Intern",
            location="SF",
            url="https://stripe.com/1",
            source="greenhouse_api",
        )
        b = RawListing(
            company="stripe",
            company_slug="stripe",
            title="swe intern",
            location="sf",
            url="https://stripe.com/1",
            source="greenhouse_api",
        )
        assert a.content_hash == b.content_hash

    def test_default_discovered_at(self):
        """discovered_at defaults to now."""
        listing = RawListing(
            company="Test",
            company_slug="test",
            title="Intern",
            location="NYC",
            url="https://test.com/1",
            source="lever_api",
        )
        assert isinstance(listing.discovered_at, datetime)
