"""Tests for the community issue processing pipeline.

Tests cover:
- _parse_issue_body: valid body, missing fields, empty body, malformed body
- _validate_url: valid/invalid URLs
- _map_category: all valid categories + unknown
- _parse_locations: delimiters, city/state, multiple
- _build_job_listing: correct construction, flags mapping
- _get_missing_fields: None input, partial fields
- process_issues: no issues, valid submission, invalid submissions,
                  multiple issues, error isolation, duplicate detection
"""

import json
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from scripts.process_issues import (
    _build_job_listing,
    _get_missing_fields,
    _map_category,
    _parse_issue_body,
    _parse_locations,
    _validate_url,
    process_issues,
)
from scripts.utils.models import (
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_issue_body(
    company: str = "Stripe",
    role: str = "Software Engineer Intern",
    url: str = "https://stripe.com/jobs/123",
    location: str = "San Francisco, CA / Remote",
    category: str = "Software Engineering",
    sponsors: bool = False,
    us_citizenship: bool = False,
    remote: bool = True,
    advanced_degree: bool = False,
    open_to_international: bool = False,
) -> str:
    """Build a realistic GitHub issue form body."""
    flags = []
    flags.append(f"- [{'X' if sponsors else ' '}] Offers visa sponsorship")
    flags.append(f"- [{'X' if us_citizenship else ' '}] Requires U.S. citizenship")
    flags.append(f"- [{'X' if remote else ' '}] Remote friendly")
    flags.append(f"- [{'X' if advanced_degree else ' '}] Requires advanced degree (Master's/PhD)")
    flags.append(f"- [{'X' if open_to_international else ' '}] Open to international students")

    return (
        f"### Company Name\n\n{company}\n\n"
        f"### Role Title\n\n{role}\n\n"
        f"### Application URL\n\n{url}\n\n"
        f"### Location(s)\n\n{location}\n\n"
        f"### Role Category\n\n{category}\n\n"
        f"### Additional Info\n\n" + "\n".join(flags)
    )


def _make_issue(
    number: int = 1,
    title: str = "[New] Stripe -- SWE Intern",
    body: str = None,
    **body_kwargs,
) -> dict:
    """Build a mock GitHub issue dict."""
    if body is None:
        body = _make_issue_body(**body_kwargs)
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": "new-internship"}],
    }


# ======================================================================
# _parse_issue_body
# ======================================================================


class TestParseIssueBody:
    """Tests for _parse_issue_body."""

    def test_valid_body(self):
        """A complete issue body is parsed correctly."""
        body = _make_issue_body()
        result = _parse_issue_body(body)
        assert result is not None
        assert result["company"] == "Stripe"
        assert result["role"] == "Software Engineer Intern"
        assert result["url"] == "https://stripe.com/jobs/123"
        assert result["location"] == "San Francisco, CA / Remote"
        assert result["category"] == "Software Engineering"

    def test_missing_company(self):
        """Missing company returns None."""
        body = _make_issue_body(company="")
        result = _parse_issue_body(body)
        assert result is None

    def test_missing_role(self):
        """Missing role returns None."""
        body = _make_issue_body(role="")
        result = _parse_issue_body(body)
        assert result is None

    def test_missing_url(self):
        """Missing URL returns None."""
        body = _make_issue_body(url="")
        result = _parse_issue_body(body)
        assert result is None

    def test_missing_location(self):
        """Missing location returns None."""
        body = _make_issue_body(location="")
        result = _parse_issue_body(body)
        assert result is None

    def test_empty_body(self):
        """An empty string returns None."""
        result = _parse_issue_body("")
        assert result is None

    def test_none_body(self):
        """A None-ish empty body returns None."""
        result = _parse_issue_body("   ")
        assert result is None

    def test_malformed_body_no_headers(self):
        """A body without ### headers returns None."""
        result = _parse_issue_body("This is just random text without any structure.")
        assert result is None

    def test_checkbox_sponsors_checked(self):
        """Sponsorship checkbox is parsed when checked."""
        body = _make_issue_body(sponsors=True)
        result = _parse_issue_body(body)
        assert result["flags"]["sponsors"] is True

    def test_checkbox_us_citizenship_checked(self):
        """US citizenship checkbox is parsed when checked."""
        body = _make_issue_body(us_citizenship=True)
        result = _parse_issue_body(body)
        assert result["flags"]["us_citizenship"] is True

    def test_checkbox_remote_checked(self):
        """Remote checkbox is parsed when checked."""
        body = _make_issue_body(remote=True)
        result = _parse_issue_body(body)
        assert result["flags"]["remote_friendly"] is True

    def test_checkbox_advanced_degree_checked(self):
        """Advanced degree checkbox is parsed when checked."""
        body = _make_issue_body(advanced_degree=True)
        result = _parse_issue_body(body)
        assert result["flags"]["advanced_degree"] is True

    def test_checkbox_open_to_international_checked(self):
        """Open to international checkbox is parsed when checked."""
        body = _make_issue_body(open_to_international=True)
        result = _parse_issue_body(body)
        assert result["flags"]["open_to_international"] is True

    def test_all_checkboxes_unchecked(self):
        """All unchecked checkboxes return False."""
        body = _make_issue_body(
            sponsors=False,
            us_citizenship=False,
            remote=False,
            advanced_degree=False,
            open_to_international=False,
        )
        result = _parse_issue_body(body)
        assert result["flags"]["sponsors"] is False
        assert result["flags"]["us_citizenship"] is False
        assert result["flags"]["remote_friendly"] is False
        assert result["flags"]["advanced_degree"] is False
        assert result["flags"]["open_to_international"] is False

    def test_lowercase_x_in_checkbox(self):
        """Lowercase [x] is also recognized as checked."""
        body = "### Company Name\n\nTest\n\n### Role Title\n\nIntern\n\n### Application URL\n\nhttps://test.com\n\n### Location(s)\n\nNYC\n\n### Role Category\n\nOther\n\n### Additional Info\n\n- [x] Offers visa sponsorship\n- [ ] Requires U.S. citizenship\n- [ ] Remote friendly\n- [ ] Requires advanced degree (Master's/PhD)"
        result = _parse_issue_body(body)
        assert result["flags"]["sponsors"] is True


# ======================================================================
# _validate_url
# ======================================================================


class TestValidateUrl:
    """Tests for _validate_url."""

    def test_https_valid(self):
        assert _validate_url("https://example.com/jobs") is True

    def test_http_valid(self):
        assert _validate_url("http://example.com/jobs") is True

    def test_no_scheme_invalid(self):
        assert _validate_url("example.com/jobs") is False

    def test_ftp_invalid(self):
        assert _validate_url("ftp://example.com") is False

    def test_empty_invalid(self):
        assert _validate_url("") is False

    def test_relative_path_invalid(self):
        assert _validate_url("/jobs/123") is False


# ======================================================================
# _map_category
# ======================================================================


class TestMapCategory:
    """Tests for _map_category."""

    def test_software_engineering(self):
        assert _map_category("Software Engineering") == RoleCategory.SWE

    def test_ml_ai(self):
        assert _map_category("ML / AI / Data Science") == RoleCategory.ML_AI

    def test_quant(self):
        assert _map_category("Quantitative Finance") == RoleCategory.QUANT

    def test_pm(self):
        assert _map_category("Product Management") == RoleCategory.PM

    def test_hardware(self):
        assert _map_category("Hardware Engineering") == RoleCategory.HARDWARE

    def test_other(self):
        assert _map_category("Other") == RoleCategory.OTHER

    def test_unknown_defaults_to_other(self):
        assert _map_category("Basket Weaving") == RoleCategory.OTHER

    def test_case_insensitive(self):
        assert _map_category("SOFTWARE ENGINEERING") == RoleCategory.SWE

    def test_with_whitespace(self):
        assert _map_category("  Software Engineering  ") == RoleCategory.SWE


# ======================================================================
# _parse_locations
# ======================================================================


class TestParseLocations:
    """Tests for _parse_locations."""

    def test_slash_delimiter(self):
        result = _parse_locations("SF / NYC / Remote")
        assert result == ["SF", "NYC", "Remote"]

    def test_pipe_delimiter(self):
        result = _parse_locations("Atlanta | Remote")
        assert result == ["Atlanta", "Remote"]

    def test_semicolon_delimiter(self):
        result = _parse_locations("NYC ; Boston")
        assert result == ["NYC", "Boston"]

    def test_city_state_preserved(self):
        result = _parse_locations("San Francisco, CA")
        assert result == ["San Francisco, CA"]

    def test_multiple_comma_locations(self):
        result = _parse_locations("NYC, SF, Remote")
        assert result == ["NYC", "SF", "Remote"]

    def test_single_location(self):
        result = _parse_locations("Atlanta, GA")
        assert result == ["Atlanta, GA"]

    def test_empty_returns_unknown(self):
        result = _parse_locations("")
        assert result == ["Unknown"]


# ======================================================================
# _build_job_listing
# ======================================================================


class TestBuildJobListing:
    """Tests for _build_job_listing."""

    def test_builds_correct_listing(self):
        parsed = {
            "company": "Stripe",
            "role": "SWE Intern",
            "url": "https://stripe.com/jobs/1",
            "location": "SF / Remote",
            "category": "Software Engineering",
            "flags": {
                "sponsors": True,
                "us_citizenship": False,
                "remote_friendly": True,
                "advanced_degree": False,
            },
        }
        job = _build_job_listing(parsed)
        assert job.company == "Stripe"
        assert job.company_slug == "stripe"
        assert job.role == "SWE Intern"
        assert job.category == RoleCategory.SWE
        assert job.locations == ["SF", "Remote"]
        assert str(job.apply_url) == "https://stripe.com/jobs/1"
        assert job.sponsorship == SponsorshipStatus.SPONSORS
        assert job.remote_friendly is True
        assert job.requires_advanced_degree is False
        assert job.source == "community"
        assert job.status == ListingStatus.OPEN
        assert job.date_added == date.today()

    def test_us_citizenship_flag(self):
        parsed = {
            "company": "DefenseCo",
            "role": "Intern",
            "url": "https://defense.com/1",
            "location": "DC",
            "category": "Other",
            "flags": {
                "sponsors": False,
                "us_citizenship": True,
                "remote_friendly": False,
                "advanced_degree": False,
            },
        }
        job = _build_job_listing(parsed)
        assert job.sponsorship == SponsorshipStatus.US_CITIZENSHIP
        assert job.requires_us_citizenship is True

    def test_advanced_degree_flag(self):
        parsed = {
            "company": "ResearchLab",
            "role": "ML Intern",
            "url": "https://research.com/1",
            "location": "Boston, MA",
            "category": "ML / AI / Data Science",
            "flags": {
                "sponsors": False,
                "us_citizenship": False,
                "remote_friendly": False,
                "advanced_degree": True,
            },
        }
        job = _build_job_listing(parsed)
        assert job.requires_advanced_degree is True
        assert job.category == RoleCategory.ML_AI

    def test_open_to_international_flag(self):
        parsed = {
            "company": "GlobalCo",
            "role": "SWE Intern",
            "url": "https://globalco.com/1",
            "location": "Atlanta, GA",
            "category": "Software Engineering",
            "flags": {
                "sponsors": False,
                "us_citizenship": False,
                "remote_friendly": False,
                "advanced_degree": False,
                "open_to_international": True,
            },
        }
        job = _build_job_listing(parsed)
        assert job.open_to_international is True


# ======================================================================
# _get_missing_fields
# ======================================================================


class TestGetMissingFields:
    """Tests for _get_missing_fields."""

    def test_none_input(self):
        result = _get_missing_fields(None)
        assert set(result) == {"company", "role", "url", "location"}

    def test_all_present(self):
        parsed = {
            "company": "Test",
            "role": "Intern",
            "url": "https://test.com",
            "location": "NYC",
        }
        result = _get_missing_fields(parsed)
        assert result == []

    def test_missing_company(self):
        parsed = {"company": "", "role": "Intern", "url": "https://test.com", "location": "NYC"}
        result = _get_missing_fields(parsed)
        assert "company" in result

    def test_missing_multiple(self):
        parsed = {"company": "", "role": "", "url": "https://test.com", "location": "NYC"}
        result = _get_missing_fields(parsed)
        assert "company" in result
        assert "role" in result


# ======================================================================
# process_issues (async integration)
# ======================================================================


class TestProcessIssues:
    """Tests for the process_issues async function."""

    @pytest.mark.asyncio
    async def test_no_issues_returns_zero(self, tmp_path):
        """When there are no open issues, returns 0."""
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()
        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[]),
        ):
            result = await process_issues()
        assert result == 0

    @pytest.mark.asyncio
    async def test_no_token_returns_zero(self):
        """When GITHUB_TOKEN is not set, returns 0."""
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()
        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value=None),
        ):
            result = await process_issues()
        assert result == 0

    @pytest.mark.asyncio
    async def test_valid_submission_accepted(self, tmp_path):
        """A valid submission is added to jobs.json and issue is closed."""
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        }))

        issue = _make_issue(number=10)
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 1
        mock_comment.assert_called_once()
        comment_body = mock_comment.call_args[0][2]
        assert "Added" in comment_body
        mock_close.assert_called_once()

        saved = json.loads(jobs_path.read_text())
        assert len(saved["listings"]) == 1
        assert saved["listings"][0]["company"] == "Stripe"

    @pytest.mark.asyncio
    async def test_missing_fields_rejected(self, tmp_path):
        """An issue with missing required fields is rejected."""
        body = "### Company Name\n\n\n\n### Role Title\n\nIntern\n\n### Application URL\n\nhttps://test.com\n\n### Location(s)\n\nNYC"
        issue = _make_issue(number=11, body=body)
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 0
        mock_comment.assert_called_once()
        comment_body = mock_comment.call_args[0][2]
        assert "required fields" in comment_body.lower() or "template" in comment_body.lower()

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self, tmp_path):
        """An issue with an invalid URL is rejected."""
        issue = _make_issue(number=12, url="not-a-url")
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 0
        comment_body = mock_comment.call_args[0][2]
        assert "url" in comment_body.lower() or "URL" in comment_body

    @pytest.mark.asyncio
    async def test_malformed_body_rejected(self, tmp_path):
        """An issue with a totally malformed body is rejected."""
        issue = _make_issue(number=13, body="random text no structure at all")
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 0

    @pytest.mark.asyncio
    async def test_multiple_issues_processed(self, tmp_path):
        """Multiple valid issues are all processed correctly."""
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        }))

        issues = [
            _make_issue(number=20, company="Stripe", role="SWE Intern",
                       url="https://stripe.com/1", location="SF"),
            _make_issue(number=21, company="Notion", role="Backend Intern",
                       url="https://notion.so/1", location="NYC"),
        ]
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=issues),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 2
        assert mock_comment.call_count == 2
        assert mock_close.call_count == 2

        saved = json.loads(jobs_path.read_text())
        assert len(saved["listings"]) == 2

    @pytest.mark.asyncio
    async def test_error_isolated_per_issue(self, tmp_path):
        """An error on one issue does not prevent processing of others."""
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        }))

        # First issue will cause an error (missing 'number' key)
        bad_issue = {"title": "Bad", "body": _make_issue_body()}
        good_issue = _make_issue(number=30)
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[bad_issue, good_issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        # The good issue should still be processed
        assert result >= 1

    @pytest.mark.asyncio
    async def test_duplicate_listing_rejected(self, tmp_path):
        """A submission that duplicates an existing listing is rejected."""
        # Pre-populate jobs.json with a listing
        body = _make_issue_body(
            company="Stripe",
            role="SWE Intern",
            location="SF / Remote",
        )
        parsed = _parse_issue_body(body)
        from scripts.process_issues import _build_job_listing
        job = _build_job_listing(parsed)

        jobs_path = tmp_path / "jobs.json"
        db = JobsDatabase(
            listings=[job],
            last_updated=datetime.now(timezone.utc),
        )
        db.compute_stats()
        jobs_path.write_text(json.dumps(db.model_dump(mode="json"), default=str))

        # Submit the exact same listing via an issue
        issue = _make_issue(
            number=40,
            company="Stripe",
            role="SWE Intern",
            url="https://stripe.com/jobs/123",
            location="SF / Remote",
        )
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 0
        comment_body = mock_comment.call_args[0][2]
        assert "already exist" in comment_body.lower()

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid(self, tmp_path):
        """Mix of valid and invalid issues; only valid ones are accepted."""
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        }))

        valid_issue = _make_issue(number=50, company="ValidCo", role="Intern",
                                  url="https://valid.com/1", location="NYC")
        invalid_issue = _make_issue(number=51, url="not-a-url")

        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[valid_issue, invalid_issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 1

    @pytest.mark.asyncio
    async def test_github_api_error_during_comment(self, tmp_path):
        """If commenting fails, the issue processing still continues."""
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        }))

        issue = _make_issue(number=60)
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=False)  # comment fails
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        # Listing should still be added even if comment fails
        assert result == 1

    @pytest.mark.asyncio
    async def test_category_mapping_all_values(self):
        """All CATEGORY_MAP values map correctly."""
        assert _map_category("Software Engineering") == RoleCategory.SWE
        assert _map_category("ML / AI / Data Science") == RoleCategory.ML_AI
        assert _map_category("Quantitative Finance") == RoleCategory.QUANT
        assert _map_category("Product Management") == RoleCategory.PM
        assert _map_category("Hardware Engineering") == RoleCategory.HARDWARE
        assert _map_category("Other") == RoleCategory.OTHER

    @pytest.mark.asyncio
    async def test_database_not_saved_when_no_accepted(self, tmp_path):
        """When no issues are accepted, the database is not saved."""
        jobs_path = tmp_path / "jobs.json"
        original_content = json.dumps({
            "listings": [],
            "last_updated": "2026-01-01T00:00:00",
            "total_open": 0,
        })
        jobs_path.write_text(original_content)

        issue = _make_issue(number=70, url="not-a-url")
        config_mock = type("Config", (), {
            "project": type("Project", (), {"github_repo": "owner/repo", "season": "summer_2026"})()
        })()

        mock_comment = AsyncMock(return_value=True)
        mock_close = AsyncMock(return_value=True)

        with (
            patch("scripts.process_issues.get_config", return_value=config_mock),
            patch("scripts.process_issues.get_secret", return_value="ghp_test"),
            patch("scripts.process_issues.fetch_issues", new_callable=AsyncMock, return_value=[issue]),
            patch("scripts.process_issues.comment_on_issue", mock_comment),
            patch("scripts.process_issues.close_issue", mock_close),
            patch("scripts.process_issues.JOBS_PATH", jobs_path),
            patch("scripts.process_issues.DATA_DIR", tmp_path),
        ):
            result = await process_issues()

        assert result == 0
        # jobs.json should not have been re-written
        assert jobs_path.read_text() == original_content
