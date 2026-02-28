"""Tests for the async link health checker (scripts/check_links.py)."""

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scripts.check_links import (
    DATA_DIR,
    DEAD_STATUSES,
    LINK_HEALTH_PATH,
    MAX_CONCURRENT,
    TRANSIENT_STATUSES,
    _check_single_link,
    _load_database,
    _load_link_health,
    _save_database,
    _save_link_health,
    check_all_links,
)
from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_listing(
    listing_id: str = "abc123",
    company: str = "TestCo",
    role: str = "SWE Intern",
    status: ListingStatus = ListingStatus.OPEN,
    url: str = "https://example.com/apply",
    date_added: date | None = None,
    date_last_verified: date | None = None,
) -> JobListing:
    """Create a JobListing for testing."""
    return JobListing(
        id=listing_id,
        company=company,
        company_slug=company.lower().replace(" ", "-"),
        role=role,
        category=RoleCategory.SWE,
        locations=["Remote"],
        apply_url=url,
        sponsorship=SponsorshipStatus.UNKNOWN,
        requires_us_citizenship=False,
        is_faang_plus=False,
        requires_advanced_degree=False,
        remote_friendly=True,
        date_added=date_added or date(2026, 1, 15),
        date_last_verified=date_last_verified or date(2026, 2, 20),
        source="greenhouse_api",
        status=status,
        tech_stack=[],
        season="summer_2026",
    )


def _make_db(listings: list[JobListing] | None = None) -> JobsDatabase:
    """Create a JobsDatabase for testing."""
    db = JobsDatabase(
        listings=listings or [],
        last_updated=datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc),
    )
    db.compute_stats()
    return db


def _db_to_dict(db: JobsDatabase) -> dict:
    """Serialize a JobsDatabase to dict (as written to JSON)."""
    return json.loads(db.model_dump_json())


# ---------------------------------------------------------------------------
# _load_database
# ---------------------------------------------------------------------------

class TestLoadDatabase:
    def test_loads_valid_database(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = tmp_path / "jobs.json"
        jobs_path.write_text(json.dumps(_db_to_dict(db), default=str))

        with patch("scripts.check_links.JOBS_PATH", jobs_path):
            result = _load_database()
        assert len(result.listings) == 1
        assert result.listings[0].id == "abc123"

    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        missing_path = tmp_path / "nonexistent.json"
        with patch("scripts.check_links.JOBS_PATH", missing_path):
            result = _load_database()
        assert result.listings == []

    def test_returns_empty_on_invalid_json(self, tmp_path: Path):
        bad_path = tmp_path / "jobs.json"
        bad_path.write_text("{invalid json!!!")
        with patch("scripts.check_links.JOBS_PATH", bad_path):
            result = _load_database()
        assert result.listings == []


# ---------------------------------------------------------------------------
# _load_link_health / _save_link_health
# ---------------------------------------------------------------------------

class TestLinkHealth:
    def test_load_returns_empty_when_missing(self, tmp_path: Path):
        missing = tmp_path / "link_health.json"
        with patch("scripts.check_links.LINK_HEALTH_PATH", missing):
            result = _load_link_health()
        assert result == {}

    def test_load_returns_data_from_file(self, tmp_path: Path):
        health_path = tmp_path / "link_health.json"
        data = {"id1": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path.write_text(json.dumps(data))
        with patch("scripts.check_links.LINK_HEALTH_PATH", health_path):
            result = _load_link_health()
        assert result == data

    def test_load_handles_invalid_json(self, tmp_path: Path):
        bad_path = tmp_path / "link_health.json"
        bad_path.write_text("not json")
        with patch("scripts.check_links.LINK_HEALTH_PATH", bad_path):
            result = _load_link_health()
        assert result == {}

    def test_load_returns_empty_for_non_dict(self, tmp_path: Path):
        health_path = tmp_path / "link_health.json"
        health_path.write_text(json.dumps([1, 2, 3]))
        with patch("scripts.check_links.LINK_HEALTH_PATH", health_path):
            result = _load_link_health()
        assert result == {}

    def test_save_creates_file(self, tmp_path: Path):
        health_path = tmp_path / "data" / "link_health.json"
        data = {"id1": {"consecutive_failures": 2, "last_checked": "2026-02-28"}}
        with patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path / "data"):
            _save_link_health(data)
        assert health_path.exists()
        loaded = json.loads(health_path.read_text())
        assert loaded == data

    def test_save_overwrites_existing(self, tmp_path: Path):
        health_path = tmp_path / "link_health.json"
        health_path.write_text(json.dumps({"old": "data"}))
        new_data = {"new": {"consecutive_failures": 0, "last_checked": "2026-02-28"}}
        with patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path):
            _save_link_health(new_data)
        loaded = json.loads(health_path.read_text())
        assert loaded == new_data


# ---------------------------------------------------------------------------
# _check_single_link
# ---------------------------------------------------------------------------

class TestCheckSingleLink:
    @pytest.mark.asyncio
    async def test_healthy_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        lid, result_type, status, err = await _check_single_link(
            client, sem, "id1", "https://example.com", "TestCo", "SWE Intern"
        )
        assert lid == "id1"
        assert result_type == "healthy"
        assert status == 200
        assert err is None

    @pytest.mark.asyncio
    async def test_dead_404(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        lid, result_type, status, err = await _check_single_link(
            client, sem, "id1", "https://example.com", "TestCo", "SWE Intern"
        )
        assert result_type == "dead"
        assert status == 404

    @pytest.mark.asyncio
    async def test_dead_410(self):
        mock_response = MagicMock()
        mock_response.status_code = 410
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "dead"
        assert status == 410

    @pytest.mark.asyncio
    async def test_dead_403(self):
        mock_response = MagicMock()
        mock_response.status_code = 403
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "dead"
        assert status == 403

    @pytest.mark.asyncio
    async def test_transient_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "transient"
        assert status == 429

    @pytest.mark.asyncio
    async def test_transient_500(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "transient"
        assert status == 500

    @pytest.mark.asyncio
    async def test_transient_502(self):
        mock_response = MagicMock()
        mock_response.status_code = 502
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "transient"

    @pytest.mark.asyncio
    async def test_transient_503(self):
        mock_response = MagicMock()
        mock_response.status_code = 503
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "transient"

    @pytest.mark.asyncio
    async def test_unknown_status_301(self):
        mock_response = MagicMock()
        mock_response.status_code = 301
        client = AsyncMock()
        client.head = AsyncMock(return_value=mock_response)
        sem = asyncio.Semaphore(10)

        _, result_type, status, _ = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "unknown"
        assert status == 301

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        client = AsyncMock()
        client.head = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        sem = asyncio.Semaphore(10)

        lid, result_type, status, err = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "error"
        assert status is None
        assert err == "timeout"

    @pytest.mark.asyncio
    async def test_connection_error_returns_error(self):
        client = AsyncMock()
        client.head = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        sem = asyncio.Semaphore(10)

        _, result_type, status, err = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "error"
        assert status is None
        assert "connection refused" in err

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_error(self):
        client = AsyncMock()
        client.head = AsyncMock(side_effect=RuntimeError("unexpected"))
        sem = asyncio.Semaphore(10)

        _, result_type, status, err = await _check_single_link(
            client, sem, "id1", "https://example.com", "Co", "Role"
        )
        assert result_type == "error"
        assert status is None
        assert "unexpected" in err


# ---------------------------------------------------------------------------
# check_all_links — integration tests
# ---------------------------------------------------------------------------

def _write_db_file(tmp_path: Path, db: JobsDatabase) -> Path:
    """Write a database to a temp jobs.json and return the path."""
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(_db_to_dict(db), default=str))
    return jobs_path


def _write_health_file(tmp_path: Path, health: dict) -> Path:
    """Write a health dict to a temp link_health.json and return the path."""
    health_path = tmp_path / "link_health.json"
    health_path.write_text(json.dumps(health))
    return health_path


class TestCheckAllLinks:
    """Integration tests for the full check_all_links() function."""

    @pytest.mark.asyncio
    async def test_empty_database_returns_zero_stats(self, tmp_path: Path):
        db = _make_db([])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path):
            stats = await check_all_links()

        assert stats["checked"] == 0
        assert stats["healthy"] == 0
        assert stats["closed"] == 0
        assert stats["transient_errors"] == 0
        assert stats["unknown"] == 0

    @pytest.mark.asyncio
    async def test_healthy_link_updates_verified_date(self, tmp_path: Path):
        listing = _make_listing(
            date_last_verified=date(2026, 1, 1),
        )
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(200, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["healthy"] == 1
        assert stats["checked"] == 1

        # Verify the saved database has updated date_last_verified
        saved = json.loads(jobs_path.read_text())
        assert saved["listings"][0]["date_last_verified"] == date.today().isoformat()

    @pytest.mark.asyncio
    async def test_dead_link_first_failure_stays_open(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(404, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        # Should NOT be closed yet (first failure)
        assert stats["closed"] == 0
        saved = json.loads(jobs_path.read_text())
        assert saved["listings"][0]["status"] == "open"

        # But health should track the failure
        saved_health = json.loads(health_path.read_text())
        assert saved_health["abc123"]["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_dead_link_second_failure_marks_closed(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        # Pre-existing health: 1 prior failure
        health_data = {"abc123": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        mock_response = httpx.Response(404, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["closed"] == 1
        saved = json.loads(jobs_path.read_text())
        assert saved["listings"][0]["status"] == "closed"

        saved_health = json.loads(health_path.read_text())
        assert saved_health["abc123"]["consecutive_failures"] == 2

    @pytest.mark.asyncio
    async def test_410_second_failure_marks_closed(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_data = {"abc123": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        mock_response = httpx.Response(410, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["closed"] == 1

    @pytest.mark.asyncio
    async def test_403_second_failure_marks_closed(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_data = {"abc123": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        mock_response = httpx.Response(403, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["closed"] == 1

    @pytest.mark.asyncio
    async def test_healthy_resets_failure_counter(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        # 1 prior failure
        health_data = {"abc123": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        mock_response = httpx.Response(200, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["healthy"] == 1
        saved_health = json.loads(health_path.read_text())
        assert saved_health["abc123"]["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_transient_errors_not_marked_closed(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_data = {"abc123": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        mock_response = httpx.Response(503, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["transient_errors"] == 1
        assert stats["closed"] == 0
        saved = json.loads(jobs_path.read_text())
        assert saved["listings"][0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_unknown_status_not_marked_closed(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(301, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["unknown"] == 1
        assert stats["closed"] == 0

    @pytest.mark.asyncio
    async def test_skips_closed_listings(self, tmp_path: Path):
        open_listing = _make_listing(listing_id="open1")
        closed_listing = _make_listing(
            listing_id="closed1",
            status=ListingStatus.CLOSED,
            url="https://example.com/closed",
        )
        db = _make_db([open_listing, closed_listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(200, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        # Only the open listing should be checked
        assert stats["checked"] == 1
        assert mock_client.head.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_counted_as_transient(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(
                side_effect=httpx.TimeoutException("timed out")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["transient_errors"] == 1
        assert stats["closed"] == 0

    @pytest.mark.asyncio
    async def test_network_error_counted_as_transient(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(
                side_effect=httpx.ConnectError("connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["transient_errors"] == 1
        assert stats["closed"] == 0

    @pytest.mark.asyncio
    async def test_link_health_created_when_missing(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"
        # Do NOT create health file beforehand

        mock_response = httpx.Response(404, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await check_all_links()

        assert health_path.exists()
        saved_health = json.loads(health_path.read_text())
        assert "abc123" in saved_health

    @pytest.mark.asyncio
    async def test_multiple_listings_mixed_results(self, tmp_path: Path):
        healthy_listing = _make_listing(
            listing_id="healthy1", url="https://example.com/healthy"
        )
        dead_listing = _make_listing(
            listing_id="dead1", url="https://example.com/dead"
        )
        transient_listing = _make_listing(
            listing_id="trans1", url="https://example.com/transient"
        )
        db = _make_db([healthy_listing, dead_listing, transient_listing])
        jobs_path = _write_db_file(tmp_path, db)
        # dead1 already has 1 failure
        health_data = {"dead1": {"consecutive_failures": 1, "last_checked": "2026-02-27"}}
        health_path = _write_health_file(tmp_path, health_data)

        def mock_head(url, **kwargs):
            url_str = str(url)
            if "healthy" in url_str:
                return httpx.Response(200, request=httpx.Request("HEAD", url_str))
            elif "dead" in url_str:
                return httpx.Response(404, request=httpx.Request("HEAD", url_str))
            else:
                return httpx.Response(503, request=httpx.Request("HEAD", url_str))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=mock_head)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["checked"] == 3
        assert stats["healthy"] == 1
        assert stats["closed"] == 1
        assert stats["transient_errors"] == 1

    @pytest.mark.asyncio
    async def test_stats_dict_has_all_keys(self, tmp_path: Path):
        db = _make_db([])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path):
            stats = await check_all_links()

        expected_keys = {"checked", "healthy", "closed", "transient_errors", "unknown"}
        assert set(stats.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_saves_updated_jobs_json(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(200, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await check_all_links()

        # Verify jobs.json was rewritten
        saved = json.loads(jobs_path.read_text())
        assert "listings" in saved
        assert "last_updated" in saved

    @pytest.mark.asyncio
    async def test_saves_link_health_json(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        mock_response = httpx.Response(200, request=httpx.Request("HEAD", "https://example.com/apply"))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await check_all_links()

        assert health_path.exists()


# ---------------------------------------------------------------------------
# Concurrency and constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_concurrent_is_10(self):
        assert MAX_CONCURRENT == 10

    def test_dead_statuses(self):
        assert DEAD_STATUSES == {404, 410, 403}

    def test_transient_statuses(self):
        assert TRANSIENT_STATUSES == {429, 500, 502, 503}


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, tmp_path: Path):
        """Verify that at most MAX_CONCURRENT requests run simultaneously."""
        listings = [
            _make_listing(
                listing_id=f"id{i}",
                url=f"https://example.com/job{i}",
            )
            for i in range(20)
        ]
        db = _make_db(listings)
        jobs_path = _write_db_file(tmp_path, db)
        health_path = tmp_path / "link_health.json"

        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_head(url, **kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            await asyncio.sleep(0.01)  # Small delay to actually test concurrency
            async with lock:
                current_concurrent -= 1
            return httpx.Response(200, request=httpx.Request("HEAD", str(url)))

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.LINK_HEALTH_PATH", health_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.head = mock_head
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stats = await check_all_links()

        assert stats["checked"] == 20
        assert max_concurrent_seen <= MAX_CONCURRENT


# ---------------------------------------------------------------------------
# _save_database
# ---------------------------------------------------------------------------

class TestSaveDatabase:
    def test_saves_and_updates_stats(self, tmp_path: Path):
        listing = _make_listing()
        db = _make_db([listing])
        jobs_path = tmp_path / "jobs.json"

        with patch("scripts.check_links.JOBS_PATH", jobs_path), \
             patch("scripts.check_links.DATA_DIR", tmp_path):
            _save_database(db)

        assert jobs_path.exists()
        saved = json.loads(jobs_path.read_text())
        assert saved["total_open"] == 1
        assert len(saved["listings"]) == 1
