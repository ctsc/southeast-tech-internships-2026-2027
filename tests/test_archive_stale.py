"""Tests for scripts/archive_stale.py — stale listing archival."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from scripts.archive_stale import (
    CLOSED_ARCHIVE_DAYS,
    STALE_ARCHIVE_DAYS,
    _should_archive,
    archive_stale,
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

TODAY = date(2026, 2, 28)


def _make_listing(
    *,
    id: str = "test-id",
    company: str = "TestCo",
    role: str = "SWE Intern",
    status: ListingStatus = ListingStatus.OPEN,
    date_added: date = TODAY,
    date_last_verified: date = TODAY,
) -> JobListing:
    """Create a JobListing with sensible defaults for testing."""
    return JobListing(
        id=id,
        company=company,
        company_slug=company.lower().replace(" ", "-"),
        role=role,
        category=RoleCategory.SWE,
        locations=["Atlanta, GA"],
        apply_url="https://example.com/apply",
        sponsorship=SponsorshipStatus.UNKNOWN,
        date_added=date_added,
        date_last_verified=date_last_verified,
        source="test",
        status=status,
    )


def _make_database(listings: list[JobListing] | None = None) -> JobsDatabase:
    """Create a JobsDatabase from a list of listings."""
    db = JobsDatabase(
        listings=listings or [],
        last_updated=datetime(2026, 2, 28, 0, 0, 0, tzinfo=timezone.utc),
    )
    db.compute_stats()
    return db


def _write_db(path: Path, db: JobsDatabase) -> None:
    """Write a JobsDatabase to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db.model_dump(mode="json"), f, indent=2, default=str)


def _read_db(path: Path) -> JobsDatabase:
    """Read a JobsDatabase from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return JobsDatabase.model_validate(json.load(f))


# ---------------------------------------------------------------------------
# Tests for _should_archive
# ---------------------------------------------------------------------------


class TestShouldArchive:
    """Unit tests for the _should_archive decision function."""

    def test_open_listing_within_120_days(self):
        listing = _make_listing(date_added=TODAY)
        assert _should_archive(listing, TODAY) is None

    def test_open_listing_exactly_120_days(self):
        listing = _make_listing(date_added=date(2025, 10, 31))  # exactly 120 days
        assert _should_archive(listing, TODAY) is None

    def test_open_listing_older_than_120_days(self):
        listing = _make_listing(date_added=date(2025, 10, 30))  # 121 days
        reason = _should_archive(listing, TODAY)
        assert reason is not None
        assert "stale" in reason

    def test_closed_listing_within_7_days(self):
        listing = _make_listing(
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 22),  # 6 days ago
        )
        assert _should_archive(listing, TODAY) is None

    def test_closed_listing_exactly_7_days(self):
        listing = _make_listing(
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 21),  # exactly 7 days
        )
        assert _should_archive(listing, TODAY) is None

    def test_closed_listing_older_than_7_days(self):
        listing = _make_listing(
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 20),  # 8 days ago
        )
        reason = _should_archive(listing, TODAY)
        assert reason is not None
        assert "closed" in reason

    def test_unknown_status_not_archived_within_120_days(self):
        listing = _make_listing(status=ListingStatus.UNKNOWN, date_added=TODAY)
        assert _should_archive(listing, TODAY) is None

    def test_unknown_status_archived_after_120_days(self):
        listing = _make_listing(
            status=ListingStatus.UNKNOWN,
            date_added=date(2025, 10, 1),  # 150 days ago
        )
        reason = _should_archive(listing, TODAY)
        assert reason is not None
        assert "stale" in reason

    def test_closed_and_stale_returns_closed_reason(self):
        """Listing that meets both criteria should return closed reason (checked first)."""
        listing = _make_listing(
            status=ListingStatus.CLOSED,
            date_added=date(2025, 6, 1),  # >120 days
            date_last_verified=date(2025, 6, 1),  # >7 days
        )
        reason = _should_archive(listing, TODAY)
        assert reason is not None
        assert "closed" in reason


# ---------------------------------------------------------------------------
# Tests for archive_stale (integration with file I/O)
# ---------------------------------------------------------------------------


class TestArchiveStale:
    """Integration tests for the archive_stale function."""

    def test_empty_database_archives_nothing(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        _write_db(jobs_path, _make_database([]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0

    def test_open_listing_within_120_days_not_archived(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(id="keep", date_added=date(2026, 1, 1))
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        db = _read_db(jobs_path)
        assert len(db.listings) == 1
        assert db.listings[0].id == "keep"

    def test_open_listing_older_than_120_days_archived(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(id="old", date_added=date(2025, 9, 1))
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 0
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 1
        assert arch_db.listings[0].id == "old"

    def test_closed_listing_within_7_days_not_archived(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="recent-closed",
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 25),  # 3 days ago
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        db = _read_db(jobs_path)
        assert len(db.listings) == 1

    def test_closed_listing_older_than_7_days_archived(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="old-closed",
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 15),  # 13 days ago
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 0
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 1
        assert arch_db.listings[0].id == "old-closed"

    def test_mixed_some_archived_some_kept(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listings = [
            _make_listing(id="keep-1", date_added=date(2026, 1, 15)),
            _make_listing(
                id="archive-closed",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 2, 10),
            ),
            _make_listing(id="keep-2", date_added=date(2026, 2, 1)),
            _make_listing(id="archive-stale", date_added=date(2025, 8, 1)),
        ]
        _write_db(jobs_path, _make_database(listings))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 2
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 2
        remaining_ids = {l.id for l in jobs_db.listings}
        assert remaining_ids == {"keep-1", "keep-2"}

        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 2
        archived_ids = {l.id for l in arch_db.listings}
        assert archived_ids == {"archive-closed", "archive-stale"}

    def test_appended_to_existing_archive(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        # Pre-populate archive with one entry
        existing_archived = _make_listing(id="already-archived", date_added=date(2025, 6, 1))
        _write_db(archived_path, _make_database([existing_archived]))

        # Jobs has one listing to archive
        listing = _make_listing(
            id="to-archive",
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 10),
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 2
        archived_ids = {l.id for l in arch_db.listings}
        assert "already-archived" in archived_ids
        assert "to-archive" in archived_ids

    def test_archived_json_created_if_missing(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listing = _make_listing(id="stale", date_added=date(2025, 8, 1))
        _write_db(jobs_path, _make_database([listing]))

        assert not archived_path.exists()

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1
        assert archived_path.exists()
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 1

    def test_jobs_json_updated_after_archival(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _make_listing(id="stay", date_added=date(2026, 2, 1)),
            _make_listing(id="go", date_added=date(2025, 7, 1)),
        ]
        _write_db(jobs_path, _make_database(listings))

        archive_stale(jobs_path, archived_path, today=TODAY)

        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 1
        assert jobs_db.listings[0].id == "stay"

    def test_stats_recomputed_after_archival(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _make_listing(id="open-keep", date_added=date(2026, 1, 1)),
            _make_listing(
                id="closed-archive",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 2, 1),
            ),
        ]
        _write_db(jobs_path, _make_database(listings))

        archive_stale(jobs_path, archived_path, today=TODAY)

        jobs_db = _read_db(jobs_path)
        assert jobs_db.total_open == 1  # only the open one remains

        arch_db = _read_db(archived_path)
        assert arch_db.total_open == 0  # closed listing, 0 open

    def test_boundary_exactly_7_days_not_archived(self, tmp_path: Path):
        """Listing closed exactly 7 days ago should NOT be archived (> not >=)."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="boundary-7",
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 21),  # exactly 7 days before TODAY
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 1

    def test_boundary_exactly_120_days_not_archived(self, tmp_path: Path):
        """Listing added exactly 120 days ago should NOT be archived (> not >=)."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="boundary-120",
            date_added=date(2025, 10, 31),  # exactly 120 days before TODAY
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 1

    def test_all_listings_archived_leaves_empty_jobs(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _make_listing(id="stale-1", date_added=date(2025, 7, 1)),
            _make_listing(id="stale-2", date_added=date(2025, 8, 1)),
            _make_listing(
                id="closed-old",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 1, 1),
            ),
        ]
        _write_db(jobs_path, _make_database(listings))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 3
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 0
        assert jobs_db.total_open == 0

    def test_closed_and_stale_archived_once_not_duplicated(self, tmp_path: Path):
        """A listing that is both closed>7d and stale>120d should only appear once in archive."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listing = _make_listing(
            id="both-criteria",
            status=ListingStatus.CLOSED,
            date_added=date(2025, 6, 1),
            date_last_verified=date(2025, 6, 1),
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1  # archived exactly once
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 1

    def test_archive_preserves_existing_entries(self, tmp_path: Path):
        """Existing archived entries must not be modified or removed."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        existing = _make_listing(id="old-archive-1", company="OldCo", date_added=date(2025, 1, 1))
        _write_db(archived_path, _make_database([existing]))

        # Nothing to archive from jobs
        fresh = _make_listing(id="fresh", date_added=date(2026, 2, 20))
        _write_db(jobs_path, _make_database([fresh]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == 1
        assert arch_db.listings[0].id == "old-archive-1"

    def test_return_value_matches_archived_count(self, tmp_path: Path):
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _make_listing(id="a1", date_added=date(2025, 7, 1)),
            _make_listing(id="a2", date_added=date(2025, 8, 1)),
            _make_listing(id="keep", date_added=date(2026, 2, 1)),
        ]
        _write_db(jobs_path, _make_database(listings))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 2
        arch_db = _read_db(archived_path)
        assert len(arch_db.listings) == count

    def test_closed_listing_1_day_past_threshold_archived(self, tmp_path: Path):
        """Closed listing 8 days ago (1 past threshold) should be archived."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="just-past",
            status=ListingStatus.CLOSED,
            date_last_verified=date(2026, 2, 20),  # 8 days ago
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1

    def test_stale_listing_1_day_past_threshold_archived(self, tmp_path: Path):
        """Listing added 121 days ago (1 past threshold) should be archived."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listing = _make_listing(
            id="just-past-stale",
            date_added=date(2025, 10, 30),  # 121 days ago
        )
        _write_db(jobs_path, _make_database([listing]))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 1

    def test_no_changes_when_nothing_to_archive(self, tmp_path: Path):
        """When nothing qualifies for archival, jobs.json should remain unchanged."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listing = _make_listing(id="fresh", date_added=date(2026, 2, 20))
        db = _make_database([listing])
        _write_db(jobs_path, db)

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 0
        # archived.json should not be created when nothing to archive
        # (function returns early)
        assert not archived_path.exists()

    def test_multiple_closed_mixed_dates(self, tmp_path: Path):
        """Multiple closed listings: some within 7d, some beyond."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _make_listing(
                id="closed-recent",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 2, 25),  # 3 days ago
            ),
            _make_listing(
                id="closed-old-1",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 2, 10),  # 18 days ago
            ),
            _make_listing(
                id="closed-old-2",
                status=ListingStatus.CLOSED,
                date_last_verified=date(2026, 1, 1),  # 58 days ago
            ),
        ]
        _write_db(jobs_path, _make_database(listings))

        count = archive_stale(jobs_path, archived_path, today=TODAY)

        assert count == 2
        jobs_db = _read_db(jobs_path)
        assert len(jobs_db.listings) == 1
        assert jobs_db.listings[0].id == "closed-recent"

    def test_listing_data_integrity_after_archival(self, tmp_path: Path):
        """Archived listing should preserve all original fields."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listing = _make_listing(
            id="integrity-check",
            company="IntegrityCorp",
            role="ML Intern",
            status=ListingStatus.CLOSED,
            date_added=date(2026, 1, 1),
            date_last_verified=date(2026, 2, 10),
        )
        _write_db(jobs_path, _make_database([listing]))

        archive_stale(jobs_path, archived_path, today=TODAY)

        arch_db = _read_db(archived_path)
        archived = arch_db.listings[0]
        assert archived.id == "integrity-check"
        assert archived.company == "IntegrityCorp"
        assert archived.role == "ML Intern"
        assert archived.status == ListingStatus.CLOSED
        assert archived.date_added == date(2026, 1, 1)
        assert archived.date_last_verified == date(2026, 2, 10)
