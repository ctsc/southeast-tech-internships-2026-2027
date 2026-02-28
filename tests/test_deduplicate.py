"""Comprehensive tests for the deduplication engine.

Tests cover:
- _load_database: valid file, missing file, empty file/listings
- _load_archived_hashes: valid file, missing file, empty file/listings
- _save_database: persists JSON, updates timestamp and stats
- _dedup_by_content_hash: no dupes, exact dupes, multiple dupes of same ID
- _dedup_by_url: no dupes, same URL keeps newer, different URLs kept
- _compute_token_overlap: identical, no overlap, partial, empty, case insensitive
- _dedup_fuzzy: similar co+role deduped, different co not deduped, archived reposts,
                single listing, empty list
- deduplicate_all: empty database, runs all 3 stages, saves result
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.deduplicate import (
    _compute_token_overlap,
    _dedup_by_content_hash,
    _dedup_by_url,
    _dedup_fuzzy,
    _load_archived_hashes,
    _load_database,
    _save_database,
    deduplicate_all,
)
from scripts.utils.models import JobListing, JobsDatabase, ListingStatus, RoleCategory


# ======================================================================
# Helpers
# ======================================================================


def _make_listing(
    *,
    id: str = "hash1",
    company: str = "TestCo",
    company_slug: str = "testco",
    role: str = "Software Engineer Intern",
    category: RoleCategory = RoleCategory.SWE,
    locations: list[str] | None = None,
    apply_url: str = "https://example.com/jobs/1",
    date_added: date = date(2026, 1, 15),
    date_last_verified: date = date(2026, 2, 20),
    source: str = "greenhouse_api",
    status: ListingStatus = ListingStatus.OPEN,
) -> JobListing:
    """Build a JobListing with sensible defaults, overriding any field."""
    return JobListing(
        id=id,
        company=company,
        company_slug=company_slug,
        role=role,
        category=category,
        locations=locations or ["Atlanta, GA"],
        apply_url=apply_url,
        date_added=date_added,
        date_last_verified=date_last_verified,
        source=source,
        status=status,
    )


def _write_jobs_json(
    path: Path,
    listings: list[dict] | None = None,
    raw: dict | None = None,
) -> None:
    """Write a jobs.json file to the given path."""
    if raw is not None:
        data = raw
    else:
        data = {
            "listings": listings or [],
            "last_updated": "2026-02-20T12:00:00",
            "total_open": 0,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_archived_json(path: Path, listings: list[dict] | None = None) -> None:
    """Write an archived.json file to the given path."""
    data = {
        "listings": listings or [],
        "last_updated": "2026-02-20T12:00:00",
        "total_open": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _listing_dict(
    *,
    id: str = "hash1",
    company: str = "TestCo",
    company_slug: str = "testco",
    role: str = "Software Engineer Intern",
    category: str = "swe",
    locations: list[str] | None = None,
    apply_url: str = "https://example.com/jobs/1",
    date_added: str = "2026-01-15",
    date_last_verified: str = "2026-02-20",
    source: str = "greenhouse_api",
    status: str = "open",
) -> dict:
    """Return a raw dict suitable for writing into jobs.json / archived.json."""
    return {
        "id": id,
        "company": company,
        "company_slug": company_slug,
        "role": role,
        "category": category,
        "locations": locations or ["Atlanta, GA"],
        "apply_url": apply_url,
        "date_added": date_added,
        "date_last_verified": date_last_verified,
        "source": source,
        "status": status,
        "sponsorship": "unknown",
        "requires_us_citizenship": False,
        "is_faang_plus": False,
        "requires_advanced_degree": False,
        "remote_friendly": False,
        "tech_stack": [],
        "season": "summer_2026",
    }


# ======================================================================
# _load_database
# ======================================================================


class TestLoadDatabase:
    """Tests for _load_database."""

    def test_loads_valid_database(self, tmp_path):
        """A well-formed jobs.json is loaded into a JobsDatabase."""
        jobs_path = tmp_path / "jobs.json"
        listing = _listing_dict(id="aaa")
        _write_jobs_json(jobs_path, listings=[listing])

        with patch("scripts.deduplicate.JOBS_PATH", jobs_path):
            db = _load_database()

        assert isinstance(db, JobsDatabase)
        assert len(db.listings) == 1
        assert db.listings[0].id == "aaa"

    def test_missing_file_returns_empty(self, tmp_path):
        """When jobs.json does not exist, return empty database."""
        jobs_path = tmp_path / "nonexistent" / "jobs.json"

        with patch("scripts.deduplicate.JOBS_PATH", jobs_path):
            db = _load_database()

        assert isinstance(db, JobsDatabase)
        assert db.listings == []

    def test_empty_listings_returns_empty(self, tmp_path):
        """When jobs.json has an empty listings array, return empty database."""
        jobs_path = tmp_path / "jobs.json"
        _write_jobs_json(jobs_path, listings=[])

        with patch("scripts.deduplicate.JOBS_PATH", jobs_path):
            db = _load_database()

        assert isinstance(db, JobsDatabase)
        assert db.listings == []

    def test_empty_object_returns_empty(self, tmp_path):
        """When jobs.json contains an empty JSON object, return empty database."""
        jobs_path = tmp_path / "jobs.json"
        _write_jobs_json(jobs_path, raw={})

        with patch("scripts.deduplicate.JOBS_PATH", jobs_path):
            db = _load_database()

        assert isinstance(db, JobsDatabase)
        assert db.listings == []

    def test_loads_multiple_listings(self, tmp_path):
        """Multiple listings are all loaded correctly."""
        jobs_path = tmp_path / "jobs.json"
        listings = [
            _listing_dict(id="a1", apply_url="https://example.com/1"),
            _listing_dict(
                id="a2", company="OtherCo", apply_url="https://example.com/2"
            ),
        ]
        _write_jobs_json(jobs_path, listings=listings)

        with patch("scripts.deduplicate.JOBS_PATH", jobs_path):
            db = _load_database()

        assert len(db.listings) == 2


# ======================================================================
# _load_archived_hashes
# ======================================================================


class TestLoadArchivedHashes:
    """Tests for _load_archived_hashes."""

    def test_loads_valid_hashes(self, tmp_path):
        """Extracts all listing IDs from archived.json."""
        archived_path = tmp_path / "archived.json"
        listings = [
            _listing_dict(id="hash_old_1"),
            _listing_dict(id="hash_old_2", apply_url="https://example.com/2"),
        ]
        _write_archived_json(archived_path, listings=listings)

        with patch("scripts.deduplicate.ARCHIVED_PATH", archived_path):
            hashes = _load_archived_hashes()

        assert hashes == {"hash_old_1", "hash_old_2"}

    def test_missing_file_returns_empty_set(self, tmp_path):
        """When archived.json does not exist, return empty set."""
        archived_path = tmp_path / "nonexistent" / "archived.json"

        with patch("scripts.deduplicate.ARCHIVED_PATH", archived_path):
            hashes = _load_archived_hashes()

        assert hashes == set()

    def test_empty_listings_returns_empty_set(self, tmp_path):
        """When archived.json has an empty listings array, return empty set."""
        archived_path = tmp_path / "archived.json"
        _write_archived_json(archived_path, listings=[])

        with patch("scripts.deduplicate.ARCHIVED_PATH", archived_path):
            hashes = _load_archived_hashes()

        assert hashes == set()

    def test_empty_object_returns_empty_set(self, tmp_path):
        """When archived.json contains an empty object, return empty set."""
        archived_path = tmp_path / "archived.json"
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        with open(archived_path, "w") as f:
            json.dump({}, f)

        with patch("scripts.deduplicate.ARCHIVED_PATH", archived_path):
            hashes = _load_archived_hashes()

        assert hashes == set()

    def test_listings_without_id_skipped(self, tmp_path):
        """Listings that lack an 'id' key are silently skipped."""
        archived_path = tmp_path / "archived.json"
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "listings": [
                {"id": "good_hash", "company": "Good"},
                {"company": "NoId"},  # missing 'id'
            ],
            "last_updated": "2026-01-01T00:00:00",
        }
        with open(archived_path, "w") as f:
            json.dump(data, f)

        with patch("scripts.deduplicate.ARCHIVED_PATH", archived_path):
            hashes = _load_archived_hashes()

        assert hashes == {"good_hash"}


# ======================================================================
# _save_database
# ======================================================================


class TestSaveDatabase:
    """Tests for _save_database."""

    def test_saves_valid_json(self, tmp_path):
        """Database is persisted as valid JSON to disk."""
        jobs_path = tmp_path / "jobs.json"
        listing = _make_listing(id="save_test")
        db = JobsDatabase(
            listings=[listing],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            _save_database(db)

        assert jobs_path.exists()
        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["listings"]) == 1
        assert saved["listings"][0]["id"] == "save_test"

    def test_updates_stats(self, tmp_path):
        """compute_stats is called so total_open reflects open listings."""
        jobs_path = tmp_path / "jobs.json"
        open_listing = _make_listing(id="open1", status=ListingStatus.OPEN)
        closed_listing = _make_listing(
            id="closed1",
            status=ListingStatus.CLOSED,
            apply_url="https://example.com/2",
        )
        db = JobsDatabase(
            listings=[open_listing, closed_listing],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            _save_database(db)

        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["total_open"] == 1

    def test_updates_timestamp(self, tmp_path):
        """last_updated is set to a recent UTC time on save."""
        jobs_path = tmp_path / "jobs.json"
        db = JobsDatabase(
            listings=[],
            last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        before = datetime.now(timezone.utc)

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            _save_database(db)

        # The model's last_updated should have been refreshed
        assert db.last_updated >= before

    def test_creates_data_dir_if_missing(self, tmp_path):
        """DATA_DIR is created if it does not exist."""
        data_dir = tmp_path / "nested" / "data"
        jobs_path = data_dir / "jobs.json"
        db = JobsDatabase(
            listings=[],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.DATA_DIR", data_dir),
        ):
            _save_database(db)

        assert data_dir.exists()
        assert jobs_path.exists()


# ======================================================================
# _dedup_by_content_hash
# ======================================================================


class TestDedupByContentHash:
    """Tests for _dedup_by_content_hash."""

    def test_no_duplicates(self):
        """All unique IDs produces zero removals."""
        listings = [
            _make_listing(id="a", apply_url="https://example.com/a"),
            _make_listing(id="b", apply_url="https://example.com/b"),
            _make_listing(id="c", apply_url="https://example.com/c"),
        ]
        result, removed = _dedup_by_content_hash(listings)
        assert len(result) == 3
        assert removed == 0

    def test_exact_duplicate_keeps_newer(self):
        """When two listings share an ID, the one with the later date_added is kept."""
        older = _make_listing(id="dup", date_added=date(2026, 1, 1))
        newer = _make_listing(id="dup", date_added=date(2026, 2, 1))
        result, removed = _dedup_by_content_hash([older, newer])

        assert len(result) == 1
        assert removed == 1
        assert result[0].date_added == date(2026, 2, 1)

    def test_exact_duplicate_newer_first(self):
        """When the newer listing appears first, it is still kept."""
        newer = _make_listing(id="dup", date_added=date(2026, 2, 1))
        older = _make_listing(id="dup", date_added=date(2026, 1, 1))
        result, removed = _dedup_by_content_hash([newer, older])

        assert len(result) == 1
        assert removed == 1
        assert result[0].date_added == date(2026, 2, 1)

    def test_multiple_duplicates_of_same_id(self):
        """Three listings with the same ID collapses to one; count is 2."""
        l1 = _make_listing(id="dup", date_added=date(2026, 1, 1))
        l2 = _make_listing(id="dup", date_added=date(2026, 2, 1))
        l3 = _make_listing(id="dup", date_added=date(2026, 3, 1))
        result, removed = _dedup_by_content_hash([l1, l2, l3])

        assert len(result) == 1
        assert removed == 2
        assert result[0].date_added == date(2026, 3, 1)

    def test_empty_list(self):
        """Empty input returns empty output with zero removals."""
        result, removed = _dedup_by_content_hash([])
        assert result == []
        assert removed == 0

    def test_mixed_duplicates_and_uniques(self):
        """Only duplicated IDs are collapsed; unique ones remain."""
        a = _make_listing(
            id="a", date_added=date(2026, 1, 1), apply_url="https://example.com/a"
        )
        b1 = _make_listing(
            id="b", date_added=date(2026, 1, 1), apply_url="https://example.com/b1"
        )
        b2 = _make_listing(
            id="b", date_added=date(2026, 2, 1), apply_url="https://example.com/b2"
        )
        c = _make_listing(
            id="c", date_added=date(2026, 1, 1), apply_url="https://example.com/c"
        )

        result, removed = _dedup_by_content_hash([a, b1, b2, c])
        assert len(result) == 3
        assert removed == 1
        ids = {r.id for r in result}
        assert ids == {"a", "b", "c"}


# ======================================================================
# _dedup_by_url
# ======================================================================


class TestDedupByUrl:
    """Tests for _dedup_by_url."""

    def test_no_duplicates(self):
        """All unique URLs produces zero removals."""
        listings = [
            _make_listing(id="a", apply_url="https://example.com/a"),
            _make_listing(id="b", apply_url="https://example.com/b"),
        ]
        result, removed = _dedup_by_url(listings)
        assert len(result) == 2
        assert removed == 0

    def test_same_url_keeps_newer(self):
        """When two listings share a URL, the newer one is kept."""
        older = _make_listing(
            id="a",
            apply_url="https://example.com/same",
            date_added=date(2026, 1, 1),
        )
        newer = _make_listing(
            id="b",
            apply_url="https://example.com/same",
            date_added=date(2026, 2, 1),
        )
        result, removed = _dedup_by_url([older, newer])

        assert len(result) == 1
        assert removed == 1
        assert result[0].id == "b"

    def test_same_url_newer_first(self):
        """Newer listing appearing first is still kept when URL dupes exist."""
        newer = _make_listing(
            id="a",
            apply_url="https://example.com/same",
            date_added=date(2026, 2, 1),
        )
        older = _make_listing(
            id="b",
            apply_url="https://example.com/same",
            date_added=date(2026, 1, 1),
        )
        result, removed = _dedup_by_url([newer, older])

        assert len(result) == 1
        assert removed == 1
        assert result[0].id == "a"

    def test_different_urls_all_kept(self):
        """Listings with distinct URLs are all preserved."""
        listings = [
            _make_listing(id="a", apply_url="https://example.com/1"),
            _make_listing(id="b", apply_url="https://example.com/2"),
            _make_listing(id="c", apply_url="https://example.com/3"),
        ]
        result, removed = _dedup_by_url(listings)
        assert len(result) == 3
        assert removed == 0

    def test_empty_list(self):
        """Empty input produces empty output."""
        result, removed = _dedup_by_url([])
        assert result == []
        assert removed == 0

    def test_three_same_urls(self):
        """Three listings sharing the same URL collapse to one."""
        l1 = _make_listing(
            id="a",
            apply_url="https://example.com/dup",
            date_added=date(2026, 1, 1),
        )
        l2 = _make_listing(
            id="b",
            apply_url="https://example.com/dup",
            date_added=date(2026, 3, 1),
        )
        l3 = _make_listing(
            id="c",
            apply_url="https://example.com/dup",
            date_added=date(2026, 2, 1),
        )
        result, removed = _dedup_by_url([l1, l2, l3])

        assert len(result) == 1
        assert removed == 2
        assert result[0].id == "b"


# ======================================================================
# _compute_token_overlap
# ======================================================================


class TestComputeTokenOverlap:
    """Tests for _compute_token_overlap (Jaccard similarity)."""

    def test_identical_strings(self):
        """Identical strings produce overlap of 1.0."""
        assert (
            _compute_token_overlap(
                "Software Engineer Intern", "Software Engineer Intern"
            )
            == 1.0
        )

    def test_no_overlap(self):
        """Completely disjoint tokens produce 0.0."""
        assert _compute_token_overlap("software engineer", "product manager") == 0.0

    def test_partial_overlap(self):
        """Shared tokens produce correct Jaccard coefficient."""
        # tokens_a = {"software", "engineer", "intern"}
        # tokens_b = {"software", "engineer", "co-op"}
        # intersection = {"software", "engineer"} = 2
        # union = {"software", "engineer", "intern", "co-op"} = 4
        # Jaccard = 2/4 = 0.5
        result = _compute_token_overlap(
            "Software Engineer Intern", "Software Engineer Co-op"
        )
        assert result == pytest.approx(0.5)

    def test_both_empty(self):
        """Two empty strings produce 0.0."""
        assert _compute_token_overlap("", "") == 0.0

    def test_one_empty(self):
        """One empty string and one non-empty produces 0.0."""
        assert _compute_token_overlap("", "software engineer") == 0.0
        assert _compute_token_overlap("software engineer", "") == 0.0

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert (
            _compute_token_overlap("Software ENGINEER", "software engineer") == 1.0
        )

    def test_superset_overlap(self):
        """When one string is a subset of the other."""
        # tokens_a = {"software"}, tokens_b = {"software", "engineer"}
        # intersection = 1, union = 2, Jaccard = 0.5
        result = _compute_token_overlap("Software", "Software Engineer")
        assert result == pytest.approx(0.5)


# ======================================================================
# _dedup_fuzzy
# ======================================================================


class TestDedupFuzzy:
    """Tests for _dedup_fuzzy (fuzzy company name + role title matching)."""

    def test_similar_company_similar_role_deduped(self):
        """Very similar company names and overlapping role titles are deduped."""
        l1 = _make_listing(
            id="f1",
            company="Stripe Inc",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
            date_added=date(2026, 1, 1),
        )
        l2 = _make_listing(
            id="f2",
            company="Stripe Inc.",
            role="Software Engineer Intern",
            apply_url="https://example.com/2",
            date_added=date(2026, 2, 1),
        )
        result, removed = _dedup_fuzzy([l1, l2], archived_hashes=set())
        assert len(result) == 1
        assert removed == 1
        # Keeps the newer one
        assert result[0].id == "f2"

    def test_different_companies_not_deduped(self):
        """Different company names are never matched as duplicates."""
        l1 = _make_listing(
            id="f1",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
        )
        l2 = _make_listing(
            id="f2",
            company="Google",
            role="Software Engineer Intern",
            apply_url="https://example.com/2",
        )
        result, removed = _dedup_fuzzy([l1, l2], archived_hashes=set())
        assert len(result) == 2
        assert removed == 0

    def test_same_company_different_roles_not_deduped(self):
        """Same company but sufficiently different roles are kept."""
        l1 = _make_listing(
            id="f1",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
        )
        l2 = _make_listing(
            id="f2",
            company="Stripe",
            role="Product Manager Intern",
            apply_url="https://example.com/2",
        )
        result, removed = _dedup_fuzzy([l1, l2], archived_hashes=set())
        assert len(result) == 2
        assert removed == 0

    def test_archived_listing_not_deduped_repost(self):
        """Listings whose IDs appear in archived_hashes are treated as reposts."""
        l1 = _make_listing(
            id="archived_id",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
            date_added=date(2026, 1, 1),
        )
        l2 = _make_listing(
            id="new_id",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/2",
            date_added=date(2026, 2, 1),
        )
        # l1 is in archived hashes, so it should not be deduped against l2
        result, removed = _dedup_fuzzy([l1, l2], archived_hashes={"archived_id"})
        assert len(result) == 2
        assert removed == 0

    def test_both_archived_not_deduped(self):
        """When both listings are in archived_hashes, neither is deduped."""
        l1 = _make_listing(
            id="arch1",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
        )
        l2 = _make_listing(
            id="arch2",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/2",
        )
        result, removed = _dedup_fuzzy(
            [l1, l2], archived_hashes={"arch1", "arch2"}
        )
        assert len(result) == 2
        assert removed == 0

    def test_single_listing_returned_as_is(self):
        """A list with a single listing is returned unchanged."""
        l1 = _make_listing(id="only")
        result, removed = _dedup_fuzzy([l1], archived_hashes=set())
        assert len(result) == 1
        assert removed == 0
        assert result[0].id == "only"

    def test_empty_list(self):
        """An empty list returns empty with zero removals."""
        result, removed = _dedup_fuzzy([], archived_hashes=set())
        assert result == []
        assert removed == 0

    def test_fuzzy_keeps_newer_when_j_is_newer(self):
        """When the second listing (j) is newer, the older one (i) is removed."""
        older = _make_listing(
            id="f1",
            company="Coinbase",
            role="Backend Engineer Intern",
            apply_url="https://example.com/1",
            date_added=date(2026, 1, 1),
        )
        newer = _make_listing(
            id="f2",
            company="Coinbase",
            role="Backend Engineer Intern",
            apply_url="https://example.com/2",
            date_added=date(2026, 3, 1),
        )
        result, removed = _dedup_fuzzy([older, newer], archived_hashes=set())
        assert len(result) == 1
        assert removed == 1
        assert result[0].id == "f2"

    def test_fuzzy_keeps_newer_when_i_is_newer(self):
        """When the first listing (i) is newer, the second (j) is removed."""
        newer = _make_listing(
            id="f1",
            company="Coinbase",
            role="Backend Engineer Intern",
            apply_url="https://example.com/1",
            date_added=date(2026, 3, 1),
        )
        older = _make_listing(
            id="f2",
            company="Coinbase",
            role="Backend Engineer Intern",
            apply_url="https://example.com/2",
            date_added=date(2026, 1, 1),
        )
        result, removed = _dedup_fuzzy([newer, older], archived_hashes=set())
        assert len(result) == 1
        assert removed == 1
        assert result[0].id == "f1"

    def test_company_similarity_below_threshold_not_deduped(self):
        """Company names below the fuzz.ratio > 90 threshold are not deduped."""
        l1 = _make_listing(
            id="f1",
            company="Stripe",
            role="Software Engineer Intern",
            apply_url="https://example.com/1",
        )
        l2 = _make_listing(
            id="f2",
            company="Strips",
            role="Software Engineer Intern",
            apply_url="https://example.com/2",
        )
        result, removed = _dedup_fuzzy([l1, l2], archived_hashes=set())
        assert len(result) == 2
        assert removed == 0


# ======================================================================
# deduplicate_all
# ======================================================================


class TestDeduplicateAll:
    """Tests for the deduplicate_all orchestrator."""

    def test_empty_database_returns_zero(self, tmp_path):
        """When jobs.json has no listings, return 0 without saving."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        _write_jobs_json(jobs_path, listings=[])

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        assert total == 0

    def test_no_duplicates_returns_zero(self, tmp_path):
        """With unique listings, total removed is 0 but database is still saved."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"
        listings = [
            _listing_dict(
                id="u1", company="Alpha", apply_url="https://example.com/1"
            ),
            _listing_dict(
                id="u2", company="Beta", apply_url="https://example.com/2"
            ),
        ]
        _write_jobs_json(jobs_path, listings=listings)

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        assert total == 0
        # Database should still be saved (file is updated)
        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["listings"]) == 2

    def test_runs_all_three_stages(self, tmp_path):
        """Duplicates caught by different stages are all counted."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        # Two listings with the same hash ID (caught by stage 1)
        # Plus another pair with same URL but different IDs (caught by stage 2)
        # Use different companies/roles so fuzzy dedup does not trigger between pairs
        listings = [
            _listing_dict(
                id="dup_hash",
                company="AlphaCorp",
                role="Software Engineer Intern",
                date_added="2026-01-01",
                apply_url="https://example.com/1",
            ),
            _listing_dict(
                id="dup_hash",
                company="AlphaCorp",
                role="Software Engineer Intern",
                date_added="2026-02-01",
                apply_url="https://example.com/2",
            ),
            _listing_dict(
                id="url_dup_a",
                company="ZetaGames",
                role="Product Manager Intern",
                date_added="2026-01-01",
                apply_url="https://example.com/same",
            ),
            _listing_dict(
                id="url_dup_b",
                company="ZetaGames",
                role="Product Manager Intern",
                date_added="2026-02-01",
                apply_url="https://example.com/same",
            ),
        ]
        _write_jobs_json(jobs_path, listings=listings)

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        # 1 hash dup + 1 URL dup = 2 total
        assert total == 2
        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["listings"]) == 2

    def test_saves_updated_database(self, tmp_path):
        """After deduplication, the saved database reflects the cleaned listings."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        listings = [
            _listing_dict(id="keep1", apply_url="https://example.com/1"),
            _listing_dict(
                id="keep2",
                company="OtherCo",
                apply_url="https://example.com/2",
            ),
            _listing_dict(
                id="keep1",
                apply_url="https://example.com/3",
                date_added="2026-02-01",
            ),
        ]
        _write_jobs_json(jobs_path, listings=listings)

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        assert total == 1
        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["listings"]) == 2
        assert saved["total_open"] == 2

    def test_missing_jobs_file_returns_zero(self, tmp_path):
        """When jobs.json does not exist, return 0 immediately."""
        jobs_path = tmp_path / "nonexistent" / "jobs.json"
        archived_path = tmp_path / "archived.json"

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        assert total == 0

    def test_with_archived_hashes_present(self, tmp_path):
        """Archived hashes are loaded and passed to fuzzy dedup for repost detection."""
        jobs_path = tmp_path / "jobs.json"
        archived_path = tmp_path / "archived.json"

        # Two fuzzy-similar listings; one has an archived hash so NOT deduped
        listings = [
            _listing_dict(
                id="previously_archived",
                company="Stripe",
                role="Software Engineer Intern",
                apply_url="https://example.com/1",
                date_added="2026-01-01",
            ),
            _listing_dict(
                id="new_posting",
                company="Stripe",
                role="Software Engineer Intern",
                apply_url="https://example.com/2",
                date_added="2026-02-01",
            ),
        ]
        _write_jobs_json(jobs_path, listings=listings)

        # Mark one listing as archived
        archived_listings = [_listing_dict(id="previously_archived")]
        _write_archived_json(archived_path, listings=archived_listings)

        with (
            patch("scripts.deduplicate.JOBS_PATH", jobs_path),
            patch("scripts.deduplicate.ARCHIVED_PATH", archived_path),
            patch("scripts.deduplicate.DATA_DIR", tmp_path),
        ):
            total = deduplicate_all()

        # Should be 0 since the archived listing is treated as a repost
        assert total == 0
        with open(jobs_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved["listings"]) == 2
