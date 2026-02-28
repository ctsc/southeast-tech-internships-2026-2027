"""Comprehensive tests for the validation and enrichment pipeline.

Tests cover:
- _find_latest_raw_discovery: file discovery, empty directory
- _load_raw_listings: valid file, malformed entries
- _load_existing_database: valid db, missing file, corrupt file
- _get_existing_hashes: hash extraction
- _generate_listing_id: determinism, case insensitivity, location sorting
- _map_category: all valid categories + unknown
- _map_sponsorship: all valid statuses + unknown
- _slugify: basic slugs, special chars, consecutive hyphens
- _parse_locations: AI locations, delimiter splitting, City/State patterns
- _build_job_listing: correct JobListing construction
- _save_database: JSON output, timestamp and stats
- validate_all: end-to-end async tests with mocked AI
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    RawListing,
    RoleCategory,
    SponsorshipStatus,
)
from scripts.validate import (
    _build_job_listing,
    _find_latest_raw_discovery,
    _generate_listing_id,
    _get_existing_hashes,
    _load_existing_database,
    _load_raw_listings,
    _map_category,
    _map_sponsorship,
    _parse_locations,
    _save_database,
    _slugify,
    validate_all,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_raw_listing(**overrides) -> RawListing:
    """Create a RawListing with sensible defaults, overriding as needed."""
    defaults = {
        "company": "TestCorp",
        "company_slug": "testcorp",
        "title": "Software Engineer Intern",
        "location": "San Francisco, CA",
        "url": "https://example.com/apply",
        "source": "greenhouse_api",
        "is_faang_plus": False,
    }
    defaults.update(overrides)
    return RawListing(**defaults)


def _make_raw_listing_dict(**overrides) -> dict:
    """Create a raw listing dict suitable for writing to discovery JSON."""
    defaults = {
        "company": "TestCorp",
        "company_slug": "testcorp",
        "title": "Software Engineer Intern",
        "location": "San Francisco, CA",
        "url": "https://example.com/apply",
        "source": "greenhouse_api",
        "is_faang_plus": False,
    }
    defaults.update(overrides)
    return defaults


def _write_raw_discovery(tmp_dir: Path, filename: str, listings: list[dict]) -> Path:
    """Write a raw discovery JSON file to the tmp directory."""
    filepath = tmp_dir / filename
    data = {"listings": listings}
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


def _write_jobs_json(tmp_dir: Path, db_dict: dict) -> Path:
    """Write a jobs.json file to the tmp directory."""
    filepath = tmp_dir / "jobs.json"
    filepath.write_text(json.dumps(db_dict, default=str), encoding="utf-8")
    return filepath


def _make_valid_metadata(**overrides) -> dict:
    """Create a valid AI enrichment metadata dict."""
    defaults = {
        "is_internship": True,
        "is_summer_2026": True,
        "category": "swe",
        "locations": ["San Francisco, CA"],
        "sponsorship": "unknown",
        "requires_advanced_degree": False,
        "remote_friendly": False,
        "tech_stack": ["Python"],
        "confidence": 0.95,
    }
    defaults.update(overrides)
    return defaults


def _make_jobs_db_dict(listings: list[dict] | None = None) -> dict:
    """Create a jobs database dict suitable for writing to jobs.json."""
    return {
        "listings": listings or [],
        "last_updated": "2026-01-01T00:00:00",
        "total_open": 0,
    }


# ======================================================================
# Tests for _find_latest_raw_discovery
# ======================================================================


class TestFindLatestRawDiscovery:
    """Tests for _find_latest_raw_discovery."""

    def test_returns_none_when_no_files(self, tmp_path):
        """Returns None when the data directory has no raw_discovery files."""
        with patch("scripts.validate.DATA_DIR", tmp_path):
            result = _find_latest_raw_discovery()
        assert result is None

    def test_finds_single_file(self, tmp_path):
        """Returns the only file when exactly one raw discovery file exists."""
        raw_file = tmp_path / "raw_discovery_20260101_120000.json"
        raw_file.write_text("{}")
        with patch("scripts.validate.DATA_DIR", tmp_path):
            result = _find_latest_raw_discovery()
        assert result == raw_file

    def test_finds_latest_of_multiple(self, tmp_path):
        """Returns the lexicographically latest file among several."""
        early = tmp_path / "raw_discovery_20260101_060000.json"
        middle = tmp_path / "raw_discovery_20260101_120000.json"
        latest = tmp_path / "raw_discovery_20260102_060000.json"
        for f in [early, middle, latest]:
            f.write_text("{}")
        with patch("scripts.validate.DATA_DIR", tmp_path):
            result = _find_latest_raw_discovery()
        assert result == latest

    def test_ignores_non_matching_files(self, tmp_path):
        """Ignores files that do not match the raw_discovery_*.json pattern."""
        (tmp_path / "jobs.json").write_text("{}")
        (tmp_path / "archived.json").write_text("{}")
        raw_file = tmp_path / "raw_discovery_20260201_000000.json"
        raw_file.write_text("{}")
        with patch("scripts.validate.DATA_DIR", tmp_path):
            result = _find_latest_raw_discovery()
        assert result == raw_file


# ======================================================================
# Tests for _load_raw_listings
# ======================================================================


class TestLoadRawListings:
    """Tests for _load_raw_listings."""

    def test_loads_valid_listings(self, tmp_path):
        """Successfully loads valid raw listings from JSON."""
        listings_data = [
            _make_raw_listing_dict(company="Anthropic", title="SWE Intern"),
            _make_raw_listing_dict(company="Stripe", title="ML Intern"),
        ]
        filepath = _write_raw_discovery(tmp_path, "raw_discovery_test.json", listings_data)
        result = _load_raw_listings(filepath)
        assert len(result) == 2
        assert result[0].company == "Anthropic"
        assert result[1].company == "Stripe"
        assert all(isinstance(r, RawListing) for r in result)

    def test_skips_malformed_entries(self, tmp_path):
        """Skips entries that fail Pydantic validation without crashing."""
        listings_data = [
            _make_raw_listing_dict(company="ValidCo"),
            {"bad": "entry", "no_required_fields": True},  # malformed
            _make_raw_listing_dict(company="AnotherValid"),
        ]
        filepath = _write_raw_discovery(tmp_path, "raw_discovery_test.json", listings_data)
        result = _load_raw_listings(filepath)
        assert len(result) == 2
        assert result[0].company == "ValidCo"
        assert result[1].company == "AnotherValid"

    def test_returns_empty_for_no_listings(self, tmp_path):
        """Returns empty list when the listings array is empty."""
        filepath = _write_raw_discovery(tmp_path, "raw_discovery_test.json", [])
        result = _load_raw_listings(filepath)
        assert result == []

    def test_returns_empty_when_listings_key_missing(self, tmp_path):
        """Returns empty list when the JSON has no 'listings' key."""
        filepath = tmp_path / "raw_discovery_test.json"
        filepath.write_text(json.dumps({"other_key": []}), encoding="utf-8")
        result = _load_raw_listings(filepath)
        assert result == []


# ======================================================================
# Tests for _load_existing_database
# ======================================================================


class TestLoadExistingDatabase:
    """Tests for _load_existing_database."""

    def test_returns_empty_db_when_file_missing(self, tmp_path):
        """Returns a fresh empty database when jobs.json does not exist."""
        fake_jobs = tmp_path / "jobs.json"
        with patch("scripts.validate.JOBS_PATH", fake_jobs):
            db = _load_existing_database()
        assert isinstance(db, JobsDatabase)
        assert db.listings == []

    def test_loads_valid_database(self, tmp_path):
        """Loads and parses a valid jobs.json."""
        listing_dict = {
            "id": "hash123",
            "company": "TestCo",
            "company_slug": "testco",
            "role": "Intern",
            "category": "swe",
            "locations": ["NYC"],
            "apply_url": "https://example.com/apply",
            "date_added": "2026-01-15",
            "date_last_verified": "2026-02-01",
            "source": "greenhouse_api",
            "status": "open",
        }
        db_dict = _make_jobs_db_dict([listing_dict])
        _write_jobs_json(tmp_path, db_dict)
        with patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"):
            db = _load_existing_database()
        assert len(db.listings) == 1
        assert db.listings[0].id == "hash123"

    def test_returns_empty_db_on_corrupt_file(self, tmp_path):
        """Returns empty database when jobs.json is corrupt/invalid JSON."""
        corrupt = tmp_path / "jobs.json"
        corrupt.write_text("NOT VALID JSON {{{", encoding="utf-8")
        with patch("scripts.validate.JOBS_PATH", corrupt):
            db = _load_existing_database()
        assert isinstance(db, JobsDatabase)
        assert db.listings == []

    def test_returns_empty_db_on_schema_mismatch(self, tmp_path):
        """Returns empty database when JSON is valid but doesn't match schema."""
        bad_schema = tmp_path / "jobs.json"
        bad_schema.write_text(json.dumps({"wrong": "schema"}), encoding="utf-8")
        with patch("scripts.validate.JOBS_PATH", bad_schema):
            db = _load_existing_database()
        assert isinstance(db, JobsDatabase)
        assert db.listings == []


# ======================================================================
# Tests for _get_existing_hashes
# ======================================================================


class TestGetExistingHashes:
    """Tests for _get_existing_hashes."""

    def test_extracts_ids_from_database(self, sample_jobs_database):
        """Returns a set of all listing IDs."""
        hashes = _get_existing_hashes(sample_jobs_database)
        assert "abc123hash" in hashes
        assert len(hashes) == 1

    def test_returns_empty_set_for_empty_db(self):
        """Returns an empty set when the database has no listings."""
        db = JobsDatabase(
            listings=[], last_updated=datetime.now(timezone.utc), total_open=0
        )
        hashes = _get_existing_hashes(db)
        assert hashes == set()


# ======================================================================
# Tests for _generate_listing_id
# ======================================================================


class TestGenerateListingId:
    """Tests for _generate_listing_id."""

    def test_deterministic_hash(self):
        """Same inputs produce the same hash every time."""
        id1 = _generate_listing_id("Anthropic", "SWE Intern", ["SF"])
        id2 = _generate_listing_id("Anthropic", "SWE Intern", ["SF"])
        assert id1 == id2

    def test_case_insensitive(self):
        """Hash is the same regardless of case."""
        id_lower = _generate_listing_id("anthropic", "swe intern", ["sf"])
        id_upper = _generate_listing_id("ANTHROPIC", "SWE INTERN", ["SF"])
        id_mixed = _generate_listing_id("Anthropic", "SWE Intern", ["Sf"])
        assert id_lower == id_upper == id_mixed

    def test_locations_sorted(self):
        """Location order does not affect the hash."""
        id1 = _generate_listing_id("Acme", "Intern", ["NYC", "SF", "Remote"])
        id2 = _generate_listing_id("Acme", "Intern", ["Remote", "NYC", "SF"])
        id3 = _generate_listing_id("Acme", "Intern", ["SF", "Remote", "NYC"])
        assert id1 == id2 == id3

    def test_different_inputs_different_hashes(self):
        """Different company/role/location combinations produce different hashes."""
        id1 = _generate_listing_id("Anthropic", "SWE Intern", ["SF"])
        id2 = _generate_listing_id("Stripe", "SWE Intern", ["SF"])
        id3 = _generate_listing_id("Anthropic", "ML Intern", ["SF"])
        assert id1 != id2
        assert id1 != id3

    def test_returns_hex_digest(self):
        """The returned hash is a valid hex string."""
        result = _generate_listing_id("Test", "Intern", ["NYC"])
        assert isinstance(result, str)
        # SHA-256 hex digest is 64 chars
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before hashing."""
        id1 = _generate_listing_id("Anthropic", "SWE Intern", ["SF"])
        id2 = _generate_listing_id("  Anthropic  ", "  SWE Intern  ", ["  SF  "])
        assert id1 == id2


# ======================================================================
# Tests for _map_category
# ======================================================================


class TestMapCategory:
    """Tests for _map_category."""

    def test_all_valid_categories(self):
        """Maps every known category string to the correct enum value."""
        assert _map_category("swe") == RoleCategory.SWE
        assert _map_category("ml_ai") == RoleCategory.ML_AI
        assert _map_category("data_science") == RoleCategory.DATA_SCIENCE
        assert _map_category("quant") == RoleCategory.QUANT
        assert _map_category("pm") == RoleCategory.PM
        assert _map_category("hardware") == RoleCategory.HARDWARE
        assert _map_category("other") == RoleCategory.OTHER

    def test_unknown_defaults_to_other(self):
        """Unknown category strings default to OTHER."""
        assert _map_category("unknown_category") == RoleCategory.OTHER
        assert _map_category("marketing") == RoleCategory.OTHER
        assert _map_category("") == RoleCategory.OTHER

    def test_case_insensitive(self):
        """Category mapping is case insensitive."""
        assert _map_category("SWE") == RoleCategory.SWE
        assert _map_category("ML_AI") == RoleCategory.ML_AI
        assert _map_category("Pm") == RoleCategory.PM

    def test_strips_whitespace(self):
        """Category mapping strips leading/trailing whitespace."""
        assert _map_category("  swe  ") == RoleCategory.SWE
        assert _map_category("\tquant\n") == RoleCategory.QUANT


# ======================================================================
# Tests for _map_sponsorship
# ======================================================================


class TestMapSponsorship:
    """Tests for _map_sponsorship."""

    def test_all_valid_statuses(self):
        """Maps every known sponsorship string to the correct enum value."""
        assert _map_sponsorship("sponsors") == SponsorshipStatus.SPONSORS
        assert _map_sponsorship("no_sponsorship") == SponsorshipStatus.NO_SPONSORSHIP
        assert _map_sponsorship("us_citizenship") == SponsorshipStatus.US_CITIZENSHIP
        assert _map_sponsorship("unknown") == SponsorshipStatus.UNKNOWN

    def test_unknown_defaults_to_unknown(self):
        """Unrecognized strings default to UNKNOWN."""
        assert _map_sponsorship("maybe") == SponsorshipStatus.UNKNOWN
        assert _map_sponsorship("") == SponsorshipStatus.UNKNOWN
        assert _map_sponsorship("idk") == SponsorshipStatus.UNKNOWN

    def test_case_insensitive(self):
        """Sponsorship mapping is case insensitive."""
        assert _map_sponsorship("SPONSORS") == SponsorshipStatus.SPONSORS
        assert _map_sponsorship("No_Sponsorship") == SponsorshipStatus.NO_SPONSORSHIP

    def test_strips_whitespace(self):
        """Sponsorship mapping strips leading/trailing whitespace."""
        assert _map_sponsorship("  sponsors  ") == SponsorshipStatus.SPONSORS


# ======================================================================
# Tests for _slugify
# ======================================================================


class TestSlugify:
    """Tests for _slugify."""

    def test_basic_slug(self):
        """Simple company name converts to kebab-case."""
        assert _slugify("Anthropic") == "anthropic"
        assert _slugify("Scale AI") == "scale-ai"
        assert _slugify("Open AI") == "open-ai"

    def test_special_characters_removed(self):
        """Periods and apostrophes are removed."""
        assert _slugify("AT&T") == "at&t"  # & is kept
        assert _slugify("O'Reilly") == "oreilly"
        assert _slugify("Inc.") == "inc"

    def test_consecutive_hyphens_collapsed(self):
        """Multiple consecutive hyphens are collapsed to one."""
        assert _slugify("A  B") == "a-b"  # double space -> double hyphen -> collapsed
        assert _slugify("Test  .  Company") == "test-company"

    def test_leading_trailing_hyphens_stripped(self):
        """Leading and trailing hyphens are stripped."""
        assert _slugify(" Test ") == "test"
        assert _slugify(".Company.") == "company"

    def test_lowercase(self):
        """Output is always lowercase."""
        assert _slugify("GOOGLE") == "google"
        assert _slugify("McKinsey") == "mckinsey"


# ======================================================================
# Tests for _parse_locations
# ======================================================================


class TestParseLocations:
    """Tests for _parse_locations."""

    def test_ai_locations_preferred(self):
        """When AI provides locations, they are used directly."""
        result = _parse_locations("some raw text", ["NYC", "Remote"])
        assert result == ["NYC", "Remote"]

    def test_ai_locations_empty_list_falls_back(self):
        """When AI locations is an empty list, falls back to raw parsing."""
        result = _parse_locations("NYC / SF", [])
        assert result == ["NYC", "SF"]

    def test_ai_locations_none_falls_back(self):
        """When AI locations is None, falls back to raw parsing."""
        result = _parse_locations("NYC / SF", None)
        assert result == ["NYC", "SF"]

    def test_slash_delimiter(self):
        """Splits on ' / ' delimiter."""
        result = _parse_locations("NYC / SF / Remote")
        assert result == ["NYC", "SF", "Remote"]

    def test_pipe_delimiter(self):
        """Splits on ' | ' delimiter."""
        result = _parse_locations("NYC | SF | Remote")
        assert result == ["NYC", "SF", "Remote"]

    def test_semicolon_delimiter(self):
        """Splits on ' ; ' delimiter."""
        result = _parse_locations("NYC ; SF ; Remote")
        assert result == ["NYC", "SF", "Remote"]

    def test_city_state_kept_intact(self):
        """A 'City, ST' pattern (2-3 char state code) is kept as one location."""
        result = _parse_locations("San Francisco, CA")
        assert result == ["San Francisco, CA"]

    def test_city_state_three_letter(self):
        """A 'City, XXX' pattern with a 3-char state is kept as one location."""
        result = _parse_locations("Atlanta, USA")
        assert result == ["Atlanta, USA"]

    def test_multi_location_commas(self):
        """Multiple comma-separated locations (>2 parts) are split."""
        result = _parse_locations("NYC, SF, Remote")
        assert result == ["NYC", "SF", "Remote"]

    def test_empty_string_returns_unknown(self):
        """Empty raw location returns ['Unknown']."""
        result = _parse_locations("")
        assert result == ["Unknown"]

    def test_whitespace_only_returns_unknown(self):
        """Whitespace-only raw location returns ['Unknown']."""
        result = _parse_locations("   ")
        assert result == ["Unknown"]

    def test_single_location(self):
        """A single location with no delimiters is returned as-is."""
        result = _parse_locations("Remote")
        assert result == ["Remote"]

    def test_slash_without_spaces(self):
        """Splits on '/' without spaces as well."""
        result = _parse_locations("NYC/SF")
        assert result == ["NYC", "SF"]


# ======================================================================
# Tests for _build_job_listing
# ======================================================================


class TestBuildJobListing:
    """Tests for _build_job_listing."""

    def test_builds_correct_listing(self):
        """Creates a JobListing with all fields correctly populated."""
        raw = _make_raw_listing(
            company="Anthropic",
            company_slug="anthropic",
            title="Software Engineer Intern",
            location="San Francisco, CA",
            url="https://boards.greenhouse.io/anthropic/jobs/123",
            source="greenhouse_api",
            is_faang_plus=True,
        )
        metadata = _make_valid_metadata(
            category="swe",
            locations=["San Francisco, CA"],
            sponsorship="sponsors",
            tech_stack=["Python", "TypeScript"],
            remote_friendly=True,
            requires_advanced_degree=False,
        )

        job = _build_job_listing(raw, metadata)

        assert isinstance(job, JobListing)
        assert job.company == "Anthropic"
        assert job.company_slug == "anthropic"
        assert job.role == "Software Engineer Intern"
        assert job.category == RoleCategory.SWE
        assert job.locations == ["San Francisco, CA"]
        assert str(job.apply_url) == "https://boards.greenhouse.io/anthropic/jobs/123"
        assert job.sponsorship == SponsorshipStatus.SPONSORS
        assert job.is_faang_plus is True
        assert job.requires_advanced_degree is False
        assert job.remote_friendly is True
        assert job.source == "greenhouse_api"
        assert job.status == ListingStatus.OPEN
        assert job.tech_stack == ["Python", "TypeScript"]
        assert job.season == "summer_2026"
        assert job.date_added == date.today()
        assert job.date_last_verified == date.today()

    def test_uses_ai_locations_when_present(self):
        """AI-provided locations override the raw location string."""
        raw = _make_raw_listing(location="Some Raw Location")
        metadata = _make_valid_metadata(locations=["NYC", "Remote"])
        job = _build_job_listing(raw, metadata)
        assert job.locations == ["NYC", "Remote"]

    def test_falls_back_to_raw_location_parsing(self):
        """Falls back to parsing raw location when AI locations are empty."""
        raw = _make_raw_listing(location="NYC / SF")
        metadata = _make_valid_metadata(locations=[])
        job = _build_job_listing(raw, metadata)
        assert job.locations == ["NYC", "SF"]

    def test_us_citizenship_flag(self):
        """Sets requires_us_citizenship when sponsorship is us_citizenship."""
        raw = _make_raw_listing()
        metadata = _make_valid_metadata(sponsorship="us_citizenship")
        job = _build_job_listing(raw, metadata)
        assert job.requires_us_citizenship is True
        assert job.sponsorship == SponsorshipStatus.US_CITIZENSHIP

    def test_defaults_for_missing_metadata_keys(self):
        """Uses sensible defaults when metadata keys are missing."""
        raw = _make_raw_listing()
        metadata = {
            "is_internship": True,
            "is_summer_2026": True,
            "confidence": 0.9,
            # No category, sponsorship, tech_stack, etc.
        }
        job = _build_job_listing(raw, metadata)
        assert job.category == RoleCategory.OTHER
        assert job.sponsorship == SponsorshipStatus.UNKNOWN
        assert job.tech_stack == []
        assert job.remote_friendly is False
        assert job.requires_advanced_degree is False


# ======================================================================
# Tests for _save_database
# ======================================================================


class TestSaveDatabase:
    """Tests for _save_database."""

    def test_saves_valid_json(self, tmp_path):
        """Saves the database as valid JSON to the configured path."""
        jobs_path = tmp_path / "jobs.json"
        listing_data = {
            "id": "hash1",
            "company": "TestCo",
            "company_slug": "testco",
            "role": "Intern",
            "category": "swe",
            "locations": ["NYC"],
            "apply_url": "https://example.com/apply",
            "date_added": "2026-01-15",
            "date_last_verified": "2026-02-01",
            "source": "greenhouse_api",
            "status": "open",
        }
        listing = JobListing(**listing_data)
        db = JobsDatabase(
            listings=[listing],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
            total_open=0,
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", jobs_path),
        ):
            _save_database(db)

        assert jobs_path.exists()
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert len(data["listings"]) == 1
        assert data["listings"][0]["company"] == "TestCo"

    def test_updates_stats(self, tmp_path):
        """Recomputes total_open before saving."""
        jobs_path = tmp_path / "jobs.json"
        listing_open = JobListing(
            id="h1", company="A", company_slug="a", role="Intern", category="swe",
            locations=["NYC"], apply_url="https://example.com/1",
            date_added="2026-01-15", date_last_verified="2026-02-01",
            source="test", status=ListingStatus.OPEN,
        )
        listing_closed = JobListing(
            id="h2", company="B", company_slug="b", role="Intern", category="swe",
            locations=["SF"], apply_url="https://example.com/2",
            date_added="2026-01-15", date_last_verified="2026-02-01",
            source="test", status=ListingStatus.CLOSED,
        )
        db = JobsDatabase(
            listings=[listing_open, listing_closed],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
            total_open=0,
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", jobs_path),
        ):
            _save_database(db)

        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert data["total_open"] == 1

    def test_creates_data_dir_if_missing(self, tmp_path):
        """Creates the DATA_DIR if it does not already exist."""
        data_dir = tmp_path / "nonexistent" / "data"
        jobs_path = data_dir / "jobs.json"
        db = JobsDatabase(
            listings=[],
            last_updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
            total_open=0,
        )

        with (
            patch("scripts.validate.DATA_DIR", data_dir),
            patch("scripts.validate.JOBS_PATH", jobs_path),
        ):
            _save_database(db)

        assert data_dir.exists()
        assert jobs_path.exists()


# ======================================================================
# Tests for validate_all (async)
# ======================================================================


class TestValidateAll:
    """Tests for the async validate_all entry point."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_raw_files(self, tmp_path):
        """Returns an empty list when no raw discovery files are found."""
        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
        ):
            result = await validate_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_raw_file_has_no_listings(self, tmp_path):
        """Returns empty when raw discovery file exists but has no listings."""
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [])
        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
        ):
            result = await validate_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_already_existing_listings(self, tmp_path):
        """Skips listings whose content_hash already exists in the database."""
        raw = _make_raw_listing(company="Anthropic", title="SWE Intern", location="SF")
        raw_dict = _make_raw_listing_dict(company="Anthropic", title="SWE Intern", location="SF")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        # Create existing db with same content hash
        existing_listing = {
            "id": raw.content_hash,
            "company": "Anthropic",
            "company_slug": "anthropic",
            "role": "SWE Intern",
            "category": "swe",
            "locations": ["SF"],
            "apply_url": "https://example.com/apply",
            "date_added": "2026-01-01",
            "date_last_verified": "2026-01-01",
            "source": "greenhouse_api",
            "status": "open",
        }
        _write_jobs_json(tmp_path, _make_jobs_db_dict([existing_listing]))

        mock_enrich = MagicMock(return_value=_make_valid_metadata())

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []
        mock_enrich.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_non_internships(self, tmp_path):
        """Rejects listings where AI says is_internship is False."""
        raw_dict = _make_raw_listing_dict(company="FullTime Inc", title="Senior Engineer")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(is_internship=False)
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_rejects_non_summer_2026(self, tmp_path):
        """Rejects listings where AI says is_summer_2026 is False."""
        raw_dict = _make_raw_listing_dict(company="OldCo", title="Fall 2025 Intern")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(is_summer_2026=False)
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_rejects_low_confidence(self, tmp_path):
        """Rejects listings with confidence below 0.7 threshold."""
        raw_dict = _make_raw_listing_dict(company="MaybeCo", title="Intern?")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(confidence=0.5)
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_accepts_valid_listings(self, tmp_path):
        """Accepts listings that pass all validation checks and appends to db."""
        raw_dict = _make_raw_listing_dict(
            company="Anthropic",
            title="SWE Intern",
            location="San Francisco, CA",
            url="https://example.com/anthropic/apply",
        )
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(return_value=_make_valid_metadata())

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert len(result) == 1
        assert result[0].company == "Anthropic"
        assert result[0].status == ListingStatus.OPEN

        # Verify saved to disk
        saved = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
        assert len(saved["listings"]) == 1

    @pytest.mark.asyncio
    async def test_handles_enrichment_errors_gracefully(self, tmp_path):
        """Continues processing when enrichment raises an exception."""
        raw_dicts = [
            _make_raw_listing_dict(company="ErrorCo", title="Intern", url="https://example.com/1"),
            _make_raw_listing_dict(company="GoodCo", title="Intern", url="https://example.com/2",
                                   location="NYC"),
        ]
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", raw_dicts)

        call_count = 0

        def side_effect(raw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API failure")
            return _make_valid_metadata()

        mock_enrich = MagicMock(side_effect=side_effect)

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        # Only the second listing should succeed
        assert len(result) == 1
        assert result[0].company == "GoodCo"

    @pytest.mark.asyncio
    async def test_handles_enrichment_returning_none(self, tmp_path):
        """Skips listing when enrichment returns None."""
        raw_dict = _make_raw_listing_dict(company="NoneCo", title="Intern")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(return_value=None)

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_deduplicates_within_single_run(self, tmp_path):
        """Removes duplicate listings (same content hash) within one validation run."""
        # Two raw listings that will produce the same content hash
        raw_dicts = [
            _make_raw_listing_dict(company="Dupe", title="Intern", location="SF",
                                   url="https://example.com/1"),
            _make_raw_listing_dict(company="Dupe", title="Intern", location="SF",
                                   url="https://example.com/2"),
        ]
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", raw_dicts)

        # AI returns locations = ["SF"] for both, making content hashes identical
        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(locations=["SF"])
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        # Only one should be kept
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_appends_to_existing_database(self, tmp_path):
        """New validated listings are appended to existing ones in the database."""
        # Pre-existing listing in jobs.json
        existing_listing = {
            "id": "existing_hash",
            "company": "OldCo",
            "company_slug": "oldco",
            "role": "Old Intern",
            "category": "swe",
            "locations": ["Boston"],
            "apply_url": "https://example.com/old",
            "date_added": "2026-01-01",
            "date_last_verified": "2026-01-01",
            "source": "greenhouse_api",
            "status": "open",
        }
        _write_jobs_json(tmp_path, _make_jobs_db_dict([existing_listing]))

        # New listing from discovery
        raw_dict = _make_raw_listing_dict(
            company="NewCo", title="New Intern", location="NYC",
            url="https://example.com/new",
        )
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(return_value=_make_valid_metadata())

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert len(result) == 1
        saved = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
        assert len(saved["listings"]) == 2  # old + new

    @pytest.mark.asyncio
    async def test_calls_reset_budget(self, tmp_path):
        """Calls reset_budget before processing listings."""
        raw_dict = _make_raw_listing_dict()
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(return_value=_make_valid_metadata())
        mock_reset = MagicMock()

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget", mock_reset),
        ):
            await validate_all()

        mock_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_enrich_called_without_await(self, tmp_path):
        """Verifies enrich_listing is called as a sync function (not awaited)."""
        raw_dict = _make_raw_listing_dict(company="SyncTest", title="Intern")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(return_value=_make_valid_metadata())

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            await validate_all()

        # MagicMock (not AsyncMock) was called, confirming sync invocation
        mock_enrich.assert_called_once()
        # The argument should be a RawListing
        call_args = mock_enrich.call_args
        assert isinstance(call_args[0][0], RawListing)

    @pytest.mark.asyncio
    async def test_confidence_at_exactly_070_is_accepted(self, tmp_path):
        """Listings with confidence exactly at 0.7 are rejected (< 0.7 check)."""
        raw_dict = _make_raw_listing_dict(company="BorderCo", title="Intern")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        # confidence = 0.7 passes the < 0.7 check
        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(confidence=0.7)
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_confidence_at_069_is_rejected(self, tmp_path):
        """Listings with confidence at 0.69 are rejected."""
        raw_dict = _make_raw_listing_dict(company="LowCo", title="Intern")
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", [raw_dict])

        mock_enrich = MagicMock(
            return_value=_make_valid_metadata(confidence=0.69)
        )

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_valid_listings(self, tmp_path):
        """Processes and validates multiple distinct listings in one run."""
        raw_dicts = [
            _make_raw_listing_dict(company="Alpha", title="SWE Intern",
                                   location="SF", url="https://example.com/a"),
            _make_raw_listing_dict(company="Beta", title="ML Intern",
                                   location="NYC", url="https://example.com/b"),
            _make_raw_listing_dict(company="Gamma", title="PM Intern",
                                   location="Remote", url="https://example.com/c"),
        ]
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", raw_dicts)

        call_idx = 0

        def mock_side_effect(raw):
            nonlocal call_idx
            call_idx += 1
            categories = ["swe", "ml_ai", "pm"]
            locs = [["SF"], ["NYC"], ["Remote"]]
            return _make_valid_metadata(
                category=categories[call_idx - 1],
                locations=locs[call_idx - 1],
            )

        mock_enrich = MagicMock(side_effect=mock_side_effect)

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert len(result) == 3
        companies = {r.company for r in result}
        assert companies == {"Alpha", "Beta", "Gamma"}

    @pytest.mark.asyncio
    async def test_mixed_accept_reject(self, tmp_path):
        """Correctly filters a mix of valid and invalid listings."""
        raw_dicts = [
            _make_raw_listing_dict(company="ValidCo", title="SWE Intern",
                                   url="https://example.com/valid", location="SF"),
            _make_raw_listing_dict(company="NotIntern", title="Senior Engineer",
                                   url="https://example.com/nope", location="NYC"),
            _make_raw_listing_dict(company="LowConf", title="Maybe Intern",
                                   url="https://example.com/low", location="LA"),
        ]
        _write_raw_discovery(tmp_path, "raw_discovery_20260101_000000.json", raw_dicts)

        call_idx = 0

        def mock_side_effect(raw):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return _make_valid_metadata()
            elif call_idx == 2:
                return _make_valid_metadata(is_internship=False)
            else:
                return _make_valid_metadata(confidence=0.3)

        mock_enrich = MagicMock(side_effect=mock_side_effect)

        with (
            patch("scripts.validate.DATA_DIR", tmp_path),
            patch("scripts.validate.JOBS_PATH", tmp_path / "jobs.json"),
            patch("scripts.validate.enrich_listing", mock_enrich),
            patch("scripts.validate.reset_budget"),
        ):
            result = await validate_all()

        assert len(result) == 1
        assert result[0].company == "ValidCo"
