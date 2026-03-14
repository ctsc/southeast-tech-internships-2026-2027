"""Tests for entry-level job pipeline: models, discovery, validation, and README rendering."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from scripts.utils.models import (
    JobListing,
    ListingType,
    RawListing,
    RoleCategory,
)


# ── ListingType Enum Tests ─────────────────────────────────────────────────


class TestListingType:
    def test_all_values(self):
        expected = {"internship", "entry_level"}
        assert {e.value for e in ListingType} == expected

    def test_str_enum(self):
        assert ListingType.INTERNSHIP == "internship"
        assert ListingType.ENTRY_LEVEL == "entry_level"
        assert isinstance(ListingType.INTERNSHIP, str)

    def test_from_value(self):
        assert ListingType("internship") is ListingType.INTERNSHIP
        assert ListingType("entry_level") is ListingType.ENTRY_LEVEL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ListingType("contractor")


# ── JobListing with listing_type Tests ─────────────────────────────────────


class TestJobListingListingType:
    def test_default_listing_type_is_internship(self):
        listing = JobListing(
            id="abc123",
            company="TestCo",
            company_slug="testco",
            role="Software Engineer Intern",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://example.com/apply",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
        )
        assert listing.listing_type == ListingType.INTERNSHIP

    def test_entry_level_listing_type(self):
        listing = JobListing(
            id="def456",
            company="TestCo",
            company_slug="testco",
            role="Junior Software Engineer",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://example.com/apply",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            listing_type=ListingType.ENTRY_LEVEL,
        )
        assert listing.listing_type == ListingType.ENTRY_LEVEL

    def test_serialization_roundtrip(self):
        listing = JobListing(
            id="abc123",
            company="TestCo",
            company_slug="testco",
            role="Junior Engineer",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://example.com/apply",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            listing_type=ListingType.ENTRY_LEVEL,
        )
        data = listing.model_dump(mode="json")
        assert data["listing_type"] == "entry_level"

        restored = JobListing.model_validate(data)
        assert restored.listing_type == ListingType.ENTRY_LEVEL

    def test_backward_compatibility_without_listing_type(self):
        """Existing data without listing_type should default to INTERNSHIP."""
        data = {
            "id": "abc123",
            "company": "TestCo",
            "company_slug": "testco",
            "role": "SWE Intern",
            "category": "swe",
            "locations": ["Atlanta, GA"],
            "apply_url": "https://example.com/apply",
            "date_added": "2026-03-01",
            "date_last_verified": "2026-03-01",
            "source": "greenhouse_api",
        }
        listing = JobListing.model_validate(data)
        assert listing.listing_type == ListingType.INTERNSHIP


# ── RawListing with listing_type Tests ─────────────────────────────────────


class TestRawListingListingType:
    def test_default_listing_type_is_internship(self):
        raw = RawListing(
            company="TestCo",
            company_slug="testco",
            title="SWE Intern",
            location="Atlanta, GA",
            url="https://example.com/apply",
            source="greenhouse_api",
        )
        assert raw.listing_type == "internship"

    def test_entry_level_listing_type(self):
        raw = RawListing(
            company="TestCo",
            company_slug="testco",
            title="Junior SWE",
            location="Atlanta, GA",
            url="https://example.com/apply",
            source="greenhouse_api",
            listing_type="entry_level",
        )
        assert raw.listing_type == "entry_level"

    def test_serialization_roundtrip(self):
        raw = RawListing(
            company="TestCo",
            company_slug="testco",
            title="Junior SWE",
            location="Atlanta, GA",
            url="https://example.com/apply",
            source="greenhouse_api",
            listing_type="entry_level",
        )
        data = raw.model_dump(mode="json")
        assert data["listing_type"] == "entry_level"

        restored = RawListing.model_validate(data)
        assert restored.listing_type == "entry_level"


# ── Entry-Level Validation Tests ───────────────────────────────────────────


class TestEntryLevelValidation:
    @pytest.fixture
    def raw_entry_level_listing(self):
        return RawListing(
            company="Google",
            company_slug="google",
            title="Junior Software Engineer",
            location="Atlanta, GA",
            url="https://careers.google.com/jobs/123",
            source="greenhouse_api",
            listing_type="entry_level",
        )

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.filters.role_categories = {"swe": ["software engineer"]}
        config.company_industries = {}
        config.big_tech_companies = ["Google"]
        config.ai.entry_level_enrichment_prompt = "test prompt"
        config.entry_level_filters.keywords_include = ["junior", "entry level"]
        config.entry_level_filters.keywords_exclude = ["intern", "internship"]
        return config

    def test_build_entry_level_listing(self, raw_entry_level_listing):
        from scripts.el_validate import _build_entry_level_listing

        metadata = {
            "is_entry_level": True,
            "category": "swe",
            "locations": ["Atlanta, GA"],
            "sponsorship": "unknown",
            "remote_friendly": False,
            "open_to_international": False,
            "tech_stack": ["Python"],
            "industry": "cloud",
            "confidence": 0.95,
        }

        job = _build_entry_level_listing(raw_entry_level_listing, metadata)
        assert job.listing_type == ListingType.ENTRY_LEVEL
        assert job.season == "n/a"
        assert job.category == RoleCategory.SWE
        assert job.company == "Google"

    def test_build_entry_level_listing_is_faang(self, raw_entry_level_listing):
        from scripts.el_validate import _build_entry_level_listing

        metadata = {
            "category": "swe",
            "locations": ["Atlanta, GA"],
            "sponsorship": "unknown",
            "industry": "other",
            "confidence": 0.9,
        }

        with patch("scripts.el_validate.is_big_tech", return_value=True):
            job = _build_entry_level_listing(raw_entry_level_listing, metadata)
            assert job.is_faang_plus is True


# ── Entry-Level Discovery Tests ────────────────────────────────────────────


class TestEntryLevelDiscovery:
    def test_entry_level_filters_adapter(self):
        from scripts.el_discover import _EntryLevelFilters

        config = MagicMock()
        config.entry_level_filters.keywords_include = ["junior", "entry level"]
        config.entry_level_filters.keywords_exclude = ["intern"]
        config.filters.role_categories = {"swe": ["software"]}
        config.filters.exclude_companies = ["Revature"]

        filters = _EntryLevelFilters(config)
        assert filters.keywords_include == ["junior", "entry level"]
        assert filters.keywords_exclude == ["intern"]
        assert filters.role_categories == {"swe": ["software"]}
        assert filters.exclude_companies == ["Revature"]



# ── Config Entry-Level Tests ───────────────────────────────────────────────


class TestEntryLevelConfig:
    def test_entry_level_filters_config_loads(self):
        from scripts.utils.config import AppConfig

        config_dict = {
            "project": {
                "name": "Test",
                "season": "summer_2026",
                "github_repo": "test/repo",
            },
            "entry_level_filters": {
                "keywords_include": ["junior", "entry level"],
                "keywords_exclude": ["intern", "senior"],
            },
        }
        config = AppConfig.model_validate(config_dict)
        assert "junior" in config.entry_level_filters.keywords_include
        assert "intern" in config.entry_level_filters.keywords_exclude

    def test_entry_level_filters_defaults_to_empty(self):
        from scripts.utils.config import AppConfig

        config_dict = {
            "project": {
                "name": "Test",
                "season": "summer_2026",
                "github_repo": "test/repo",
            },
        }
        config = AppConfig.model_validate(config_dict)
        assert config.entry_level_filters.keywords_include == []
        assert config.entry_level_filters.keywords_exclude == []

    def test_ai_config_has_entry_level_prompt(self):
        from scripts.utils.config import AIConfig

        ai = AIConfig(entry_level_enrichment_prompt="Analyze entry-level job")
        assert ai.entry_level_enrichment_prompt == "Analyze entry-level job"

    def test_ai_config_entry_level_prompt_defaults_empty(self):
        from scripts.utils.config import AIConfig

        ai = AIConfig()
        assert ai.entry_level_enrichment_prompt == ""


