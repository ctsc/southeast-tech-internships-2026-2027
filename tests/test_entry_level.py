"""Tests for entry-level job pipeline: models, discovery, validation, and README rendering."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
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


# ── README Renderer Entry-Level Section Tests ──────────────────────────────


class TestReadmeEntryLevelSection:
    @pytest.fixture
    def entry_level_listing(self):
        return JobListing(
            id="el001",
            company="Google",
            company_slug="google",
            role="Junior Software Engineer",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://careers.google.com/jobs/456",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            status=ListingStatus.OPEN,
            is_faang_plus=True,
            listing_type=ListingType.ENTRY_LEVEL,
            season="n/a",
        )

    @pytest.fixture
    def non_ga_entry_level_listing(self):
        return JobListing(
            id="el002",
            company="Meta",
            company_slug="meta",
            role="Associate Software Engineer",
            category=RoleCategory.SWE,
            locations=["Menlo Park, CA"],
            apply_url="https://meta.com/jobs/789",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            status=ListingStatus.OPEN,
            listing_type=ListingType.ENTRY_LEVEL,
            season="n/a",
        )

    def test_render_entry_level_section_with_ga_listings(self, entry_level_listing):
        from scripts.utils.readme_renderer import _render_entry_level_section

        result = _render_entry_level_section([entry_level_listing])
        assert "Entry-Level Jobs in GA" in result
        assert "Google" in result
        assert "Junior Software Engineer" in result
        assert "Apply" in result

    def test_render_entry_level_section_empty(self):
        from scripts.utils.readme_renderer import _render_entry_level_section

        result = _render_entry_level_section([])
        assert "Entry-Level Jobs in GA" in result
        assert "No entry-level listings yet" in result

    def test_render_entry_level_section_filters_non_ga(
        self, entry_level_listing, non_ga_entry_level_listing
    ):
        from scripts.utils.readme_renderer import _render_entry_level_section

        result = _render_entry_level_section(
            [entry_level_listing, non_ga_entry_level_listing]
        )
        assert "Google" in result
        assert "Meta" not in result

    def test_entry_level_row_no_season_column(self, entry_level_listing):
        from scripts.utils.readme_renderer import _format_entry_level_row

        row = _format_entry_level_row(entry_level_listing)
        # Entry-level rows should not have season badges
        assert "S26" not in row
        assert "F26" not in row
        # Should have company, role, location, apply, posted
        assert "Google" in row
        assert "Apply" in row

    def test_entry_level_row_closed(self, entry_level_listing):
        entry_level_listing.status = ListingStatus.CLOSED
        from scripts.utils.readme_renderer import _format_entry_level_row

        row = _format_entry_level_row(entry_level_listing)
        assert "Closed" in row

    def test_entry_level_row_flags(self, entry_level_listing):
        entry_level_listing.remote_friendly = True
        entry_level_listing.open_to_international = True
        from scripts.utils.readme_renderer import _format_entry_level_row

        row = _format_entry_level_row(entry_level_listing)
        assert "🏠" in row
        assert "🌍" in row


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


# ── CLI --entry-level Tests ────────────────────────────────────────────────


class TestCLIEntryLevel:
    def test_parse_entry_level_flag(self):
        from main import parse_args

        args = parse_args(["--entry-level"])
        assert args.entry_level is True

    def test_entry_level_mutually_exclusive_with_full(self):
        from main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--entry-level", "--full"])

    def test_entry_level_mutually_exclusive_with_discover(self):
        from main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--entry-level", "--discover-only"])

    def test_main_dispatches_to_entry_level(self):
        from main import main

        with patch("main.run_entry_level_pipeline") as mock_run:
            with patch("main._setup_logging"):
                main(["--entry-level"])
                mock_run.assert_called_once()


# ── Stats Table Entry-Level Count Tests ────────────────────────────────────


class TestStatsTableEntryLevel:
    def test_readme_includes_entry_level_stats_row(self):
        from scripts.utils.readme_renderer import render_readme

        jobs_db = JobsDatabase(
            listings=[],
            last_updated=datetime(2026, 3, 1, 12, 0, 0),
            total_open=0,
        )

        with patch("scripts.utils.readme_renderer.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.project.github_repo = "test/repo"
            with patch("scripts.utils.readme_renderer._load_entry_level_db", return_value=None):
                readme = render_readme(jobs_db)

        assert "Entry-Level Jobs in GA" in readme

    def test_readme_includes_entry_level_section_when_data_exists(self):
        from scripts.utils.readme_renderer import render_readme

        jobs_db = JobsDatabase(
            listings=[],
            last_updated=datetime(2026, 3, 1, 12, 0, 0),
            total_open=0,
        )

        el_listing = JobListing(
            id="el001",
            company="Google",
            company_slug="google",
            role="Junior SWE",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://example.com/apply",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            status=ListingStatus.OPEN,
            listing_type=ListingType.ENTRY_LEVEL,
            season="n/a",
        )

        el_db = JobsDatabase(
            listings=[el_listing],
            last_updated=datetime(2026, 3, 1, 12, 0, 0),
            total_open=1,
        )

        with patch("scripts.utils.readme_renderer.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.project.github_repo = "test/repo"
            with patch("scripts.utils.readme_renderer._load_entry_level_db", return_value=el_db):
                readme = render_readme(jobs_db)

        # Should have both stats row and section
        assert "Entry-Level Jobs in GA" in readme
        assert "Junior SWE" in readme

    def test_readme_does_not_mix_entry_level_into_internship_sections(self):
        from scripts.utils.readme_renderer import render_readme

        internship_listing = JobListing(
            id="int001",
            company="Meta",
            company_slug="meta",
            role="SWE Intern",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://meta.com/intern",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            status=ListingStatus.OPEN,
            listing_type=ListingType.INTERNSHIP,
        )

        jobs_db = JobsDatabase(
            listings=[internship_listing],
            last_updated=datetime(2026, 3, 1, 12, 0, 0),
            total_open=1,
        )

        el_listing = JobListing(
            id="el001",
            company="Google",
            company_slug="google",
            role="Junior SWE",
            category=RoleCategory.SWE,
            locations=["Atlanta, GA"],
            apply_url="https://example.com/apply",
            date_added=date(2026, 3, 1),
            date_last_verified=date(2026, 3, 1),
            source="greenhouse_api",
            status=ListingStatus.OPEN,
            listing_type=ListingType.ENTRY_LEVEL,
            season="n/a",
        )

        el_db = JobsDatabase(
            listings=[el_listing],
            last_updated=datetime(2026, 3, 1, 12, 0, 0),
            total_open=1,
        )

        with patch("scripts.utils.readme_renderer.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.project.github_repo = "test/repo"
            with patch("scripts.utils.readme_renderer._load_entry_level_db", return_value=el_db):
                readme = render_readme(jobs_db)

        # The entry-level section should contain Junior SWE but not SWE Intern
        lines = readme.split("\n")
        el_section_started = False
        el_section_lines: list[str] = []
        for line in lines:
            if "## 🍑 Entry-Level Jobs in GA" in line:
                el_section_started = True
                continue
            if el_section_started:
                # Section ends at next --- or ## heading
                if line.strip() == "---" or (line.startswith("## ") and "Entry-Level" not in line):
                    break
                el_section_lines.append(line)

        el_section_text = "\n".join(el_section_lines)
        assert "Junior SWE" in el_section_text
        assert "SWE Intern" not in el_section_text
