"""Tests for README generation: renderer and generate_readme script."""

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

from scripts.utils.models import (
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
)
from scripts.utils.readme_renderer import (
    _count_open,
    _format_listing_row,
    _format_locations,
    _format_relative_date,
    _is_southeast_listing,
    _render_category_section,
    _render_southeast_graduate_section,
    _render_southeast_section,
    render_readme,
)
from scripts.generate_readme import (
    generate_readme,
    load_database,
    validate_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_listing(**overrides) -> JobListing:
    """Create a JobListing with sensible defaults, overridden by kwargs."""
    defaults = {
        "id": "test123",
        "company": "TestCo",
        "company_slug": "testco",
        "role": "Software Engineer Intern",
        "category": RoleCategory.SWE,
        "locations": ["San Francisco, CA"],
        "apply_url": "https://example.com/apply",
        "sponsorship": SponsorshipStatus.UNKNOWN,
        "requires_us_citizenship": False,
        "is_faang_plus": False,
        "requires_advanced_degree": False,
        "remote_friendly": False,
        "date_added": date(2026, 2, 15),
        "date_last_verified": date(2026, 2, 20),
        "source": "greenhouse_api",
        "status": ListingStatus.OPEN,
        "tech_stack": [],
        "season": "summer_2026",
    }
    defaults.update(overrides)
    return JobListing(**defaults)


def _make_db(listings: list[JobListing] | None = None) -> JobsDatabase:
    """Create a JobsDatabase with given listings."""
    db = JobsDatabase(
        listings=listings or [],
        last_updated=datetime(2026, 2, 20, 12, 0, 0),
    )
    db.compute_stats()
    return db


# Mock config for all render_readme tests
_MOCK_CONFIG = MagicMock()
_MOCK_CONFIG.georgia_focus.georgia_section_in_readme = True
_MOCK_CONFIG.georgia_focus.priority_locations = ["Atlanta, GA", "Alpharetta, GA"]
_MOCK_CONFIG.georgia_focus.highlight_georgia = True
_MOCK_CONFIG.project.github_repo = "ctsc/atlanta-tech-internships-2026"


# ---------------------------------------------------------------------------
# _format_locations
# ---------------------------------------------------------------------------

class TestFormatLocations:
    def test_empty_locations(self):
        assert _format_locations([]) == "Unknown"

    def test_single_location(self):
        assert _format_locations(["NYC"]) == "NYC"

    def test_two_locations(self):
        assert _format_locations(["NYC", "SF"]) == "NYC, SF"

    def test_three_locations(self):
        assert _format_locations(["NYC", "SF", "LA"]) == "NYC, SF, LA"

    def test_four_locations_truncated(self):
        result = _format_locations(["NYC", "SF", "LA", "Chicago"])
        assert result == "NYC, SF, LA and 1 more"

    def test_six_locations_truncated(self):
        locs = ["NYC", "SF", "LA", "Chicago", "Boston", "Seattle"]
        result = _format_locations(locs)
        assert result == "NYC, SF, LA and 3 more"

    def test_custom_max_display(self):
        locs = ["NYC", "SF", "LA", "Chicago"]
        result = _format_locations(locs, max_display=2)
        assert result == "NYC, SF and 2 more"


# ---------------------------------------------------------------------------
# _format_relative_date
# ---------------------------------------------------------------------------

class TestFormatRelativeDate:
    def test_today(self):
        assert _format_relative_date(date.today()) == "today"

    def test_one_day_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=1)) == "1d ago"

    def test_five_days_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=5)) == "5d ago"

    def test_one_week_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=7)) == "1w ago"

    def test_two_weeks_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=14)) == "2w ago"

    def test_one_month_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=30)) == "1mo ago"

    def test_three_months_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=90)) == "3mo ago"

    def test_one_year_ago(self):
        assert _format_relative_date(date.today() - timedelta(days=365)) == "1y ago"

    def test_future_date_shows_today(self):
        assert _format_relative_date(date.today() + timedelta(days=5)) == "today"


# ---------------------------------------------------------------------------
# _format_listing_row
# ---------------------------------------------------------------------------

class TestFormatListingRow:
    def test_basic_open_listing(self):
        listing = _make_listing()
        row = _format_listing_row(listing)
        assert "**TestCo**" in row
        assert "Software Engineer Intern" in row
        assert "[Apply](" in row
        # Date column uses relative time
        assert "ago" in row or "today" in row

    def test_faang_plus_emoji(self):
        listing = _make_listing(is_faang_plus=True)
        row = _format_listing_row(listing)
        assert "🔥 **TestCo**" in row

    def test_no_industry_emoji_in_row(self):
        listing = _make_listing()
        row = _format_listing_row(listing)
        # Should not have any industry emoji prefix before company name
        assert row.startswith("| **TestCo**") or "| **TestCo**" in row
        # Specifically check no emoji appears right before the bold company name
        assert "🏷️" not in row
        assert "💳" not in row
        assert "🧠" not in row

    def test_no_sponsorship_no_emoji(self):
        listing = _make_listing(sponsorship=SponsorshipStatus.NO_SPONSORSHIP)
        row = _format_listing_row(listing)
        assert "🛂" not in row

    def test_us_citizenship_no_emoji(self):
        listing = _make_listing(requires_us_citizenship=True)
        row = _format_listing_row(listing)
        assert "🇺🇸" not in row

    def test_closed_listing_emoji(self):
        listing = _make_listing(status=ListingStatus.CLOSED)
        row = _format_listing_row(listing)
        assert "🔒" in row
        assert "🔒 Closed" in row
        assert "[Apply](" not in row

    def test_advanced_degree_emoji(self):
        listing = _make_listing(requires_advanced_degree=True)
        row = _format_listing_row(listing)
        assert "🎓" in row

    def test_remote_friendly_emoji(self):
        listing = _make_listing(remote_friendly=True)
        row = _format_listing_row(listing)
        assert "🏠" in row

    def test_multiple_flags(self):
        listing = _make_listing(
            is_faang_plus=True,
            requires_advanced_degree=True,
            remote_friendly=True,
        )
        row = _format_listing_row(listing)
        assert "🔥" in row
        assert "🎓" in row
        assert "🏠" in row

    def test_row_is_pipe_delimited(self):
        listing = _make_listing()
        row = _format_listing_row(listing)
        assert row.startswith("|")
        assert row.endswith("|")
        # 6 columns = 7 pipes
        assert row.count("|") == 7

    def test_season_badge_in_row(self):
        listing = _make_listing(season="summer_2026")
        row = _format_listing_row(listing)
        assert "S26" in row

    def test_fall_season_badge(self):
        listing = _make_listing(season="fall_2026")
        row = _format_listing_row(listing)
        assert "F26" in row


# ---------------------------------------------------------------------------
# _count_open
# ---------------------------------------------------------------------------

class TestCountOpen:
    def test_empty_list(self):
        assert _count_open([], RoleCategory.SWE) == 0

    def test_counts_only_open(self):
        listings = [
            _make_listing(category=RoleCategory.SWE, status=ListingStatus.OPEN),
            _make_listing(id="closed1", category=RoleCategory.SWE, status=ListingStatus.CLOSED),
            _make_listing(id="open2", category=RoleCategory.SWE, status=ListingStatus.OPEN),
        ]
        assert _count_open(listings, RoleCategory.SWE) == 2

    def test_counts_correct_category(self):
        listings = [
            _make_listing(category=RoleCategory.SWE, status=ListingStatus.OPEN),
            _make_listing(id="ml1", category=RoleCategory.ML_AI, status=ListingStatus.OPEN),
        ]
        assert _count_open(listings, RoleCategory.SWE) == 1
        assert _count_open(listings, RoleCategory.ML_AI) == 1


# ---------------------------------------------------------------------------
# _render_category_section
# ---------------------------------------------------------------------------

class TestRenderCategorySection:
    def test_empty_category(self):
        section = _render_category_section(RoleCategory.SWE, "💻", "Software Engineering", [])
        assert "## 💻 Software Engineering" in section
        assert "No listings yet" in section

    def test_with_listings(self):
        listings = [_make_listing(), _make_listing(id="x2", role="Backend Intern")]
        section = _render_category_section(RoleCategory.SWE, "💻", "Software Engineering", listings)
        assert "| Company | Role | Location | Season | Apply | Posted |" in section
        assert "|---------|------|----------|--------|-------|------------|" in section
        assert "**TestCo**" in section

    def test_sorted_by_date_desc(self):
        older = _make_listing(id="old", role="Old Intern", date_added=date(2026, 1, 1))
        newer = _make_listing(id="new", role="New Intern", date_added=date(2026, 2, 20))
        section = _render_category_section(RoleCategory.SWE, "💻", "Software Engineering", [older, newer])
        # Newer should appear before older
        new_pos = section.index("New Intern")
        old_pos = section.index("Old Intern")
        assert new_pos < old_pos


# ---------------------------------------------------------------------------
# render_readme — full integration
# ---------------------------------------------------------------------------

class TestRenderReadme:
    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_empty_database(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "# Atlanta Tech Internships" in readme
        assert "### 📊 Stats" in readme
        assert "### Legend" in readme
        assert "## How This Works" in readme
        assert "**Total** | **0**" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_single_swe_listing(self, mock_config):
        db = _make_db([_make_listing()])
        readme = render_readme(db)
        assert "**TestCo**" in readme
        assert "Software Engineer Intern" in readme
        assert "[Apply](" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_multiple_categories(self, mock_config):
        listings = [
            _make_listing(id="swe1", category=RoleCategory.SWE),
            _make_listing(id="ml1", category=RoleCategory.ML_AI, role="ML Intern"),
            _make_listing(id="pm1", category=RoleCategory.PM, role="PM Intern"),
        ]
        db = _make_db(listings)
        readme = render_readme(db)
        assert "## 💻 Software Engineering" in readme
        assert "## 🤖 ML / AI / Data Science" in readme
        assert "## 📱 Product Management" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_stats_counts_correct(self, mock_config):
        listings = [
            _make_listing(id="s1", category=RoleCategory.SWE),
            _make_listing(id="s2", category=RoleCategory.SWE),
            _make_listing(id="m1", category=RoleCategory.ML_AI),
        ]
        db = _make_db(listings)
        readme = render_readme(db)
        assert "**Total** | **3**" in readme
        # SWE count should be 2
        assert "| 2 |" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_header_with_timestamp(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "Last updated: February 20, 2026" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_header_has_carter_bio(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "Carter" in readme
        assert "IEEE" in readme
        assert "Georgia State" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_legend_table_present(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "| Symbol | Meaning |" in readme
        assert "| 🔥 | FAANG+ company |" in readme
        assert "🛂" not in readme
        assert "🇺🇸" not in readme
        assert "| 🔒 | Application closed |" in readme
        assert "| 🎓 | Advanced degree required |" in readme
        assert "| 🏠 | Remote friendly |" in readme
        assert "| S26 | Summer 2026 |" in readme
        assert "| F26 | Fall 2026 |" in readme
        assert "| Sp27 | Spring 2027 |" in readme
        assert "| S27 | Summer 2027 |" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_no_georgia_section(self, mock_config):
        listings = [_make_listing(locations=["Atlanta, GA"])]
        db = _make_db(listings)
        readme = render_readme(db)
        assert "## 🍑 Georgia Internships" not in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_contributing_section(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "## Contributing" in readme
        assert "Submit an issue" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_star_repo_footer(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "Star this repo" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_how_this_works_section(self, mock_config):
        db = _make_db([])
        readme = render_readme(db)
        assert "## How This Works" in readme
        assert "automatically maintained by AI" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_company_names_bold(self, mock_config):
        db = _make_db([_make_listing(company="Anthropic")])
        readme = render_readme(db)
        assert "**Anthropic**" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_apply_link_format(self, mock_config):
        db = _make_db([_make_listing(apply_url="https://example.com/job/123")])
        readme = render_readme(db)
        assert "[Apply](https://example.com/job/123)" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_closed_listing_shows_closed_instead_of_apply(self, mock_config):
        db = _make_db([_make_listing(status=ListingStatus.CLOSED)])
        readme = render_readme(db)
        assert "🔒 Closed" in readme
        assert "[Apply](" not in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_other_category_hidden_when_empty(self, mock_config):
        db = _make_db([_make_listing(category=RoleCategory.SWE)])
        readme = render_readme(db)
        assert "## 🔹 Other" not in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_other_category_shown_when_populated(self, mock_config):
        db = _make_db([_make_listing(category=RoleCategory.OTHER)])
        readme = render_readme(db)
        assert "## 🔹 Other" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_locations_truncated_in_row(self, mock_config):
        listing = _make_listing(locations=["NYC", "SF", "LA", "Chicago", "Seattle"])
        db = _make_db([listing])
        readme = render_readme(db)
        assert "and 2 more" in readme

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_sort_newest_first(self, mock_config):
        older = _make_listing(id="old", role="Old Role", date_added=date(2026, 1, 1))
        newer = _make_listing(id="new", role="New Role", date_added=date(2026, 2, 25))
        db = _make_db([older, newer])
        readme = render_readme(db)
        new_pos = readme.index("New Role")
        old_pos = readme.index("Old Role")
        assert new_pos < old_pos

    def test_render_readme_config_failure_uses_defaults(self):
        """If config fails to load, renderer should still produce output."""
        with patch("scripts.utils.readme_renderer.get_config", side_effect=Exception("no config")):
            db = _make_db([])
            readme = render_readme(db)
            assert "# Atlanta Tech Internships" in readme


# ---------------------------------------------------------------------------
# validate_markdown
# ---------------------------------------------------------------------------

class TestValidateMarkdown:
    def test_empty_string_fails(self):
        assert validate_markdown("") is False

    def test_whitespace_only_fails(self):
        assert validate_markdown("   \n  ") is False

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_valid_readme_passes(self, mock_config):
        db = _make_db([_make_listing()])
        content = render_readme(db)
        assert validate_markdown(content) is True

    def test_missing_header_fails(self):
        content = "### Legend\n## How This Works\n"
        assert validate_markdown(content) is False

    def test_missing_legend_fails(self):
        content = "# Atlanta Tech Internships\n## How This Works\n"
        assert validate_markdown(content) is False

    def test_inconsistent_table_pipes_fails(self):
        content = (
            "# Atlanta Tech Internships\n"
            "### Legend\n"
            "## How This Works\n"
            "| A | B | C |\n"
            "| D | E |\n"
        )
        assert validate_markdown(content) is False


# ---------------------------------------------------------------------------
# load_database
# ---------------------------------------------------------------------------

class TestLoadDatabase:
    def test_load_missing_file_returns_empty(self, tmp_path):
        db = load_database(tmp_path / "nonexistent.json")
        assert len(db.listings) == 0

    def test_load_valid_file(self, tmp_path):
        jobs_file = tmp_path / "jobs.json"
        data = {
            "listings": [
                {
                    "id": "h1",
                    "company": "Acme",
                    "company_slug": "acme",
                    "role": "SWE Intern",
                    "category": "swe",
                    "locations": ["NYC"],
                    "apply_url": "https://example.com/apply",
                    "date_added": "2026-02-01",
                    "date_last_verified": "2026-02-20",
                    "source": "greenhouse_api",
                }
            ],
            "last_updated": "2026-02-20T12:00:00",
            "total_open": 0,
        }
        jobs_file.write_text(json.dumps(data), encoding="utf-8")
        db = load_database(jobs_file)
        assert len(db.listings) == 1
        assert db.listings[0].company == "Acme"
        assert db.total_open == 1  # compute_stats recalculates


# ---------------------------------------------------------------------------
# generate_readme (end-to-end)
# ---------------------------------------------------------------------------

class TestGenerateReadme:
    @patch("scripts.generate_readme.load_database")
    @patch("scripts.generate_readme.render_readme")
    def test_writes_file(self, mock_render, mock_load, tmp_path):
        mock_load.return_value = _make_db([])
        mock_render.return_value = (
            "# Atlanta Tech Internships\n### Legend\n## How This Works\n"
        )
        readme_path = tmp_path / "README.md"
        generate_readme(
            jobs_path=tmp_path / "jobs.json",
            readme_path=readme_path,
        )
        assert readme_path.exists()
        assert "Atlanta Tech Internships" in readme_path.read_text(encoding="utf-8")

    @patch("scripts.generate_readme.render_readme")
    @patch("scripts.generate_readme.load_database")
    def test_returns_content(self, mock_load, mock_render, tmp_path):
        mock_load.return_value = _make_db([])
        expected = "# Atlanta Tech Internships\n### Legend\n## How This Works\n"
        mock_render.return_value = expected
        result = generate_readme(
            jobs_path=tmp_path / "jobs.json",
            readme_path=tmp_path / "README.md",
        )
        assert result == expected

    @patch("scripts.utils.readme_renderer.get_config", return_value=_MOCK_CONFIG)
    def test_full_end_to_end(self, mock_config, tmp_path):
        """Full pipeline: write jobs.json, generate README, verify output."""
        jobs_file = tmp_path / "jobs.json"
        data = {
            "listings": [
                {
                    "id": "e2e1",
                    "company": "Stripe",
                    "company_slug": "stripe",
                    "role": "Backend Intern",
                    "category": "swe",
                    "locations": ["Seattle, WA"],
                    "apply_url": "https://stripe.com/apply",
                    "date_added": "2026-02-10",
                    "date_last_verified": "2026-02-20",
                    "source": "greenhouse_api",
                    "status": "open",
                }
            ],
            "last_updated": "2026-02-20T12:00:00",
            "total_open": 1,
        }
        jobs_file.write_text(json.dumps(data), encoding="utf-8")
        readme_path = tmp_path / "README.md"

        result = generate_readme(jobs_path=jobs_file, readme_path=readme_path)

        assert "**Stripe**" in result
        assert "Backend Intern" in result
        assert "[Apply](https://stripe.com/apply)" in result
        assert readme_path.exists()
        assert validate_markdown(result) is True


# ---------------------------------------------------------------------------
# Industry Emoji Tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _is_southeast_listing
# ---------------------------------------------------------------------------

class TestIsSoutheastListing:
    def test_florida_match(self):
        listing = _make_listing(locations=["Miami, FL"])
        assert _is_southeast_listing(listing) is True

    def test_texas_city_match(self):
        listing = _make_listing(locations=["Austin, TX"])
        assert _is_southeast_listing(listing) is True

    def test_north_carolina_match(self):
        listing = _make_listing(locations=["Charlotte, NC"])
        assert _is_southeast_listing(listing) is True

    def test_georgia_match(self):
        listing = _make_listing(locations=["Atlanta, GA"])
        assert _is_southeast_listing(listing) is True

    def test_tennessee_match(self):
        listing = _make_listing(locations=["Nashville, TN"])
        assert _is_southeast_listing(listing) is True

    def test_alabama_match(self):
        listing = _make_listing(locations=["Huntsville, AL"])
        assert _is_southeast_listing(listing) is True

    def test_south_carolina_match(self):
        listing = _make_listing(locations=["Charleston, SC"])
        assert _is_southeast_listing(listing) is True

    def test_non_southeast_rejected(self):
        listing = _make_listing(locations=["San Francisco, CA"])
        assert _is_southeast_listing(listing) is False

    def test_non_southeast_northeast(self):
        listing = _make_listing(locations=["Boston, MA"])
        assert _is_southeast_listing(listing) is False

    def test_empty_locations(self):
        listing = _make_listing(locations=[])
        assert _is_southeast_listing(listing) is False

    def test_multiple_locations_one_se(self):
        listing = _make_listing(locations=["NYC", "Orlando, FL"])
        assert _is_southeast_listing(listing) is True

    def test_state_name_match(self):
        listing = _make_listing(locations=["Tennessee"])
        assert _is_southeast_listing(listing) is True


# ---------------------------------------------------------------------------
# _render_southeast_section
# ---------------------------------------------------------------------------

class TestRenderSoutheastSection:
    def test_empty_when_no_se_listings(self):
        listings = [_make_listing(locations=["San Francisco, CA"])]
        section = _render_southeast_section(listings)
        assert "## 🌴 Southeast Internships" in section
        assert "No Southeast-based listings" in section

    def test_with_se_listings(self):
        listings = [_make_listing(locations=["Atlanta, GA"])]
        section = _render_southeast_section(listings)
        assert "| Company | Role | Location | Season | Apply | Posted |" in section
        assert "**TestCo**" in section

    def test_excludes_closed_listings(self):
        closed = _make_listing(locations=["Miami, FL"], status=ListingStatus.CLOSED)
        section = _render_southeast_section([closed])
        assert "No Southeast-based listings" in section

    def test_includes_multiple_states(self):
        listings = [
            _make_listing(id="ga1", locations=["Atlanta, GA"]),
            _make_listing(id="fl1", locations=["Miami, FL"]),
            _make_listing(id="nc1", locations=["Charlotte, NC"]),
        ]
        section = _render_southeast_section(listings)
        # All three should be in the section
        assert section.count("**TestCo**") == 3


class TestReadmeHeader:
    @patch("scripts.utils.readme_renderer.get_config")
    def test_header_contains_georgia_southeast(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing()])
        readme = render_readme(db)
        assert "Catered to Georgia / Southeast" in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_header_contains_star_message(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing()])
        readme = render_readme(db)
        assert "Leave a star on the repo" in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_legend_has_no_industry_section(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing()])
        readme = render_readme(db)
        assert "**Industry**" not in readme
        assert "| 💳 | Fintech |" not in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_stats_table_has_no_georgia_link(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing(locations=["Atlanta, GA"])])
        readme = render_readme(db)
        assert "🍑" not in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_stats_table_has_se_graduate_link(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing(locations=["Atlanta, GA"], graduate_friendly=True)])
        readme = render_readme(db)
        assert "🎓 [SE Graduate & PhD](#-southeast-graduate--phd-internships)" in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_stats_table_has_southeast_link(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        db = _make_db([_make_listing(locations=["Miami, FL"])])
        readme = render_readme(db)
        assert "🌴 [Southeast Internships](#-southeast-internships)" in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_southeast_section_present(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        listings = [_make_listing(locations=["Orlando, FL"])]
        db = _make_db(listings)
        readme = render_readme(db)
        assert "## 🌴 Southeast Internships" in readme

    @patch("scripts.utils.readme_renderer.get_config")
    def test_se_graduate_section_present(self, mock_config):
        mock_config.return_value = _MOCK_CONFIG
        listings = [_make_listing(locations=["Atlanta, GA"], graduate_friendly=True)]
        db = _make_db(listings)
        readme = render_readme(db)
        assert "## 🎓 Southeast Graduate & PhD Internships" in readme


# ---------------------------------------------------------------------------
# _render_southeast_graduate_section
# ---------------------------------------------------------------------------

class TestRenderSoutheastGraduateSection:
    def test_empty_when_no_grad_se_listings(self):
        listings = [_make_listing(locations=["Atlanta, GA"])]
        section = _render_southeast_graduate_section(listings)
        assert "## 🎓 Southeast Graduate & PhD Internships" in section
        assert "No graduate-friendly Southeast listings" in section

    def test_with_graduate_friendly_listing(self):
        listings = [_make_listing(locations=["Atlanta, GA"], graduate_friendly=True)]
        section = _render_southeast_graduate_section(listings)
        assert "| Company | Role | Location | Season | Apply | Posted |" in section
        assert "**TestCo**" in section

    def test_with_advanced_degree_listing(self):
        listings = [_make_listing(locations=["Miami, FL"], requires_advanced_degree=True)]
        section = _render_southeast_graduate_section(listings)
        assert "**TestCo**" in section

    def test_excludes_non_se_graduate_listings(self):
        listings = [_make_listing(locations=["San Francisco, CA"], graduate_friendly=True)]
        section = _render_southeast_graduate_section(listings)
        assert "No graduate-friendly Southeast listings" in section

    def test_excludes_closed_listings(self):
        closed = _make_listing(
            locations=["Atlanta, GA"],
            graduate_friendly=True,
            status=ListingStatus.CLOSED,
        )
        section = _render_southeast_graduate_section([closed])
        assert "No graduate-friendly Southeast listings" in section

    def test_includes_both_grad_types(self):
        listings = [
            _make_listing(id="gf1", locations=["Atlanta, GA"], graduate_friendly=True),
            _make_listing(id="ad1", locations=["Nashville, TN"], requires_advanced_degree=True),
        ]
        section = _render_southeast_graduate_section(listings)
        assert section.count("**TestCo**") == 2
