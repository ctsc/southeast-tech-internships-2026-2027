"""Tests for config loader and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.utils.config import (
    AIConfig,
    AppConfig,
    AshbyBoard,
    FiltersConfig,
    GeorgiaFocusConfig,
    GitHubMonitor,
    GreenhouseBoard,
    LeverBoard,
    ProjectConfig,
    ScheduleConfig,
    ScrapeSource,
    is_big_tech,
    load_config,
)


# ── Section Model Tests ─────────────────────────────────────────────────────


class TestProjectConfig:
    def test_valid(self):
        p = ProjectConfig(name="Test", season="summer_2026", github_repo="a/b")
        assert p.name == "Test"

    def test_active_seasons_default(self):
        p = ProjectConfig(name="Test", season="summer_2026", github_repo="a/b")
        assert p.active_seasons == ["summer_2026"]

    def test_active_seasons_custom(self):
        p = ProjectConfig(
            name="Test", season="summer_2026", github_repo="a/b",
            active_seasons=["summer_2026", "fall_2026", "spring_2027", "summer_2027"],
        )
        assert len(p.active_seasons) == 4
        assert "fall_2026" in p.active_seasons

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            ProjectConfig(season="summer_2026", github_repo="a/b")

    def test_missing_season(self):
        with pytest.raises(ValidationError):
            ProjectConfig(name="Test", github_repo="a/b")

    def test_missing_github_repo(self):
        with pytest.raises(ValidationError):
            ProjectConfig(name="Test", season="summer_2026")


class TestGeorgiaFocusConfig:
    def test_defaults(self):
        g = GeorgiaFocusConfig()
        assert g.priority_locations == []
        assert g.highlight_georgia is True
        assert g.georgia_section_in_readme is True

    def test_custom_locations(self):
        g = GeorgiaFocusConfig(priority_locations=["Atlanta, GA", "Alpharetta, GA"])
        assert len(g.priority_locations) == 2


class TestGreenhouseBoard:
    def test_valid(self):
        b = GreenhouseBoard(token="anthropic", company="Anthropic")
        assert b.token == "anthropic"
        assert b.is_faang_plus is False

    def test_faang_plus(self):
        b = GreenhouseBoard(token="openai", company="OpenAI", is_faang_plus=True)
        assert b.is_faang_plus is True

    def test_missing_token(self):
        with pytest.raises(ValidationError):
            GreenhouseBoard(company="Anthropic")

    def test_missing_company(self):
        with pytest.raises(ValidationError):
            GreenhouseBoard(token="anthropic")


class TestLeverBoard:
    def test_valid(self):
        b = LeverBoard(company_slug="netflix", company="Netflix", is_faang_plus=True)
        assert b.company_slug == "netflix"
        assert b.is_faang_plus is True

    def test_missing_slug(self):
        with pytest.raises(ValidationError):
            LeverBoard(company="Netflix")

    def test_missing_company(self):
        with pytest.raises(ValidationError):
            LeverBoard(company_slug="netflix")


class TestAshbyBoard:
    def test_valid(self):
        b = AshbyBoard(company_slug="ramp", company="Ramp")
        assert b.company_slug == "ramp"
        assert b.is_faang_plus is False

    def test_missing_slug(self):
        with pytest.raises(ValidationError):
            AshbyBoard(company="Ramp")


class TestScrapeSource:
    def test_valid(self):
        s = ScrapeSource(company="Google", url="https://careers.google.com")
        assert s.company == "Google"
        assert s.is_faang_plus is False

    def test_faang_plus(self):
        s = ScrapeSource(
            company="Google", url="https://careers.google.com", is_faang_plus=True
        )
        assert s.is_faang_plus is True

    def test_missing_company(self):
        with pytest.raises(ValidationError):
            ScrapeSource(url="https://example.com")

    def test_missing_url(self):
        with pytest.raises(ValidationError):
            ScrapeSource(company="Google")


class TestGitHubMonitor:
    def test_valid(self):
        m = GitHubMonitor(repo="SimplifyJobs/Summer2026-Internships")
        assert m.branch == "main"
        assert m.file == "README.md"

    def test_custom_branch_and_file(self):
        m = GitHubMonitor(
            repo="owner/repo", branch="dev", file="data/listings.md"
        )
        assert m.branch == "dev"
        assert m.file == "data/listings.md"

    def test_missing_repo(self):
        with pytest.raises(ValidationError):
            GitHubMonitor()


class TestFiltersConfig:
    def test_defaults(self):
        f = FiltersConfig()
        assert f.keywords_include == []
        assert f.keywords_exclude == []
        assert f.role_categories == {}
        assert f.exclude_companies == []

    def test_populated(self):
        f = FiltersConfig(
            keywords_include=["intern"],
            keywords_exclude=["senior"],
            role_categories={"swe": ["software engineer"]},
            exclude_companies=["Revature"],
        )
        assert "intern" in f.keywords_include
        assert "swe" in f.role_categories
        assert len(f.role_categories["swe"]) == 1


class TestAIConfig:
    def test_defaults(self):
        a = AIConfig()
        assert a.model == "gemini-2.0-flash"
        assert a.max_tokens == 1024
        assert a.enrichment_prompt == ""

    def test_custom(self):
        a = AIConfig(model="custom-model", max_tokens=512, enrichment_prompt="Go!")
        assert a.model == "custom-model"
        assert a.max_tokens == 512


class TestScheduleConfig:
    def test_defaults(self):
        s = ScheduleConfig()
        assert s.update_interval_hours == 6
        assert s.link_check_interval_hours == 24
        assert s.archive_after_days == 7

    def test_custom(self):
        s = ScheduleConfig(
            update_interval_hours=12,
            link_check_interval_hours=48,
            archive_after_days=14,
        )
        assert s.update_interval_hours == 12


# ── AppConfig Tests ──────────────────────────────────────────────────────────


class TestAppConfig:
    def test_minimal_config(self, minimal_config_dict):
        config = AppConfig.model_validate(minimal_config_dict)
        assert config.project.name == "Test Project"
        assert config.greenhouse_boards == []
        assert config.lever_boards == []
        assert config.ashby_boards == []
        assert config.scrape_sources == []
        assert config.total_sources == 0

    def test_full_config(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        assert config.project.github_repo == "ctsc/atlanta-tech-internships-2026"
        assert len(config.greenhouse_boards) == 2
        assert len(config.lever_boards) == 1
        assert len(config.ashby_boards) == 1
        assert len(config.scrape_sources) == 1
        assert len(config.github_monitors) == 1
        assert config.total_sources == 6  # 2 + 1 + 1 + 1 + 1

    def test_total_sources_property(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        expected = (
            len(config.greenhouse_boards)
            + len(config.lever_boards)
            + len(config.ashby_boards)
            + len(config.scrape_sources)
            + len(config.github_monitors)
        )
        assert config.total_sources == expected

    def test_missing_project_section(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({})

    def test_invalid_project_section(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"project": {"name": "No season or repo"}})

    def test_georgia_focus_defaults(self, minimal_config_dict):
        config = AppConfig.model_validate(minimal_config_dict)
        assert config.georgia_focus.highlight_georgia is True
        assert config.georgia_focus.priority_locations == []

    def test_greenhouse_faang_detection(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        faang = [b for b in config.greenhouse_boards if b.is_faang_plus]
        assert len(faang) == 1
        assert faang[0].company == "OpenAI"

    def test_filters_populated(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        assert "intern" in config.filters.keywords_include
        assert "senior" in config.filters.keywords_exclude
        assert "Revature" in config.filters.exclude_companies

    def test_ai_model(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        assert config.ai.model == "gemini-2.0-flash"

    def test_schedule_values(self, full_config_dict):
        config = AppConfig.model_validate(full_config_dict)
        assert config.schedule.update_interval_hours == 6
        assert config.schedule.archive_after_days == 7


# ── load_config() Tests ─────────────────────────────────────────────────────


class TestLoadConfig:
    def test_load_valid_config(self, config_yaml_file):
        config = load_config(config_path=config_yaml_file)
        assert config.project.name == "Atlanta Tech Internships"
        assert len(config.greenhouse_boards) == 2

    def test_load_minimal_config(self, minimal_config_yaml_file):
        config = load_config(config_path=minimal_config_yaml_file)
        assert config.project.name == "Test Project"
        assert config.greenhouse_boards == []

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(config_path=Path("/nonexistent/path/config.yaml"))

    def test_load_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(config_path=empty)

    def test_load_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("project:\n  name: [unterminated")
        with pytest.raises(Exception):
            load_config(config_path=bad)

    def test_load_yaml_missing_project(self, tmp_path):
        no_project = tmp_path / "noproj.yaml"
        no_project.write_text("greenhouse_boards: []\n")
        with pytest.raises(ValidationError):
            load_config(config_path=no_project)

    def test_load_real_config_yaml(self):
        """Load the actual project config.yaml and verify it parses."""
        real_config = Path(__file__).resolve().parent.parent / "config.yaml"
        if not real_config.exists():
            pytest.skip("Real config.yaml not found")
        config = load_config(config_path=real_config)
        assert config.project.github_repo == "ctsc/atlanta-tech-internships-2026"
        assert len(config.greenhouse_boards) >= 80
        assert len(config.lever_boards) >= 20
        assert len(config.ashby_boards) >= 15
        assert len(config.scrape_sources) >= 30
        assert config.total_sources >= 100
        assert len(config.georgia_focus.priority_locations) >= 10
        assert config.ai.model == "gemini-2.0-flash"

    def test_load_config_returns_app_config_type(self, config_yaml_file):
        config = load_config(config_path=config_yaml_file)
        assert isinstance(config, AppConfig)


# ── Big Tech Companies Tests ──────────────────────────────────────────────


class TestBigTechCompanies:
    """Tests for the big_tech_companies list and is_big_tech() helper."""

    def test_big_tech_defaults_to_empty(self, minimal_config_dict):
        """big_tech_companies defaults to empty list."""
        config = AppConfig.model_validate(minimal_config_dict)
        assert config.big_tech_companies == []

    def test_big_tech_loaded_from_config(self, minimal_config_dict):
        """big_tech_companies loads a list of company names."""
        minimal_config_dict["big_tech_companies"] = ["Google", "Meta", "Apple"]
        config = AppConfig.model_validate(minimal_config_dict)
        assert len(config.big_tech_companies) == 3
        assert "Google" in config.big_tech_companies

    def test_is_big_tech_exact_match(self, minimal_config_dict):
        """is_big_tech returns True for exact match."""
        minimal_config_dict["big_tech_companies"] = ["Google", "Meta"]
        config = AppConfig.model_validate(minimal_config_dict)
        assert is_big_tech("Google", config) is True
        assert is_big_tech("Meta", config) is True

    def test_is_big_tech_case_insensitive(self, minimal_config_dict):
        """is_big_tech is case insensitive."""
        minimal_config_dict["big_tech_companies"] = ["Google"]
        config = AppConfig.model_validate(minimal_config_dict)
        assert is_big_tech("google", config) is True
        assert is_big_tech("GOOGLE", config) is True
        assert is_big_tech("GoOgLe", config) is True

    def test_is_big_tech_strips_whitespace(self, minimal_config_dict):
        """is_big_tech strips leading/trailing whitespace."""
        minimal_config_dict["big_tech_companies"] = ["Google"]
        config = AppConfig.model_validate(minimal_config_dict)
        assert is_big_tech("  Google  ", config) is True

    def test_is_big_tech_returns_false_for_non_match(self, minimal_config_dict):
        """is_big_tech returns False for companies not on the list."""
        minimal_config_dict["big_tech_companies"] = ["Google", "Meta"]
        config = AppConfig.model_validate(minimal_config_dict)
        assert is_big_tech("SmallStartup", config) is False
        assert is_big_tech("", config) is False

    def test_is_big_tech_empty_list(self, minimal_config_dict):
        """is_big_tech returns False when list is empty."""
        config = AppConfig.model_validate(minimal_config_dict)
        assert is_big_tech("Google", config) is False

    def test_real_config_has_big_tech(self):
        """Real config.yaml should have big_tech_companies populated."""
        real_config = Path(__file__).resolve().parent.parent / "config.yaml"
        if not real_config.exists():
            pytest.skip("Real config.yaml not found")
        config = load_config(config_path=real_config)
        assert len(config.big_tech_companies) >= 50
        assert is_big_tech("Google", config) is True
        assert is_big_tech("Anthropic", config) is True
        assert is_big_tech("Stripe", config) is True
