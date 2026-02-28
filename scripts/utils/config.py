"""Configuration loader and validator for config.yaml.

Loads config.yaml from the project root and provides typed access to all
configuration sections via Pydantic models. Also loads .env for secrets.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root: two levels up from scripts/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Config section models
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level project metadata."""
    name: str
    season: str
    github_repo: str
    active_seasons: list[str] = ["summer_2026"]


class GeorgiaFocusConfig(BaseModel):
    """Georgia-specific location prioritization settings."""
    priority_locations: list[str] = []
    highlight_georgia: bool = True
    georgia_section_in_readme: bool = True


class GreenhouseBoard(BaseModel):
    """A single Greenhouse ATS board source."""
    token: str
    company: str
    is_faang_plus: bool = False


class LeverBoard(BaseModel):
    """A single Lever ATS board source."""
    company_slug: str
    company: str
    is_faang_plus: bool = False


class AshbyBoard(BaseModel):
    """A single Ashby ATS board source."""
    company_slug: str
    company: str
    is_faang_plus: bool = False


class ScrapeSource(BaseModel):
    """A career page that requires web scraping."""
    company: str
    url: str
    is_faang_plus: bool = False


class GitHubMonitor(BaseModel):
    """A GitHub repo to monitor for new listings."""
    repo: str
    branch: str = "main"
    file: str = "README.md"


class FiltersConfig(BaseModel):
    """Keyword and company filtering rules."""
    keywords_include: list[str] = []
    keywords_exclude: list[str] = []
    role_categories: dict[str, list[str]] = {}
    exclude_companies: list[str] = []


class AIConfig(BaseModel):
    """AI enrichment settings."""
    model: str = "gemini-2.0-flash"
    max_tokens: int = 1024
    enrichment_prompt: str = ""


class ScheduleConfig(BaseModel):
    """Cron schedule settings."""
    update_interval_hours: int = 6
    link_check_interval_hours: int = 24
    archive_after_days: int = 7


# ---------------------------------------------------------------------------
# Top-level config model
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Complete application configuration loaded from config.yaml."""
    project: ProjectConfig
    georgia_focus: GeorgiaFocusConfig = Field(default_factory=GeorgiaFocusConfig)
    greenhouse_boards: list[GreenhouseBoard] = []
    lever_boards: list[LeverBoard] = []
    ashby_boards: list[AshbyBoard] = []
    scrape_sources: list[ScrapeSource] = []
    github_monitors: list[GitHubMonitor] = []
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    company_industries: dict[str, str] = {}

    @property
    def total_sources(self) -> int:
        """Total number of configured discovery sources."""
        return (
            len(self.greenhouse_boards)
            + len(self.lever_boards)
            + len(self.ashby_boards)
            + len(self.scrape_sources)
            + len(self.github_monitors)
        )


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Also loads environment variables from .env if the file exists.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        A validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the config fails validation.
    """
    global _config

    path = config_path or CONFIG_PATH

    # Load .env for secrets (GEMINI_API_KEY, GITHUB_TOKEN, etc.)
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        logger.debug("Loaded environment variables from %s", ENV_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    _config = AppConfig.model_validate(raw)
    logger.info(
        "Config loaded: %d total sources (%d greenhouse, %d lever, %d ashby, %d scrape, %d monitors)",
        _config.total_sources,
        len(_config.greenhouse_boards),
        len(_config.lever_boards),
        len(_config.ashby_boards),
        len(_config.scrape_sources),
        len(_config.github_monitors),
    )
    return _config


def get_config() -> AppConfig:
    """Return the cached config, loading it if necessary.

    Returns:
        The current AppConfig singleton.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a secret from environment variables.

    Ensures .env has been loaded first.

    Args:
        key: Environment variable name (e.g. "GEMINI_API_KEY").
        default: Fallback value if the variable is not set.

    Returns:
        The secret value or the default.
    """
    # Ensure .env is loaded
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    return os.environ.get(key, default)
