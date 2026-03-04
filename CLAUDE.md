# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated internship listing aggregator that discovers, validates, and publishes tech internship listings. Runs every 6 hours via GitHub Actions. Geographic focus on Georgia/Southeast US.

**Repo**: https://github.com/ctsc/atlanta-tech-internships-2026
**Owner**: Carter | **Python**: 3.12+ | **AI Backend**: Google Gemini 2.0 Flash (free tier)

## Autonomous Operation

Claude Code is authorized to operate autonomously in this repo. When working unattended:

- **Read/Write/Edit files** freely — no confirmation needed for any file operations
- **Run Bash commands** freely — install deps, run tests, run pipeline, lint, git operations
- **Create commits** after completing work — use descriptive messages
- **Run the full pipeline** or individual steps as needed
- **Fix lint/test failures** immediately without asking
- **Create new files** when needed for features, tests, or fixes
- **Modify** any source, test, config, or data file as the task requires

**Always ask before**: deleting files, pushing to remote, or creating PRs. All other operations are pre-approved.

## Commands

```bash
pip install -r requirements.txt          # Install dependencies

# Pipeline
python main.py --full                    # Run complete pipeline (default)
python main.py --discover-only           # Discovery only
python main.py --readme-only             # Regenerate README only
python main.py --check-links-only        # Link checking only
python main.py --clean                   # Re-filter existing jobs.json

# Testing
python -m pytest tests/                  # Run all tests (~560)
python -m pytest tests/test_models.py    # Run a single test file
python -m pytest tests/ -k "test_name"   # Run a specific test

# Linting
ruff check scripts/ tests/               # Lint all source
```

## Architecture

Multi-phase pipeline orchestrated by `main.py`:

```
ATS APIs (Greenhouse/Lever/Ashby) ─┐
Career page scraping ──────────────┤
GitHub repo monitors ──────────────┘
           │
    scripts/discover.py          → data/raw_discovery_{timestamp}.json
    scripts/validate.py          → Gemini AI confirms real internships
    scripts/deduplicate.py       → Content-hash + fuzzy dedup
    scripts/check_links.py       → Async HEAD requests, 2-failure threshold
    scripts/archive_stale.py     → Closed >7d or stale >120d → archived.json
    scripts/generate_readme.py   → data/jobs.json → README.md
    scripts/process_issues.py    → GitHub Issues → validation pipeline
```

**Data flow**: `data/jobs.json` is the single source of truth. README.md is auto-generated from it. Each pipeline step is error-isolated — one failing source never crashes the run.

### Key modules in `scripts/utils/`

- **models.py** — Pydantic v2 models (`JobListing`, `JobsDatabase`, `RawListing`, enums for `RoleCategory`, `ListingStatus`, `SponsorshipStatus`, `InternSeason`, `IndustrySector`)
- **config.py** — Loads `config.yaml` into typed `AppConfig` model. Exports `PROJECT_ROOT`, `load_config()`, `get_config()`
- **ats_clients.py** — Async Greenhouse/Lever/Ashby API clients with rate limiting (2 req/sec) and tenacity retry
- **ai_enrichment.py** — Gemini API integration with response caching (`data/.cache/`) and budget cap (200 calls/run)
- **readme_renderer.py** — Markdown table renderer with emoji indicators, grouped by `RoleCategory`
- **scraper.py** — Generic career page scraper + GitHub repo monitor
- **github_utils.py** — GitHub API v3 helpers for issue processing

## Key Conventions

- **HTTP**: Always `httpx.AsyncClient`, never `requests`. Rate limit 2 req/sec per domain
- **Logging**: Python `logging` module only, never `print()`
- **Data validation**: All data goes through Pydantic models in `models.py`
- **Config**: Everything configurable lives in `config.yaml` — company lists, filters, AI settings, Georgia focus locations. Never hardcode
- **Secrets**: Environment variables only (`GEMINI_API_KEY`, `GITHUB_TOKEN`). `.env` is gitignored
- **Error isolation**: Each pipeline step and each discovery source is wrapped in try/except — partial failures are logged, not fatal
- **Link health**: `data/link_health.json` tracks consecutive failures. A link must fail 2 runs in a row before being marked closed
- **Testing**: `pytest` + `pytest-asyncio`. All HTTP and Gemini calls are mocked. Shared fixtures in `tests/conftest.py`

## Data Files

- `data/jobs.json` — All active listings (source of truth)
- `data/archived.json` — Closed/expired listings
- `data/companies.json` — Company metadata
- `data/link_health.json` — Consecutive link failure tracking
- `data/monitor_state.json` — Last-known state for GitHub repo monitors
- `data/raw_discovery_*.json` — Debug snapshots from discovery runs
- `data/.cache/` — Gemini API response cache (gitignored)

## Multi-Season Support

Active seasons configured in `config.yaml` under `project.active_seasons`:
`summer_2026`, `fall_2026`, `spring_2027`, `summer_2027`

The `InternSeason` enum in `models.py` and filter keywords in `config.yaml` cover all active seasons.

## Georgia Focus

`config.yaml` has a `georgia_focus` section with priority locations (Atlanta, Alpharetta, Marietta, etc.) and a `highlight_georgia` flag. The README renderer surfaces Georgia listings prominently.
