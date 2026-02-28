"""AI enrichment module using Gemini 2.0 Flash for listing validation and metadata extraction.

Provides functions to enrich raw job listings with structured metadata via the
Google Gemini API. Includes response caching, budget tracking, and graceful
degradation when the API is unavailable.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from scripts.utils.config import AppConfig, get_config, get_secret

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level budget tracking
# ---------------------------------------------------------------------------

_api_call_count: int = 0
MAX_API_CALLS_PER_RUN: int = 200

# Cached Gemini client (lazy-initialized)
_gemini_client: Optional[object] = None
_gemini_client_initialized: bool = False

# Cache directory relative to project root
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / ".cache"

# Default metadata returned when the API is unavailable or budget is exceeded
DEFAULT_METADATA: dict = {
    "is_internship": True,
    "season": "none",
    "category": "other",
    "locations": [],
    "sponsorship": "unknown",
    "requires_advanced_degree": False,
    "remote_friendly": False,
    "tech_stack": [],
    "confidence": 0.0,
    "industry": "other",
}


def reset_budget() -> None:
    """Reset the API call counter for a new pipeline run."""
    global _api_call_count
    _api_call_count = 0
    logger.info("AI enrichment budget reset (0 / %d)", MAX_API_CALLS_PER_RUN)


def get_api_call_count() -> int:
    """Return the current number of API calls made this run."""
    return _api_call_count


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _get_gemini_client() -> Optional[object]:
    """Get or create a cached Gemini API client.

    Returns:
        A google.genai.Client instance, or None if no API key is configured.
    """
    global _gemini_client, _gemini_client_initialized

    if _gemini_client_initialized:
        return _gemini_client

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — AI enrichment disabled")
        _gemini_client_initialized = True
        _gemini_client = None
        return None

    try:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
        _gemini_client_initialized = True
        logger.info("Gemini client initialized successfully")
        return _gemini_client
    except Exception:
        logger.exception("Failed to initialize Gemini client")
        _gemini_client_initialized = True
        _gemini_client = None
        return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _get_cache_path(content_hash: str) -> Path:
    """Return the cache file path for a given content hash."""
    return _CACHE_DIR / f"{content_hash}.json"


def _load_cached(content_hash: str) -> Optional[dict]:
    """Load a cached enrichment result if it exists.

    Args:
        content_hash: The SHA-256 content hash of the listing.

    Returns:
        The cached metadata dict, or None on miss or error.
    """
    cache_path = _get_cache_path(content_hash)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Cache hit for %s", content_hash[:12])
        return data
    except (json.JSONDecodeError, OSError):
        logger.debug("Cache read error for %s — treating as miss", content_hash[:12])
        return None


def _save_to_cache(content_hash: str, data: dict) -> None:
    """Save an enrichment result to the cache.

    Args:
        content_hash: The SHA-256 content hash of the listing.
        data: The metadata dict to cache.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(content_hash)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Cached enrichment for %s", content_hash[:12])
    except OSError:
        logger.warning("Failed to write cache for %s", content_hash[:12])


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_gemini_response(text: str) -> dict:
    """Parse a Gemini response into a metadata dict.

    Handles responses wrapped in ```json ... ``` markdown code blocks.

    Args:
        text: Raw text from the Gemini API response.

    Returns:
        Parsed metadata dict, or a default dict with is_internship=False on failure.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            logger.warning("Gemini response parsed but is not a dict")
            return {**DEFAULT_METADATA, "is_internship": False}
        return result
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse Gemini response as JSON: %.100s...", cleaned
        )
        return {**DEFAULT_METADATA, "is_internship": False}


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------


def _format_listing_prompt(raw_listing: object) -> str:
    """Format a RawListing into a user prompt for Gemini.

    Args:
        raw_listing: A RawListing instance.

    Returns:
        Formatted string with listing details.
    """
    return (
        f"Company: {raw_listing.company}\n"
        f"Title: {raw_listing.title}\n"
        f"Location: {raw_listing.location}\n"
        f"URL: {raw_listing.url}"
    )


def enrich_listing(
    raw_listing: object,
    config: Optional[AppConfig] = None,
) -> dict:
    """Enrich a single raw listing with AI-extracted metadata.

    Checks the cache first, respects the per-run budget cap, and gracefully
    degrades when the Gemini API is unavailable.

    Args:
        raw_listing: A RawListing instance with company, title, location, url,
                     and content_hash.
        config: Optional AppConfig. If None, loads via get_config().

    Returns:
        A metadata dict with keys: is_internship, is_summer_2026, category,
        locations, sponsorship, requires_advanced_degree, remote_friendly,
        tech_stack, confidence.
    """
    global _api_call_count

    if config is None:
        config = get_config()

    content_hash = raw_listing.content_hash

    # 1. Check cache
    cached = _load_cached(content_hash)
    if cached is not None:
        return cached

    # 2. Check budget
    if _api_call_count >= MAX_API_CALLS_PER_RUN:
        logger.warning(
            "API budget exhausted (%d / %d) — returning default metadata for %s",
            _api_call_count,
            MAX_API_CALLS_PER_RUN,
            raw_listing.title,
        )
        return {**DEFAULT_METADATA}

    # 3. Get Gemini client
    client = _get_gemini_client()
    if client is None:
        logger.info(
            "No Gemini client available — returning default metadata for %s",
            raw_listing.title,
        )
        return {**DEFAULT_METADATA}

    # 4. Call Gemini API
    system_prompt = config.ai.enrichment_prompt
    user_message = _format_listing_prompt(raw_listing)

    try:
        from google import genai

        response = client.models.generate_content(
            model=config.ai.model,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=config.ai.max_tokens,
            ),
        )
        result_text = response.text
        _api_call_count += 1
        logger.info(
            "Gemini API call %d / %d for: %s — %s",
            _api_call_count,
            MAX_API_CALLS_PER_RUN,
            raw_listing.company,
            raw_listing.title,
        )
    except Exception:
        logger.exception(
            "Gemini API error for %s — %s — returning default metadata",
            raw_listing.company,
            raw_listing.title,
        )
        return {**DEFAULT_METADATA}

    # 5. Parse response
    metadata = _parse_gemini_response(result_text)

    # 6. Cache the result
    _save_to_cache(content_hash, metadata)

    return metadata


async def enrich_batch(
    listings: list,
    config: Optional[AppConfig] = None,
) -> list[dict]:
    """Enrich a batch of raw listings, processing in groups of 10.

    Processes listings sequentially within each group (Gemini client is sync),
    with a 1-second delay between groups.

    Args:
        listings: List of RawListing instances.
        config: Optional AppConfig. If None, loads via get_config().

    Returns:
        List of metadata dicts in the same order as the input listings.
    """
    if config is None:
        config = get_config()

    results: list[dict] = []
    batch_size = 10
    total_batches = (len(listings) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(listings))
        batch = listings[start:end]

        logger.info(
            "Processing enrichment batch %d / %d (%d listings)",
            batch_idx + 1,
            total_batches,
            len(batch),
        )

        for listing in batch:
            metadata = enrich_listing(listing, config=config)
            results.append(metadata)

        # Delay between batches (skip after the last batch)
        if batch_idx < total_batches - 1:
            await asyncio.sleep(1.0)

    logger.info(
        "Enrichment complete: %d listings processed, %d API calls used",
        len(results),
        _api_call_count,
    )
    return results
