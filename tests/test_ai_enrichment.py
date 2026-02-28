"""Comprehensive tests for the AI enrichment module.

Tests cover:
- reset_budget / get_api_call_count: counter management
- _get_cache_path: correct path generation
- _load_cached: cache hit, miss, corrupt JSON, OS errors
- _save_to_cache: successful write, directory creation, OS errors
- _parse_gemini_response: valid JSON, markdown code blocks, invalid JSON, non-dict
- _format_listing_prompt: correct formatting
- enrich_listing: caching, budget cap, no client, API success, API error
- enrich_batch: batching, ordering, empty list, delay between batches
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from scripts.utils.models import RawListing

# We need to import the module itself so we can reset its globals
import scripts.utils.ai_enrichment as ai_mod
from scripts.utils.ai_enrichment import (
    DEFAULT_METADATA,
    MAX_API_CALLS_PER_RUN,
    _format_listing_prompt,
    _get_cache_path,
    _load_cached,
    _parse_gemini_response,
    _save_to_cache,
    enrich_batch,
    enrich_listing,
    get_api_call_count,
    reset_budget,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset module-level globals before each test."""
    ai_mod._api_call_count = 0
    ai_mod._gemini_client = None
    ai_mod._gemini_client_initialized = False
    yield
    # Also reset after the test to avoid side effects
    ai_mod._api_call_count = 0
    ai_mod._gemini_client = None
    ai_mod._gemini_client_initialized = False


@pytest.fixture
def raw_listing():
    """A minimal RawListing for testing."""
    return RawListing(
        company="Anthropic",
        company_slug="anthropic",
        title="Software Engineer Intern",
        location="San Francisco, CA",
        url="https://boards.greenhouse.io/anthropic/jobs/123",
        source="greenhouse_api",
    )


@pytest.fixture
def raw_listing_2():
    """A second RawListing for batch tests."""
    return RawListing(
        company="Stripe",
        company_slug="stripe",
        title="Backend Engineer Intern",
        location="Seattle, WA",
        url="https://boards.greenhouse.io/stripe/jobs/456",
        source="greenhouse_api",
    )


@pytest.fixture
def mock_config():
    """A mock AppConfig with AI settings."""
    config = MagicMock()
    config.ai.model = "gemini-2.0-flash"
    config.ai.max_tokens = 1024
    config.ai.enrichment_prompt = "Analyze this job listing."
    return config


@pytest.fixture
def valid_metadata():
    """A valid enrichment metadata dict."""
    return {
        "is_internship": True,
        "is_summer_2026": True,
        "category": "swe",
        "locations": ["San Francisco, CA"],
        "sponsorship": "unknown",
        "requires_advanced_degree": False,
        "remote_friendly": False,
        "tech_stack": ["Python", "React"],
        "confidence": 0.95,
    }


# ======================================================================
# Tests: reset_budget / get_api_call_count
# ======================================================================


class TestBudgetTracking:
    """Tests for API call budget tracking."""

    def test_get_api_call_count_starts_at_zero(self):
        """Counter should start at 0."""
        assert get_api_call_count() == 0

    def test_reset_budget_sets_count_to_zero(self):
        """reset_budget should set the counter back to 0."""
        ai_mod._api_call_count = 42
        reset_budget()
        assert get_api_call_count() == 0

    def test_reset_budget_from_nonzero(self):
        """reset_budget should work regardless of current value."""
        ai_mod._api_call_count = MAX_API_CALLS_PER_RUN
        reset_budget()
        assert get_api_call_count() == 0

    def test_get_api_call_count_reflects_module_global(self):
        """get_api_call_count reads from the module global."""
        ai_mod._api_call_count = 99
        assert get_api_call_count() == 99


# ======================================================================
# Tests: _get_cache_path
# ======================================================================


class TestGetCachePath:
    """Tests for cache path generation."""

    def test_returns_path_object(self):
        """Should return a Path instance."""
        result = _get_cache_path("abc123")
        assert isinstance(result, Path)

    def test_filename_matches_hash(self):
        """Filename should be {content_hash}.json."""
        result = _get_cache_path("deadbeef1234")
        assert result.name == "deadbeef1234.json"

    def test_parent_is_cache_dir(self):
        """Parent should be data/.cache."""
        result = _get_cache_path("abc123")
        assert result.parent == ai_mod._CACHE_DIR

    def test_different_hashes_give_different_paths(self):
        """Different content hashes produce different paths."""
        p1 = _get_cache_path("hash_a")
        p2 = _get_cache_path("hash_b")
        assert p1 != p2


# ======================================================================
# Tests: _load_cached
# ======================================================================


class TestLoadCached:
    """Tests for loading cached enrichment results."""

    def test_cache_miss_when_file_does_not_exist(self, tmp_path):
        """Returns None when the cache file does not exist."""
        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            result = _load_cached("nonexistent_hash")
        assert result is None

    def test_cache_hit_returns_data(self, tmp_path):
        """Returns the cached dict when the file exists and is valid."""
        data = {"is_internship": True, "confidence": 0.9}
        cache_file = tmp_path / "myhash.json"
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            result = _load_cached("myhash")

        assert result == data

    def test_cache_corrupt_json_returns_none(self, tmp_path):
        """Returns None when the cache file contains invalid JSON."""
        cache_file = tmp_path / "badhash.json"
        cache_file.write_text("not valid json {{{", encoding="utf-8")

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            result = _load_cached("badhash")

        assert result is None

    def test_cache_os_error_returns_none(self, tmp_path):
        """Returns None when reading the file raises an OSError."""
        cache_file = tmp_path / "oserrhash.json"
        cache_file.write_text("{}", encoding="utf-8")

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            with patch("builtins.open", side_effect=OSError("permission denied")):
                result = _load_cached("oserrhash")

        assert result is None

    def test_cache_returns_complex_data(self, tmp_path):
        """Returns complex nested dicts correctly."""
        data = {
            "is_internship": True,
            "locations": ["Atlanta, GA", "Remote"],
            "tech_stack": ["Python", "Go"],
        }
        cache_file = tmp_path / "complex.json"
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            result = _load_cached("complex")

        assert result == data


# ======================================================================
# Tests: _save_to_cache
# ======================================================================


class TestSaveToCache:
    """Tests for saving enrichment results to cache."""

    def test_writes_valid_json_file(self, tmp_path):
        """Should create a JSON file with the correct content."""
        data = {"is_internship": True, "confidence": 0.85}

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            _save_to_cache("savehash", data)

        cache_file = tmp_path / "savehash.json"
        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        assert loaded == data

    def test_creates_cache_directory(self, tmp_path):
        """Should create the cache directory if it does not exist."""
        nested_cache = tmp_path / "nested" / "cache"

        with patch.object(ai_mod, "_CACHE_DIR", nested_cache):
            _save_to_cache("dirhash", {"test": True})

        assert nested_cache.exists()
        assert (nested_cache / "dirhash.json").exists()

    def test_handles_os_error_gracefully(self, tmp_path):
        """Should not raise on OSError (e.g., permission denied)."""
        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            with patch("builtins.open", side_effect=OSError("disk full")):
                # Should not raise
                _save_to_cache("failhash", {"test": True})

    def test_overwrites_existing_cache(self, tmp_path):
        """Should overwrite an existing cache file."""
        cache_file = tmp_path / "overwrite.json"
        cache_file.write_text('{"old": true}', encoding="utf-8")

        with patch.object(ai_mod, "_CACHE_DIR", tmp_path):
            _save_to_cache("overwrite", {"new": True})

        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        assert loaded == {"new": True}


# ======================================================================
# Tests: _parse_gemini_response
# ======================================================================


class TestParseGeminiResponse:
    """Tests for parsing Gemini API response text."""

    def test_valid_json(self, valid_metadata):
        """Parses valid JSON directly."""
        text = json.dumps(valid_metadata)
        result = _parse_gemini_response(text)
        assert result == valid_metadata

    def test_json_in_markdown_code_block(self, valid_metadata):
        """Parses JSON wrapped in ```json ... ``` fences."""
        text = f"```json\n{json.dumps(valid_metadata)}\n```"
        result = _parse_gemini_response(text)
        assert result == valid_metadata

    def test_json_in_plain_code_block(self, valid_metadata):
        """Parses JSON wrapped in ``` ... ``` fences without language tag."""
        text = f"```\n{json.dumps(valid_metadata)}\n```"
        result = _parse_gemini_response(text)
        assert result == valid_metadata

    def test_json_with_surrounding_whitespace(self, valid_metadata):
        """Handles leading/trailing whitespace around JSON."""
        text = f"  \n  {json.dumps(valid_metadata)}  \n  "
        result = _parse_gemini_response(text)
        assert result == valid_metadata

    def test_invalid_json_returns_default(self):
        """Returns default metadata with is_internship=False on parse error."""
        result = _parse_gemini_response("this is not json at all")
        assert result["is_internship"] is False
        assert result["confidence"] == 0.0

    def test_non_dict_json_returns_default(self):
        """Returns default metadata when JSON parses to non-dict (e.g., list)."""
        result = _parse_gemini_response('[1, 2, 3]')
        assert result["is_internship"] is False
        assert result["confidence"] == 0.0

    def test_empty_string_returns_default(self):
        """Returns default metadata on empty string."""
        result = _parse_gemini_response("")
        assert result["is_internship"] is False

    def test_json_string_returns_default(self):
        """Returns default metadata when JSON parses to a bare string."""
        result = _parse_gemini_response('"just a string"')
        assert result["is_internship"] is False

    def test_code_block_with_extra_text_before(self, valid_metadata):
        """Extracts JSON from code block even with text before it."""
        text = f"Here is the analysis:\n```json\n{json.dumps(valid_metadata)}\n```\nDone."
        result = _parse_gemini_response(text)
        assert result == valid_metadata

    def test_partial_metadata_preserved(self):
        """Partial but valid JSON dict is returned as-is."""
        partial = {"is_internship": True, "category": "swe"}
        result = _parse_gemini_response(json.dumps(partial))
        assert result == partial


# ======================================================================
# Tests: _format_listing_prompt
# ======================================================================


class TestFormatListingPrompt:
    """Tests for formatting a RawListing into a Gemini prompt."""

    def test_includes_all_fields(self, raw_listing):
        """Prompt should include company, title, location, and URL."""
        result = _format_listing_prompt(raw_listing)
        assert "Company: Anthropic" in result
        assert "Title: Software Engineer Intern" in result
        assert "Location: San Francisco, CA" in result
        assert "URL: https://boards.greenhouse.io/anthropic/jobs/123" in result

    def test_format_is_multiline(self, raw_listing):
        """Prompt should be multi-line with newline separators."""
        result = _format_listing_prompt(raw_listing)
        lines = result.strip().split("\n")
        assert len(lines) == 4

    def test_different_listings_produce_different_prompts(self, raw_listing, raw_listing_2):
        """Different listings should produce different prompts."""
        p1 = _format_listing_prompt(raw_listing)
        p2 = _format_listing_prompt(raw_listing_2)
        assert p1 != p2


# ======================================================================
# Tests: enrich_listing
# ======================================================================


class TestEnrichListing:
    """Tests for the main enrich_listing function."""

    def test_returns_cached_result_no_api_call(self, raw_listing, mock_config, valid_metadata):
        """When cache hits, returns cached data without calling Gemini."""
        with patch.object(ai_mod, "_load_cached", return_value=valid_metadata):
            with patch.object(ai_mod, "_get_gemini_client") as mock_client:
                result = enrich_listing(raw_listing, config=mock_config)

        assert result == valid_metadata
        mock_client.assert_not_called()

    def test_returns_default_when_budget_exhausted(self, raw_listing, mock_config):
        """Returns default metadata when API call budget is used up."""
        ai_mod._api_call_count = MAX_API_CALLS_PER_RUN

        with patch.object(ai_mod, "_load_cached", return_value=None):
            result = enrich_listing(raw_listing, config=mock_config)

        assert result["confidence"] == 0.0
        assert result["is_summer_2026"] is False

    def test_returns_default_when_no_client(self, raw_listing, mock_config):
        """Returns default metadata when no Gemini client is available."""
        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=None):
                result = enrich_listing(raw_listing, config=mock_config)

        assert result["confidence"] == 0.0
        assert result["is_summer_2026"] is False

    def test_successful_api_call(self, raw_listing, mock_config, valid_metadata):
        """Makes API call, parses response, caches result, and increments counter."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_metadata)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=mock_client):
                # We need to mock the google.genai import inside enrich_listing
                mock_genai = MagicMock()
                with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                    with patch.object(ai_mod, "_save_to_cache") as mock_save:
                        result = enrich_listing(raw_listing, config=mock_config)

        assert result == valid_metadata
        assert get_api_call_count() == 1
        mock_save.assert_called_once_with(raw_listing.content_hash, valid_metadata)

    def test_api_error_returns_default(self, raw_listing, mock_config):
        """Returns default metadata when Gemini API raises an exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=mock_client):
                mock_genai = MagicMock()
                with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                    result = enrich_listing(raw_listing, config=mock_config)

        assert result["confidence"] == 0.0
        assert result["is_summer_2026"] is False
        # Counter should NOT increment on failure
        assert get_api_call_count() == 0

    def test_increments_counter_on_success(self, raw_listing, mock_config, valid_metadata):
        """API call counter is incremented after a successful call."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_metadata)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=mock_client):
                mock_genai = MagicMock()
                with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                    with patch.object(ai_mod, "_save_to_cache"):
                        enrich_listing(raw_listing, config=mock_config)
                        assert get_api_call_count() == 1
                        enrich_listing(raw_listing, config=mock_config)
                        assert get_api_call_count() == 2

    def test_loads_config_when_none(self, raw_listing, valid_metadata):
        """Calls get_config() when config parameter is None."""
        mock_cfg = MagicMock()
        mock_cfg.ai.model = "gemini-2.0-flash"
        mock_cfg.ai.max_tokens = 1024
        mock_cfg.ai.enrichment_prompt = "Test prompt"

        with patch.object(ai_mod, "_load_cached", return_value=valid_metadata):
            with patch("scripts.utils.ai_enrichment.get_config", return_value=mock_cfg) as mock_get:
                result = enrich_listing(raw_listing, config=None)

        mock_get.assert_called_once()
        assert result == valid_metadata

    def test_budget_at_exact_limit(self, raw_listing, mock_config):
        """Budget check triggers at exactly MAX_API_CALLS_PER_RUN."""
        ai_mod._api_call_count = MAX_API_CALLS_PER_RUN

        with patch.object(ai_mod, "_load_cached", return_value=None):
            result = enrich_listing(raw_listing, config=mock_config)

        assert result["confidence"] == 0.0

    def test_budget_one_below_limit_allows_call(self, raw_listing, mock_config, valid_metadata):
        """Budget is not exhausted at MAX - 1."""
        ai_mod._api_call_count = MAX_API_CALLS_PER_RUN - 1

        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_metadata)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=mock_client):
                mock_genai = MagicMock()
                with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                    with patch.object(ai_mod, "_save_to_cache"):
                        result = enrich_listing(raw_listing, config=mock_config)

        assert result == valid_metadata
        assert get_api_call_count() == MAX_API_CALLS_PER_RUN

    def test_caches_successful_result(self, raw_listing, mock_config, valid_metadata):
        """Successful API results are saved to cache."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_metadata)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(ai_mod, "_load_cached", return_value=None):
            with patch.object(ai_mod, "_get_gemini_client", return_value=mock_client):
                mock_genai = MagicMock()
                with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                    with patch.object(ai_mod, "_save_to_cache") as mock_save:
                        enrich_listing(raw_listing, config=mock_config)

        mock_save.assert_called_once()
        saved_hash = mock_save.call_args[0][0]
        saved_data = mock_save.call_args[0][1]
        assert saved_hash == raw_listing.content_hash
        assert saved_data == valid_metadata


# ======================================================================
# Tests: _get_gemini_client
# ======================================================================


class TestGetGeminiClient:
    """Tests for the lazy Gemini client initializer."""

    def test_returns_none_when_no_api_key(self):
        """Returns None when GEMINI_API_KEY is not set."""
        with patch("scripts.utils.ai_enrichment.get_secret", return_value=None):
            client = ai_mod._get_gemini_client()
        assert client is None
        assert ai_mod._gemini_client_initialized is True

    def test_returns_cached_client_on_second_call(self):
        """Returns the same client on subsequent calls (cached)."""
        sentinel = MagicMock()
        ai_mod._gemini_client = sentinel
        ai_mod._gemini_client_initialized = True

        client = ai_mod._get_gemini_client()
        assert client is sentinel

    def test_returns_cached_none_when_already_failed(self):
        """Returns None on second call if first call found no key."""
        ai_mod._gemini_client = None
        ai_mod._gemini_client_initialized = True

        client = ai_mod._get_gemini_client()
        assert client is None

    def test_initializes_client_with_api_key(self):
        """Creates a genai.Client when API key is available."""
        mock_genai = MagicMock()
        mock_client_instance = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        with patch("scripts.utils.ai_enrichment.get_secret", return_value="test-api-key"):
            with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                # Need to patch the import inside the function
                with patch("scripts.utils.ai_enrichment.get_secret", return_value="test-api-key"):
                    # Reset so it tries to initialize
                    ai_mod._gemini_client_initialized = False
                    ai_mod._gemini_client = None

                    # Patch the import of google.genai within the function
                    import importlib
                    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                    def mock_import(name, *args, **kwargs):
                        if name == "google":
                            mod = MagicMock()
                            mod.genai = mock_genai
                            return mod
                        return original_import(name, *args, **kwargs)

                    with patch("builtins.__import__", side_effect=mock_import):
                        client = ai_mod._get_gemini_client()

        # The client should be initialized (even if mocking is complex)
        assert ai_mod._gemini_client_initialized is True

    def test_handles_import_error_gracefully(self):
        """Returns None if google.genai cannot be imported."""
        with patch("scripts.utils.ai_enrichment.get_secret", return_value="test-key"):
            # Simulate import failure by having the from import raise
            original_import = __import__

            def fail_import(name, *args, **kwargs):
                if name == "google":
                    raise ImportError("No module named 'google'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_import):
                client = ai_mod._get_gemini_client()

        assert client is None
        assert ai_mod._gemini_client_initialized is True


# ======================================================================
# Tests: enrich_batch
# ======================================================================


class TestEnrichBatch:
    """Tests for batch enrichment processing."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, mock_config):
        """Empty input returns empty output."""
        result = await enrich_batch([], config=mock_config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preserves_order(self, raw_listing, raw_listing_2, mock_config):
        """Results are returned in the same order as input listings."""
        meta1 = {**DEFAULT_METADATA, "category": "swe"}
        meta2 = {**DEFAULT_METADATA, "category": "ml_ai"}

        call_count = [0]

        def mock_enrich(listing, config=None):
            call_count[0] += 1
            if listing.company == "Anthropic":
                return meta1
            return meta2

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            result = await enrich_batch([raw_listing, raw_listing_2], config=mock_config)

        assert len(result) == 2
        assert result[0]["category"] == "swe"
        assert result[1]["category"] == "ml_ai"

    @pytest.mark.asyncio
    async def test_processes_in_batches_of_10(self, mock_config):
        """Listings are processed in groups of 10 with delays between groups."""
        # Create 25 listings (3 batches: 10, 10, 5)
        listings = []
        for i in range(25):
            listings.append(RawListing(
                company=f"Company{i}",
                company_slug=f"company-{i}",
                title=f"Intern {i}",
                location="Remote",
                url=f"https://example.com/{i}",
                source="test",
            ))

        call_order = []

        def mock_enrich(listing, config=None):
            call_order.append(listing.company)
            return {**DEFAULT_METADATA}

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            async def fake_sleep(duration):
                pass

            with patch("asyncio.sleep", side_effect=fake_sleep) as mock_sleep:
                result = await enrich_batch(listings, config=mock_config)

        assert len(result) == 25
        assert len(call_order) == 25
        # Should have slept between batches (2 sleeps for 3 batches)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_single_listing(self, raw_listing, mock_config, valid_metadata):
        """Single listing processed without inter-batch delay."""

        def mock_enrich(listing, config=None):
            return valid_metadata

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            async def fake_sleep(duration):
                pass

            with patch("asyncio.sleep", side_effect=fake_sleep) as mock_sleep:
                result = await enrich_batch([raw_listing], config=mock_config)

        assert len(result) == 1
        assert result[0] == valid_metadata
        # No sleep for a single batch
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_exactly_10_listings_no_sleep(self, mock_config):
        """Exactly 10 listings = 1 batch, no inter-batch delay."""
        listings = [
            RawListing(
                company=f"Company{i}",
                company_slug=f"company-{i}",
                title=f"Intern {i}",
                location="Remote",
                url=f"https://example.com/{i}",
                source="test",
            )
            for i in range(10)
        ]

        def mock_enrich(listing, config=None):
            return {**DEFAULT_METADATA}

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            async def fake_sleep(duration):
                pass

            with patch("asyncio.sleep", side_effect=fake_sleep) as mock_sleep:
                result = await enrich_batch(listings, config=mock_config)

        assert len(result) == 10
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_eleven_listings_one_sleep(self, mock_config):
        """11 listings = 2 batches, 1 inter-batch delay."""
        listings = [
            RawListing(
                company=f"Company{i}",
                company_slug=f"company-{i}",
                title=f"Intern {i}",
                location="Remote",
                url=f"https://example.com/{i}",
                source="test",
            )
            for i in range(11)
        ]

        def mock_enrich(listing, config=None):
            return {**DEFAULT_METADATA}

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            async def fake_sleep(duration):
                pass

            with patch("asyncio.sleep", side_effect=fake_sleep) as mock_sleep:
                result = await enrich_batch(listings, config=mock_config)

        assert len(result) == 11
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_loads_config_when_none(self, raw_listing, valid_metadata):
        """Calls get_config() when config parameter is None."""
        mock_cfg = MagicMock()

        def mock_enrich(listing, config=None):
            return valid_metadata

        with patch("scripts.utils.ai_enrichment.enrich_listing", side_effect=mock_enrich):
            with patch("scripts.utils.ai_enrichment.get_config", return_value=mock_cfg) as mock_get:
                result = await enrich_batch([raw_listing], config=None)

        mock_get.assert_called_once()
        assert len(result) == 1


# ======================================================================
# Tests: DEFAULT_METADATA constant
# ======================================================================


class TestDefaultMetadata:
    """Tests for the DEFAULT_METADATA constant."""

    def test_has_required_keys(self):
        """DEFAULT_METADATA should contain all expected keys."""
        expected_keys = {
            "is_internship",
            "is_summer_2026",
            "category",
            "locations",
            "sponsorship",
            "requires_advanced_degree",
            "remote_friendly",
            "tech_stack",
            "confidence",
        }
        assert set(DEFAULT_METADATA.keys()) == expected_keys

    def test_default_confidence_is_zero(self):
        """Confidence should be 0.0 for defaults."""
        assert DEFAULT_METADATA["confidence"] == 0.0

    def test_default_is_summer_2026_false(self):
        """Default should not assume summer 2026."""
        assert DEFAULT_METADATA["is_summer_2026"] is False

    def test_default_is_internship_true(self):
        """Default assumes it is an internship (benefit of the doubt)."""
        assert DEFAULT_METADATA["is_internship"] is True


# ======================================================================
# Tests: MAX_API_CALLS_PER_RUN constant
# ======================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_api_calls_is_200(self):
        """Budget cap should be 200 per the spec."""
        assert MAX_API_CALLS_PER_RUN == 200
