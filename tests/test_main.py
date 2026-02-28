"""Tests for the CLI entry point (main.py)."""

import json
from unittest.mock import patch

import pytest

from main import main, parse_args, run_clean, run_full_pipeline


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_no_args_defaults_to_full(self):
        args = parse_args([])
        assert not args.discover_only
        assert not args.readme_only
        assert not args.check_links_only
        assert not args.full

    def test_full_flag(self):
        args = parse_args(["--full"])
        assert args.full is True

    def test_discover_only(self):
        args = parse_args(["--discover-only"])
        assert args.discover_only is True

    def test_readme_only(self):
        args = parse_args(["--readme-only"])
        assert args.readme_only is True

    def test_check_links_only(self):
        args = parse_args(["--check-links-only"])
        assert args.check_links_only is True

    def test_mutually_exclusive_flags(self):
        with pytest.raises(SystemExit):
            parse_args(["--full", "--discover-only"])

    def test_mutually_exclusive_readme_and_links(self):
        with pytest.raises(SystemExit):
            parse_args(["--readme-only", "--check-links-only"])

    def test_clean_flag(self):
        args = parse_args(["--clean"])
        assert args.clean is True

    def test_mutually_exclusive_clean_and_full(self):
        with pytest.raises(SystemExit):
            parse_args(["--clean", "--full"])


class TestMainDispatch:
    """Tests for dispatch logic in main()."""

    @patch("main.run_full_pipeline")
    def test_default_runs_full_pipeline(self, mock_full):
        main([])
        mock_full.assert_called_once()

    @patch("main.run_full_pipeline")
    def test_full_flag_runs_full_pipeline(self, mock_full):
        main(["--full"])
        mock_full.assert_called_once()

    @patch("main.run_discover_only")
    def test_discover_only_flag(self, mock_discover):
        main(["--discover-only"])
        mock_discover.assert_called_once()

    @patch("main.run_readme_only")
    def test_readme_only_flag(self, mock_readme):
        main(["--readme-only"])
        mock_readme.assert_called_once()

    @patch("main.run_check_links_only")
    def test_check_links_only_flag(self, mock_links):
        main(["--check-links-only"])
        mock_links.assert_called_once()

    @patch("main.run_clean")
    def test_clean_flag(self, mock_clean):
        main(["--clean"])
        mock_clean.assert_called_once()


class TestRunStep:
    """Tests for the _run_step error isolation."""

    @patch("main._run_step")
    def test_full_pipeline_calls_all_steps(self, mock_step):
        mock_step.return_value = True
        run_full_pipeline()
        assert mock_step.call_count == 6

    @patch("main._run_step")
    def test_full_pipeline_continues_after_failure(self, mock_step):
        mock_step.side_effect = [False, True, True, True, True, True]
        run_full_pipeline()
        assert mock_step.call_count == 6

    def test_run_step_catches_exceptions(self):
        from main import _run_step

        def exploding():
            raise RuntimeError("boom")

        result = _run_step("test step", exploding, is_async=False)
        assert result is False

    def test_run_step_returns_true_on_success(self):
        from main import _run_step

        result = _run_step("test step", lambda: None, is_async=False)
        assert result is True

    def test_run_step_handles_async(self):
        from main import _run_step

        async def async_fn():
            return 42

        result = _run_step("async step", async_fn, is_async=True)
        assert result is True


class TestRunClean:
    """Tests for the --clean flag functionality."""

    def test_clean_removes_non_intern_titles(self, tmp_path):
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.filters.keywords_exclude = ["sales", "accounting", "internal audit"]
        cfg.filters.keywords_include = ["intern", "internship"]

        # Create fake jobs.json
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        jobs = {
            "listings": [
                {"role": "Software Engineer Intern", "company": "Stripe", "status": "open"},
                {"role": "Internal Audit Specialist", "company": "BigBank", "status": "open"},
                {"role": "Sales Development Intern", "company": "SalesCo", "status": "open"},
                {"role": "ML Internship", "company": "AI Inc", "status": "open"},
            ],
            "last_updated": "2026-02-28T00:00:00",
            "total_open": 4,
        }
        jobs_file = data_dir / "jobs.json"
        jobs_file.write_text(json.dumps(jobs), encoding="utf-8")

        with patch("scripts.utils.config.PROJECT_ROOT", tmp_path), \
             patch("scripts.utils.config.get_config", return_value=cfg):
            run_clean()

        result = json.loads(jobs_file.read_text(encoding="utf-8"))
        roles = [x["role"] for x in result["listings"]]
        # "Software Engineer Intern" and "ML Internship" should remain
        assert "Software Engineer Intern" in roles
        assert "ML Internship" in roles
        # "Internal Audit Specialist" should be removed (no word-boundary "intern" match,
        # and "internal audit" is in exclude list)
        assert "Internal Audit Specialist" not in roles
        # "Sales Development Intern" should be removed (contains "sales" exclude keyword)
        assert "Sales Development Intern" not in roles
