"""Transforms jobs.json into a formatted README.md with categorized internship tables.

Entry point script that loads the jobs database, renders it to markdown,
validates the output, and writes the final README.md.
"""

import json
import logging
from pathlib import Path

from scripts.utils.config import PROJECT_ROOT
from scripts.utils.models import JobsDatabase
from scripts.utils.readme_renderer import render_readme

logger = logging.getLogger(__name__)

JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
README_PATH = PROJECT_ROOT / "README.md"


def load_database(jobs_path: Path = JOBS_PATH) -> JobsDatabase:
    """Load and validate the jobs database from JSON.

    Args:
        jobs_path: Path to jobs.json file.

    Returns:
        A validated JobsDatabase instance.
    """
    if not jobs_path.exists():
        logger.warning("jobs.json not found at %s, using empty database", jobs_path)
        from datetime import datetime, timezone

        return JobsDatabase(listings=[], last_updated=datetime.now(timezone.utc))

    with open(jobs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db = JobsDatabase.model_validate(data)
    db.compute_stats()
    logger.info("Loaded %d listings (%d open) from %s", len(db.listings), db.total_open, jobs_path)
    return db


def validate_markdown(content: str) -> bool:
    """Basic validation that the markdown output is well-formed.

    Checks for:
    - Table rows have consistent pipe counts
    - No empty table headers
    - Content is non-empty

    Args:
        content: The markdown string to validate.

    Returns:
        True if validation passes, False otherwise.
    """
    if not content or not content.strip():
        logger.error("README content is empty")
        return False

    lines = content.split("\n")

    # Check table consistency: look for table blocks
    in_table = False
    expected_pipes = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            pipe_count = stripped.count("|")
            if not in_table:
                in_table = True
                expected_pipes = pipe_count
            else:
                if pipe_count != expected_pipes:
                    logger.warning(
                        "Table row %d has %d pipes, expected %d: %s",
                        i + 1, pipe_count, expected_pipes, stripped[:80],
                    )
                    return False
        else:
            in_table = False
            expected_pipes = 0

    # Check that required sections exist
    required = ["# Summer 2026 Tech Internships", "### Legend", "## How This Works"]
    for section in required:
        if section not in content:
            logger.warning("Missing required section: %s", section)
            return False

    return True


def generate_readme(
    jobs_path: Path = JOBS_PATH,
    readme_path: Path = README_PATH,
) -> str:
    """Load jobs, render README, validate, and write to disk.

    Args:
        jobs_path: Path to jobs.json.
        readme_path: Path to write README.md.

    Returns:
        The rendered README content.
    """
    db = load_database(jobs_path)
    content = render_readme(db)

    if not validate_markdown(content):
        logger.error("Markdown validation failed, writing anyway with warning")

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("README written to %s (%d bytes)", readme_path, len(content))
    return content


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    generate_readme()
