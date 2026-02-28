"""Tests for Pydantic data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from scripts.utils.models import (
    ATSType,
    Company,
    IndustrySector,
    InternSeason,
    JobListing,
    JobsDatabase,
    ListingStatus,
    RoleCategory,
    SponsorshipStatus,
)


# ── Enum Tests ──────────────────────────────────────────────────────────────


class TestInternSeason:
    def test_all_values(self):
        expected = {"summer_2026", "fall_2026", "spring_2027", "summer_2027"}
        assert {e.value for e in InternSeason} == expected

    def test_str_enum(self):
        assert InternSeason.SUMMER_2026 == "summer_2026"
        assert isinstance(InternSeason.SUMMER_2026, str)

    def test_from_value(self):
        assert InternSeason("fall_2026") is InternSeason.FALL_2026
        assert InternSeason("spring_2027") is InternSeason.SPRING_2027

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            InternSeason("winter_2026")


class TestRoleCategory:
    def test_all_values(self):
        expected = {"swe", "ml_ai", "data_science", "quant", "pm", "hardware", "other"}
        assert {e.value for e in RoleCategory} == expected

    def test_str_enum(self):
        assert RoleCategory.SWE == "swe"
        assert isinstance(RoleCategory.SWE, str)

    def test_from_value(self):
        assert RoleCategory("swe") is RoleCategory.SWE
        assert RoleCategory("ml_ai") is RoleCategory.ML_AI

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            RoleCategory("invalid_category")


class TestSponsorshipStatus:
    def test_all_values(self):
        expected = {"sponsors", "no_sponsorship", "us_citizenship", "unknown"}
        assert {e.value for e in SponsorshipStatus} == expected

    def test_str_enum(self):
        assert SponsorshipStatus.SPONSORS == "sponsors"
        assert isinstance(SponsorshipStatus.NO_SPONSORSHIP, str)

    def test_from_value(self):
        assert SponsorshipStatus("unknown") is SponsorshipStatus.UNKNOWN


class TestListingStatus:
    def test_all_values(self):
        expected = {"open", "closed", "unknown"}
        assert {e.value for e in ListingStatus} == expected

    def test_str_enum(self):
        assert ListingStatus.OPEN == "open"
        assert ListingStatus.CLOSED == "closed"


class TestATSType:
    def test_all_values(self):
        expected = {
            "greenhouse",
            "lever",
            "ashby",
            "workday",
            "smartrecruiters",
            "icims",
            "custom",
        }
        assert {e.value for e in ATSType} == expected

    def test_str_enum(self):
        assert ATSType.GREENHOUSE == "greenhouse"


# ── Company Model Tests ─────────────────────────────────────────────────────


class TestCompany:
    def test_minimal_company(self):
        c = Company(name="Stripe", slug="stripe")
        assert c.name == "Stripe"
        assert c.slug == "stripe"
        assert c.careers_url is None
        assert c.ats_type is None
        assert c.ats_identifier is None
        assert c.is_faang_plus is False

    def test_full_company(self):
        c = Company(
            name="Google",
            slug="google",
            careers_url="https://careers.google.com",
            ats_type=ATSType.CUSTOM,
            ats_identifier="google-careers",
            is_faang_plus=True,
        )
        assert c.name == "Google"
        assert c.is_faang_plus is True
        assert c.ats_type == ATSType.CUSTOM
        assert str(c.careers_url) == "https://careers.google.com/"

    def test_missing_required_name(self):
        with pytest.raises(ValidationError):
            Company(slug="test")

    def test_missing_required_slug(self):
        with pytest.raises(ValidationError):
            Company(name="Test")

    def test_invalid_ats_type(self):
        with pytest.raises(ValidationError):
            Company(name="Test", slug="test", ats_type="nonexistent")

    def test_invalid_careers_url(self):
        with pytest.raises(ValidationError):
            Company(name="Test", slug="test", careers_url="not-a-url")


# ── JobListing Model Tests ──────────────────────────────────────────────────


class TestJobListing:
    def test_valid_listing(self, sample_job_listing):
        assert sample_job_listing.id == "abc123hash"
        assert sample_job_listing.company == "Anthropic"
        assert sample_job_listing.category == RoleCategory.SWE
        assert sample_job_listing.status == ListingStatus.OPEN

    def test_defaults(self, sample_job_listing_data):
        """Test that optional fields have correct defaults."""
        minimal = {
            "id": "hash1",
            "company": "Acme",
            "company_slug": "acme",
            "role": "Intern",
            "category": "swe",
            "locations": ["Remote"],
            "apply_url": "https://example.com/apply",
            "date_added": "2026-01-01",
            "date_last_verified": "2026-01-01",
            "source": "scrape",
        }
        listing = JobListing(**minimal)
        assert listing.sponsorship == SponsorshipStatus.UNKNOWN
        assert listing.requires_us_citizenship is False
        assert listing.is_faang_plus is False
        assert listing.requires_advanced_degree is False
        assert listing.remote_friendly is False
        assert listing.status == ListingStatus.OPEN
        assert listing.tech_stack == []
        assert listing.season == "summer_2026"

    def test_all_categories(self, sample_job_listing_data):
        """Every RoleCategory value should be accepted."""
        for cat in RoleCategory:
            data = {**sample_job_listing_data, "category": cat}
            listing = JobListing(**data)
            assert listing.category == cat

    def test_all_sponsorship_statuses(self, sample_job_listing_data):
        for status in SponsorshipStatus:
            data = {**sample_job_listing_data, "sponsorship": status}
            listing = JobListing(**data)
            assert listing.sponsorship == status

    def test_all_listing_statuses(self, sample_job_listing_data):
        for status in ListingStatus:
            data = {**sample_job_listing_data, "status": status}
            listing = JobListing(**data)
            assert listing.status == status

    def test_multiple_locations(self, sample_job_listing_data):
        data = {
            **sample_job_listing_data,
            "locations": ["SF", "NYC", "Remote", "Austin, TX"],
        }
        listing = JobListing(**data)
        assert len(listing.locations) == 4

    def test_empty_locations_list(self, sample_job_listing_data):
        data = {**sample_job_listing_data, "locations": []}
        listing = JobListing(**data)
        assert listing.locations == []

    def test_tech_stack(self, sample_job_listing_data):
        data = {
            **sample_job_listing_data,
            "tech_stack": ["Python", "Go", "Kubernetes", "PostgreSQL"],
        }
        listing = JobListing(**data)
        assert "Python" in listing.tech_stack
        assert len(listing.tech_stack) == 4

    def test_missing_required_id(self, sample_job_listing_data):
        del sample_job_listing_data["id"]
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_missing_required_company(self, sample_job_listing_data):
        del sample_job_listing_data["company"]
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_missing_required_role(self, sample_job_listing_data):
        del sample_job_listing_data["role"]
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_missing_required_apply_url(self, sample_job_listing_data):
        del sample_job_listing_data["apply_url"]
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_missing_required_category(self, sample_job_listing_data):
        del sample_job_listing_data["category"]
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_invalid_category(self, sample_job_listing_data):
        sample_job_listing_data["category"] = "not_a_category"
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_invalid_apply_url(self, sample_job_listing_data):
        sample_job_listing_data["apply_url"] = "not-a-valid-url"
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_invalid_date(self, sample_job_listing_data):
        sample_job_listing_data["date_added"] = "not-a-date"
        with pytest.raises(ValidationError):
            JobListing(**sample_job_listing_data)

    def test_faang_plus_flag(self, sample_job_listing_data):
        data = {**sample_job_listing_data, "is_faang_plus": True}
        listing = JobListing(**data)
        assert listing.is_faang_plus is True

    def test_boolean_flags(self, sample_job_listing_data):
        data = {
            **sample_job_listing_data,
            "requires_us_citizenship": True,
            "requires_advanced_degree": True,
            "remote_friendly": True,
        }
        listing = JobListing(**data)
        assert listing.requires_us_citizenship is True
        assert listing.requires_advanced_degree is True
        assert listing.remote_friendly is True

    def test_serialization_roundtrip(self, sample_job_listing):
        """Serialize to dict and back — should produce identical model."""
        data = sample_job_listing.model_dump(mode="json")
        restored = JobListing(**data)
        assert restored.id == sample_job_listing.id
        assert restored.company == sample_job_listing.company
        assert restored.role == sample_job_listing.role
        assert restored.category == sample_job_listing.category
        assert str(restored.apply_url) == str(sample_job_listing.apply_url)


# ── JobsDatabase Model Tests ────────────────────────────────────────────────


class TestJobsDatabase:
    def test_valid_database(self, sample_jobs_database):
        assert len(sample_jobs_database.listings) == 1
        assert isinstance(sample_jobs_database.last_updated, datetime)

    def test_compute_stats_all_open(self, sample_job_listing_data):
        listings = []
        for i in range(5):
            data = {
                **sample_job_listing_data,
                "id": f"hash_{i}",
                "status": ListingStatus.OPEN,
            }
            listings.append(JobListing(**data))
        db = JobsDatabase(listings=listings, last_updated=datetime.now())
        db.compute_stats()
        assert db.total_open == 5

    def test_compute_stats_mixed(self, sample_job_listing_data):
        listings = []
        for i in range(3):
            data = {
                **sample_job_listing_data,
                "id": f"open_{i}",
                "status": ListingStatus.OPEN,
            }
            listings.append(JobListing(**data))
        for i in range(2):
            data = {
                **sample_job_listing_data,
                "id": f"closed_{i}",
                "status": ListingStatus.CLOSED,
            }
            listings.append(JobListing(**data))
        db = JobsDatabase(listings=listings, last_updated=datetime.now())
        db.compute_stats()
        assert db.total_open == 3

    def test_compute_stats_all_closed(self, sample_job_listing_data):
        listings = []
        for i in range(3):
            data = {
                **sample_job_listing_data,
                "id": f"closed_{i}",
                "status": ListingStatus.CLOSED,
            }
            listings.append(JobListing(**data))
        db = JobsDatabase(listings=listings, last_updated=datetime.now())
        db.compute_stats()
        assert db.total_open == 0

    def test_compute_stats_empty(self):
        db = JobsDatabase(listings=[], last_updated=datetime.now())
        db.compute_stats()
        assert db.total_open == 0

    def test_compute_stats_with_unknown_status(self, sample_job_listing_data):
        data_open = {
            **sample_job_listing_data,
            "id": "open1",
            "status": ListingStatus.OPEN,
        }
        data_unknown = {
            **sample_job_listing_data,
            "id": "unk1",
            "status": ListingStatus.UNKNOWN,
        }
        db = JobsDatabase(
            listings=[JobListing(**data_open), JobListing(**data_unknown)],
            last_updated=datetime.now(),
        )
        db.compute_stats()
        assert db.total_open == 1  # only OPEN, not UNKNOWN

    def test_total_open_default(self, sample_jobs_database):
        """total_open defaults to 0 before compute_stats is called."""
        db = JobsDatabase(
            listings=sample_jobs_database.listings,
            last_updated=datetime.now(),
        )
        assert db.total_open == 0

    def test_empty_listings(self):
        db = JobsDatabase(listings=[], last_updated=datetime.now())
        assert db.listings == []
        assert db.total_open == 0

    def test_missing_required_listings(self):
        with pytest.raises(ValidationError):
            JobsDatabase(last_updated=datetime.now())

    def test_missing_required_last_updated(self, sample_job_listing):
        with pytest.raises(ValidationError):
            JobsDatabase(listings=[sample_job_listing])

    def test_serialization_roundtrip(self, sample_jobs_database):
        sample_jobs_database.compute_stats()
        data = sample_jobs_database.model_dump(mode="json")
        restored = JobsDatabase(**data)
        assert len(restored.listings) == len(sample_jobs_database.listings)
        assert restored.total_open == sample_jobs_database.total_open


# ── IndustrySector Tests ──────────────────────────────────────────────────


class TestIndustrySector:
    def test_all_values(self):
        expected = {
            "fintech", "healthcare", "energy", "ecommerce", "banking",
            "automotive", "gaming", "social_media", "cybersecurity", "cloud",
            "enterprise", "ai_ml", "aerospace", "telecom", "media",
            "food", "logistics", "semiconductor", "other",
        }
        assert {e.value for e in IndustrySector} == expected

    def test_str_enum(self):
        assert IndustrySector.FINTECH == "fintech"
        assert isinstance(IndustrySector.FINTECH, str)

    def test_from_value(self):
        assert IndustrySector("ai_ml") is IndustrySector.AI_ML
        assert IndustrySector("cybersecurity") is IndustrySector.CYBERSECURITY

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            IndustrySector("invalid_industry")


class TestJobListingIndustry:
    def test_default_industry(self):
        """JobListing defaults to IndustrySector.OTHER."""
        listing = JobListing(
            id="abc123",
            company="TestCo",
            company_slug="testco",
            role="SWE Intern",
            category=RoleCategory.SWE,
            locations=["NYC"],
            apply_url="https://example.com/apply",
            date_added="2026-02-15",
            date_last_verified="2026-02-20",
            source="greenhouse_api",
        )
        assert listing.industry == IndustrySector.OTHER

    def test_explicit_industry(self):
        """JobListing accepts an explicit industry value."""
        listing = JobListing(
            id="abc123",
            company="Stripe",
            company_slug="stripe",
            role="SWE Intern",
            category=RoleCategory.SWE,
            locations=["SF"],
            apply_url="https://stripe.com/apply",
            date_added="2026-02-15",
            date_last_verified="2026-02-20",
            source="greenhouse_api",
            industry=IndustrySector.FINTECH,
        )
        assert listing.industry == IndustrySector.FINTECH

    def test_industry_serialization(self):
        """Industry field round-trips through serialization."""
        listing = JobListing(
            id="abc123",
            company="CrowdStrike",
            company_slug="crowdstrike",
            role="Security Intern",
            category=RoleCategory.SWE,
            locations=["Austin, TX"],
            apply_url="https://crowdstrike.com/apply",
            date_added="2026-02-15",
            date_last_verified="2026-02-20",
            source="greenhouse_api",
            industry=IndustrySector.CYBERSECURITY,
        )
        data = listing.model_dump(mode="json")
        assert data["industry"] == "cybersecurity"
        restored = JobListing(**data)
        assert restored.industry == IndustrySector.CYBERSECURITY
