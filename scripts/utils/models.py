"""Pydantic v2 data models for the Autonomous Internship Board.

Defines all core models: enums for role categories, sponsorship status,
listing status, and ATS types; plus Company, JobListing, JobsDatabase,
and RawListing.
"""

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class InternSeason(str, Enum):
    """Internship season identifiers."""

    SUMMER_2026 = "summer_2026"
    FALL_2026 = "fall_2026"
    SPRING_2027 = "spring_2027"
    SUMMER_2027 = "summer_2027"


class RoleCategory(str, Enum):
    """Categories for internship roles."""

    SWE = "swe"
    ML_AI = "ml_ai"
    DATA_SCIENCE = "data_science"
    QUANT = "quant"
    PM = "pm"
    HARDWARE = "hardware"
    OTHER = "other"


class SponsorshipStatus(str, Enum):
    """Visa sponsorship status for a listing."""

    SPONSORS = "sponsors"
    NO_SPONSORSHIP = "no_sponsorship"
    US_CITIZENSHIP = "us_citizenship"
    UNKNOWN = "unknown"


class ListingStatus(str, Enum):
    """Current status of a job listing."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class IndustrySector(str, Enum):
    """Industry/sector classification for companies."""

    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    ECOMMERCE = "ecommerce"
    BANKING = "banking"
    AUTOMOTIVE = "automotive"
    GAMING = "gaming"
    SOCIAL_MEDIA = "social_media"
    CYBERSECURITY = "cybersecurity"
    CLOUD = "cloud"
    ENTERPRISE = "enterprise"
    AI_ML = "ai_ml"
    AEROSPACE = "aerospace"
    TELECOM = "telecom"
    MEDIA = "media"
    FOOD = "food"
    LOGISTICS = "logistics"
    SEMICONDUCTOR = "semiconductor"
    OTHER = "other"


class ATSType(str, Enum):
    """Applicant Tracking System types."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    SMARTRECRUITERS = "smartrecruiters"
    ICIMS = "icims"
    CUSTOM = "custom"


class Company(BaseModel):
    """Company metadata including ATS information."""

    name: str
    slug: str  # kebab-case identifier
    careers_url: Optional[HttpUrl] = None
    ats_type: Optional[ATSType] = None
    ats_identifier: Optional[str] = None  # e.g., greenhouse board token
    is_faang_plus: bool = False


class JobListing(BaseModel):
    """A single internship job listing with all metadata."""

    id: str  # content hash of (company + role + location)
    company: str
    company_slug: str
    role: str
    category: RoleCategory
    locations: list[str]  # ["SF", "Remote", "NYC"]
    apply_url: HttpUrl
    sponsorship: SponsorshipStatus = SponsorshipStatus.UNKNOWN
    requires_us_citizenship: bool = False
    is_faang_plus: bool = False
    requires_advanced_degree: bool = False
    graduate_friendly: bool = False
    remote_friendly: bool = False
    open_to_international: bool = False
    date_added: date
    date_last_verified: date
    source: str  # "greenhouse_api", "lever_api", "scrape", "community"
    status: ListingStatus = ListingStatus.OPEN
    tech_stack: list[str] = []
    season: str = "summer_2026"
    industry: IndustrySector = IndustrySector.OTHER


class JobsDatabase(BaseModel):
    """Top-level container for all job listings."""

    listings: list[JobListing]
    last_updated: datetime
    total_open: int = 0

    def compute_stats(self) -> None:
        """Recompute the total_open count from current listings."""
        self.total_open = len(
            [j for j in self.listings if j.status == ListingStatus.OPEN]
        )


class RawListing(BaseModel):
    """A raw job listing discovered before AI validation."""

    company: str
    company_slug: str
    title: str
    location: str
    url: str
    source: str  # "greenhouse_api", "lever_api", "ashby_api", "scrape", "github_monitor"
    is_faang_plus: bool = False
    raw_data: dict = {}
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of normalized company + title + location."""
        raw = f"{self.company.lower().strip()}|{self.title.lower().strip()}|{self.location.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()
