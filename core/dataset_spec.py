"""
DataSmith AI - Dataset Specification Schema

Universal data contract that ALL agents read and write.
Grows through the pipeline, with each agent adding its section.
"""

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
import uuid
import re
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlsplit
from typing import List, Dict, Any, Optional, Literal, Callable
from datetime import datetime, timezone
from enum import Enum
import unicodedata
from uuid import UUID


# ============================================================================
# ENUMS
# ============================================================================
class SpecModel(BaseModel):
    """Base for every spec model: values are re-validated when a field is assigned,
    not only when the object is created."""
    model_config = ConfigDict(validate_assignment=True)

class DataModality(str, Enum):
    """Supported data modalities."""
    TABULAR = "tabular"
    TEXT = "text"
    TIME_SERIES = "time_series"
    IMAGE = "image"  # Future
    VIDEO = "video"  # Future
    AUDIO = "audio"  # Future


class PipelineStatus(str, Enum):
    """Overall pipeline status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    PAUSED = "paused"
    REJECTED = "rejected"


class AgentStatus(str, Enum):
    """Individual agent execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


EvaluationDimension = Literal[
    "purpose",
    "publisher_reliability",
    "variables",
    "geography",
    "temporal_coverage",
    "resolution",
    "freshness",
    "update_frequency",
    "formats",
    "retrieval_methods",
    "access",
    "license_usage",
    "retrieval_complexity",
    "connector",
]
Criticality = Literal["essential", "advisory"]
DiscoveryMethod = Literal[
    "catalog_hint",
    "web_search",
    "open_data_catalog",
    "provider_documentation",
    "repository_search",
    "manual",
    "other",
]
RetrievalMethod = Literal[
    "http_api",
    "direct_download",
    "catalog_export",
    "manual",
    "other",
]
UsageRight = Literal["use", "modify", "redistribute", "commercial_use"]
EvidenceStatus = Literal["verified", "unverified", "unknown", "conflicting"]
EvidenceType = Literal[
    "official_documentation",
    "api_documentation",
    "license_page",
    "dataset_page",
    "catalog_record",
    "observed_response",
    "research_paper",
    "secondary_report",
    "connector_test",
    "other",
]
EvidenceCaptureMethod = Literal[
    "catalog_hint",
    "search_result",
    "fetched_content",
    "observed_response",
    "manual_capture",
]
EvidenceAuthority = Literal[
    "official_provider",
    "official_publisher",
    "secondary",
    "catalog",
    "observed",
    "unknown",
]
SourceEvaluationStatus = Literal[
    # not_started: No research activity has been attempted on this candidate.
    #   The candidate may have been discovered but evaluation has not begun.
    #   Must not contain findings or an assessment.
    "not_started",

    # in_progress: Research is underway; some dimensions may have been
    #   investigated but the work is incomplete and intermediate results
    #   should not be used for final decisions.
    "in_progress",

    # complete: Every required dimension has been investigated and a finding
    #   has been recorded for it. Some findings may still be 'unknown'—
    #   meaning the investigation ran but could not establish a value—which
    #   is a valid research outcome, not a sign of an uninvestigated dimension.
    "complete",

    # partial: At least one required dimension has been investigated and
    #   recorded in the assessment, but at least one required dimension has
    #   not yet been investigated at all. Use 'complete' (with 'unknown'
    #   findings) for dimensions that were investigated but yielded no data.
    "partial",

    # failed: Evaluation could not be completed due to an error (e.g., the
    #   source was unreachable or produced a fatal parse error).
    "failed",
]
AccessClass = Literal[
    "free",
    "free_with_registration",
    "free_tier",
    "paid",
    "unknown",
]
AuthenticationRequirement = Literal[
    "none",
    "free_registration",
    "api_key",
    "oauth",
    "other",
    "unknown",
]
QuotaStatus = Literal[
    "unknown",
    "not_applicable",
    "confirmed_within_limit",
    "exceeds_limit",
]
LicensePermissionStatus = Literal["allowed", "prohibited", "conditional", "unknown"]
LicenseConditionsStatus = Literal["satisfied", "unsatisfied", "unknown"]
ConnectorAssessmentStatus = Literal[
    "unknown",
    "not_implemented",
    "not_required",
    "planned",
    "implemented_untested",
    "implemented_tested",
    "unsupported",
]
BlockedFetchReasonCode = Literal[
    "allowlist_blocked",
    "ssrf_blocked",
    "scheme_blocked",
    "redirect_blocked",
    "timeout",
    "too_large",
    "bad_content_type",
    "http_error",
    "network_error",
    "budget_exhausted",
]
BlockedFetchStage = Literal[
    "pre_request_policy",
    "pre_request_dns",
    "redirect_validation",
    "network_request",
]
TemporalClaimType = Literal["documented", "observed"]
ProviderDiscoveryStatus = Literal[
    "results",
    "empty",
    "error",
    "invalid_response",
    "not_configured",
]

DEFAULT_FRESHNESS_MAX_AGE_HOURS: Dict[str, int] = {
    "live": 1,
    "recent": 24,
}


# ============================================================================
# REQUIREMENT SECTION (Phase 1: Understanding)
# ============================================================================

class RequirementSpec(SpecModel):
    """Requirement understanding from user input."""
    
    # User input
    dataset_name: str = Field(..., description="Name of the dataset")
    description: str = Field(..., description="User's description of what they need")
    topic_summary: Optional[str] = Field(None, description="LLM's short summary of the topic; never overwrites description")
    # Inferred by agents
    domain: Optional[str] = Field(None, description="Domain (e.g., healthcare, finance)")
    subdomain: Optional[str] = Field(None, description="More specific area")
    problem_type: Optional[str] = Field(None, description="ML task type")
    data_modality: Optional[DataModality] = Field(None, description="Type of data")
    
    # Expected characteristics
    expected_features: List[str] = Field(default_factory=list, description="Expected input features/measurements")
    target_variable: Optional[str] = Field(None, description="Target variable for prediction problems")
    target_requires_derivation: bool = Field(False, description="True if target variable must be derived from real events or thresholds")
    expected_size: Optional[str] = Field(None, description="Expected dataset size")
    expected_rows: Optional[int] = Field(None, ge=1, description="Expected number of rows as a number (derived from expected_size)")
    # Temporal requirements
    freshness_need: Literal["live", "recent", "historical", "unspecified"] = Field(
        "unspecified", 
        description="How fresh the data needs to be"
    )
    max_data_age: Optional[str] = Field(None, description="Maximum data age (e.g., '24 hours', '7 days')")
    max_data_age_hours: Optional[int] = Field(
        None,
        ge=1,
        description="max_data_age as integer hours >= 1 (live=1, recent=24)",
    )
    requested_update_frequency: Optional[str] = Field(
        None,
        description="Explicit user requirement for how often source data is updated",
    )
    # Geographic requirements
    geography: Optional[str] = Field(None, description="Geographic area (country, region, city)")
    
    # Time range
    time_range: Optional[str] = Field(None, description="Time range (dates or description)")
    start_year: Optional[int] = Field(None, description="First year of a range like '2010 to 2020'")
    end_year: Optional[int] = Field(None, description="Last year of a range like '2010 to 2020'")
    time_window_days: Optional[int] = Field(None, ge=1, description="Length in days of a relative window like 'last 7 days'")
    time_granularity: Optional[str] = Field(None, description="Time step between records (Hourly, Daily, Monthly...)")
    # Output preferences
    output_format: Literal["csv", "excel", "json"] = Field("csv", description="Desired output format")
    
    # Constraints
    constraints: List[str] = Field(default_factory=list, description="Additional constraints (e.g., 'free sources only')")
    
    # Clarifications
    missing_or_unclear: List[str] = Field(default_factory=list, description="Information that is missing or unclear")
    clarifying_questions: List[str] = Field(default_factory=list, description="Questions to ask the user")
    
    # Metadata
    is_supported: bool = Field(True, description="Whether this is supported in current version")
    unsupported_reason: Optional[str] = Field(None, description="Why not supported")
    explicitly_stated: List[str] = Field(default_factory=list, description="What user explicitly said")
    accepted_defaults: List[str] = Field(
        default_factory=list, description="Values the user accepted from displayed defaults"
    )
    inferred_by_model: List[str] = Field(default_factory=list, description="What model inferred")
    collection_status: Literal[
        "collected", "collected_with_defaults", "needs_clarification", "rejected"
    ] = Field("needs_clarification", description="Requirement collection outcome")
    
    # Completeness (how much detail the user provided)
    completeness: float = Field(0.0, ge=0.0, le=1.0, description="Request completeness (0-1, based on detail provided)")
    
    # Clarification tracking
    clarification_round: int = Field(0, description="Number of clarification rounds completed (0 or 1)")

    @field_validator("requested_update_frequency")
    @classmethod
    def validate_requested_update_frequency(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("requested_update_frequency must not be blank")
        return value


def _normalize_policy_name(value: str) -> str:
    """Normalize identifiers for deterministic duplicate detection."""
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in normalized):
        raise ValueError("identifiers must not contain control or formatting characters")
    return " ".join(normalized.casefold().split())


_SCHEMA_VARIABLE_ALIASES: Dict[str, str] = {}


def _normalize_variable_name(value: str) -> str:
    normalized = _normalize_policy_name(value)
    if not any(character.isalnum() for character in normalized):
        raise ValueError("variable identifiers must contain a letter or digit")
    return _SCHEMA_VARIABLE_ALIASES.get(normalized, normalized)


def _normalize_format_identifier(value: str) -> str:
    """Canonicalize format identity after syntax-checking optional MIME parameters.

    MIME parameters are ignored for format identity. Their syntax is checked,
    but their values are not interpreted or semantically verified.
    """
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in normalized):
        raise ValueError("format identifiers must not contain control or formatting characters")
    parts = _split_mime_parameters(normalized)
    media_type = parts[0].strip().casefold()
    mime_token = r"[!#$%&'*+.^_`|~0-9a-z-]+"
    if "/" in media_type:
        if not re.fullmatch(rf"{mime_token}/{mime_token}", media_type):
            raise ValueError(f"invalid format identifier: {value!r}")
        parameter_pattern = re.compile(
            rf'^\s*{mime_token}\s*=\s*(?:{mime_token}|"(?:\\.|[^"\\\r\n])*")\s*$',
            re.IGNORECASE,
        )
        if any(
            not parameter_pattern.fullmatch(parameter.strip())
            for parameter in parts[1:]
        ):
            raise ValueError(f"invalid format identifier: {value!r}")
        normalized = media_type
    elif len(parts) != 1 or not re.fullmatch(mime_token, media_type):
        raise ValueError(f"invalid format identifier: {value!r}")
    else:
        normalized = media_type

    aliases = {
        "csv": "text/csv",
        "json": "application/json",
        "ndjson": "application/x-ndjson",
    }
    return aliases.get(normalized, normalized)


def _split_mime_parameters(value: str) -> List[str]:
    parts: List[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
        elif quoted and character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == ";" and not quoted:
            parts.append(value[start:index])
            start = index + 1
    if quoted or escaped:
        raise ValueError(f"invalid format identifier: {value!r}")
    parts.append(value[start:])
    return parts


def _validate_unique_policy_names(
    values: List[str],
    field_name: str,
    normalizer: Callable[[str], str] = _normalize_policy_name,
) -> None:
    normalized = [normalizer(value) for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} entries must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} entries must be unique after normalization")


class UsageIntent(SpecModel):
    """Application-injected data-use rights derived from the user's request."""

    required_rights: List[UsageRight] = Field(default_factory=lambda: ["use"])

    @model_validator(mode="after")
    def validate_required_rights(self):
        if "use" not in self.required_rights:
            raise ValueError("UsageIntent must always require the 'use' right")
        if len(self.required_rights) != len(set(self.required_rights)):
            raise ValueError("UsageIntent.required_rights must be unique")
        return self


class EvaluationRequirements(SpecModel):
    """Server-owned evaluation scope and essential/advisory policy."""

    policy_version: StrictStr
    required_dimensions: List[EvaluationDimension]
    dimension_criticality: Dict[EvaluationDimension, Criticality]
    required_variables: List[StrictStr] = Field(default_factory=list)
    required_formats: List[StrictStr] = Field(default_factory=list)
    required_retrieval_methods: List[RetrievalMethod] = Field(default_factory=list)

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("policy_version must not be blank")
        return value

    @field_validator("required_variables", "required_formats", mode="before")
    @classmethod
    def require_string_lists(cls, value: Any) -> Any:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("constraint identifiers must be provided as a list of strings")
        return value

    @model_validator(mode="after")
    def validate_policy(self):
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError("required_dimensions must be unique")
        if set(self.dimension_criticality) != set(self.required_dimensions):
            raise ValueError(
                "dimension_criticality must classify every required dimension exactly once"
            )

        _validate_unique_policy_names(
            self.required_variables,
            "required_variables",
            normalizer=_normalize_variable_name,
        )
        _validate_unique_policy_names(
            self.required_formats,
            "required_formats",
            normalizer=_normalize_format_identifier,
        )
        if len(self.required_retrieval_methods) != len(
            set(self.required_retrieval_methods)
        ):
            raise ValueError("required_retrieval_methods must be unique")

        dimension_requirements = (
            ("required_variables", "variables"),
            ("required_formats", "formats"),
            ("required_retrieval_methods", "retrieval_methods"),
        )
        for field_name, dimension in dimension_requirements:
            if getattr(self, field_name) and dimension not in self.required_dimensions:
                raise ValueError(
                    f"{field_name} requires '{dimension}' in required_dimensions"
                )
        return self


class ApplicationPlanningPolicy(SpecModel):
    """Application-generated policy from the actual user request.

    This bundle must never be accepted as LLM-authored policy.
    """

    evaluation_requirements: EvaluationRequirements
    usage_intent: UsageIntent


def derive_evaluation_requirements(
    requirement: RequirementSpec,
    usage_intent: UsageIntent,
    *,
    required_formats: Optional[List[str]] = None,
    required_retrieval_methods: Optional[List[RetrievalMethod]] = None,
    policy_version: str = "1",
) -> ApplicationPlanningPolicy:
    """Build policy only from the actual user request and application-injected intent.

    Source formats and retrieval methods are accepted only as explicit,
    application-extracted constraints. RequirementSpec.output_format describes
    the final dataset output and is not treated as a source-format requirement.
    Never accept this policy or its requirements as LLM-authored input.
    """
    validated_usage_intent = UsageIntent.model_validate(usage_intent)
    dimensions: Dict[EvaluationDimension, Criticality] = {
        "purpose": "essential",
        "publisher_reliability": "advisory",
        "formats": "essential",
        "retrieval_methods": "essential",
        "access": "essential",
        "license_usage": "essential",
        "retrieval_complexity": "advisory",
        "connector": "advisory",
    }

    variables = list(requirement.expected_features)
    if requirement.target_variable and not requirement.target_requires_derivation:
        variables.append(requirement.target_variable)
    variable_keys = [_normalize_variable_name(value) for value in variables]
    variables = [
        value for index, value in enumerate(variables)
        if variable_keys.index(_normalize_variable_name(value)) == index
    ]
    if variables:
        dimensions["variables"] = "essential"
    if requirement.geography:
        dimensions["geography"] = "essential"
    if (
        requirement.freshness_need == "historical"
        or requirement.time_range
        or requirement.start_year is not None
        or requirement.end_year is not None
        or requirement.time_window_days is not None
    ):
        dimensions["temporal_coverage"] = "essential"
    if requirement.time_granularity:
        dimensions["resolution"] = "essential"
    if (
        requirement.freshness_need in {"live", "recent"}
        or requirement.max_data_age
        or requirement.max_data_age_hours is not None
    ):
        dimensions["freshness"] = "essential"
    if (
        requirement.freshness_need in {"live", "recent"}
        or requirement.requested_update_frequency
    ):
        dimensions["update_frequency"] = "essential"

    return ApplicationPlanningPolicy(
        usage_intent=validated_usage_intent,
        evaluation_requirements=EvaluationRequirements(
            policy_version=policy_version,
            required_dimensions=list(dimensions),
            dimension_criticality=dimensions,
            required_variables=variables,
            required_formats=required_formats or [],
            required_retrieval_methods=required_retrieval_methods or [],
        ),
    )


# ============================================================================
# PLANNING SECTION (Phase 2: Finding data)
# ============================================================================

class SourceRequirement(SpecModel):
    """A required source type."""
    source_type: str = Field(..., description="Type of source (e.g., meteorological_data_provider)")
    why_needed: str = Field(..., description="Why this source is relevant")
    information_expected: List[str] = Field(default_factory=list, description="What info it should provide")
    importance: Literal["required", "useful", "fallback", "optional"] = Field(..., description="How important")
    preferred_access_methods: List[str] = Field(default_factory=list, description="How to access (api, download, etc)")
    priority: int = Field(..., ge=1, description="Priority (1=highest)")
    data_origin: str = Field("measured", description="Origin of data: measured, modeled, or mixed")
    connector_available: bool = Field(
        False,
        description="Whether DataSmith has an implemented connector for this source type",
    )


class SearchQuery(SpecModel):
    """A search query for finding sources."""
    query: str = Field(..., description="The actual search query")
    purpose: str = Field(..., description="What this query aims to find")
    related_source_type: str = Field(..., description="Which source type this targets")
    priority: int = Field(..., ge=1, description="Priority (1=highest)")


CandidateSelectionStatus = Literal[
    # not_selected: Candidate has not been selected for retrieval.
    "not_selected",
    # selected: Candidate has been selected for retrieval based on its assessment.
    "selected",
    # rejected: Candidate was evaluated but rejected (e.g., failed constraints,
    #   insufficient evidence, unavailable connector, out of scope).
    "rejected",
    # deferred: Candidate is a valid match but retrieval is deferred (e.g., no
    #   implemented connector yet, quota exceeded, manual retrieval needed).
    "deferred",
]


class PlanSpec(SpecModel):
    """Planning output - what sources to find and how.

    Increment 3 additions (all optional with backward-compatible defaults):
    - ``planning_policy``: the application-derived evaluation policy used by the
      planner.  This must be derived from the actual RequirementSpec by the
      application and must never be authored or modified by an LLM.
    - ``candidate_sources``: discovered CandidateSource objects, each carrying
      their own evidence, findings, and assessment.  Research and selection
      status are stored here, independent of each other.
    - ``research_status``: overall research progress across all candidates.
    """
    
    # Core strategy
    source_requirements: List[SourceRequirement] = Field(default_factory=list, description="Required source types")
    search_queries: List[SearchQuery] = Field(default_factory=list, description="Queries to execute")
    uncovered_features: List[str] = Field(
        default_factory=list,
        description="Requested features not advertised by any selected source",
    )
    
    # Strategy
    primary_source_types: List[str] = Field(
        default_factory=list,
        description=(
            "Sources prioritized for discovery; some may not have implemented connectors"
        ),
    )
    fallback_source_types: List[str] = Field(default_factory=list, description="Fallback if primary fails")
    reasoning: Optional[str] = Field(None, description="Overall strategy reasoning")
    
    # Confidence
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Plan match confidence based on topic, geography, feature coverage, "
            "freshness, and primary-source fit"
        ),
    )

    # ---- Increment 3 additions (optional; default to None / empty list) ----

    # Application-derived evaluation policy.  Must be set by the application from
    # the actual RequirementSpec; never accepted from LLM output.
    planning_policy: Optional["ApplicationPlanningPolicy"] = Field(
        None,
        description=(
            "Application-owned evaluation policy derived from the actual requirement. "
            "Controls which dimensions are essential or advisory. "
            "Must not be authored or modified by an LLM."
        ),
    )

    # Discovered candidate sources, each carrying evidence, findings, and assessment.
    # Research status and selection status are stored per-candidate and are independent.
    candidate_sources: List["CandidateSource"] = Field(
        default_factory=list,
        description=(
            "Discovered CandidateSource objects from source research. "
            "Each carries its own evidence, findings, assessment, evaluation_status, "
            "and selection_status.  Do not duplicate evidence or findings here."
        ),
    )

    # Overall research progress across all candidates.
    research_status: "SourceEvaluationStatus" = Field(
        "not_started",
        description=(
            "Overall research progress: not_started before any discovery, in_progress "
            "while research is running, complete when every candidate has been fully "
            "evaluated, partial when some are evaluated, failed on a fatal error."
        ),
    )

    # ---- Increment 4.0 additions (backward-compatible defaults) ----
    retrieval_instructions: List["RetrievalInstruction"] = Field(
        default_factory=list,
        description=(
            "Executable retrieval instructions for the selected primary candidate. "
            "At most one instruction is emitted per plan."
        ),
    )
    blocked_fetches: List["BlockedFetchRecord"] = Field(
        default_factory=list,
        description="Urls blocked or failed during discovery/evidence capture.",
    )
    provider_statuses: Dict[StrictStr, ProviderDiscoveryStatus] = Field(
        default_factory=dict,
        description="Per-provider discovery outcome status.",
    )


# ============================================================================
# SOURCE RESEARCH AND EVIDENCE
# ============================================================================

def _validate_safe_relative_path(value: str, field_name: str = "relative_path") -> str:
    """Validate and normalize a relative artifact path (prevents traversal/absolute paths)."""
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or re.match(r"^[a-zA-Z]:", normalized)
        or ".." in path.parts
        or not path.parts
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    return path.as_posix()


class SourceEvidence(SpecModel):
    """Traceable, application-collected material that may support a finding.

    Instances must be constructed from actually captured material:
    - Catalog hints recorded from the static source catalog.
    - Search results returned by a search engine or catalog API.
    - Content fetched from a URL (pages, documentation, license text).
    - Observed responses received from a live API or connector test.
    - Material recorded explicitly by a human operator (manual_capture).

    LLM-authored assertions are not captured evidence.  An LLM supplying
    the correct ``EvidenceType`` or ``EvidenceAuthority`` values does not
    make its output into captured evidence.  The calling code is responsible
    for ensuring that a ``SourceEvidence`` record corresponds to genuinely
    captured material.

    Provenance semantics of each field:
    - ``source_url``: The URL from which the material was obtained or the
      canonical URL of the resource it describes.  This is a recorded claim,
      not a verified assertion that the URL is accessible or accurate.
    - ``retrieved_at``: The timestamp at which the application captured the
      material, as recorded by the collecting code.
    - ``evidence_type``: The kind of document or response (e.g., license
      page, API documentation, connector test result).
    - ``capture_method``: How the material entered the system (fetched from
      a URL, observed as an API response, recorded as a catalog hint, etc.).
    - ``authority``: The assessed trustworthiness of the publisher relative
      to the claim being supported.  This is an application-recorded
      classification; Pydantic cannot verify that a publisher is truthful or
      that the content is genuine.
    - ``publisher``: The organization or service that published the material.
      This is a recorded metadata claim from the content itself or the
      collecting code.  It is not independently verified.
    - ``content_sha256``: A hex-encoded SHA-256 digest of the captured
      content, when computed by the collecting code.  Its presence indicates
      that the content was hashed at capture time; it does not prove
      authenticity or that the content has not been modified since.

    No cryptographic verification mechanism exists in this schema.
    """

    evidence_id: UUID = Field(default_factory=uuid.uuid4)
    source_url: AnyHttpUrl = Field(
        ...,
        description=(
            "URL from which the material was obtained or the canonical URL of the "
            "resource it describes.  Recorded claim; not independently verified."
        ),
    )
    publisher: Optional[StrictStr] = Field(
        None,
        description=(
            "Organization or service that published the material.  Metadata recorded "
            "from the content or the collecting code; not independently verified."
        ),
    )
    title: StrictStr = Field(
        ...,
        min_length=1,
        description="Human-readable title of the evidence item.",
    )
    discovery_method: DiscoveryMethod = Field(
        ...,
        description="How this evidence item was initially located.",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description=(
            "Kind of document or response (e.g., license_page, api_documentation, "
            "connector_test, secondary_report)."
        ),
    )
    capture_method: EvidenceCaptureMethod = Field(
        ...,
        description=(
            "How the material entered the system: fetched_content, observed_response, "
            "catalog_hint, search_result, or manual_capture."
        ),
    )
    authority: EvidenceAuthority = Field(
        "unknown",
        description=(
            "Application-recorded assessment of the publisher's trustworthiness relative "
            "to the claim being supported.  official_provider / official_publisher mean "
            "the material comes directly from the data provider or its publisher.  "
            "observed means the evidence was produced by an actual live interaction "
            "(e.g., a connector test).  secondary means the evidence is a report about "
            "the source, not the source itself.  catalog means the evidence originated "
            "as a static catalog hint.  Pydantic cannot verify that a publisher is "
            "truthful or that the authority classification is accurate."
        ),
    )
    excerpt: Optional[StrictStr] = Field(
        None,
        max_length=4000,
        description="Verbatim or representative excerpt from the captured material.",
    )
    published_at: Optional[datetime] = Field(
        None,
        description="Date the source material was published, if known.",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Date the source material was last updated, if known.",
    )
    retrieved_at: datetime = Field(
        ...,
        description=(
            "Timestamp at which the application captured this material, as recorded "
            "by the collecting code."
        ),
    )
    content_sha256: Optional[StrictStr] = Field(
        None,
        pattern=r"^[a-fA-F0-9]{64}$",
        description=(
            "Hex-encoded SHA-256 digest of the captured content, when computed by the "
            "collecting code at capture time.  When artifact_relative_path is set, this "
            "hashes the exact UTF-8 bytes of that artifact file."
        ),
    )
    artifact_relative_path: Optional[StrictStr] = Field(
        None,
        description=(
            "Safe path relative to the run's artifact directory pointing to the full "
            "captured UTF-8 text artifact."
        ),
    )

    @field_validator("artifact_relative_path")
    @classmethod
    def validate_artifact_relative_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_safe_relative_path(value, "artifact_relative_path")

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence title must not be blank")
        return value

    @field_validator("publisher")
    @classmethod
    def validate_publisher(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("publisher must not be blank when provided")
        return value

    @field_validator("excerpt")
    @classmethod
    def normalize_excerpt(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                return None
        return value

    @field_validator("content_sha256")
    @classmethod
    def normalize_content_hash(cls, value: Optional[str]) -> Optional[str]:
        return value.lower() if value is not None else None


class SourceFinding(SpecModel):
    """One evidence-linked claim about a discovered source."""

    finding_id: UUID = Field(default_factory=uuid.uuid4)
    dimension: EvaluationDimension
    claim: StrictStr = Field(..., min_length=1)
    value: Optional[Dict[str, Any]] = None
    claim_type: Optional[TemporalClaimType] = Field(
        None,
        description=(
            "For temporal_coverage and freshness findings: 'documented' for source "
            "claims vs 'observed' for parsed timestamped records."
        ),
    )
    evidence_ids: List[UUID] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    status: EvidenceStatus
    explanation: StrictStr = Field(..., min_length=1)

    @field_validator("claim", "explanation")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim and explanation must not be blank")
        return value

    @model_validator(mode="after")
    def validate_finding(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        if self.status == "unknown":
            if self.value is not None:
                raise ValueError("unknown findings must not assert a value")
        else:
            if not self.value:
                raise ValueError(
                    "verified, unverified, and conflicting findings require a non-empty value"
                )
        if self.status == "verified" and not self.evidence_ids:
            raise ValueError("verified findings require supporting evidence")
        if self.status == "conflicting" and not self.evidence_ids:
            raise ValueError("conflicting findings require supporting evidence")
        return self


class QuotaAssessment(SpecModel):
    """Evidence-linked source quota findings; unknown is not not_applicable."""

    status: QuotaStatus = "unknown"
    expected_usage: Optional[float] = Field(None, ge=0.0)
    expected_usage_unit: Optional[StrictStr] = None
    limit_value: Optional[float] = Field(None, gt=0.0)
    limit_unit: Optional[StrictStr] = None
    finding_ids: List[UUID] = Field(default_factory=list)

    @field_validator("expected_usage", mode="before")
    @classmethod
    def validate_expected_usage(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        import math
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value  # let Pydantic report the type error
        if math.isnan(value) or math.isinf(value):
            raise ValueError("expected_usage must be a finite number, not NaN or infinity")
        return value

    @field_validator("limit_value", mode="before")
    @classmethod
    def validate_limit_value(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        import math
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value  # let Pydantic report the type error
        if math.isnan(value) or math.isinf(value):
            raise ValueError("limit_value must be a finite number, not NaN or infinity")
        return value

    @field_validator("expected_usage_unit", "limit_unit")
    @classmethod
    def validate_unit(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("quota units must not be blank")
        return value

    @model_validator(mode="after")
    def validate_quota(self):
        if len(self.finding_ids) != len(set(self.finding_ids)):
            raise ValueError("quota finding_ids must be unique")

        measured_statuses = {"confirmed_within_limit", "exceeds_limit"}
        if self.status in measured_statuses:
            if (
                self.expected_usage is None
                or self.expected_usage_unit is None
                or self.limit_value is None
                or self.limit_unit is None
                or not self.finding_ids
            ):
                raise ValueError(
                    f"{self.status} requires usage, limit, units, and supporting findings"
                )
            if self.expected_usage_unit != self.limit_unit:
                raise ValueError("quota quantities must use the same unit")
            if self.status == "confirmed_within_limit" and self.expected_usage > self.limit_value:
                raise ValueError("expected usage must not exceed the confirmed quota")
            if self.status == "exceeds_limit" and self.expected_usage <= self.limit_value:
                raise ValueError("exceeds_limit requires expected usage above the quota")
        elif self.status == "not_applicable":
            if any(
                value is not None
                for value in (
                    self.expected_usage,
                    self.expected_usage_unit,
                    self.limit_value,
                    self.limit_unit,
                )
            ):
                raise ValueError("not_applicable quota must not include usage or limit values")
            if not self.finding_ids:
                raise ValueError("not_applicable quota requires supporting findings")
        return self


class LicensePermissionAssessment(SpecModel):
    """Assessment of one specific usage right under a source's license."""

    status: LicensePermissionStatus = "unknown"
    verification_status: EvidenceStatus = "unknown"
    finding_ids: List[UUID] = Field(default_factory=list)
    conditions: Optional[StrictStr] = None
    conditions_status: Optional[LicenseConditionsStatus] = None

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("license conditions must not be blank")
        return value

    @model_validator(mode="after")
    def validate_permission(self):
        if len(self.finding_ids) != len(set(self.finding_ids)):
            raise ValueError("license finding_ids must be unique")

        if self.status in {"allowed", "prohibited", "conditional"}:
            if self.verification_status != "verified" or not self.finding_ids:
                raise ValueError(
                    f"{self.status} license permissions require verified supporting findings"
                )
        elif self.verification_status == "verified":
            raise ValueError("unknown license permissions cannot be verified")

        if self.status == "conditional":
            if self.conditions is None or self.conditions_status is None:
                raise ValueError(
                    "conditional permissions require conditions and conditions_status"
                )
        elif self.conditions is not None or self.conditions_status is not None:
            raise ValueError("conditions are only valid for conditional permissions")
        return self


class ConnectorAssessment(SpecModel):
    """Connector implementation status, distinct from source relevance."""

    status: ConnectorAssessmentStatus = "unknown"
    finding_ids: List[UUID] = Field(default_factory=list)
    test_reference: Optional[StrictStr] = None
    tested_at: Optional[datetime] = None

    @field_validator("test_reference")
    @classmethod
    def validate_test_reference(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("test_reference must not be blank")
        return value

    @model_validator(mode="after")
    def validate_connector(self):
        if len(self.finding_ids) != len(set(self.finding_ids)):
            raise ValueError("connector finding_ids must be unique")
        if self.status == "implemented_tested":
            if self.test_reference is None or self.tested_at is None or not self.finding_ids:
                raise ValueError(
                    "implemented_tested connectors require a test reference, time, "
                    "and supporting findings"
                )
        elif self.test_reference is not None or self.tested_at is not None:
            raise ValueError(
                "test_reference and tested_at are only valid for implemented_tested connectors"
            )
        if self.status != "unknown" and not self.finding_ids:
            raise ValueError("known connector statuses require supporting findings")
        return self


class SourceAssessment(SpecModel):
    """Structured assessment summary linked to findings for each dimension.

    required_dimensions must be derived by application policy from the actual
    request; do not accept it as LLM-authored policy. An unknown finding still
    records that a dimension was investigated, but it does not verify a claim.
    """

    assessment_id: UUID = Field(default_factory=uuid.uuid4)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    required_dimensions: List[EvaluationDimension]
    findings_by_dimension: Dict[EvaluationDimension, List[UUID]] = Field(
        default_factory=dict
    )
    access_class: AccessClass = "unknown"
    access_class_finding_ids: List[UUID] = Field(default_factory=list)
    authentication_requirement: AuthenticationRequirement = "unknown"
    authentication_finding_ids: List[UUID] = Field(default_factory=list)
    quota: QuotaAssessment = Field(default_factory=QuotaAssessment)
    license_permissions: Dict[UsageRight, LicensePermissionAssessment] = Field(
        default_factory=dict
    )
    connector: ConnectorAssessment = Field(default_factory=ConnectorAssessment)

    @model_validator(mode="after")
    def validate_assessment(self):
        if not self.required_dimensions:
            raise ValueError("required_dimensions must not be empty")
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError("required_dimensions must be unique")

        if not set(self.findings_by_dimension).issubset(self.required_dimensions):
            raise ValueError("findings_by_dimension contains a non-required dimension")

        grouped_ids: List[UUID] = []
        for dimension, finding_ids in self.findings_by_dimension.items():
            if not finding_ids:
                raise ValueError(f"{dimension} must reference at least one finding")
            if len(finding_ids) != len(set(finding_ids)):
                raise ValueError(f"{dimension} finding IDs must be unique")
            grouped_ids.extend(finding_ids)
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError("a finding may belong to only one assessment dimension")

        for label, finding_ids, dimension in (
            ("access_class", self.access_class_finding_ids, "access"),
            (
                "authentication_requirement",
                self.authentication_finding_ids,
                "access",
            ),
            ("quota", self.quota.finding_ids, "access"),
        ):
            if len(finding_ids) != len(set(finding_ids)):
                raise ValueError(f"{label} finding IDs must be unique")
            if not set(finding_ids).issubset(
                self.findings_by_dimension.get(dimension, [])
            ):
                raise ValueError(f"{label} findings must be grouped under '{dimension}'")

        if self.access_class != "unknown" and not self.access_class_finding_ids:
            raise ValueError("known access classes require supporting access findings")
        if (
            self.authentication_requirement != "unknown"
            and not self.authentication_finding_ids
        ):
            raise ValueError(
                "known authentication requirements require supporting access findings"
            )

        for permission in self.license_permissions.values():
            if not set(permission.finding_ids).issubset(
                self.findings_by_dimension.get("license_usage", [])
            ):
                raise ValueError(
                    "license permission findings must be grouped under 'license_usage'"
                )

        if not set(self.connector.finding_ids).issubset(
            self.findings_by_dimension.get("connector", [])
        ):
            raise ValueError("connector findings must be grouped under 'connector'")
        return self


class CandidateSource(SpecModel):
    """A discovered source, kept distinct from its evidence-backed assessment."""

    candidate_id: UUID = Field(default_factory=uuid.uuid4)
    source_url: AnyHttpUrl
    title: StrictStr = Field(..., min_length=1)
    publisher: Optional[StrictStr] = None
    description: Optional[StrictStr] = None
    source_type: Optional[StrictStr] = None
    source_category: Optional[StrictStr] = None
    discovery_method: DiscoveryMethod
    discovery_query: Optional[StrictStr] = None
    discovered_at: datetime
    evidence: List[SourceEvidence] = Field(default_factory=list)
    findings: List[SourceFinding] = Field(default_factory=list)
    assessment: Optional[SourceAssessment] = None
    evaluation_status: SourceEvaluationStatus = Field(
        "not_started",
        description=(
            "Research progress for this candidate: not_started, in_progress, "
            "complete, partial, or failed.  Independent of selection_status."
        ),
    )
    selection_status: "CandidateSelectionStatus" = Field(
        "not_selected",
        description=(
            "Whether this candidate has been selected, rejected, or deferred for "
            "retrieval.  Independent of evaluation_status: a candidate may be "
            "deferred because it lacks a connector even when fully evaluated."
        ),
    )
    fallback_rank: Optional[int] = Field(
        None,
        ge=1,
        description="Ranked fallback order (1=next best) when selection_status='deferred'.",
    )
    selection_reason: Optional[StrictStr] = Field(
        None,
        description="Human-readable rationale for selection, ranked deferral, or rejection.",
    )

    @field_validator("title")
    @classmethod
    def validate_candidate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("candidate title must not be blank")
        return value

    @field_validator(
        "publisher",
        "description",
        "source_type",
        "source_category",
        "discovery_query",
        "selection_reason",
    )
    @classmethod
    def validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("optional text fields must not be blank when provided")
        return value

    @model_validator(mode="after")
    def validate_candidate(self):
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("candidate evidence IDs must be unique")

        findings_by_id = {item.finding_id: item for item in self.findings}
        if len(findings_by_id) != len(self.findings):
            raise ValueError("candidate finding IDs must be unique")

        for finding in self.findings:
            missing_evidence = set(finding.evidence_ids) - set(evidence_by_id)
            if missing_evidence:
                raise ValueError(
                    f"finding {finding.finding_id} references missing evidence"
                )
            if finding.status == "verified":
                supporting = [evidence_by_id[item] for item in finding.evidence_ids]
                # ----------------------------------------------------------------
                # Universal guards (apply to every dimension)
                # ----------------------------------------------------------------
                # G1: Catalog hints alone can never verify any finding.
                if all(item.capture_method == "catalog_hint" for item in supporting):
                    raise ValueError(
                        "catalog hints alone cannot verify source findings"
                    )
                # G2: search_result snippets alone cannot verify findings; at
                #     least one piece of evidence must have been actually fetched,
                #     observed as a live response, or manually captured.
                _SUBSTANTIVE_CAPTURE = frozenset(
                    {"fetched_content", "observed_response", "manual_capture"}
                )
                if not any(item.capture_method in _SUBSTANTIVE_CAPTURE for item in supporting):
                    raise ValueError(
                        "search-result snippets alone cannot verify source findings; "
                        "at least one piece of evidence must be fetched, observed, or "
                        "manually captured"
                    )

                # ----------------------------------------------------------------
                # Dimension-specific evidence-type + capture-method requirements
                # ----------------------------------------------------------------
                # The checks below enforce that the evidence is *relevant to the
                # dimension* and was *actually obtained*, not just labelled as such.
                # Both evidence_type and capture_method are checked together because:
                #   - authority alone can be mis-labelled by the caller
                #   - an official document is only relevant when its type matches
                #     what the dimension needs
                #   - observed_response requires capture_method == "observed_response"

                # Evidence type sets that apply to each group of dimensions.
                # Fetched/observed/manual means capture_method in _SUBSTANTIVE_CAPTURE.

                # Connector: only an actual observed live connector-test result.
                if finding.dimension == "connector":
                    if not any(
                        item.evidence_type == "connector_test"
                        and item.capture_method == "observed_response"
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified connector findings require actual observed "
                            "connector-test evidence (evidence_type='connector_test', "
                            "capture_method='observed_response'); "
                            "documentation alone is not sufficient"
                        )

                # License usage: must come from a license page or official
                # documentation that actually covers licensing, obtained by
                # fetching, observing, or manual capture — not a mere search
                # snippet, catalog hint, or unrelated document type.
                elif finding.dimension == "license_usage":
                    _LICENSE_TYPES = frozenset({"license_page", "official_documentation"})
                    if not any(
                        item.evidence_type in _LICENSE_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified license_usage findings require a fetched or "
                            "manually captured license page or official documentation "
                            "(evidence_type in {'license_page','official_documentation'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "secondary reports, catalog hints, and search snippets are "
                            "not sufficient"
                        )

                # Variables: what the dataset actually contains.  Acceptable:
                #   - Observed live API/dataset response (data in hand).
                #   - A fetched/manual dataset page or official documentation that
                #     describes the specific variable list.
                #   - A fetched/manual API-documentation page.
                # NOT acceptable: secondary reports, catalog hints, unrelated pages.
                elif finding.dimension == "variables":
                    _VAR_TYPES = frozenset(
                        {
                            "observed_response",
                            "dataset_page",
                            "official_documentation",
                            "api_documentation",
                        }
                    )
                    if not any(
                        item.evidence_type in _VAR_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified variables findings require an observed response "
                            "or fetched dataset/API documentation "
                            "(evidence_type in {'observed_response','dataset_page',"
                            "'official_documentation','api_documentation'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "secondary reports and catalog hints are not sufficient"
                        )

                # Formats: what serialisation formats the source actually provides.
                # Acceptable: observed response (content in hand proves the format)
                # or fetched official/API/dataset documentation.
                elif finding.dimension == "formats":
                    _FMT_TYPES = frozenset(
                        {
                            "observed_response",
                            "official_documentation",
                            "api_documentation",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _FMT_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified formats findings require an observed response or "
                            "fetched official/API/dataset documentation "
                            "(evidence_type in {'observed_response','official_documentation',"
                            "'api_documentation','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "secondary reports are not sufficient"
                        )

                # Geography and resolution: properties described in provider/dataset
                # documentation or directly observable in live responses.
                # NOT acceptable: secondary reports, license pages, connector tests.
                elif finding.dimension in {"geography", "resolution"}:
                    _GEO_RES_TYPES = frozenset(
                        {
                            "observed_response",
                            "official_documentation",
                            "api_documentation",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _GEO_RES_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            f"verified {finding.dimension} findings require an observed "
                            "response or fetched provider/dataset documentation "
                            "(evidence_type in {'observed_response','official_documentation',"
                            "'api_documentation','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "secondary reports, license pages, and catalog hints are "
                            "not sufficient"
                        )

                # update_frequency: the provider's documented update schedule.
                # Acceptable: official/API/dataset documentation (the provider's
                # stated policy) OR an observed response that includes scheduling
                # metadata.  Documentation of intent is sufficient here because
                # update_frequency describes what the provider claims to do, not
                # whether a specific record was present at a specific time.
                # NOT acceptable: secondary reports, license pages, catalog hints.
                elif finding.dimension == "update_frequency":
                    _UPDATE_FREQ_TYPES = frozenset(
                        {
                            "observed_response",
                            "official_documentation",
                            "api_documentation",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _UPDATE_FREQ_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified update_frequency findings require fetched official/"
                            "API/dataset documentation or an observed response "
                            "(evidence_type in {'observed_response','official_documentation',"
                            "'api_documentation','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "documentation of the provider's update schedule is acceptable "
                            "here, but secondary reports and catalog hints are not"
                        )

                # freshness: whether data as recent as required is currently
                # available — not merely how often the provider intends to update.
                #
                # API documentation describing an endpoint or its expected update
                # schedule must NOT, by itself, verify actual data freshness.
                # The evidence must reflect actual available data, such as:
                #   - an observed API/file response containing relevant timestamps,
                #   - a fetched dataset page that directly states a publication or
                #     'last updated' timestamp,
                #   - or a manually captured record from the live dataset.
                #
                # NOT acceptable: api_documentation (describes intent/schedule,
                # not actual current data), official_documentation about the
                # provider's update policy, secondary reports, catalog hints.
                #
                # Note: these checks validate recorded evidence metadata.  They
                # cannot prove that the evidence was genuinely fetched or that its
                # timestamps are truthful.
                elif finding.dimension == "freshness":
                    _FRESHNESS_TYPES = frozenset(
                        {
                            "observed_response",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _FRESHNESS_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified freshness findings require an observed response "
                            "or a fetched dataset page containing relevant publication/"
                            "update timestamps "
                            "(evidence_type in {'observed_response','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "api_documentation describes an update schedule but cannot "
                            "prove that current data is as fresh as claimed; "
                            "secondary reports and catalog hints are not sufficient"
                        )

                # temporal_coverage: whether records for a requested historical
                # period are currently available — not merely whether the provider
                # documents an intended archive range.
                #
                # Endpoint documentation alone must NOT verify that records for a
                # historical period are present; it describes intended or nominal
                # coverage, which may not reflect what is actually queryable.
                # Acceptable evidence must demonstrate actual retrieved records:
                #   - an observed API/file response containing records from the
                #     relevant period,
                #   - a fetched dataset page that directly states the actual
                #     available date range (not just the nominal archive policy).
                #
                # NOT acceptable: api_documentation, official_documentation about
                # provider coverage policy, secondary reports, or catalog hints.
                elif finding.dimension == "temporal_coverage":
                    _TEMPORAL_TYPES = frozenset(
                        {
                            "observed_response",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _TEMPORAL_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified temporal_coverage findings require an observed "
                            "response containing records from the relevant period, or a "
                            "fetched dataset page stating the actual available date range "
                            "(evidence_type in {'observed_response','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "api_documentation or official_documentation describes "
                            "intended coverage but cannot prove that historical records "
                            "are currently available; "
                            "secondary reports and catalog hints are not sufficient"
                        )

                # Retrieval methods: how to retrieve the data.  Acceptable:
                # fetched API or official documentation, or an observed retrieval
                # response.  A secondary review or license page is not sufficient.
                elif finding.dimension == "retrieval_methods":
                    _RETRIEVAL_TYPES = frozenset(
                        {
                            "observed_response",
                            "api_documentation",
                            "official_documentation",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _RETRIEVAL_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified retrieval_methods findings require an observed "
                            "response or fetched API/official documentation "
                            "(evidence_type in {'observed_response','api_documentation',"
                            "'official_documentation','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'})"
                        )

                # Access: authentication, pricing, and registration requirements.
                # Acceptable: fetched access/API/official documentation, or an
                # observed API response that demonstrates access behaviour.
                # NOT acceptable: secondary reviews, license pages, connector tests.
                elif finding.dimension == "access":
                    _ACCESS_TYPES = frozenset(
                        {
                            "observed_response",
                            "api_documentation",
                            "official_documentation",
                            "dataset_page",
                        }
                    )
                    if not any(
                        item.evidence_type in _ACCESS_TYPES
                        and item.capture_method in _SUBSTANTIVE_CAPTURE
                        for item in supporting
                    ):
                        raise ValueError(
                            "verified access findings require an observed response or "
                            "fetched API/official documentation "
                            "(evidence_type in {'observed_response','api_documentation',"
                            "'official_documentation','dataset_page'}, "
                            "capture_method in {'fetched_content','observed_response',"
                            "'manual_capture'}); "
                            "secondary reports and license pages are not sufficient"
                        )

                # publisher_reliability, purpose, retrieval_complexity, and other
                # advisory/informational dimensions allow a broader range of evidence
                # (including secondary reports and research papers) but are still
                # subject to the universal guards above: catalog hints alone and
                # search-result snippets alone cannot verify any finding.

        if self.findings and self.assessment is None:
            raise ValueError("candidate findings require a source assessment")
        if self.evaluation_status == "not_started" and (
            self.assessment is not None or self.findings
        ):
            raise ValueError("not_started candidates must not contain findings or assessment")
        if self.evaluation_status in {"complete", "partial"} and self.assessment is None:
            raise ValueError(
                f"{self.evaluation_status} candidates require a source assessment"
            )

        if self.assessment is not None:
            assessment = self.assessment
            mapped_ids = [
                finding_id
                for finding_ids in assessment.findings_by_dimension.values()
                for finding_id in finding_ids
            ]
            if len(mapped_ids) != len(set(mapped_ids)):
                raise ValueError("assessment finding references must be unique")
            if set(mapped_ids) != set(findings_by_id):
                raise ValueError(
                    "assessment must reference every candidate finding exactly once"
                )
            for dimension, finding_ids in assessment.findings_by_dimension.items():
                if any(findings_by_id[item].dimension != dimension for item in finding_ids):
                    raise ValueError(
                        f"assessment findings do not match dimension '{dimension}'"
                    )

            def validate_assessment_findings(
                finding_ids: List[UUID],
                dimension: EvaluationDimension,
                require_verified: bool,
            ) -> None:
                for finding_id in finding_ids:
                    finding = findings_by_id.get(finding_id)
                    if finding is None or finding.dimension != dimension:
                        raise ValueError(
                            f"{dimension} assessment references a missing or mismatched finding"
                        )
                    if require_verified and (
                        finding.status != "verified" or not finding.evidence_ids
                    ):
                        raise ValueError(
                            f"{dimension} assessment requires verified findings with evidence"
                        )

            validate_assessment_findings(
                assessment.access_class_finding_ids,
                "access",
                assessment.access_class != "unknown",
            )
            validate_assessment_findings(
                assessment.authentication_finding_ids,
                "access",
                assessment.authentication_requirement != "unknown",
            )
            validate_assessment_findings(
                assessment.quota.finding_ids,
                "access",
                assessment.quota.status
                in {"not_applicable", "confirmed_within_limit", "exceeds_limit"},
            )
            for permission in assessment.license_permissions.values():
                validate_assessment_findings(
                    permission.finding_ids,
                    "license_usage",
                    permission.status in {"allowed", "prohibited", "conditional"},
                )
            validate_assessment_findings(
                assessment.connector.finding_ids,
                "connector",
                assessment.connector.status not in {"unknown", "not_required"},
            )
            if assessment.connector.status == "not_required":
                for finding_id in assessment.connector.finding_ids:
                    finding = findings_by_id[finding_id]
                    if finding.status == "unknown" or not finding.evidence_ids:
                        raise ValueError(
                            "not_required connector assessment requires evidence-backed "
                            "connector findings"
                        )

            assessed_dimensions = set(assessment.findings_by_dimension)
            required_dimensions = set(assessment.required_dimensions)
            if self.evaluation_status == "complete" and assessed_dimensions != required_dimensions:
                raise ValueError(
                    "complete evaluation requires findings for every required dimension"
                )
            if self.evaluation_status == "partial" and assessed_dimensions == required_dimensions:
                raise ValueError(
                    "partial evaluation must leave at least one required dimension unassessed"
                )
            if self.evaluation_status == "partial" and not assessed_dimensions:
                raise ValueError("partial evaluation requires at least one assessed dimension")
        if self.fallback_rank is not None and self.selection_status != "deferred":
            raise ValueError("fallback_rank may only be set when selection_status='deferred'")
        return self


_SENSITIVE_QUERY_KEYS = frozenset({
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "token",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "subscription_key",
    "subscription-key",
})
_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
})


class BlockedFetchRecord(SpecModel):
    """Audit record for a URL blocked pre-request or failed during discovery/evidence capture."""

    url: AnyHttpUrl
    candidate_id: Optional[UUID] = None
    reason_code: BlockedFetchReasonCode
    network_attempted: bool = Field(
        ...,
        description=(
            "Whether a network request was actually attempted to this specific URL "
            "(False if blocked pre-connection, including a blocked redirect destination)."
        ),
    )
    block_stage: BlockedFetchStage
    detail: StrictStr = Field(..., min_length=1)
    recorded_at: datetime
    redirect_chain: List[StrictStr] = Field(default_factory=list)

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("detail must not be blank")
        return value

    @field_validator("redirect_chain")
    @classmethod
    def validate_redirect_chain(cls, values: List[str]) -> List[str]:
        cleaned = []
        for item in values:
            stripped = item.strip()
            if not stripped:
                raise ValueError("redirect_chain entries must not be blank")
            cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def validate_stage_consistency(self):
        if self.block_stage in {"pre_request_policy", "pre_request_dns"} and self.network_attempted:
            raise ValueError(
                f"network_attempted must be False for block_stage='{self.block_stage}'"
            )
        if self.block_stage == "redirect_validation" and self.network_attempted:
            raise ValueError(
                "network_attempted must be False when a redirect destination is blocked before connection"
            )
        return self


class RetrievalInstruction(SpecModel):
    """Executable, secret-free retrieval handoff for the single selected primary candidate."""

    instruction_id: UUID = Field(default_factory=uuid.uuid4)
    candidate_id: UUID
    source_url: AnyHttpUrl
    retrieval_method: Literal["direct_download", "http_api"]
    http_method: Literal["GET"] = "GET"
    query_params: Dict[StrictStr, StrictStr] = Field(default_factory=dict)
    headers: Dict[StrictStr, StrictStr] = Field(default_factory=dict)
    expected_format: StrictStr = Field(..., min_length=1)
    parser_hint: StrictStr = Field(..., min_length=1)
    access_class: Literal["free", "free_with_registration", "free_tier"]
    authentication_requirement: Literal["none", "free_registration", "api_key"]
    credential_env_var: Optional[StrictStr] = None
    credential_location: Optional[Literal["header", "query_param"]] = None
    credential_field_name: Optional[StrictStr] = None
    license_spdx_id: Optional[StrictStr] = None
    license_conditions_by_right: Dict[UsageRight, StrictStr] = Field(default_factory=dict)
    requires_runtime_freshness_verification: bool = False
    max_data_age_hours: Optional[int] = Field(None, ge=1)
    rate_limit_rpm: Optional[int] = Field(None, ge=1)
    max_bytes: int = Field(50_000_000, ge=1)
    pagination_mode: Literal["single_request"] = "single_request"
    rationale: StrictStr = Field(..., min_length=1)
    evidence_ids: List[UUID] = Field(..., min_length=1)

    @field_validator(
        "expected_format",
        "parser_hint",
        "rationale",
    )
    @classmethod
    def validate_required_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required string field must not be blank")
        return value

    @field_validator("credential_env_var", "credential_field_name", "license_spdx_id")
    @classmethod
    def validate_optional_nonblank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("optional string field must not be blank when provided")
        return value

    @field_validator("license_conditions_by_right")
    @classmethod
    def validate_license_conditions(
        cls, value: Dict[UsageRight, str]
    ) -> Dict[UsageRight, str]:
        cleaned: Dict[UsageRight, str] = {}
        for right, text in value.items():
            stripped = text.strip()
            if not stripped:
                raise ValueError(
                    f"license condition for right '{right}' must not be blank"
                )
            cleaned[right] = stripped
        return cleaned

    @model_validator(mode="after")
    def validate_instruction(self):
        parsed = urlsplit(str(self.source_url))
        if parsed.scheme.lower() != "https":
            raise ValueError("RetrievalInstruction.source_url must use https")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("RetrievalInstruction.source_url must not contain credentials")
        for key, val in parse_qsl(parsed.query, keep_blank_values=True):
            if key.strip().lower() in _SENSITIVE_QUERY_KEYS and val.strip():
                raise ValueError(
                    f"RetrievalInstruction.source_url must not embed credential parameter '{key}'"
                )

        fixed_query_keys = {
            key.strip().lower()
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
            if key.strip()
        }
        for key in self.query_params:
            lowered_key = key.strip().lower()
            if lowered_key in _SENSITIVE_QUERY_KEYS:
                raise ValueError(
                    f"Credentials must use credential_env_var, not query_params['{key}']"
                )
            if lowered_key in fixed_query_keys:
                raise ValueError(
                    f"query_params['{key}'] must not duplicate or override fixed "
                    "query parameters in source_url"
                )
        for key in self.headers:
            if key.strip().lower() in _SENSITIVE_HEADER_KEYS:
                raise ValueError(
                    f"Credentials or cookies must not be stored in headers['{key}']"
                )

        cred_tuple = (
            self.credential_env_var,
            self.credential_location,
            self.credential_field_name,
        )
        any_cred = any(item is not None for item in cred_tuple)
        all_cred = all(item is not None for item in cred_tuple)
        if any_cred and not all_cred:
            raise ValueError(
                "credential_env_var, credential_location, and credential_field_name "
                "must all be provided together"
            )
        if self.authentication_requirement == "none" and any_cred:
            raise ValueError(
                "credential_env_var must not be set when authentication_requirement='none'"
            )
        if self.authentication_requirement == "api_key" and not all_cred:
            raise ValueError(
                "authentication_requirement='api_key' requires credential_env_var, "
                "credential_location, and credential_field_name"
            )
        if self.credential_env_var is not None and not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", self.credential_env_var
        ):
            raise ValueError(
                "credential_env_var must be an uppercase environment variable name"
            )

        if self.requires_runtime_freshness_verification and self.max_data_age_hours is None:
            raise ValueError(
                "requires_runtime_freshness_verification=True requires max_data_age_hours"
            )
        if not self.requires_runtime_freshness_verification and self.max_data_age_hours is not None:
            raise ValueError(
                "max_data_age_hours may only be set when requires_runtime_freshness_verification=True"
            )
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("RetrievalInstruction.evidence_ids must be unique")
        return self


# ============================================================================
# RETRIEVAL SECTION (Phase 2: Collecting data)
# ============================================================================

class SourceRecord(SpecModel):
    """A discovered data source with provenance."""
    
    # Identity
    source_id: str = Field(..., description="Unique source identifier")
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    
    # Classification
    source_type: str = Field(..., description="Classified source type")
    source_category: str = Field(..., description="Which source category this satisfies")
    
    # Discovery
    discovered_via: str = Field(..., description="Which search query found this")
    discovered_at: str = Field(..., description="When discovered (ISO timestamp)")
    
    # Relevance
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How relevant (0-1)")
    
    # Data availability
    data_available: bool = Field(False, description="Whether data was successfully retrieved")
    data_format: Optional[str] = Field(None, description="Format (csv, json, html, etc)")
    
    # Confidence & provenance
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in this source")
    source_url: str = Field(..., min_length=1, description="Original URL (for provenance)")

class DataRecord(SpecModel):
    """A raw data record extracted from a source."""
    
    # Identity
    record_id: str = Field(..., description="Unique record identifier")
    source_id: str = Field(..., description="Which source this came from")
    
    # Content
    raw_data: Dict[str, Any] = Field(..., description="The actual data")
    extracted_at: str = Field(..., description="When extracted (ISO timestamp)")
    
    # Quality
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in extraction")
    extraction_method: str = Field(..., description="How extracted (csv_parser, json_parser, etc)")
    
    # Provenance (CRITICAL)
    source_url: str = Field(..., min_length=1, description="Original source URL")
    source_type: str = Field(..., description="Source type")


class DownloadedArtifact(SpecModel):
    """Metadata for a downloaded file stored under a run's raw artifact directory."""

    relative_path: str = Field(
        ...,
        min_length=1,
        description="Path relative to the run's artifact directory",
    )
    sha256: str = Field(..., pattern=r"^[a-fA-F0-9]{64}$")
    bytes: int = Field(..., ge=0, description="Downloaded file size in bytes")
    content_type: Optional[str] = Field(None, description="MIME content type, when known")
    source_url: str = Field(..., min_length=1, description="Original source URL")
    retrieved_at: datetime = Field(..., description="Retrieval timestamp")
    extraction_method: str = Field(..., min_length=1, description="How the file was obtained")
    row_count: Optional[int] = Field(None, ge=0, description="Parsed row count, when available")
    license: Optional[str] = Field(None, description="License information, when known")

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_safe_relative_path(value, "relative_path")


class FailedSource(SpecModel):
    """A source retrieval failure with a constrained machine-readable reason."""

    source_url: Optional[str] = Field(None, min_length=1)
    source_title: Optional[str] = Field(None, min_length=1)
    reason_code: Literal[
        "timeout",
        "too_large",
        "bad_content_type",
        "auth_required",
        "parse_error",
    ]
    error_message: str = Field(..., min_length=1)
    retrieved_at: datetime = Field(..., description="Retrieval failure timestamp")


class RetrievalSpec(SpecModel):
    """Retrieval output - discovered sources and raw data."""
    
    # Sources
    discovered_sources: List[SourceRecord] = Field(default_factory=list, description="All discovered sources")
    failed_sources: List[FailedSource] = Field(default_factory=list, description="Sources that failed")
    downloaded_artifacts: List[DownloadedArtifact] = Field(
        default_factory=list,
        description="Downloaded files stored separately from the JSON specification",
    )
    
    # Data
    raw_records: List[DataRecord] = Field(
        default_factory=list,
        description="Small sample only (maximum 50 rows per source); not the complete dataset",
    )
    
    # Coverage
    requirements_satisfied: List[str] = Field(default_factory=list, description="Which requirements met")
    requirements_missing: List[str] = Field(default_factory=list, description="Which requirements not met")
    overall_completeness: float = Field(0.0, ge=0.0, le=1.0, description="Coverage completeness")
    
    # Metadata
    total_sources_discovered: int = Field(0, description="Number of sources found")
    total_records_extracted: int = Field(0, description="Number of records extracted")

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_failed_sources(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "failed_sources" not in value:
            return value

        entries = value["failed_sources"]
        if not isinstance(entries, list):
            return value

        allowed_fields = set(FailedSource.model_fields)
        converted_entries = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                converted_entries.append(entry)
                continue

            unsupported_fields = set(entry) - allowed_fields
            if unsupported_fields:
                fields = ", ".join(sorted(unsupported_fields))
                raise ValueError(
                    f"failed_sources[{index}] cannot be migrated safely; "
                    f"unsupported fields: {fields}"
                )

            try:
                converted_entries.append(FailedSource.model_validate(entry))
            except ValidationError as exc:
                raise ValueError(
                    f"failed_sources[{index}] cannot be migrated safely: {exc}"
                ) from exc

        converted_value = dict(value)
        converted_value["failed_sources"] = converted_entries
        return converted_value


# ============================================================================
# EXECUTION HISTORY
# ============================================================================

class AgentExecution(SpecModel):
    """Record of a single agent execution."""
    
    agent_name: str = Field(..., description="Which agent ran")
    status: AgentStatus = Field(..., description="Execution status")
    started_at: str = Field(..., description="Start timestamp (ISO)")
    completed_at: Optional[str] = Field(None, description="End timestamp (ISO)")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Outputs
    outputs_written: List[str] = Field(default_factory=list, description="Which spec sections were modified")
    
    # Errors
    error_message: Optional[str] = Field(None, description="Error if failed")
    
    # LLM usage
    llm_calls: int = Field(0, description="Number of LLM calls made")
    llm_seconds: float = Field(0.0, description="Total time spent inside LLM calls")
    user_wait_seconds: float = Field(0.0, description="Time spent waiting for user input")
    http_requests: int = Field(0, description="Number of HTTP requests made")


# ============================================================================
# MAIN DATASET SPECIFICATION
# ============================================================================

def _normalize_endpoint_url(url: str) -> str:
    """Normalize an endpoint URL for deterministic equality/exclusion checks."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    query = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def _endpoints_match(instruction_url: str, documented_endpoint_url: str) -> bool:
    """True when instruction_url matches an explicitly verified documented endpoint.

    Both URLs must use HTTPS, match on host and normalized path, and carry the
    exact same fixed query parameters (an undocumented URL query string is
    rejected; ordinary request filters belong in RetrievalInstruction.query_params).
    """
    iu = urlsplit(instruction_url.strip())
    ep = urlsplit(documented_endpoint_url.strip())
    if iu.scheme.lower() != "https" or ep.scheme.lower() != "https":
        return False
    if iu.netloc.lower() != ep.netloc.lower():
        return False
    if (iu.path.rstrip("/") or "/") != (ep.path.rstrip("/") or "/"):
        return False
    return sorted(parse_qsl(iu.query, keep_blank_values=True)) == sorted(
        parse_qsl(ep.query, keep_blank_values=True)
    )


_TEXTUAL_AGE_RE = re.compile(
    r"^\s*(?:within\s+|max\s+|last\s+)?(-?\d+)\s*"
    r"(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|wks?|w)\s*$",
    re.IGNORECASE,
)


def _parse_textual_age_hours(text: str) -> tuple[Optional[int], Optional[str]]:
    """Parse an unambiguous textual max_data_age into positive integer hours."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None, "max_data_age is blank; clarification is required"

    match = _TEXTUAL_AGE_RE.match(cleaned)
    if not match:
        return (
            None,
            f"max_data_age='{text}' is ambiguous or unsupported; clarification is required",
        )

    amount = int(match.group(1))
    unit = match.group(2).lower()
    if amount <= 0:
        return None, f"max_data_age='{text}' must be at least 1 hour"

    if unit.startswith("h"):
        return amount, None
    if unit.startswith("d"):
        return amount * 24, None
    if unit.startswith("w"):
        return amount * 24 * 7, None
    if unit.startswith("m"):
        if amount % 60 == 0 and amount >= 60:
            return amount // 60, None
        return (
            None,
            f"max_data_age='{text}' requires sub-hour or non-integer precision that "
            "cannot be represented accurately in integer hours; clarification is required",
        )
    if unit.startswith("s"):
        if amount % 3600 == 0 and amount >= 3600:
            return amount // 3600, None
        return (
            None,
            f"max_data_age='{text}' requires sub-hour or non-integer precision that "
            "cannot be represented accurately in integer hours; clarification is required",
        )
    return None, f"max_data_age='{text}' is unsupported; clarification is required"


def resolve_freshness_threshold_hours(
    requirement: RequirementSpec,
) -> tuple[Optional[int], Optional[str]]:
    """Resolve the numeric freshness threshold in hours without guessing or ignoring conflicts."""
    numeric_hours = requirement.max_data_age_hours
    if numeric_hours is not None and numeric_hours < 1:
        return None, f"max_data_age_hours ({numeric_hours}) must be at least 1 hour"

    text_age = (requirement.max_data_age or "").strip()
    if text_age:
        parsed_hours, parse_err = _parse_textual_age_hours(text_age)
        if numeric_hours is not None:
            if parse_err is not None or parsed_hours != numeric_hours:
                return (
                    None,
                    f"Conflicting freshness thresholds: max_data_age='{text_age}' "
                    f"disagrees with max_data_age_hours={numeric_hours}",
                )
            return numeric_hours, None
        return parsed_hours, parse_err

    if numeric_hours is not None:
        return numeric_hours, None

    default_hours = DEFAULT_FRESHNESS_MAX_AGE_HOURS.get(requirement.freshness_need)
    if default_hours is not None:
        return default_hours, None
    return None, None


class ReplanAttempt(SpecModel):
    """Audit record of one Orchestrator-triggered bounded replanning attempt."""

    attempt_number: int = Field(..., ge=1, le=2)
    triggered_at: datetime
    reason: StrictStr = Field(..., min_length=1)
    failed_candidate_id: Optional[UUID] = None
    failed_endpoint_url: Optional[AnyHttpUrl] = None
    excluded_candidate_ids: List[UUID] = Field(default_factory=list)
    excluded_urls: List[StrictStr] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ReplanAttempt.reason must not be blank")
        return value

    @field_validator("excluded_urls")
    @classmethod
    def validate_excluded_urls(cls, values: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in values:
            stripped = item.strip()
            if not stripped:
                raise ValueError("excluded_urls entries must not be blank")
            cleaned.append(stripped)
        return cleaned

    @model_validator(mode="after")
    def validate_attempt(self):
        if len(self.excluded_candidate_ids) != len(set(self.excluded_candidate_ids)):
            raise ValueError("excluded_candidate_ids must be unique")
        if not self.excluded_candidate_ids and not self.excluded_urls:
            raise ValueError(
                "ReplanAttempt must exclude at least one candidate ID or endpoint URL"
            )
        if (
            self.failed_candidate_id is not None
            and self.failed_candidate_id not in self.excluded_candidate_ids
        ):
            raise ValueError(
                "failed_candidate_id must be included in excluded_candidate_ids"
            )
        if self.failed_endpoint_url is not None:
            norm_failed = _normalize_endpoint_url(str(self.failed_endpoint_url))
            norm_excluded = {_normalize_endpoint_url(u) for u in self.excluded_urls}
            if norm_failed not in norm_excluded:
                raise ValueError(
                    "failed_endpoint_url must be included in excluded_urls"
                )
        return self


class DatasetSpec(SpecModel):
    """
    Complete dataset specification.
    
    This is the ONE object that flows through the entire pipeline.
    Each agent reads earlier sections and writes its own section.
    """
    
    # Identity
    spec_id: str = Field(..., description="Unique specification ID")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")
    
    # Status
    pipeline_status: PipelineStatus = Field(PipelineStatus.PENDING, description="Overall pipeline status")
    current_stage: str = Field("initialization", description="Current pipeline stage")
    
    # Stage outputs (written by respective agents)
    requirement: Optional[RequirementSpec] = Field(None, description="Phase 1: Requirement understanding")
    plan: Optional[PlanSpec] = Field(None, description="Phase 2: Planning")
    retrieval: Optional[RetrievalSpec] = Field(None, description="Phase 2: Retrieval")
    cleaning: Optional[Dict[str, Any]] = Field(None, description="Phase 3: Data cleaning (future)")
    validation: Optional[Dict[str, Any]] = Field(None, description="Phase 4: Quality validation (future)")
    synthetic: Optional[Dict[str, Any]] = Field(None, description="Phase 4: Synthetic generation (future)")
    export: Optional[Dict[str, Any]] = Field(None, description="Phase 5: Export (future)")
    
    # Execution history
    execution_history: List[AgentExecution] = Field(default_factory=list, description="All agent executions")
    replan_count: int = Field(0, ge=0, le=2, description="Orchestrator-owned replanning counter (max 2)")
    replan_history: List[ReplanAttempt] = Field(default_factory=list, description="Bounded replanning audit trail")
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal errors")
    
    def mark_updated(self):
        """Mark spec as updated."""
        self.updated_at = datetime.now().isoformat()
    
    def add_execution(self, execution: AgentExecution):
        """Add an agent execution record."""
        self.execution_history.append(execution)
        self.mark_updated()
    
    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)
        self.mark_updated()
    
    def add_error(self, error: str):
        """Add an error."""
        self.errors.append(error)
        self.mark_updated()
    
    def get_latest_execution(self, agent_name: str) -> Optional[AgentExecution]:
        """Get the most recent execution of an agent."""
        for exec in reversed(self.execution_history):
            if exec.agent_name == agent_name:
                return exec
        return None
    
    def has_agent_succeeded(self, agent_name: str) -> bool:
        """Check if an agent has successfully completed."""
        exec = self.get_latest_execution(agent_name)
        return exec is not None and exec.status == AgentStatus.SUCCESS


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _validate_plan_invariants(spec: DatasetSpec, errors: List[str]) -> None:
    """Validate Increment 4 planning, selection, instruction, and provenance invariants."""
    plan = spec.plan
    if plan is None:
        return

    source_types = {source.source_type for source in plan.source_requirements}
    source_priorities = [source.priority for source in plan.source_requirements]
    query_priorities = [query.priority for query in plan.search_queries]
    if len(source_priorities) != len(set(source_priorities)):
        errors.append("Plan source requirement priorities must be unique")
    if len(query_priorities) != len(set(query_priorities)):
        errors.append("Plan search query priorities must be unique")
    for query in plan.search_queries:
        if query.related_source_type not in source_types:
            errors.append(
                f"Search query source type '{query.related_source_type}' "
                "does not match a source requirement"
            )

    candidates_by_id = {c.candidate_id: c for c in plan.candidate_sources}
    if len(candidates_by_id) != len(plan.candidate_sources):
        errors.append("Plan candidate_sources must have unique candidate_id values")

    # 1. Selection and retrieval-instruction cardinality
    selected = [c for c in plan.candidate_sources if c.selection_status == "selected"]
    if len(selected) > 1:
        errors.append("At most one CandidateSource may have selection_status='selected'")
    if len(plan.retrieval_instructions) > 1:
        errors.append(
            "PlanSpec.retrieval_instructions may contain at most one primary instruction"
        )
    if len(plan.retrieval_instructions) == 1 and len(selected) != 1:
        errors.append(
            "PlanSpec.retrieval_instructions requires exactly one selected CandidateSource"
        )

    # 2. Unique fallback ranks for deferred candidates
    fallback_ranks = [
        c.fallback_rank
        for c in plan.candidate_sources
        if c.selection_status == "deferred" and c.fallback_rank is not None
    ]
    if len(fallback_ranks) != len(set(fallback_ranks)):
        errors.append("Deferred candidate fallback_rank values must be unique")

    # 2b. Deferred-candidate usability: a deferred candidate is a valid
    # fallback only while its own assessment keeps it usable.  Replan
    # exclusion is enforced separately in _validate_replan_invariants
    # (an excluded candidate must already be 'rejected', never 'deferred').
    if plan.planning_policy is not None:
        required_usage_rights: List[UsageRight] = list(
            plan.planning_policy.usage_intent.required_rights
        )
    else:
        # UsageIntent always requires the 'use' right, so a plan without an
        # explicit policy still carries that minimum requirement.
        required_usage_rights = ["use"]
    for cand in plan.candidate_sources:
        if cand.selection_status != "deferred" or cand.assessment is None:
            continue
        assessment = cand.assessment

        # Quota: a candidate known to exceed its applicable quota limit
        # cannot serve as a fallback; it must be rejected, not deferred.
        if assessment.quota.status == "exceeds_limit":
            errors.append(
                f"Deferred candidate {cand.candidate_id} has quota.status="
                "'exceeds_limit' and cannot be a usable fallback; it must be "
                "selection_status='rejected'"
            )

        # License: a fully evaluated deferred fallback must establish every
        # required usage right as allowed or conditional.  A prohibited or
        # unknown required right makes it unusable as a fallback.
        if cand.evaluation_status == "complete":
            for right in required_usage_rights:
                permission = assessment.license_permissions.get(right)
                if permission is None or permission.status not in {"allowed", "conditional"}:
                    errors.append(
                        f"Deferred candidate {cand.candidate_id} does not establish "
                        f"required usage right '{right}' as allowed or conditional; "
                        "a fully evaluated fallback with a prohibited or unknown "
                        "required right must be selection_status='rejected'"
                    )

        # Connector: an unsupported or not-implemented connector disqualifies
        # a deferred fallback only when its verified retrieval_methods
        # findings document no manual retrieval path.  When no verified
        # retrieval finding exists, the manual path cannot be established
        # either way and no error is raised (recorded limitation).
        if assessment.connector.status in {"unsupported", "not_implemented"}:
            findings_by_id = {f.finding_id: f for f in cand.findings}
            documented_methods: set = set()
            for fid in assessment.findings_by_dimension.get("retrieval_methods", []):
                finding = findings_by_id.get(fid)
                if finding is None or finding.status != "verified" or not finding.value:
                    continue
                methods = finding.value.get("methods")
                if isinstance(methods, list):
                    documented_methods.update(
                        str(m) for m in methods if isinstance(m, str)
                    )
                for key in ("method", "retrieval_method"):
                    single = finding.value.get(key)
                    if isinstance(single, str):
                        documented_methods.add(single)
            if documented_methods and "manual" not in documented_methods:
                errors.append(
                    f"Deferred candidate {cand.candidate_id} has connector.status="
                    f"'{assessment.connector.status}' with no documented manual "
                    "retrieval path; it cannot serve as a usable fallback and must "
                    "be selection_status='rejected'"
                )

    # 8. Provider-aware research-status consistency
    # not_configured optional providers do not count as attempted or failed.
    eval_statuses = [c.evaluation_status for c in plan.candidate_sources]
    configured_providers = {
        name: status
        for name, status in plan.provider_statuses.items()
        if status != "not_configured"
    }
    failed_providers = [
        name
        for name, status in configured_providers.items()
        if status in {"error", "invalid_response"}
    ]
    has_incomplete_candidates = any(
        s in {"not_started", "in_progress", "partial"} for s in eval_statuses
    )
    has_failed_candidates = any(s == "failed" for s in eval_statuses)
    has_complete_candidates = any(s == "complete" for s in eval_statuses)
    has_blocked_fetches = bool(plan.blocked_fetches)
    # A potentially actionable candidate: fully evaluated and either selected
    # or deferred for retrieval.  While such a candidate remains, a blocked
    # primary is a recoverable setback ('partial'), never a terminal 'failed'
    # research outcome.  Per-candidate usability (quota, license, connector)
    # is enforced separately in section 2b above.
    has_actionable_candidate = any(
        c.evaluation_status == "complete" and c.selection_status in {"selected", "deferred"}
        for c in plan.candidate_sources
    )
    all_configured_failed_no_results = (
        bool(configured_providers)
        and len(failed_providers) == len(configured_providers)
        and not plan.candidate_sources
    )
    all_candidates_failed_no_results = (
        bool(plan.candidate_sources)
        and all(s == "failed" for s in eval_statuses)
    )

    if plan.research_status == "not_started":
        if (
            plan.candidate_sources
            or plan.retrieval_instructions
            or plan.blocked_fetches
            or configured_providers
        ):
            errors.append(
                "PlanSpec with research_status='not_started' must not contain "
                "candidate_sources, retrieval_instructions, blocked_fetches, or "
                "configured provider results"
            )
    elif plan.research_status == "complete":
        if not configured_providers:
            errors.append(
                "PlanSpec with research_status='complete' requires at least one "
                "configured discovery provider result"
            )
        if failed_providers:
            errors.append(
                "PlanSpec with research_status='complete' cannot have failed providers"
            )
        if has_blocked_fetches:
            errors.append(
                "PlanSpec with research_status='complete' cannot contain blocked "
                "fetches; a blocked fetch is a recoverable failure, not a "
                "completed research run"
            )
        if has_incomplete_candidates or has_failed_candidates:
            errors.append(
                "PlanSpec with research_status='complete' cannot contain "
                "not_started, in_progress, partial, or failed candidates"
            )
    elif plan.research_status == "partial":
        # A blocked fetch alone keeps research recoverable only while a
        # potentially actionable candidate remains.  With no actionable
        # candidate, no failed provider, and no failed candidate, a blocked
        # primary with no usable fallback is a terminal failure.
        has_recoverable_failure_or_progress = (
            bool(failed_providers)
            or has_incomplete_candidates
            or has_failed_candidates
            or (has_blocked_fetches and has_actionable_candidate)
        )
        if not has_recoverable_failure_or_progress:
            errors.append(
                "PlanSpec with research_status='partial' requires a recoverable "
                "provider or candidate-level failure, an incomplete candidate "
                "evaluation, or a blocked fetch while a selected or deferred "
                "candidate remains actionable"
            )
        elif all_configured_failed_no_results or all_candidates_failed_no_results:
            errors.append(
                "PlanSpec with research_status='partial' is invalid when all "
                "configured providers or all candidates failed with no usable results"
            )
    elif plan.research_status == "failed":
        if plan.retrieval_instructions:
            errors.append(
                "PlanSpec with research_status='failed' must not contain retrieval_instructions"
            )
        if has_actionable_candidate:
            errors.append(
                "PlanSpec with research_status='failed' cannot contain a fully "
                "evaluated selected or deferred candidate; a plan with a valid "
                "selected or deferred candidate remains actionable and must not "
                "be classified as failed"
            )
        elif has_incomplete_candidates:
            errors.append(
                "PlanSpec with research_status='failed' cannot contain "
                "not_started, in_progress, or partial candidates; research is "
                "not terminal while a candidate evaluation is still underway"
            )
        else:
            # A terminal 'failed' outcome requires a total failure: every
            # configured provider failed with no candidates, every discovered
            # candidate failed evaluation, a blocked fetch as the sole failure
            # with no actionable candidate, or a fatal pipeline failure.
            blocked_with_no_usable_path = (
                has_blocked_fetches
                and not failed_providers
                and not has_failed_candidates
            )
            fatal_pipeline_failure = (
                spec.pipeline_status == PipelineStatus.FAILED
                or bool(spec.errors)
            )
            if not (
                all_configured_failed_no_results
                or all_candidates_failed_no_results
                or blocked_with_no_usable_path
                or fatal_pipeline_failure
            ):
                errors.append(
                    "PlanSpec with research_status='failed' requires a total "
                    "failure: every configured provider failed with no candidates, "
                    "every discovered candidate failed evaluation, a blocked fetch "
                    "with no actionable candidate and no other recoverable failure, "
                    "or a fatal pipeline failure with no usable research result "
                    "(not_configured alone is not a failure)"
                )

    # 9. Blocked-fetch candidate references
    for blocked in plan.blocked_fetches:
        if blocked.candidate_id is not None and blocked.candidate_id not in candidates_by_id:
            errors.append(
                f"BlockedFetchRecord references unknown candidate_id '{blocked.candidate_id}'"
            )

    # 7b. Claim-type validation on evaluated temporal/freshness findings
    if plan.research_status != "not_started":
        for cand in plan.candidate_sources:
            ev_map = {e.evidence_id: e for e in cand.evidence}
            for finding in cand.findings:
                if finding.dimension in {"temporal_coverage", "freshness"} and finding.status != "unknown":
                    if finding.claim_type not in {"documented", "observed"}:
                        errors.append(
                            f"Candidate {cand.candidate_id} {finding.dimension} finding "
                            f"{finding.finding_id} must specify claim_type ('documented' or 'observed')"
                        )
                    elif finding.claim_type == "observed":
                        has_observed = any(
                            ev_map[eid].evidence_type == "observed_response"
                            and ev_map[eid].capture_method == "observed_response"
                            for eid in finding.evidence_ids
                            if eid in ev_map
                        )
                        if not has_observed:
                            errors.append(
                                f"Candidate {cand.candidate_id} observed {finding.dimension} "
                                "finding requires observed_response evidence"
                            )

    if not plan.retrieval_instructions:
        return

    instr = plan.retrieval_instructions[0]
    if len(selected) != 1:
        return
    primary = selected[0]

    # 3. Retrieval instruction and selected-candidate consistency
    if instr.candidate_id != primary.candidate_id:
        errors.append(
            "RetrievalInstruction.candidate_id does not match the selected CandidateSource"
        )
    if primary.evaluation_status != "complete":
        errors.append(
            "RetrievalInstruction requires the selected CandidateSource to have evaluation_status='complete'"
        )
    if primary.assessment is None:
        errors.append("Selected CandidateSource with RetrievalInstruction must have an assessment")
        return

    cand_evidence_ids = {e.evidence_id for e in primary.evidence}
    missing_ev = [eid for eid in instr.evidence_ids if eid not in cand_evidence_ids]
    if missing_ev:
        errors.append(
            "RetrievalInstruction.evidence_ids references evidence missing from selected candidate"
        )

    # 4. Evidence-backed endpoint provenance and method matching
    # Note: In Increment 4.0, schema validation checks structured finding metadata,
    # evidence types, capture methods, and exact endpoint/method match. Increment 4c/4d
    # (EvidenceCapturer + FindingExtractor/SourceEvaluator) is responsible for verifying
    # that the captured document/artifact text itself actually substantiates the
    # claimed endpoint_url and retrieval_method before emitting a verified finding.
    findings_by_id = {f.finding_id: f for f in primary.findings}
    ev_by_id = {e.evidence_id: e for e in primary.evidence}
    retrieval_fids = primary.assessment.findings_by_dimension.get("retrieval_methods", [])
    instr_ev_set = set(instr.evidence_ids)
    _SUBSTANTIVE_CAPTURES = {"fetched_content", "observed_response", "manual_capture"}
    _ELIGIBLE_RETRIEVAL_EV_TYPES = {
        "observed_response",
        "api_documentation",
        "official_documentation",
        "dataset_page",
    }
    endpoint_verified = False
    for fid in retrieval_fids:
        finding = findings_by_id.get(fid)
        if not finding or finding.status != "verified" or not finding.value:
            continue
        overlapping_ids = set(finding.evidence_ids) & instr_ev_set
        if not overlapping_ids:
            continue
        overlapping_ev = [ev_by_id[eid] for eid in overlapping_ids if eid in ev_by_id]
        if not any(
            ev.evidence_type in _ELIGIBLE_RETRIEVAL_EV_TYPES
            and ev.capture_method in _SUBSTANTIVE_CAPTURES
            for ev in overlapping_ev
        ):
            continue
        # If the substantiating evidence is an observed_response, its captured URL
        # must match the instruction endpoint (one endpoint's response cannot authorize another).
        if all(ev.evidence_type == "observed_response" for ev in overlapping_ev) and not any(
            _endpoints_match(str(instr.source_url), str(ev.source_url))
            for ev in overlapping_ev
        ):
            continue
        val = finding.value
        method_match = (
            val.get("method") == instr.retrieval_method
            or val.get("retrieval_method") == instr.retrieval_method
            or instr.retrieval_method in (val.get("methods") or [])
        )
        endpoints: List[str] = []
        if isinstance(val.get("endpoint_url"), str):
            endpoints.append(val["endpoint_url"])
        if isinstance(val.get("endpoint_urls"), list):
            endpoints.extend(x for x in val["endpoint_urls"] if isinstance(x, str))
        if method_match and any(_endpoints_match(str(instr.source_url), ep) for ep in endpoints):
            endpoint_verified = True
            break
    if not endpoint_verified:
        errors.append(
            "RetrievalInstruction source_url and retrieval_method must be backed by a "
            "verified retrieval_methods finding with matching endpoint_url and evidence_ids"
        )
    if primary.assessment.connector.status not in {"implemented_tested", "not_required"}:
        errors.append(
            "RetrievalInstruction requires connector assessment status "
            "'implemented_tested' or 'not_required'"
        )

    # 5. Access-class & authentication consistency
    if primary.assessment.access_class != instr.access_class:
        errors.append(
            "RetrievalInstruction.access_class must match selected candidate access_class"
        )
    if primary.assessment.authentication_requirement != instr.authentication_requirement:
        errors.append(
            "RetrievalInstruction.authentication_requirement must match selected candidate"
        )
    if primary.assessment.quota.status == "exceeds_limit":
        errors.append("Selected candidate exceeding quota limit cannot have a RetrievalInstruction")

    # 6. Required license permissions and per-right conditions
    if plan.planning_policy is None:
        errors.append("PlanSpec.planning_policy is required when retrieval_instructions are emitted")
    else:
        required_rights = plan.planning_policy.usage_intent.required_rights
        expected_conditional_rights: set[UsageRight] = set()
        for right in required_rights:
            perm = primary.assessment.license_permissions.get(right)
            if perm is None:
                errors.append(f"Selected candidate assessment missing required usage right '{right}'")
                continue
            if perm.verification_status != "verified" or not perm.finding_ids:
                errors.append(f"Required usage right '{right}' must be verified with findings")
                continue
            if perm.status not in {"allowed", "conditional"}:
                errors.append(
                    f"Required usage right '{right}' has ineligible license status '{perm.status}'"
                )
                continue
            if perm.status == "conditional":
                expected_conditional_rights.add(right)
                if perm.conditions_status != "satisfied" or not perm.conditions:
                    errors.append(
                        f"Conditional usage right '{right}' must have satisfied conditions"
                    )
                elif instr.license_conditions_by_right.get(right) != perm.conditions:
                    errors.append(
                        f"RetrievalInstruction.license_conditions_by_right['{right}'] must "
                        "preserve the verified condition text"
                    )
            elif right in instr.license_conditions_by_right:
                errors.append(
                    f"Allowed usage right '{right}' must not carry conditional restrictions"
                )
        extra_rights = set(instr.license_conditions_by_right) - expected_conditional_rights
        if extra_rights:
            errors.append(
                "RetrievalInstruction.license_conditions_by_right contains non-conditional "
                f"or unrequired rights: {sorted(extra_rights)}"
            )

    # 7a. Live/recent freshness threshold consistency
    if spec.requirement:
        needs_freshness = (
            spec.requirement.freshness_need in {"live", "recent"}
            or spec.requirement.max_data_age_hours is not None
            or bool(spec.requirement.max_data_age)
        )
        if needs_freshness:
            expected_age, err = resolve_freshness_threshold_hours(spec.requirement)
            if err:
                errors.append(err)
            elif expected_age is None:
                errors.append("Live/recent freshness requirement has no resolvable max_data_age_hours")
            else:
                if not instr.requires_runtime_freshness_verification:
                    errors.append(
                        "RetrievalInstruction must set requires_runtime_freshness_verification=True "
                        "for live/recent freshness requirements"
                    )
                if instr.max_data_age_hours != expected_age:
                    errors.append(
                        f"RetrievalInstruction.max_data_age_hours ({instr.max_data_age_hours}) "
                        f"must preserve requirement threshold ({expected_age})"
                    )
        elif instr.requires_runtime_freshness_verification or instr.max_data_age_hours is not None:
            errors.append(
                "RetrievalInstruction must not set runtime freshness verification when no "
                "live/recent freshness requirement exists"
            )


def _validate_replan_invariants(spec: DatasetSpec, errors: List[str]) -> None:
    """Validate Orchestrator-owned bounded replanning history and exclusions."""
    if spec.replan_count != len(spec.replan_history):
        errors.append(
            f"replan_count ({spec.replan_count}) must equal len(replan_history) ({len(spec.replan_history)})"
        )
    if spec.replan_count > 2:
        errors.append("replan_count cannot exceed maximum of 2")

    expected_numbers = list(range(1, len(spec.replan_history) + 1))
    actual_numbers = [attempt.attempt_number for attempt in spec.replan_history]
    if actual_numbers != expected_numbers:
        errors.append("ReplanAttempt.attempt_number values must be sequential starting at 1")

    if not spec.replan_history:
        return

    candidates_by_id = (
        {c.candidate_id: c for c in spec.plan.candidate_sources} if spec.plan else {}
    )
    instructions = spec.plan.retrieval_instructions if spec.plan else []

    for attempt in spec.replan_history:
        excluded_urls_norm = {_normalize_endpoint_url(u) for u in attempt.excluded_urls}
        for cid in attempt.excluded_candidate_ids:
            cand = candidates_by_id.get(cid)
            if cand is not None and cand.selection_status != "rejected":
                errors.append(
                    f"Excluded candidate {cid} from replan attempt {attempt.attempt_number} "
                    "must have selection_status='rejected'"
                )
            for instr in instructions:
                if instr.candidate_id == cid:
                    errors.append(
                        f"RetrievalInstruction cannot target excluded candidate {cid}"
                    )
        for instr in instructions:
            if _normalize_endpoint_url(str(instr.source_url)) in excluded_urls_norm:
                errors.append(
                    f"RetrievalInstruction cannot target excluded endpoint URL '{instr.source_url}'"
                )


def validate_spec(spec: DatasetSpec) -> tuple[bool, List[str]]:
    """
    Validate a dataset spec.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []
    
    # Check logical consistency
    if spec.requirement and not spec.requirement.is_supported:
        if spec.plan or spec.retrieval:
            errors.append("Unsupported requirement should not have plan or retrieval")
    
    # Check agent execution order
    if spec.plan and not spec.requirement:
        errors.append("Plan exists but requirement missing (agents out of order)")
    
    if spec.retrieval and not spec.plan:
        errors.append("Retrieval exists but plan missing (agents out of order)")

    if spec.plan:
        _validate_plan_invariants(spec, errors)

    _validate_replan_invariants(spec, errors)
    
    # Check provenance on sources
    if spec.retrieval:
        for source in spec.retrieval.discovered_sources:
            if not source.source_url:
                errors.append(f"Source {source.source_id} missing source_url (provenance required)")
    
    # Check provenance on records
    if spec.retrieval:
        for record in spec.retrieval.raw_records:
            if not record.source_url:
                errors.append(f"Record {record.record_id} missing source_url (provenance required)")
    
    return len(errors) == 0, errors


def create_initial_spec(dataset_name: str, description: str) -> DatasetSpec:
    """
    Create a new dataset specification from user input.
    
    Args:
        dataset_name: Name of the dataset
        description: User's description
        
    Returns:
        Initial DatasetSpec with requirement section populated
    """
    spec_id = f"DS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"    
    return DatasetSpec(
        spec_id=spec_id,
        requirement=RequirementSpec(
            dataset_name=dataset_name,
            description=description
        )
    )