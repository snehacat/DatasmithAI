"""
DataSmith AI - Dataset Specification Schema

Universal data contract that ALL agents read and write.
Grows through the pipeline, with each agent adding its section.
"""

from pydantic import BaseModel, Field
import uuid
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

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


# ============================================================================
# REQUIREMENT SECTION (Phase 1: Understanding)
# ============================================================================

class RequirementSpec(BaseModel):
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
    max_data_age_hours: Optional[int] = Field(None, description="max_data_age as hours (live=1, recent=24)")
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
    inferred_by_model: List[str] = Field(default_factory=list, description="What model inferred")
    
    # Completeness (how much detail the user provided)
    completeness: float = Field(0.0, ge=0.0, le=1.0, description="Request completeness (0-1, based on detail provided)")
    
    # Clarification tracking
    clarification_round: int = Field(0, description="Number of clarification rounds completed (0 or 1)")


# ============================================================================
# PLANNING SECTION (Phase 2: Finding data)
# ============================================================================

class SourceRequirement(BaseModel):
    """A required source type."""
    source_type: str = Field(..., description="Type of source (e.g., meteorological_data_provider)")
    why_needed: str = Field(..., description="Why this source is relevant")
    information_expected: List[str] = Field(default_factory=list, description="What info it should provide")
    importance: Literal["required", "useful", "fallback", "optional"] = Field(..., description="How important")
    preferred_access_methods: List[str] = Field(default_factory=list, description="How to access (api, download, etc)")
    priority: int = Field(..., ge=1, description="Priority (1=highest)")


class SearchQuery(BaseModel):
    """A search query for finding sources."""
    query: str = Field(..., description="The actual search query")
    purpose: str = Field(..., description="What this query aims to find")
    related_source_type: str = Field(..., description="Which source type this targets")
    priority: int = Field(..., ge=1, description="Priority (1=highest)")


class PlanSpec(BaseModel):
    """Planning output - what sources to find and how."""
    
    # Core strategy
    source_requirements: List[SourceRequirement] = Field(default_factory=list, description="Required source types")
    search_queries: List[SearchQuery] = Field(default_factory=list, description="Queries to execute")
    
    # Strategy
    primary_source_types: List[str] = Field(default_factory=list, description="Primary sources to try first")
    fallback_source_types: List[str] = Field(default_factory=list, description="Fallback if primary fails")
    reasoning: Optional[str] = Field(None, description="Overall strategy reasoning")
    
    # Confidence
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Plan confidence")


# ============================================================================
# RETRIEVAL SECTION (Phase 2: Collecting data)
# ============================================================================

class SourceRecord(BaseModel):
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

class DataRecord(BaseModel):
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


class RetrievalSpec(BaseModel):
    """Retrieval output - discovered sources and raw data."""
    
    # Sources
    discovered_sources: List[SourceRecord] = Field(default_factory=list, description="All discovered sources")
    failed_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources that failed")
    
    # Data
    raw_records: List[DataRecord] = Field(default_factory=list, description="All extracted records")
    
    # Coverage
    requirements_satisfied: List[str] = Field(default_factory=list, description="Which requirements met")
    requirements_missing: List[str] = Field(default_factory=list, description="Which requirements not met")
    overall_completeness: float = Field(0.0, ge=0.0, le=1.0, description="Coverage completeness")
    
    # Metadata
    total_sources_discovered: int = Field(0, description="Number of sources found")
    total_records_extracted: int = Field(0, description="Number of records extracted")


# ============================================================================
# EXECUTION HISTORY
# ============================================================================

class AgentExecution(BaseModel):
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

class DatasetSpec(BaseModel):
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

def validate_spec(spec: DatasetSpec) -> tuple[bool, List[str]]:
    """
    Validate a dataset spec.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check basic structure
    try:
        # Pydantic validation happens automatically
        pass
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
    
    # Check logical consistency
    if spec.requirement and not spec.requirement.is_supported:
        if spec.plan or spec.retrieval:
            errors.append("Unsupported requirement should not have plan or retrieval")
    
    # Check agent execution order
    if spec.plan and not spec.requirement:
        errors.append("Plan exists but requirement missing (agents out of order)")
    
    if spec.retrieval and not spec.plan:
        errors.append("Retrieval exists but plan missing (agents out of order)")
    
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