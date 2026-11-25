"""
Tests for Dataset Specification (Data Contract)
"""

import json

import pytest
from pydantic import ValidationError
from core.dataset_spec import (
    DatasetSpec,
    RequirementSpec,
    DataModality,
    PipelineStatus,
    AgentStatus,
    AgentExecution,
    DataRecord,
    DownloadedArtifact,
    FailedSource,
    RetrievalSpec,
    validate_spec,
    create_initial_spec
)
from datetime import datetime


def test_create_initial_spec():
    """Test creating initial spec from user input."""
    spec = create_initial_spec("Test Dataset", "A test description")
    
    assert spec.spec_id.startswith("DS_")
    assert spec.requirement is not None
    assert spec.requirement.dataset_name == "Test Dataset"
    assert spec.requirement.description == "A test description"
    assert spec.pipeline_status == PipelineStatus.PENDING
    assert spec.plan is None
    assert spec.retrieval is None


def test_requirement_spec_validation():
    """Test requirement spec validation."""
    req = RequirementSpec(
        dataset_name="Flood Dataset",
        description="Flood prediction data",
        domain="environmental",
        data_modality=DataModality.TABULAR,
        completeness=0.85
    )
    
    assert req.dataset_name == "Flood Dataset"
    assert req.domain == "environmental"
    assert req.data_modality == DataModality.TABULAR
    assert 0.0 <= req.completeness <= 1.0


def test_requirement_spec_new_fields():
    """Test new RequirementSpec fields with valid values."""
    req = RequirementSpec(
        dataset_name="Weather Data",
        description="Real-time weather data",
        freshness_need="live",
        max_data_age="24 hours",
        geography="United States",
        time_range="Last 7 days",
        output_format="json",
        constraints=["free sources only", "no login required"],
        missing_or_unclear=["specific cities"],
        clarifying_questions=["Which cities do you need?"]
    )
    
    assert req.freshness_need == "live"
    assert req.max_data_age == "24 hours"
    assert req.geography == "United States"
    assert req.time_range == "Last 7 days"
    assert req.output_format == "json"
    assert len(req.constraints) == 2
    assert "free sources only" in req.constraints
    assert len(req.missing_or_unclear) == 1
    assert len(req.clarifying_questions) == 1


def test_requirement_spec_defaults():
    """Test that new fields have proper defaults."""
    req = RequirementSpec(
        dataset_name="Test Dataset",
        description="Test description"
    )
    
    # Check defaults
    assert req.freshness_need == "unspecified"
    assert req.max_data_age is None
    assert req.geography is None
    assert req.time_range is None
    assert req.output_format == "csv"
    assert req.constraints == []
    assert req.missing_or_unclear == []
    assert req.clarifying_questions == []


def test_requirement_spec_invalid_freshness():
    """Test that invalid freshness_need is rejected."""
    with pytest.raises(ValueError):
        RequirementSpec(
            dataset_name="Test",
            description="Test",
            freshness_need="immediate"  # Invalid - not in allowed values
        )


def test_requirement_spec_invalid_output_format():
    """Test that invalid output_format is rejected."""
    with pytest.raises(ValueError):
        RequirementSpec(
            dataset_name="Test",
            description="Test",
            output_format="pdf"  # Invalid - not in allowed values
        )


def test_completeness_validation():
    """Test completeness scores are in valid range."""
    # Valid completeness
    req = RequirementSpec(
        dataset_name="Test",
        description="Test",
        completeness=0.5
    )
    assert req.completeness == 0.5
    
    # Invalid completeness should fail
    with pytest.raises(ValueError):
        RequirementSpec(
            dataset_name="Test",
            description="Test",
            completeness=1.5  # > 1.0
        )
    
    with pytest.raises(ValueError):
        RequirementSpec(
            dataset_name="Test",
            description="Test",
            completeness=-0.1  # < 0.0
        )


def test_agent_execution_tracking():
    """Test tracking agent executions."""
    spec = create_initial_spec("Test", "Test description")
    
    # Add execution
    exec1 = AgentExecution(
        agent_name="RequirementAnalyzer",
        status=AgentStatus.SUCCESS,
        started_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        duration_seconds=2.5,
        outputs_written=["requirement"],
        llm_calls=1
    )
    spec.add_execution(exec1)
    
    assert len(spec.execution_history) == 1
    assert spec.has_agent_succeeded("RequirementAnalyzer")
    assert not spec.has_agent_succeeded("PlannerAgent")
    
    # Get latest execution
    latest = spec.get_latest_execution("RequirementAnalyzer")
    assert latest is not None
    assert latest.status == AgentStatus.SUCCESS


def test_provenance_requirement():
    """Test that data records require source_url."""
    # Valid record with source_url
    record = DataRecord(
        record_id="REC_001",
        source_id="SRC_001",
        raw_data={"temperature": 25.5},
        extracted_at=datetime.now().isoformat(),
        completeness=0.9,
        extraction_method="csv_parser",
        source_url="https://noaa.gov/data/temp.csv",
        source_type="meteorological_data_provider"
    )
    assert record.source_url == "https://noaa.gov/data/temp.csv"
    
    # Record without source_url should fail validation at spec level
    # (Pydantic will require it at model level)


def test_spec_validation():
    """Test spec validation logic."""
    spec = create_initial_spec("Test", "Test")
    
    # Initially valid
    is_valid, errors = validate_spec(spec)
    assert is_valid
    assert len(errors) == 0
    
    # Mark requirement as unsupported but add plan (inconsistent)
    spec.requirement.is_supported = False
    spec.requirement.unsupported_reason = "Image data not yet supported"
    spec.plan = None  # This is consistent
    
    is_valid, errors = validate_spec(spec)
    assert is_valid  # No plan, so consistent


def test_warnings_and_errors():
    """Test adding warnings and errors."""
    spec = create_initial_spec("Test", "Test")
    
    spec.add_warning("Low completeness in domain detection")
    spec.add_error("Failed to connect to data source")
    
    assert len(spec.warnings) == 1
    assert len(spec.errors) == 1
    assert "Low completeness" in spec.warnings[0]
    assert "Failed to connect" in spec.errors[0]


def test_spec_update_tracking():
    """Test that updates are tracked."""
    spec = create_initial_spec("Test", "Test")
    
    created_at = spec.created_at
    updated_at = spec.updated_at
    
    # Small delay to ensure timestamp changes
    import time
    time.sleep(0.01)
    
    spec.mark_updated()
    
    assert spec.created_at == created_at  # Unchanged
    assert spec.updated_at > updated_at  # Changed

def test_empty_source_url_is_rejected():
    import pytest
    from pydantic import ValidationError
    from core.dataset_spec import DataRecord
    with pytest.raises(ValidationError):
        DataRecord(record_id="r1", source_id="s1", raw_data={},
                   extracted_at="2026-10-07T00:00:00", extraction_method="csv_parser",
                   source_url="", source_type="api")


def test_spec_ids_are_unique():
    from core.dataset_spec import create_initial_spec
    ids = {create_initial_spec("a", "b").spec_id for _ in range(20)}
    assert len(ids) == 20

def test_invalid_assignment_is_rejected():
    """Bad values must be rejected on assignment, not only at creation."""
    r = RequirementSpec(dataset_name="a", description="b")

    with pytest.raises(ValidationError):
        r.output_format = "pdf"        # only csv / excel / json allowed
    with pytest.raises(ValidationError):
        r.completeness = 9             # must be 0 to 1
    with pytest.raises(ValidationError):
        r.time_window_days = 0         # must be >= 1

    assert r.output_format == "csv"    # the bad value was not stored
    r.output_format = "json"           # a valid value still works
    assert r.output_format == "json"


def test_downloaded_artifact_validation():
    artifact = DownloadedArtifact(
        relative_path="weather/daily.csv",
        sha256="a" * 64,
        bytes=128,
        content_type="text/csv",
        source_url="https://example.test/weather.csv",
        retrieved_at="2026-10-07T12:30:00Z",
        extraction_method="direct_download",
        row_count=4,
        license="CC-BY-4.0",
    )

    assert artifact.relative_path == "weather/daily.csv"
    assert artifact.bytes == 128
    assert artifact.row_count == 4
    assert artifact.license == "CC-BY-4.0"


def test_downloaded_artifact_accepts_nested_relative_path():
    artifact = DownloadedArtifact(
        relative_path="raw/weather/2026/daily.csv",
        sha256="a" * 64,
        bytes=128,
        source_url="https://example.test/weather.csv",
        retrieved_at="2026-10-07T12:30:00Z",
        extraction_method="direct_download",
    )

    assert artifact.relative_path == "raw/weather/2026/daily.csv"


@pytest.mark.parametrize(
    "overrides",
    [
        {"relative_path": "../outside.csv"},
        {"relative_path": "raw/../../outside.csv"},
        {"relative_path": "/absolute/path.csv"},
        {"relative_path": "C:\\outside.csv"},
        {"relative_path": "bad\x00path.csv"},
        {"sha256": "not-a-checksum"},
        {"bytes": -1},
        {"row_count": -1},
        {"retrieved_at": "not-a-timestamp"},
    ],
)
def test_downloaded_artifact_rejects_invalid_values(overrides):
    values = {
        "relative_path": "data.csv",
        "sha256": "a" * 64,
        "bytes": 0,
        "source_url": "https://example.test/data.csv",
        "retrieved_at": "2026-10-07T12:30:00Z",
        "extraction_method": "direct_download",
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        DownloadedArtifact(**values)


@pytest.mark.parametrize(
    "missing_field",
    ["relative_path", "sha256", "bytes", "source_url", "retrieved_at", "extraction_method"],
)
def test_downloaded_artifact_requires_metadata(missing_field):
    values = {
        "relative_path": "raw/data.csv",
        "sha256": "a" * 64,
        "bytes": 128,
        "source_url": "https://example.test/data.csv",
        "retrieved_at": "2026-10-07T12:30:00Z",
        "extraction_method": "direct_download",
    }
    values.pop(missing_field)

    with pytest.raises(ValidationError):
        DownloadedArtifact(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("relative_path", ""),
        ("sha256", ""),
        ("source_url", ""),
        ("extraction_method", ""),
    ],
)
def test_downloaded_artifact_rejects_empty_required_strings(field, value):
    values = {
        "relative_path": "raw/data.csv",
        "sha256": "a" * 64,
        "bytes": 128,
        "source_url": "https://example.test/data.csv",
        "retrieved_at": "2026-10-07T12:30:00Z",
        "extraction_method": "direct_download",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        DownloadedArtifact(**values)


@pytest.mark.parametrize(
    "reason_code",
    ["timeout", "too_large", "bad_content_type", "auth_required", "parse_error"],
)
def test_failed_source_accepts_supported_reason_codes(reason_code):
    failed_source = FailedSource(
        source_url="https://example.test/data.csv",
        source_title="Example data",
        reason_code=reason_code,
        error_message="Retrieval failed",
        retrieved_at="2026-10-07T12:30:00Z",
    )

    assert failed_source.reason_code == reason_code


def test_failed_source_rejects_unsupported_reason_code():
    with pytest.raises(ValidationError):
        FailedSource(
            source_url="https://example.test/data.csv",
            reason_code="not_a_reason",
            error_message="Retrieval failed",
            retrieved_at="2026-10-07T12:30:00Z",
        )


@pytest.mark.parametrize("missing_field", ["reason_code", "error_message", "retrieved_at"])
def test_failed_source_requires_metadata(missing_field):
    values = {
        "source_url": "https://example.test/data.csv",
        "source_title": "Example data",
        "reason_code": "timeout",
        "error_message": "Request timed out",
        "retrieved_at": "2026-10-07T12:30:00Z",
    }
    values.pop(missing_field)

    with pytest.raises(ValidationError):
        FailedSource(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("error_message", ""),
        ("reason_code", ""),
        ("retrieved_at", ""),
    ],
)
def test_failed_source_rejects_empty_required_values(field, value):
    values = {
        "source_url": "https://example.test/data.csv",
        "reason_code": "timeout",
        "error_message": "Request timed out",
        "retrieved_at": "2026-10-07T12:30:00Z",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        FailedSource(**values)


def test_retrieval_spec_serializes_and_deserializes_artifacts_and_failures():
    retrieval = RetrievalSpec(
        downloaded_artifacts=[
            DownloadedArtifact(
                relative_path="raw/data.json",
                sha256="b" * 64,
                bytes=64,
                source_url="https://example.test/data.json",
                retrieved_at="2026-10-07T12:30:00Z",
                extraction_method="direct_download",
            )
        ],
        failed_sources=[
            FailedSource(
                source_url="https://example.test/other.csv",
                source_title="Other data",
                reason_code="timeout",
                error_message="Request timed out",
                retrieved_at="2026-10-07T12:31:00Z",
            )
        ],
    )

    serialized = retrieval.model_dump(mode="json")
    restored = RetrievalSpec.model_validate(serialized)

    assert restored.downloaded_artifacts[0].relative_path == "raw/data.json"
    assert restored.downloaded_artifacts[0].retrieved_at.isoformat() == "2026-10-07T12:30:00+00:00"
    assert restored.failed_sources[0].reason_code == "timeout"
    assert restored.raw_records == []


def test_dataset_spec_migrates_compatible_legacy_failed_sources():
    legacy_failure = {
        "source_url": "https://example.test/data.csv",
        "source_title": "Example data",
        "reason_code": "timeout",
        "error_message": "Request timed out",
        "retrieved_at": "2026-10-07T12:30:00Z",
    }
    payload = create_initial_spec("Dataset", "Description").model_dump(mode="json")
    payload["retrieval"] = {"failed_sources": [legacy_failure]}

    restored = DatasetSpec.model_validate_json(json.dumps(payload))

    assert isinstance(restored.retrieval.failed_sources[0], FailedSource)
    assert restored.retrieval.failed_sources[0].reason_code == "timeout"
    assert restored.model_dump(mode="json")["retrieval"]["failed_sources"] == [legacy_failure]


@pytest.mark.parametrize(
    "legacy_failure",
    [
        {
            "source_url": "https://example.test/data.csv",
            "error_message": "Request timed out",
            "retrieved_at": "2026-10-07T12:30:00Z",
        },
        {
            "source_url": "https://example.test/data.csv",
            "reason_code": "timeout",
            "error_message": "Request timed out",
        },
        {
            "reason_code": "timeout",
            "error_message": "Request timed out",
            "retrieved_at": "2026-10-07T12:30:00Z",
            "legacy_payload": {"detail": "must not be discarded"},
        },
    ],
)
def test_dataset_spec_rejects_unmappable_legacy_failed_sources(legacy_failure):
    payload = create_initial_spec("Dataset", "Description").model_dump(mode="json")
    payload["retrieval"] = {"failed_sources": [legacy_failure]}

    with pytest.raises(ValidationError, match=r"failed_sources\[0\] cannot be migrated safely"):
        DatasetSpec.model_validate(payload)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
