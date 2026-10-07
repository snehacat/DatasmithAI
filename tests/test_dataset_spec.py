"""
Tests for Dataset Specification (Data Contract)
"""

import pytest
from core.dataset_spec import (
    DatasetSpec,
    RequirementSpec,
    DataModality,
    PipelineStatus,
    AgentStatus,
    AgentExecution,
    DataRecord,
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


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
