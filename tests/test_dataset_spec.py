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
    EvaluationRequirements,
    ApplicationPlanningPolicy,
    UsageIntent,
    derive_evaluation_requirements,
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
    assert req.requested_update_frequency is None
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


def test_usage_intent_requires_use_and_defaults_to_only_use():
    assert UsageIntent().required_rights == ["use"]
    assert UsageIntent(required_rights=["use", "commercial_use"]).required_rights == [
        "use", "commercial_use"
    ]

    with pytest.raises(ValidationError, match="must always require the 'use' right"):
        UsageIntent(required_rights=["modify"])

    with pytest.raises(ValidationError, match="must be unique"):
        UsageIntent(required_rights=["use", "use"])


def test_evaluation_requirements_validate_classification_and_dimensions():
    requirements = EvaluationRequirements(
        policy_version="1",
        required_dimensions=["purpose", "variables"],
        dimension_criticality={"purpose": "essential", "variables": "essential"},
        required_variables=["PM2.5"],
    )

    assert requirements.dimension_criticality["purpose"] == "essential"

    with pytest.raises(ValidationError, match="required_dimensions must be unique"):
        EvaluationRequirements(
            policy_version="1",
            required_dimensions=["purpose", "purpose"],
            dimension_criticality={"purpose": "essential"},
        )

    with pytest.raises(ValidationError, match="classify every required dimension"):
        EvaluationRequirements(
            policy_version="1",
            required_dimensions=["purpose", "access"],
            dimension_criticality={"purpose": "essential"},
        )


@pytest.mark.parametrize(
    "field,values",
    [
        ("required_variables", ["PM2.5", "PM 2.5"]),
        ("required_formats", ["CSV", "text/csv"]),
        ("required_retrieval_methods", ["direct_download", "direct_download"]),
    ],
)
def test_evaluation_requirements_reject_duplicate_constraints(field, values):
    kwargs = {
        "policy_version": "1",
        "required_dimensions": ["purpose"],
        "dimension_criticality": {"purpose": "essential"},
        field: values,
    }

    with pytest.raises(ValidationError):
        EvaluationRequirements(**kwargs)


def test_evaluation_requirements_require_dimension_for_constraints():
    with pytest.raises(ValidationError, match="requires 'variables'"):
        EvaluationRequirements(
            policy_version="1",
            required_dimensions=["purpose"],
            dimension_criticality={"purpose": "essential"},
            required_variables=["temperature"],
        )


def test_derive_evaluation_requirements_uses_structured_request_fields():
    requirement = RequirementSpec(
        dataset_name="Weather",
        description="Hourly city weather",
        expected_features=["temperature", "humidity", "Temperature"],
        target_variable="heat_index",
        target_requires_derivation=True,
        geography="Delhi",
        time_range="2020-2025",
        time_granularity="hourly",
        freshness_need="recent",
        output_format="excel",
    )

    policy = derive_evaluation_requirements(
        requirement,
        UsageIntent(required_rights=["use", "modify"]),
        required_formats=["text/csv"],
        required_retrieval_methods=["http_api"],
    )

    assert isinstance(policy, ApplicationPlanningPolicy)
    assert policy.usage_intent.required_rights == ["use", "modify"]
    requirements = policy.evaluation_requirements
    assert requirements.required_variables == ["temperature", "humidity"]
    assert requirements.required_formats == ["text/csv"]
    assert requirements.required_retrieval_methods == ["http_api"]
    assert "geography" in requirements.required_dimensions
    assert "temporal_coverage" in requirements.required_dimensions
    assert "resolution" in requirements.required_dimensions
    assert requirements.dimension_criticality["publisher_reliability"] == "advisory"
    assert "excel" not in requirements.required_formats
    restored = ApplicationPlanningPolicy.model_validate(policy.model_dump())
    assert restored.evaluation_requirements == requirements
    assert restored.usage_intent.required_rights == ["use", "modify"]


def test_derive_evaluation_requirements_accepts_usage_intent_model():
    usage_intent = UsageIntent(required_rights=["use", "redistribute"])
    requirement = RequirementSpec(dataset_name="Data", description="Data request")

    policy = derive_evaluation_requirements(requirement, usage_intent)

    assert policy.usage_intent == usage_intent


@pytest.mark.parametrize("usage_intent", [None, "use", 42, ["use"]])
def test_derive_evaluation_requirements_rejects_invalid_usage_intent_types(usage_intent):
    requirement = RequirementSpec(dataset_name="Data", description="Data request")

    with pytest.raises(ValidationError, match="UsageIntent"):
        derive_evaluation_requirements(requirement, usage_intent)


def test_schema_variable_identity_does_not_apply_catalog_aliases():
    from agents.source_catalog import normalize_feature_name

    requirements = EvaluationRequirements(
        policy_version="1",
        required_dimensions=["variables"],
        dimension_criticality={"variables": "essential"},
        required_variables=["timestamp", "time"],
    )

    assert requirements.required_variables == ["timestamp", "time"]
    assert normalize_feature_name("timestamp") == normalize_feature_name("time")


@pytest.mark.parametrize(
    "freshness_need,max_data_age,requested_update_frequency,expected",
    [
        ("live", None, None, {"freshness", "update_frequency"}),
        ("recent", None, None, {"freshness", "update_frequency"}),
        ("historical", None, None, {"temporal_coverage"}),
        ("unspecified", None, None, set()),
        ("historical", "24 hours", None, {"temporal_coverage", "freshness"}),
        ("unspecified", "24 hours", None, {"freshness"}),
        ("unspecified", None, "daily", {"update_frequency"}),
    ],
)
def test_freshness_and_update_frequency_policy(freshness_need, max_data_age,
                                              requested_update_frequency, expected):
    requirement = RequirementSpec(
        dataset_name="Data",
        description="Data request",
        freshness_need=freshness_need,
        max_data_age=max_data_age,
        requested_update_frequency=requested_update_frequency,
    )

    policy = derive_evaluation_requirements(requirement, UsageIntent())
    actual = set(policy.evaluation_requirements.required_dimensions)
    freshness_dimensions = {"freshness", "update_frequency", "temporal_coverage"}
    assert actual & freshness_dimensions == expected


def test_max_data_age_hours_alone_marks_freshness_essential():
    """Regression: max_data_age_hours populated while max_data_age is None must
    still mark freshness as essential.  Previously the condition only checked
    max_data_age (the string field), so a requirement with only max_data_age_hours
    set would silently produce a policy without a freshness dimension."""
    requirement = RequirementSpec(
        dataset_name="Data",
        description="Data request",
        freshness_need="unspecified",
        max_data_age=None,           # string field deliberately absent
        max_data_age_hours=24,       # only the integer field is set
    )

    policy = derive_evaluation_requirements(requirement, UsageIntent())
    assert "freshness" in policy.evaluation_requirements.required_dimensions, (
        "freshness must be essential when max_data_age_hours is set, "
        "even if max_data_age is None"
    )
    assert (
        policy.evaluation_requirements.dimension_criticality["freshness"] == "essential"
    )


def test_requested_update_frequency_rejects_blank_value():
    with pytest.raises(ValidationError, match="requested_update_frequency must not be blank"):
        RequirementSpec(
            dataset_name="Data",
            description="Data request",
            requested_update_frequency=" \t ",
        )


@pytest.mark.parametrize("policy_version", ["", " ", "\t\n"])
def test_evaluation_requirements_rejects_blank_policy_version(policy_version):
    with pytest.raises(ValidationError, match="policy_version must not be blank"):
        EvaluationRequirements(
            policy_version=policy_version,
            required_dimensions=["purpose"],
            dimension_criticality={"purpose": "essential"},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_variables", ["   "]),
        ("required_variables", ["\x00"]),
        ("required_variables", ["---"]),
        ("required_variables", [42]),
        ("required_variables", [None]),
        ("required_formats", [""]),
        ("required_formats", ["text/"]),
        ("required_formats", ["/csv"]),
        ("required_formats", ["text/csv; charset"]),
        ("required_formats", ['text/csv; charset="utf-8']),
        ("required_formats", ["text / csv"]),
        ("required_formats", [42]),
        ("required_formats", [None]),
    ],
)
def test_evaluation_requirements_rejects_invalid_identifiers(field, value):
    kwargs = {
        "policy_version": "1",
        "required_dimensions": ["variables", "formats"],
        "dimension_criticality": {
            "variables": "essential",
            "formats": "essential",
        },
        field: value,
    }

    with pytest.raises(ValidationError):
        EvaluationRequirements(**kwargs)


def test_evaluation_requirements_rejects_non_string_policy_version():
    with pytest.raises(ValidationError):
        EvaluationRequirements(
            policy_version=1,
            required_dimensions=["purpose"],
            dimension_criticality={"purpose": "essential"},
        )


def test_mime_parameters_are_syntax_checked_but_ignored_for_format_identity():
    requirements = EvaluationRequirements(
        policy_version="1",
        required_dimensions=["formats"],
        dimension_criticality={"formats": "essential"},
        required_formats=['application/json; PrOfIlE="https://example.test/a;b"'],
    )
    assert requirements.required_formats == [
        'application/json; PrOfIlE="https://example.test/a;b"'
    ]

    with pytest.raises(ValidationError, match="unique after normalization"):
        EvaluationRequirements(
            policy_version="1",
            required_dimensions=["formats"],
            dimension_criticality={"formats": "essential"},
            required_formats=[
                'text/csv; charset="utf-8"',
                "TEXT/CSV; CHARSET=UTF-8",
            ],
        )

    # Syntax is checked, but MIME parameter semantics are deliberately not
    # verified by this planning-policy model.
    accepted = EvaluationRequirements(
        policy_version="1",
        required_dimensions=["formats"],
        dimension_criticality={"formats": "essential"},
        required_formats=['text/csv; charset="not-a-validated-charset"'],
    )
    assert accepted.required_formats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
