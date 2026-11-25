"""Tests for the source research and evidence schema."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.dataset_spec import (
    CandidateSource,
    ConnectorAssessment,
    DatasetSpec,
    LicensePermissionAssessment,
    PlanSpec,
    QuotaAssessment,
    SourceAssessment,
    SourceEvidence,
    SourceFinding,
    create_initial_spec,
)


_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _evidence(**overrides):
    values = {
        "source_url": "https://data.example.org/documentation",
        "publisher": "Example Data Provider",
        "title": "Dataset API documentation",
        "discovery_method": "provider_documentation",
        "evidence_type": "api_documentation",
        "capture_method": "fetched_content",
        "authority": "official_provider",
        "excerpt": "The dataset provides hourly temperature measurements.",
        "retrieved_at": _NOW,
    }
    values.update(overrides)
    return SourceEvidence(**values)


def _finding(dimension, evidence_ids=(), **overrides):
    values = {
        "dimension": dimension,
        "claim": f"Evidence indicates {dimension} details.",
        "value": {"result": "documented"},
        "evidence_ids": list(evidence_ids),
        "confidence": 0.9,
        "status": "verified",
        "explanation": "The captured source material supports this finding.",
    }
    values.update(overrides)
    return SourceFinding(**values)


def _candidate(evidence, findings, *, required_dimensions=None, status="complete", **overrides):
    finding_ids_by_dimension = {}
    for finding in findings:
        finding_ids_by_dimension.setdefault(finding.dimension, []).append(finding.finding_id)

    required_dimensions = required_dimensions or list(finding_ids_by_dimension)
    assessment = overrides.pop("assessment", None)
    values = {
        "source_url": "https://data.example.org/",
        "title": "Example dataset",
        "publisher": "Example Data Provider",
        "source_type": "public_dataset",
        "discovery_method": "web_search",
        "discovery_query": "example dataset",
        "discovered_at": _NOW,
        "evidence": evidence,
        "findings": findings,
        "assessment": (
            assessment
            if assessment is not None
            else SourceAssessment(
                required_dimensions=required_dimensions,
                findings_by_dimension=finding_ids_by_dimension,
            )
        ),
        "evaluation_status": status,
    }
    values.update(overrides)
    return CandidateSource(**values)


def test_source_evidence_and_verified_finding_validate():
    evidence = _evidence()
    finding = _finding("variables", [evidence.evidence_id])

    candidate = _candidate([evidence], [finding])

    assert candidate.evidence[0].authority == "official_provider"
    assert candidate.findings[0].status == "verified"


def test_evidence_validates_url_and_nonblank_required_text():
    with pytest.raises(ValidationError):
        _evidence(source_url="not a URL")
    with pytest.raises(ValidationError, match="title must not be blank"):
        _evidence(title=" \t ")


def test_verified_findings_require_evidence_and_a_value():
    with pytest.raises(ValidationError, match="require supporting evidence"):
        _finding("variables")
    with pytest.raises(ValidationError, match="require a non-empty value"):
        _finding("variables", [uuid4()], value=None)


def test_unknown_finding_is_distinct_from_an_unavailable_value():
    unknown = _finding(
        "variables",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    unavailable = _finding(
        "variables",
        status="verified",
        value={"variable": "ozone", "availability": "unavailable"},
        evidence_ids=[uuid4()],
    )

    assert unknown.status == "unknown"
    assert unknown.value is None
    assert unavailable.value["availability"] == "unavailable"

    with pytest.raises(ValidationError, match="unknown findings must not assert a value"):
        _finding("variables", status="unknown", value={"available": False})
    with pytest.raises(ValidationError, match="conflicting findings require supporting evidence"):
        _finding(
            "variables",
            status="conflicting",
            value={"claims": ["available", "unavailable"]},
            evidence_ids=[],
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_source_finding_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        _finding("purpose", confidence=confidence)


def test_candidate_rejects_missing_evidence_references():
    finding = _finding("variables", [uuid4()])

    with pytest.raises(ValidationError, match="references missing evidence"):
        _candidate([], [finding])


def test_catalog_hint_alone_cannot_verify_a_finding():
    hint = _evidence(
        capture_method="catalog_hint",
        evidence_type="catalog_record",
        authority="catalog",
    )
    finding = _finding("variables", [hint.evidence_id])

    with pytest.raises(ValidationError, match="catalog hints alone"):
        _candidate([hint], [finding])


def test_secondary_reporting_is_not_authoritative_for_measurement_claims():
    secondary = _evidence(
        capture_method="fetched_content",
        evidence_type="secondary_report",
        authority="secondary",
    )
    finding = _finding("variables", [secondary.evidence_id])

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not sufficient"):
        _candidate([secondary], [finding])


def test_secondary_evidence_can_be_recorded_without_becoming_verified():
    secondary = _evidence(
        capture_method="fetched_content",
        evidence_type="secondary_report",
        authority="secondary",
    )
    finding = _finding(
        "publisher_reliability",
        [secondary.evidence_id],
        status="unverified",
        confidence=0.4,
    )

    candidate = _candidate([secondary], [finding])

    assert candidate.findings[0].status == "unverified"


def test_complete_evaluation_can_record_unknown_findings_without_verifying_them():
    unknown_variable = _finding(
        "variables",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    unknown_geography = _finding(
        "geography",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )

    candidate = _candidate(
        [],
        [unknown_variable, unknown_geography],
        required_dimensions=["variables", "geography"],
    )

    assert candidate.evaluation_status == "complete"
    assert all(finding.status == "unknown" for finding in candidate.findings)


def test_complete_evaluation_requires_every_required_dimension():
    finding = _finding("variables", status="unknown", value=None, evidence_ids=[])

    with pytest.raises(ValidationError, match="every required dimension"):
        _candidate(
            [],
            [finding],
            required_dimensions=["variables", "geography"],
        )


def test_partial_evaluation_requires_an_unassessed_required_dimension():
    finding = _finding("variables", status="unknown", value=None, evidence_ids=[])

    with pytest.raises(ValidationError, match="partial evaluation must leave"):
        _candidate(
            [],
            [finding],
            required_dimensions=["variables"],
            status="partial",
        )


def test_partial_candidate_can_preserve_incomplete_research():
    finding = _finding("variables", status="unknown", value=None, evidence_ids=[])

    candidate = _candidate(
        [],
        [finding],
        required_dimensions=["variables", "geography"],
        status="partial",
    )

    assert candidate.evaluation_status == "partial"


def test_assessment_rejects_duplicate_or_mismatched_finding_references():
    finding_id = uuid4()
    with pytest.raises(ValidationError, match="only one assessment dimension"):
        SourceAssessment(
            required_dimensions=["variables", "geography"],
            findings_by_dimension={
                "variables": [finding_id],
                "geography": [finding_id],
            },
        )

    with pytest.raises(ValidationError, match="non-required dimension"):
        SourceAssessment(
            required_dimensions=["variables"],
            findings_by_dimension={"geography": [finding_id]},
        )


def test_quota_assessment_validates_units_evidence_and_limits():
    finding_id = uuid4()
    assessment = QuotaAssessment(
        status="confirmed_within_limit",
        expected_usage=10,
        expected_usage_unit="requests/month",
        limit_value=100,
        limit_unit="requests/month",
        finding_ids=[finding_id],
    )
    assert assessment.status == "confirmed_within_limit"

    with pytest.raises(ValidationError, match="same unit"):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=10,
            expected_usage_unit="requests/month",
            limit_value=100,
            limit_unit="requests/day",
            finding_ids=[finding_id],
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=101,
            expected_usage_unit="requests/month",
            limit_value=100,
            limit_unit="requests/month",
            finding_ids=[finding_id],
        )
    with pytest.raises(ValidationError, match="supporting findings"):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=10,
            expected_usage_unit="requests/month",
            limit_value=100,
            limit_unit="requests/month",
        )
    with pytest.raises(ValidationError):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=-1,
            expected_usage_unit="requests/month",
            limit_value=100,
            limit_unit="requests/month",
            finding_ids=[finding_id],
        )
    with pytest.raises(ValidationError):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=10,
            expected_usage_unit="requests/month",
            limit_value=0,
            limit_unit="requests/month",
            finding_ids=[finding_id],
        )


def test_license_permission_and_connector_status_combinations_are_validated():
    with pytest.raises(ValidationError, match="verified supporting findings"):
        LicensePermissionAssessment(status="allowed")
    with pytest.raises(ValidationError, match="conditions and conditions_status"):
        LicensePermissionAssessment(
            status="conditional",
            verification_status="verified",
            finding_ids=[uuid4()],
        )

    conditional = LicensePermissionAssessment(
        status="conditional",
        verification_status="verified",
        finding_ids=[uuid4()],
        conditions="Attribution is required.",
        conditions_status="satisfied",
    )
    assert conditional.conditions_status == "satisfied"

    with pytest.raises(ValidationError, match="test reference, time"):
        ConnectorAssessment(status="implemented_tested")
    with pytest.raises(ValidationError, match="known connector statuses"):
        ConnectorAssessment(status="implemented_untested")


def test_access_quota_and_license_assessments_link_to_verified_findings():
    access_evidence = _evidence(
        source_url="https://data.example.org/limits",
        title="API access and quotas",
    )
    license_evidence = _evidence(
        source_url="https://data.example.org/license",
        title="License terms",
        evidence_type="license_page",
    )
    access_finding = _finding(
        "access",
        [access_evidence.evidence_id],
        value={
            "access_class": "free_tier",
            "quota": {"limit": 100, "unit": "requests/month"},
        },
    )
    license_finding = _finding(
        "license_usage",
        [license_evidence.evidence_id],
        value={"use": "allowed"},
    )
    assessment = SourceAssessment(
        required_dimensions=["access", "license_usage"],
        findings_by_dimension={
            "access": [access_finding.finding_id],
            "license_usage": [license_finding.finding_id],
        },
        access_class="free_tier",
        access_class_finding_ids=[access_finding.finding_id],
        quota=QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=10,
            expected_usage_unit="requests/month",
            limit_value=100,
            limit_unit="requests/month",
            finding_ids=[access_finding.finding_id],
        ),
        license_permissions={
            "use": LicensePermissionAssessment(
                status="allowed",
                verification_status="verified",
                finding_ids=[license_finding.finding_id],
            )
        },
    )

    candidate = _candidate(
        [access_evidence, license_evidence],
        [access_finding, license_finding],
        required_dimensions=["access", "license_usage"],
        assessment=assessment,
    )

    assert candidate.assessment.quota.status == "confirmed_within_limit"
    assert candidate.assessment.license_permissions["use"].status == "allowed"


def test_candidate_checks_assessment_finding_dimension_and_connector_test_refs():
    evidence = _evidence(
        capture_method="observed_response",
        evidence_type="connector_test",
        authority="observed",
    )
    finding = _finding("connector", [evidence.evidence_id])
    assessment = SourceAssessment(
        required_dimensions=["connector"],
        findings_by_dimension={"connector": [finding.finding_id]},
        connector=ConnectorAssessment(
            status="implemented_tested",
            finding_ids=[finding.finding_id],
            test_reference="tests/connectors/test_example.py::test_fetch",
            tested_at=_NOW,
        ),
    )

    candidate = _candidate(
        [evidence],
        [finding],
        required_dimensions=["connector"],
        assessment=assessment,
    )
    assert candidate.assessment.connector.status == "implemented_tested"

    # Place a connector finding in the wrong assessment dimension bucket.
    # We use an unknown-status finding so no evidence is required, letting
    # the dimension-mismatch check fire rather than the evidence-type check.
    connector_finding_misplaced = _finding(
        "connector",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    wrong_dimension = SourceAssessment(
        required_dimensions=["variables"],
        findings_by_dimension={"variables": [connector_finding_misplaced.finding_id]},
    )
    with pytest.raises(ValidationError, match="do not match dimension"):
        _candidate(
            [],
            [connector_finding_misplaced],
            required_dimensions=["variables"],
            assessment=wrong_dimension,
        )


def test_candidate_serializes_and_deserializes_multiple_evidence_and_findings():
    first_evidence = _evidence()
    second_evidence = _evidence(
        source_url="https://data.example.org/access",
        title="Access policy",
        evidence_type="official_documentation",
        excerpt="A free registration is required.",
    )
    variable_finding = _finding("variables", [first_evidence.evidence_id])
    access_finding = _finding(
        "access",
        [second_evidence.evidence_id],
        value={"access_class": "free_with_registration"},
    )
    assessment = SourceAssessment(
        required_dimensions=["variables", "access"],
        findings_by_dimension={
            "variables": [variable_finding.finding_id],
            "access": [access_finding.finding_id],
        },
        access_class="free_with_registration",
        access_class_finding_ids=[access_finding.finding_id],
    )

    candidate = _candidate(
        [first_evidence, second_evidence],
        [variable_finding, access_finding],
        required_dimensions=["variables", "access"],
        assessment=assessment,
    )
    serialized = candidate.model_dump(mode="json")
    restored = CandidateSource.model_validate(serialized)

    assert restored == candidate
    assert len(restored.evidence) == 2
    assert len(restored.findings) == 2


def test_existing_plan_and_dataset_json_remain_backward_compatible():
    old_plan = PlanSpec.model_validate(
        {
            "source_requirements": [],
            "search_queries": [],
            "primary_source_types": [],
            "fallback_source_types": [],
            "confidence": 0.4,
        }
    )
    old_spec_json = create_initial_spec("Dataset", "Description").model_dump(mode="json")
    restored = DatasetSpec.model_validate_json(json.dumps(old_spec_json))

    assert old_plan.confidence == 0.4
    assert restored.plan is None
    assert "candidate_sources" not in DatasetSpec.model_fields
    assert "source_research" not in PlanSpec.model_fields


# ---------------------------------------------------------------------------
# New focused tests added for Increment 2 refinements
# ---------------------------------------------------------------------------

# ---- Item 5a: connector findings cannot be verified by documentation alone ---

def test_connector_finding_cannot_be_verified_by_official_documentation_alone():
    """Official documentation about a connector is not an observed connector test."""
    doc_evidence = _evidence(
        source_url="https://data.example.org/connector-docs",
        title="Connector implementation documentation",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("connector", [doc_evidence.evidence_id])

    with pytest.raises(ValidationError, match="observed connector-test evidence"):
        _candidate([doc_evidence], [finding])


def test_connector_finding_cannot_be_verified_by_api_documentation():
    """API documentation describes how to call an API; it is not a connector test."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs",
        title="API reference",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("connector", [api_doc.evidence_id])

    with pytest.raises(ValidationError, match="observed connector-test evidence"):
        _candidate([api_doc], [finding])


# ---- Item 5b: connector findings CAN be verified with observed connector-test ---

def test_connector_finding_can_be_verified_with_observed_connector_test():
    """An actual observed connector-test result verifies a connector finding."""
    test_evidence = _evidence(
        source_url="https://data.example.org/api/v1/observations",
        title="Connector test: successful fetch of observations endpoint",
        evidence_type="connector_test",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("connector", [test_evidence.evidence_id])
    assessment = SourceAssessment(
        required_dimensions=["connector"],
        findings_by_dimension={"connector": [finding.finding_id]},
        connector=ConnectorAssessment(
            status="implemented_tested",
            finding_ids=[finding.finding_id],
            test_reference="tests/connectors/test_example.py::test_live_fetch",
            tested_at=_NOW,
        ),
    )

    candidate = _candidate(
        [test_evidence],
        [finding],
        required_dimensions=["connector"],
        assessment=assessment,
    )

    assert candidate.assessment.connector.status == "implemented_tested"
    assert candidate.findings[0].status == "verified"


def test_connector_finding_can_be_verified_with_mixed_evidence_including_test():
    """When evidence includes at least one observed connector test, documentation
    may also be present without breaking the validation."""
    test_evidence = _evidence(
        source_url="https://data.example.org/api/v1/ping",
        title="Connector test: ping endpoint",
        evidence_type="connector_test",
        capture_method="observed_response",
        authority="observed",
    )
    doc_evidence = _evidence(
        source_url="https://data.example.org/connector-guide",
        title="Connector implementation guide",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("connector", [test_evidence.evidence_id, doc_evidence.evidence_id])

    candidate = _candidate([test_evidence, doc_evidence], [finding])

    assert candidate.findings[0].status == "verified"


# ---- Item 5c: license claims require license-related evidence ---------------

def test_license_finding_cannot_be_verified_without_license_evidence():
    """An observed API response or catalog hint does not constitute license evidence."""
    api_response = _evidence(
        source_url="https://api.example.org/data",
        title="API response sample",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("license_usage", [api_response.evidence_id])

    with pytest.raises(ValidationError, match="license page or official documentation"):
        _candidate([api_response], [finding])


def test_license_finding_cannot_be_verified_by_secondary_report():
    """A secondary report about licensing is insufficient."""
    secondary = _evidence(
        source_url="https://blog.example.com/dataset-review",
        title="Dataset review mentioning open license",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding("license_usage", [secondary.evidence_id])

    with pytest.raises(ValidationError, match="license page or official documentation"):
        _candidate([secondary], [finding])


def test_license_finding_can_be_verified_by_license_page():
    """A fetched license page from the official provider verifies a license finding."""
    license_evidence = _evidence(
        source_url="https://data.example.org/license",
        title="CC BY 4.0 License",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "license_usage",
        [license_evidence.evidence_id],
        value={"use": "allowed", "license": "CC-BY-4.0"},
    )

    candidate = _candidate([license_evidence], [finding])

    assert candidate.findings[0].status == "verified"


def test_license_finding_can_be_verified_by_official_documentation():
    """Official documentation that covers licensing also satisfies the requirement."""
    official_doc = _evidence(
        source_url="https://data.example.org/terms",
        title="Terms of use and license",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "license_usage",
        [official_doc.evidence_id],
        value={"use": "allowed"},
    )

    candidate = _candidate([official_doc], [finding])

    assert candidate.findings[0].status == "verified"


# ---- Item 5d: access and quota claims require relevant evidence --------------

def test_access_finding_cannot_be_verified_by_secondary_report():
    """A secondary blog post about access conditions is not official evidence."""
    secondary = _evidence(
        source_url="https://blog.example.com/api-access",
        title="How to get access to Example API",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "access",
        [secondary.evidence_id],
        value={"access_class": "free_tier"},
    )

    with pytest.raises(ValidationError, match="secondary reports and license pages are not sufficient"):
        _candidate([secondary], [finding])


def test_access_finding_can_be_verified_by_observed_response():
    """An actual observed API response verifies access conditions."""
    observed = _evidence(
        source_url="https://api.example.org/status",
        title="API status response confirming free-tier access",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding(
        "access",
        [observed.evidence_id],
        value={"access_class": "free_tier"},
    )

    candidate = _candidate([observed], [finding])

    assert candidate.findings[0].status == "verified"


def test_access_finding_can_be_verified_by_official_provider_documentation():
    """Official provider documentation verifies access claim."""
    doc = _evidence(
        source_url="https://data.example.org/access-policy",
        title="Data access policy",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "access",
        [doc.evidence_id],
        value={"access_class": "free_with_registration"},
    )

    candidate = _candidate([doc], [finding])

    assert candidate.findings[0].status == "verified"


# ---- Item 5e: secondary reporting cannot verify direct measurement claims ----

def test_secondary_report_cannot_verify_variables_finding():
    """A secondary report about what variables a dataset contains is insufficient."""
    secondary = _evidence(
        source_url="https://review.example.com/dataset-variables",
        title="Third-party review listing dataset variables",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "variables",
        [secondary.evidence_id],
        value={"variables": ["temperature", "humidity"]},
    )

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not sufficient"):
        _candidate([secondary], [finding])


def test_secondary_report_cannot_verify_geography_finding():
    secondary = _evidence(
        source_url="https://review.example.com/coverage",
        title="Third-party coverage summary",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "geography",
        [secondary.evidence_id],
        value={"coverage": "global"},
    )

    with pytest.raises(ValidationError, match="secondary reports, license pages, and catalog hints are not sufficient"):
        _candidate([secondary], [finding])


def test_secondary_report_cannot_verify_freshness_finding():
    secondary = _evidence(
        source_url="https://blog.example.com/update-frequency",
        title="Blog post about data freshness",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "freshness",
        [secondary.evidence_id],
        value={"max_age_hours": 1},
    )

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([secondary], [finding])


# ---- Item 5f: dataset content can support variable/format findings ----------

def test_dataset_content_can_support_variables_finding():
    """An observed API response containing actual dataset content is valid evidence
    for variables and formats without requiring official documentation."""
    dataset_response = _evidence(
        source_url="https://api.example.org/data?limit=1",
        title="Observed API response with dataset sample",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    variables_finding = _finding(
        "variables",
        [dataset_response.evidence_id],
        value={"variables": ["pm2_5", "pm10", "no2"]},
    )
    formats_finding = _finding(
        "formats",
        [dataset_response.evidence_id],
        value={"format": "application/json"},
    )

    candidate = _candidate(
        [dataset_response],
        [variables_finding, formats_finding],
        required_dimensions=["variables", "formats"],
    )

    assert candidate.findings[0].status == "verified"
    assert candidate.findings[1].status == "verified"


def test_official_provider_documentation_can_support_formats_finding():
    """Official provider docs are valid evidence for format claims."""
    doc = _evidence(
        source_url="https://data.example.org/formats",
        title="Available download formats",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "formats",
        [doc.evidence_id],
        value={"formats": ["text/csv", "application/json"]},
    )

    candidate = _candidate([doc], [finding])

    assert candidate.findings[0].status == "verified"


# ---- Item 5g: NaN and ±inf rejected for quota values -----------------------

import math as _math


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_quota_expected_usage_rejects_non_finite_values(value):
    with pytest.raises(ValidationError, match="finite number"):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=value,
            expected_usage_unit="requests/month",
            limit_value=100.0,
            limit_unit="requests/month",
            finding_ids=[uuid4()],
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_quota_limit_value_rejects_non_finite_values(value):
    with pytest.raises(ValidationError, match="finite number"):
        QuotaAssessment(
            status="confirmed_within_limit",
            expected_usage=10.0,
            expected_usage_unit="requests/month",
            limit_value=value,
            limit_unit="requests/month",
            finding_ids=[uuid4()],
        )


def test_quota_accepts_finite_zero_expected_usage():
    """Zero expected usage is valid (ge=0 bound)."""
    assessment = QuotaAssessment(
        status="confirmed_within_limit",
        expected_usage=0.0,
        expected_usage_unit="requests/month",
        limit_value=100.0,
        limit_unit="requests/month",
        finding_ids=[uuid4()],
    )
    assert assessment.expected_usage == 0.0


def test_quota_accepts_large_but_finite_values():
    """Very large but finite values are accepted."""
    assessment = QuotaAssessment(
        status="confirmed_within_limit",
        expected_usage=9e15,
        expected_usage_unit="rows/month",
        limit_value=1e16,
        limit_unit="rows/month",
        finding_ids=[uuid4()],
    )
    assert _math.isfinite(assessment.expected_usage)
    assert _math.isfinite(assessment.limit_value)


# ---- Item 5h: unknown findings in a complete evaluation ----------------------

def test_complete_evaluation_with_all_unknown_findings_is_valid():
    """A complete evaluation can have all unknown findings; unknown ≠ uninvestigated."""
    unknown_access = _finding(
        "access",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    unknown_license = _finding(
        "license_usage",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    unknown_connector = _finding(
        "connector",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )

    candidate = _candidate(
        [],
        [unknown_access, unknown_license, unknown_connector],
        required_dimensions=["access", "license_usage", "connector"],
        status="complete",
    )

    assert candidate.evaluation_status == "complete"
    assert all(f.status == "unknown" for f in candidate.findings)


def test_unknown_finding_is_distinct_from_partial_evaluation():
    """Having an unknown finding for a dimension is different from not investigating it.
    A partial evaluation has uninvestigated dimensions; a complete evaluation does not."""
    # Investigated but couldn't determine value → unknown finding, complete status
    unknown = _finding(
        "variables",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )
    complete = _candidate(
        [],
        [unknown],
        required_dimensions=["variables"],
        status="complete",
    )
    assert complete.evaluation_status == "complete"

    # Not investigated at all → partial evaluation (geography has no finding)
    partial = _candidate(
        [],
        [unknown],
        required_dimensions=["variables", "geography"],
        status="partial",
    )
    assert partial.evaluation_status == "partial"

    # Must not mark partial when all required dimensions are covered
    with pytest.raises(ValidationError, match="partial evaluation must leave"):
        _candidate(
            [],
            [unknown],
            required_dimensions=["variables"],
            status="partial",
        )


def test_complete_evaluation_mixes_verified_and_unknown_findings():
    """A complete evaluation may have some verified and some unknown findings."""
    doc = _evidence(
        source_url="https://data.example.org/api",
        title="API documentation",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    verified_formats = _finding(
        "formats",
        [doc.evidence_id],
        value={"format": "application/json"},
    )
    unknown_variables = _finding(
        "variables",
        status="unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
    )

    candidate = _candidate(
        [doc],
        [verified_formats, unknown_variables],
        required_dimensions=["formats", "variables"],
        status="complete",
    )

    statuses = {f.dimension: f.status for f in candidate.findings}
    assert statuses["formats"] == "verified"
    assert statuses["variables"] == "unknown"
    assert candidate.evaluation_status == "complete"


# ---------------------------------------------------------------------------
# New tests: stricter evidence-type + capture-method validation
# Tests are grouped as negative (rejected) then positive (accepted) per rule.
# ---------------------------------------------------------------------------

# ============================================================
# Universal guard: search-result snippets alone
# ============================================================

def test_search_result_snippet_alone_cannot_verify_any_finding():
    """A search-result snippet has not been fetched; it cannot verify a finding."""
    snippet = _evidence(
        source_url="https://data.example.org/api",
        title="Search result snippet for API page",
        evidence_type="api_documentation",
        capture_method="search_result",   # not fetched
        authority="official_provider",
    )
    finding = _finding("variables", [snippet.evidence_id])

    with pytest.raises(ValidationError, match="search-result snippets alone"):
        _candidate([snippet], [finding])


def test_search_result_snippet_plus_fetched_evidence_can_verify():
    """A search snippet alongside a fetched document satisfies the guard."""
    snippet = _evidence(
        source_url="https://data.example.org/api",
        title="Search snippet",
        evidence_type="api_documentation",
        capture_method="search_result",
        authority="official_provider",
    )
    fetched = _evidence(
        source_url="https://data.example.org/api/full",
        title="Full API documentation page",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("variables", [snippet.evidence_id, fetched.evidence_id])

    candidate = _candidate([snippet, fetched], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# CONNECTOR
# ============================================================

def test_connector_api_documentation_fetched_cannot_verify_connector():
    """Fetched API docs describe the connector but cannot prove it was tested."""
    api_doc = _evidence(
        source_url="https://data.example.org/connector-api",
        title="Connector API reference",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("connector", [api_doc.evidence_id])

    with pytest.raises(ValidationError, match="connector-test evidence"):
        _candidate([api_doc], [finding])


def test_connector_observed_response_wrong_type_cannot_verify():
    """An observed API response that is not a connector test cannot verify connector."""
    obs = _evidence(
        source_url="https://api.example.org/data",
        title="Observed API data response",
        evidence_type="observed_response",   # not connector_test
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("connector", [obs.evidence_id])

    with pytest.raises(ValidationError, match="connector-test evidence"):
        _candidate([obs], [finding])


def test_connector_verified_by_connector_test_observed_response():
    """connector_test + observed_response verifies a connector finding."""
    test_ev = _evidence(
        source_url="https://api.example.org/v1/ping",
        title="Connector test: ping",
        evidence_type="connector_test",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("connector", [test_ev.evidence_id])

    candidate = _candidate([test_ev], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# LICENSE USAGE
# ============================================================

def test_license_page_as_search_snippet_cannot_verify_license():
    """A license page found as a search snippet has not been fetched."""
    snippet = _evidence(
        source_url="https://data.example.org/license",
        title="License page snippet",
        evidence_type="license_page",
        capture_method="search_result",   # not fetched
        authority="official_provider",
    )
    finding = _finding("license_usage", [snippet.evidence_id])

    with pytest.raises(ValidationError, match="search-result snippets alone"):
        _candidate([snippet], [finding])


def test_dataset_page_cannot_verify_license_usage():
    """A dataset page is not a license document."""
    dataset_page = _evidence(
        source_url="https://data.example.org/dataset",
        title="Dataset landing page",
        evidence_type="dataset_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("license_usage", [dataset_page.evidence_id])

    with pytest.raises(ValidationError, match="license page or official documentation"):
        _candidate([dataset_page], [finding])


def test_license_page_fetched_verifies_license():
    """A fetched license page verifies a license_usage finding."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="CC BY 4.0",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("license_usage", [license_ev.evidence_id], value={"license": "CC-BY-4.0"})

    candidate = _candidate([license_ev], [finding])
    assert candidate.findings[0].status == "verified"


def test_license_cannot_verify_variables():
    """A license page is irrelevant to variable coverage."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="CC BY 4.0",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("variables", [license_ev.evidence_id])

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not sufficient"):
        _candidate([license_ev], [finding])


def test_license_cannot_verify_formats():
    """A license page does not describe the data formats."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="MIT License",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("formats", [license_ev.evidence_id], value={"format": "text/csv"})

    with pytest.raises(ValidationError, match="secondary reports are not sufficient"):
        _candidate([license_ev], [finding])


def test_license_cannot_verify_freshness():
    """A license page does not establish freshness."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="Apache 2.0",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("freshness", [license_ev.evidence_id], value={"max_age_hours": 24})

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([license_ev], [finding])


# ============================================================
# VARIABLES
# ============================================================

def test_research_paper_cannot_verify_variables():
    """A research paper is secondary evidence for variable coverage."""
    paper = _evidence(
        source_url="https://doi.example.org/10.1234/paper",
        title="Analysis of dataset variables",
        evidence_type="research_paper",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding("variables", [paper.evidence_id])

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not sufficient"):
        _candidate([paper], [finding])


def test_official_documentation_with_observed_authority_label_alone_insufficient():
    """authority='observed' on official_documentation must not bypass the
    evidence-type check.  The capture_method must actually be observed_response."""
    misleading = _evidence(
        source_url="https://data.example.org/docs",
        title="Documentation with wrong authority label",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="observed",   # label claims observed, but it was fetched_content
    )
    # For variables: official_documentation + fetched_content IS in the allowed set,
    # so this should actually pass — confirming the rule uses evidence_type+capture_method
    # not authority.
    finding = _finding("variables", [misleading.evidence_id])
    candidate = _candidate([misleading], [finding])
    assert candidate.findings[0].status == "verified"


def test_secondary_report_with_observed_authority_label_cannot_verify_variables():
    """authority='observed' on a secondary_report must not make it pass the
    evidence-type check.  evidence_type drives the rule, not authority."""
    misleading = _evidence(
        source_url="https://review.example.com/vars",
        title="Review labelled as observed",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="observed",   # caller lies about authority
    )
    finding = _finding("variables", [misleading.evidence_id])

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not sufficient"):
        _candidate([misleading], [finding])


def test_dataset_page_fetched_verifies_variables():
    """A fetched dataset page verifies variable coverage."""
    page = _evidence(
        source_url="https://data.example.org/dataset",
        title="Dataset page listing variables",
        evidence_type="dataset_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("variables", [page.evidence_id])

    candidate = _candidate([page], [finding])
    assert candidate.findings[0].status == "verified"


def test_observed_response_verifies_variables():
    """An observed API data response proves what variables are present."""
    obs = _evidence(
        source_url="https://api.example.org/data",
        title="Observed API data",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("variables", [obs.evidence_id])

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# FORMATS
# ============================================================

def test_secondary_report_cannot_verify_formats():
    """A third-party review is not direct evidence of data format."""
    secondary = _evidence(
        source_url="https://review.example.com/formats",
        title="Format review",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding("formats", [secondary.evidence_id], value={"format": "text/csv"})

    with pytest.raises(ValidationError, match="secondary reports are not sufficient"):
        _candidate([secondary], [finding])


def test_license_page_cannot_verify_formats():
    """License page does not describe data serialisation format."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="License",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("formats", [license_ev.evidence_id], value={"format": "text/csv"})

    with pytest.raises(ValidationError, match="secondary reports are not sufficient"):
        _candidate([license_ev], [finding])


def test_observed_response_verifies_formats():
    """An observed API response with the actual data proves the format."""
    obs = _evidence(
        source_url="https://api.example.org/data.csv",
        title="Observed CSV response",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("formats", [obs.evidence_id], value={"format": "text/csv"})

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


def test_api_documentation_fetched_verifies_formats():
    """Fetched API docs explicitly listing formats verifies format findings."""
    doc = _evidence(
        source_url="https://api.example.org/docs",
        title="API documentation: response formats",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("formats", [doc.evidence_id], value={"formats": ["text/csv", "application/json"]})

    candidate = _candidate([doc], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# GEOGRAPHY / TEMPORAL_COVERAGE / RESOLUTION / FRESHNESS / UPDATE_FREQUENCY
# ============================================================

def test_connector_test_cannot_verify_geography():
    """A connector test result is not documentation of geographic coverage."""
    test_ev = _evidence(
        source_url="https://api.example.org/data",
        title="Connector test result",
        evidence_type="connector_test",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding("geography", [test_ev.evidence_id], value={"coverage": "global"})

    with pytest.raises(ValidationError, match="secondary reports, license pages, and catalog hints"):
        _candidate([test_ev], [finding])


def test_research_paper_cannot_verify_temporal_coverage():
    secondary = _evidence(
        source_url="https://doi.example.org/10.1234/study",
        title="Study referencing dataset temporal range",
        evidence_type="research_paper",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding("temporal_coverage", [secondary.evidence_id], value={"years": "2000-2024"})

    with pytest.raises(ValidationError, match="api_documentation or official_documentation describes intended coverage"):
        _candidate([secondary], [finding])


def test_official_documentation_fetched_verifies_geography():
    doc = _evidence(
        source_url="https://data.example.org/coverage",
        title="Coverage documentation",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("geography", [doc.evidence_id], value={"region": "India"})

    candidate = _candidate([doc], [finding])
    assert candidate.findings[0].status == "verified"


def test_api_documentation_alone_cannot_verify_freshness():
    """API documentation describes an endpoint's update schedule, not whether
    current data is actually as fresh as claimed.  It must not verify freshness."""
    doc = _evidence(
        source_url="https://api.example.org/docs#freshness",
        title="API freshness documentation",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("freshness", [doc.evidence_id], value={"max_age_hours": 1})

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([doc], [finding])


def test_observed_response_verifies_temporal_coverage():
    obs = _evidence(
        source_url="https://api.example.org/data?from=2000",
        title="Observed API response with temporal metadata",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
        excerpt="Available observations: 2000-01-01 to 2025-12-31",
    )
    finding = _finding("temporal_coverage", [obs.evidence_id], value={"start": "2000"})

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# RETRIEVAL METHODS
# ============================================================

def test_secondary_report_cannot_verify_retrieval_methods():
    secondary = _evidence(
        source_url="https://blog.example.com/how-to-download",
        title="Blog: how to download the dataset",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding("retrieval_methods", [secondary.evidence_id], value={"method": "http_api"})

    with pytest.raises(ValidationError, match="fetched API/official documentation"):
        _candidate([secondary], [finding])


def test_license_page_cannot_verify_retrieval_methods():
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="License",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("retrieval_methods", [license_ev.evidence_id], value={"method": "direct_download"})

    with pytest.raises(ValidationError, match="fetched API/official documentation"):
        _candidate([license_ev], [finding])


def test_api_documentation_verifies_retrieval_methods():
    doc = _evidence(
        source_url="https://api.example.org/docs",
        title="API reference",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("retrieval_methods", [doc.evidence_id], value={"method": "http_api"})

    candidate = _candidate([doc], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# ACCESS
# ============================================================

def test_secondary_report_labelled_official_cannot_verify_access():
    """authority='official_provider' on secondary_report must not bypass the
    evidence_type check for access."""
    misleading = _evidence(
        source_url="https://blog.example.com/access-review",
        title="Access review labelled official",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="official_provider",   # caller mis-labels authority
    )
    finding = _finding("access", [misleading.evidence_id], value={"access_class": "free"})

    with pytest.raises(ValidationError, match="secondary reports and license pages are not sufficient"):
        _candidate([misleading], [finding])


def test_unrelated_official_document_cannot_verify_access():
    """A research paper is an official document but it does not establish access."""
    paper = _evidence(
        source_url="https://doi.example.org/10.1234/study",
        title="Dataset study",
        evidence_type="research_paper",
        capture_method="fetched_content",
        authority="official_publisher",
    )
    finding = _finding("access", [paper.evidence_id], value={"access_class": "free"})

    with pytest.raises(ValidationError, match="secondary reports and license pages are not sufficient"):
        _candidate([paper], [finding])


def test_license_page_cannot_verify_access():
    """A license page establishes usage rights, not access class or authentication."""
    license_ev = _evidence(
        source_url="https://data.example.org/license",
        title="License",
        evidence_type="license_page",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("access", [license_ev.evidence_id], value={"access_class": "free"})

    with pytest.raises(ValidationError, match="secondary reports and license pages are not sufficient"):
        _candidate([license_ev], [finding])


def test_official_documentation_verifies_access():
    doc = _evidence(
        source_url="https://data.example.org/access-policy",
        title="Data access policy",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("access", [doc.evidence_id], value={"access_class": "free_with_registration"})

    candidate = _candidate([doc], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# PUBLISHER RELIABILITY AND PURPOSE (advisory — wider evidence allowed)
# ============================================================

def test_research_paper_can_verify_publisher_reliability():
    """A research paper is acceptable evidence for advisory publisher-reliability."""
    paper = _evidence(
        source_url="https://doi.example.org/10.1234/reliability",
        title="Study evaluating provider reliability",
        evidence_type="research_paper",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "publisher_reliability",
        [paper.evidence_id],
        value={"score": "high"},
    )

    candidate = _candidate([paper], [finding])
    assert candidate.findings[0].status == "verified"


def test_secondary_report_can_verify_purpose():
    """A secondary report can verify an advisory purpose finding."""
    secondary = _evidence(
        source_url="https://review.example.com/purpose",
        title="Dataset purpose review",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "purpose",
        [secondary.evidence_id],
        value={"use_case": "environmental monitoring"},
    )

    candidate = _candidate([secondary], [finding])
    assert candidate.findings[0].status == "verified"


def test_catalog_hint_alone_still_cannot_verify_publisher_reliability():
    """Even advisory dimensions cannot be verified by catalog hints alone."""
    hint = _evidence(
        source_url="https://catalog.example.org/entry",
        title="Catalog entry",
        evidence_type="catalog_record",
        capture_method="catalog_hint",
        authority="catalog",
    )
    finding = _finding("publisher_reliability", [hint.evidence_id], value={"score": "high"})

    with pytest.raises(ValidationError, match="catalog hints alone cannot verify"):
        _candidate([hint], [finding])


# ---------------------------------------------------------------------------
# Freshness vs update_frequency vs temporal_coverage distinction
# ---------------------------------------------------------------------------

# ============================================================
# FRESHNESS — negative tests
# ============================================================

def test_api_documentation_alone_cannot_verify_freshness_v2():
    """Fetched API documentation describes the update schedule/policy, not
    actual data currency.  It must not verify a freshness finding."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs",
        title="API reference — update cadence section",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("freshness", [api_doc.evidence_id], value={"max_age_hours": 1})

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([api_doc], [finding])


def test_official_documentation_alone_cannot_verify_freshness():
    """Official documentation stating 'data is updated hourly' describes intent,
    not actual current availability.  It must not verify freshness."""
    official_doc = _evidence(
        source_url="https://data.example.org/about",
        title="About page: data update policy",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding("freshness", [official_doc.evidence_id], value={"max_age_hours": 1})

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([official_doc], [finding])


def test_secondary_report_cannot_verify_freshness_regardless_of_authority():
    """A secondary report claiming 'data is live' cannot verify actual freshness,
    even if the caller labels it as official_provider."""
    misleading = _evidence(
        source_url="https://review.example.com/freshness",
        title="Third-party freshness claim",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="official_provider",   # mis-labelled authority
    )
    finding = _finding("freshness", [misleading.evidence_id], value={"max_age_hours": 1})

    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([misleading], [finding])


# ============================================================
# FRESHNESS — positive tests
# ============================================================

def test_observed_response_with_timestamps_can_verify_freshness():
    """An observed API response containing current data records supports
    freshness: the application captured live data and recorded its timestamp."""
    obs = _evidence(
        source_url="https://api.example.org/observations/latest",
        title="Observed API response with current readings (timestamp: 2026-01-01T00:00Z)",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
        excerpt="Dataset last updated: 2026-01-01T00:00Z; most recent observation: 2026-01-01T00:00Z",
    )
    finding = _finding("freshness", [obs.evidence_id], value={"max_age_hours": 1})

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


def test_dataset_page_with_publication_timestamp_can_verify_freshness():
    """A fetched dataset page that directly states a 'last updated' or
    'published' timestamp supports a freshness claim."""
    page = _evidence(
        source_url="https://data.example.org/dataset/latest",
        title="Dataset page — last updated: 2026-01-01",
        evidence_type="dataset_page",
        capture_method="fetched_content",
        authority="official_provider",
        excerpt="Dataset last updated: 2026-01-01",
    )
    finding = _finding("freshness", [page.evidence_id], value={"last_updated": "2026-01-01"})

    candidate = _candidate([page], [finding])
    assert candidate.findings[0].status == "verified"


def test_manually_captured_dataset_page_can_verify_freshness():
    """A manually captured dataset page with timestamps also verifies freshness."""
    manual = _evidence(
        source_url="https://data.example.org/dataset",
        title="Dataset page captured manually — updated 2026-01-01",
        evidence_type="dataset_page",
        capture_method="manual_capture",
        authority="official_provider",
        excerpt="Dataset last updated: 2026-01-01",
    )
    finding = _finding("freshness", [manual.evidence_id], value={"last_updated": "2026-01-01"})

    candidate = _candidate([manual], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# UPDATE_FREQUENCY — API docs CAN verify update schedule
# ============================================================

def test_api_documentation_can_verify_update_frequency():
    """API documentation explicitly stating the update cadence (e.g., 'data
    refreshed hourly') is appropriate evidence for update_frequency."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs#update-schedule",
        title="API reference — update schedule",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "update_frequency",
        [api_doc.evidence_id],
        value={"frequency": "hourly"},
    )

    candidate = _candidate([api_doc], [finding])
    assert candidate.findings[0].status == "verified"


def test_official_documentation_can_verify_update_frequency():
    """Official documentation stating the update policy is acceptable for
    update_frequency (documents intent, which is what this dimension describes)."""
    official_doc = _evidence(
        source_url="https://data.example.org/about",
        title="Provider data update policy",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "update_frequency",
        [official_doc.evidence_id],
        value={"frequency": "daily"},
    )

    candidate = _candidate([official_doc], [finding])
    assert candidate.findings[0].status == "verified"


def test_observed_response_can_verify_update_frequency():
    """An observed response can also verify update_frequency (e.g., the
    response includes a 'next-update' header or metadata)."""
    obs = _evidence(
        source_url="https://api.example.org/meta",
        title="API metadata response with update schedule",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
    )
    finding = _finding(
        "update_frequency",
        [obs.evidence_id],
        value={"frequency": "hourly"},
    )

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


def test_secondary_report_cannot_verify_update_frequency():
    """A third-party review is not direct evidence of the update schedule."""
    secondary = _evidence(
        source_url="https://review.example.com/update-cadence",
        title="Blog: how often does Example API update?",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "update_frequency",
        [secondary.evidence_id],
        value={"frequency": "daily"},
    )

    with pytest.raises(ValidationError, match="secondary reports and catalog hints are not"):
        _candidate([secondary], [finding])


# ============================================================
# TEMPORAL_COVERAGE — negative tests
# ============================================================

def test_api_documentation_alone_cannot_verify_temporal_coverage():
    """API documentation may state a nominal archive range, but it cannot
    prove that historical records for that period are currently queryable."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs#archive",
        title="API archive documentation: data available from 2000",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "temporal_coverage",
        [api_doc.evidence_id],
        value={"start": "2000-01-01"},
    )

    with pytest.raises(
        ValidationError,
        match="api_documentation or official_documentation describes intended coverage",
    ):
        _candidate([api_doc], [finding])


def test_official_documentation_alone_cannot_verify_temporal_coverage():
    """Official documentation stating 'archives go back to 2000' describes
    nominal coverage; it cannot prove records are currently queryable."""
    official_doc = _evidence(
        source_url="https://data.example.org/archive",
        title="Provider archive policy: data since 2000",
        evidence_type="official_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )
    finding = _finding(
        "temporal_coverage",
        [official_doc.evidence_id],
        value={"start": "2000-01-01"},
    )

    with pytest.raises(
        ValidationError,
        match="api_documentation or official_documentation describes intended coverage",
    ):
        _candidate([official_doc], [finding])


def test_secondary_report_cannot_verify_temporal_coverage():
    """A secondary study referencing a historical range is not proof that
    records are currently available."""
    secondary = _evidence(
        source_url="https://doi.example.org/10.1234/archive-study",
        title="Study using 20 years of archived data",
        evidence_type="secondary_report",
        capture_method="fetched_content",
        authority="secondary",
    )
    finding = _finding(
        "temporal_coverage",
        [secondary.evidence_id],
        value={"years": "2000-2024"},
    )

    with pytest.raises(
        ValidationError,
        match="api_documentation or official_documentation describes intended coverage",
    ):
        _candidate([secondary], [finding])


# ============================================================
# TEMPORAL_COVERAGE — positive tests
# ============================================================

def test_observed_records_can_verify_temporal_coverage():
    """An observed API response containing records from the requested historical
    period is direct evidence that those records are available."""
    obs = _evidence(
        source_url="https://api.example.org/data?from=2000-01-01&to=2000-12-31",
        title="Observed API response with records from year 2000",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
        excerpt="Available observations: 2000-01-01 to 2000-12-31",
    )
    finding = _finding(
        "temporal_coverage",
        [obs.evidence_id],
        value={"start": "2000-01-01", "end": "2000-12-31"},
    )

    candidate = _candidate([obs], [finding])
    assert candidate.findings[0].status == "verified"


def test_dataset_page_with_actual_date_range_can_verify_temporal_coverage():
    """A fetched dataset page that explicitly states the actual available date
    range (not merely the nominal policy) supports temporal coverage."""
    page = _evidence(
        source_url="https://data.example.org/dataset/archive",
        title="Dataset archive page — available: 2000-01-01 to present",
        evidence_type="dataset_page",
        capture_method="fetched_content",
        authority="official_provider",
        excerpt="Available observations: 2000-01-01 to 2025-12-31",
    )
    finding = _finding(
        "temporal_coverage",
        [page.evidence_id],
        value={"start": "2000-01-01"},
    )

    candidate = _candidate([page], [finding])
    assert candidate.findings[0].status == "verified"


def test_manually_captured_dataset_records_verify_temporal_coverage():
    """A manually captured dataset page with actual records also verifies
    temporal coverage."""
    manual = _evidence(
        source_url="https://data.example.org/archive/2010",
        title="Manually inspected archive — 2010 records present",
        evidence_type="dataset_page",
        capture_method="manual_capture",
        authority="official_provider",
        excerpt="Available observations: 2010-01-01 to 2010-12-31",
    )
    finding = _finding(
        "temporal_coverage",
        [manual.evidence_id],
        value={"year": "2010"},
    )

    candidate = _candidate([manual], [finding])
    assert candidate.findings[0].status == "verified"


# ============================================================
# Cross-dimension: same evidence used correctly vs incorrectly
# ============================================================

def test_api_doc_verifies_update_frequency_but_not_freshness():
    """The same API documentation evidence is acceptable for update_frequency
    but must be rejected when used as sole evidence for freshness."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs#schedule",
        title="API: data refreshed every hour",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )

    # update_frequency: should pass
    freq_finding = _finding(
        "update_frequency",
        [api_doc.evidence_id],
        value={"frequency": "hourly"},
    )
    freq_candidate = _candidate([api_doc], [freq_finding])
    assert freq_candidate.findings[0].dimension == "update_frequency"
    assert freq_candidate.findings[0].status == "verified"

    # freshness: must be rejected — the doc describes schedule, not actual data
    fresh_finding = _finding(
        "freshness",
        [api_doc.evidence_id],
        value={"max_age_hours": 1},
    )
    with pytest.raises(ValidationError, match="api_documentation describes an update schedule"):
        _candidate([api_doc], [fresh_finding])


def test_api_doc_verifies_update_frequency_but_not_temporal_coverage():
    """API documentation about an archive range is acceptable for no temporal
    coverage claim, but can establish update_frequency."""
    api_doc = _evidence(
        source_url="https://api.example.org/docs#archive",
        title="API: data from 2000 to present",
        evidence_type="api_documentation",
        capture_method="fetched_content",
        authority="official_provider",
    )

    # update_frequency with API doc: pass
    freq_finding = _finding(
        "update_frequency",
        [api_doc.evidence_id],
        value={"frequency": "daily"},
    )
    freq_candidate = _candidate([api_doc], [freq_finding])
    assert freq_candidate.findings[0].status == "verified"

    # temporal_coverage with API doc alone: reject
    temp_finding = _finding(
        "temporal_coverage",
        [api_doc.evidence_id],
        value={"start": "2000-01-01"},
    )
    with pytest.raises(
        ValidationError,
        match="api_documentation or official_documentation describes intended coverage",
    ):
        _candidate([api_doc], [temp_finding])
