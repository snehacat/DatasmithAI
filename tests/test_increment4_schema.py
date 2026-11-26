"""Focused tests for Increment 4.0 schema additions, invariants, and backward compatibility."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.dataset_spec import (
    ApplicationPlanningPolicy,
    BlockedFetchRecord,
    CandidateSource,
    ConnectorAssessment,
    DatasetSpec,
    EvaluationRequirements,
    LicensePermissionAssessment,
    PlanSpec,
    QuotaAssessment,
    ReplanAttempt,
    RequirementSpec,
    RetrievalInstruction,
    SourceAssessment,
    SourceEvidence,
    SourceFinding,
    UsageIntent,
    create_initial_spec,
    derive_evaluation_requirements,
    resolve_freshness_threshold_hours,
    validate_spec,
)

_NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _evidence(**overrides):
    data = {
        "source_url": "https://api.example.org/docs",
        "publisher": "Example Org",
        "title": "API docs",
        "discovery_method": "provider_documentation",
        "evidence_type": "api_documentation",
        "capture_method": "fetched_content",
        "authority": "official_provider",
        "excerpt": "GET https://api.example.org/v1/obs returns CSV.",
        "retrieved_at": _NOW,
        "content_sha256": "a" * 64,
        "artifact_relative_path": "evidence/doc1.txt",
    }
    data.update(overrides)
    return SourceEvidence(**data)


def _build_valid_plan_spec(
    *,
    freshness_need: str = "unspecified",
    max_data_age: str | None = None,
    max_data_age_hours: int | None = None,
    usage_rights: list[str] | None = None,
    conditional_modify: bool = False,
    endpoint_url: str = "https://api.example.org/v1/obs",
    retrieval_method: str = "http_api",
    connector_status: str = "not_required",
    auth_req: str = "none",
    cred_env: str | None = None,
    cred_loc: str | None = None,
    cred_field: str | None = None,
) -> DatasetSpec:
    req = RequirementSpec(
        dataset_name="Air quality",
        description="Delhi air quality observations",
        domain="environmental",
        subdomain="air quality",
        geography="Delhi",
        freshness_need=freshness_need,
        max_data_age=max_data_age,
        max_data_age_hours=max_data_age_hours,
        expected_features=["pm2.5"],
        is_supported=True,
    )
    rights = usage_rights or ["use"]
    policy = derive_evaluation_requirements(req, UsageIntent(required_rights=rights))

    ev_docs = _evidence(source_url="https://api.example.org/docs", title="API docs")
    ev_page = _evidence(
        source_url="https://data.example.org/dataset",
        title="Dataset page",
        evidence_type="dataset_page",
        excerpt="Updated hourly; records from 2020 to 2026.",
    )
    ev_license = _evidence(
        source_url="https://data.example.org/license",
        title="License page",
        evidence_type="license_page",
        excerpt="Licensed under CC-BY-4.0 with attribution.",
    )
    ev_obs = _evidence(
        source_url=endpoint_url,
        title="Live API sample",
        evidence_type="observed_response",
        capture_method="observed_response",
        authority="observed",
        excerpt='{"pm25": 42, "measured_at": "2026-11-25T11:30:00Z"}',
    )
    ev_conn = (
        _evidence(
            source_url=endpoint_url,
            title="Connector test",
            evidence_type="connector_test",
            capture_method="observed_response",
            authority="observed",
        )
        if connector_status != "not_required"
        else None
    )

    evidences = [ev_docs, ev_page, ev_license, ev_obs]
    if ev_conn is not None:
        evidences.append(ev_conn)

    findings = []
    findings_by_dim = {}

    def add_finding(dim, value, ev_ids, *, status="verified", claim_type=None):
        f = SourceFinding(
            dimension=dim,
            claim=f"Verified {dim}",
            value=value,
            claim_type=claim_type,
            evidence_ids=ev_ids,
            confidence=0.95,
            status=status,
            explanation=f"Supported by captured {dim} evidence.",
        )
        findings.append(f)
        findings_by_dim.setdefault(dim, []).append(f.finding_id)
        return f

    f_access = add_finding("access", {"access_class": "free", "auth": auth_req}, [ev_docs.evidence_id])
    f_license = add_finding("license_usage", {"spdx": "CC-BY-4.0"}, [ev_license.evidence_id])
    f_ret = add_finding(
        "retrieval_methods",
        {"method": retrieval_method, "endpoint_url": endpoint_url},
        [ev_docs.evidence_id],
    )
    add_finding("purpose", {"relevant": True}, [ev_page.evidence_id])
    add_finding("publisher_reliability", {"publisher": "Example Org"}, [ev_page.evidence_id])
    add_finding("variables", {"variables": ["pm2.5"]}, [ev_docs.evidence_id])
    add_finding("geography", {"geography": "Delhi"}, [ev_page.evidence_id])
    add_finding("formats", {"formats": ["json"]}, [ev_docs.evidence_id])
    add_finding("retrieval_complexity", {"complexity": "low"}, [ev_docs.evidence_id])

    if connector_status == "implemented_tested":
        f_conn = add_finding("connector", {"status": "implemented_tested"}, [ev_conn.evidence_id])
        conn_obj = ConnectorAssessment(
            status="implemented_tested",
            finding_ids=[f_conn.finding_id],
            test_reference="tests/test_conn.py::test_ok",
            tested_at=_NOW,
        )
    elif connector_status == "not_required":
        f_conn = add_finding(
            "connector",
            {"status": "not_required"},
            [ev_docs.evidence_id],
            status="unverified",
        )
        conn_obj = ConnectorAssessment(
            status="not_required",
            finding_ids=[f_conn.finding_id],
        )
    else:
        f_conn = add_finding("connector", {"status": connector_status}, [ev_conn.evidence_id])
        conn_obj = ConnectorAssessment(
            status=connector_status,
            finding_ids=[f_conn.finding_id],
        )

    if "freshness" in policy.evaluation_requirements.required_dimensions:
        add_finding(
            "freshness",
            {"max_age_hours": 1},
            [ev_obs.evidence_id],
            claim_type="observed",
        )
    if "update_frequency" in policy.evaluation_requirements.required_dimensions:
        add_finding("update_frequency", {"cadence": "hourly"}, [ev_docs.evidence_id])
    if "temporal_coverage" in policy.evaluation_requirements.required_dimensions:
        add_finding(
            "temporal_coverage",
            {"start": "2020"},
            [ev_page.evidence_id],
            claim_type="documented",
        )

    perms = {}
    conditions_by_right = {}
    for right in rights:
        if right == "modify" and conditional_modify:
            cond_text = "Must preserve attribution and mark changes."
            perms[right] = LicensePermissionAssessment(
                status="conditional",
                verification_status="verified",
                finding_ids=[f_license.finding_id],
                conditions=cond_text,
                conditions_status="satisfied",
            )
            conditions_by_right[right] = cond_text
        else:
            perms[right] = LicensePermissionAssessment(
                status="allowed",
                verification_status="verified",
                finding_ids=[f_license.finding_id],
            )

    assessment = SourceAssessment(
        required_dimensions=list(policy.evaluation_requirements.required_dimensions),
        findings_by_dimension=findings_by_dim,
        access_class="free",
        access_class_finding_ids=[f_access.finding_id],
        authentication_requirement=auth_req,
        authentication_finding_ids=[f_access.finding_id],
        quota=QuotaAssessment(status="not_applicable", finding_ids=[f_access.finding_id]),
        license_permissions=perms,
        connector=conn_obj,
    )

    candidate = CandidateSource(
        source_url="https://data.example.org/landing",
        title="Example AQ",
        discovery_method="repository_search",
        discovered_at=_NOW,
        evidence=evidences,
        findings=findings,
        assessment=assessment,
        evaluation_status="complete",
        selection_status="selected",
    )

    expected_age, _ = resolve_freshness_threshold_hours(req)
    requires_fresh = expected_age is not None

    instr = RetrievalInstruction(
        candidate_id=candidate.candidate_id,
        source_url=endpoint_url,
        retrieval_method=retrieval_method,
        expected_format="json",
        parser_hint="json_records",
        access_class="free",
        authentication_requirement=auth_req,
        credential_env_var=cred_env,
        credential_location=cred_loc,
        credential_field_name=cred_field,
        license_spdx_id="CC-BY-4.0",
        license_conditions_by_right=conditions_by_right,
        requires_runtime_freshness_verification=requires_fresh,
        max_data_age_hours=expected_age,
        rationale="Primary verified source",
        evidence_ids=[ev_docs.evidence_id],
    )

    spec = create_initial_spec("Air quality", "Delhi air quality observations")
    spec.requirement = req
    spec.plan = PlanSpec(
        planning_policy=policy,
        candidate_sources=[candidate],
        research_status="complete",
        retrieval_instructions=[instr],
        provider_statuses={"zenodo": "results"},
    )
    return spec


# ---------------------------------------------------------------------------
# 1. Model & field validation
# ---------------------------------------------------------------------------

def test_blocked_fetch_record_enforces_pre_network_and_redirect_semantics():
    rec = BlockedFetchRecord(
        url="https://169.254.169.254/latest/meta-data",
        reason_code="redirect_blocked",
        network_attempted=False,
        block_stage="redirect_validation",
        detail="Redirect target resolved to link-local metadata IP",
        recorded_at=_NOW,
        redirect_chain=["https://public.example.org/start"],
    )
    assert rec.network_attempted is False
    assert rec.redirect_chain == ["https://public.example.org/start"]

    for stage in ("pre_request_policy", "pre_request_dns", "redirect_validation"):
        with pytest.raises(ValidationError, match="network_attempted must be False"):
            BlockedFetchRecord(
                url="https://blocked.example.org/",
                reason_code="allowlist_blocked",
                network_attempted=True,
                block_stage=stage,
                detail="Blocked before connecting",
                recorded_at=_NOW,
            )


def test_source_evidence_artifact_relative_path_rejects_traversal():
    ev = _evidence(artifact_relative_path="evidence/page.txt")
    assert ev.artifact_relative_path == "evidence/page.txt"

    for bad_path in ("../secret.txt", "/etc/passwd", "C:\\temp\\a.txt", "a\x00b.txt"):
        with pytest.raises(ValidationError, match="safe relative path"):
            _evidence(artifact_relative_path=bad_path)


def test_candidate_fallback_rank_only_valid_when_deferred():
    c = CandidateSource(
        source_url="https://fallback.example.org/",
        title="Fallback",
        discovery_method="web_search",
        discovered_at=_NOW,
        selection_status="deferred",
        fallback_rank=1,
        selection_reason="ranked_fallback: rank=1",
    )
    assert c.fallback_rank == 1

    for status in ("not_selected", "selected", "rejected"):
        with pytest.raises(ValidationError, match="fallback_rank"):
            CandidateSource(
                source_url="https://fallback.example.org/",
                title="Bad",
                discovery_method="web_search",
                discovered_at=_NOW,
                selection_status=status,
                fallback_rank=1,
            )


def test_retrieval_instruction_rejects_secrets_and_non_https():
    base = dict(
        candidate_id=uuid4(),
        source_url="https://api.example.org/v1/obs",
        retrieval_method="http_api",
        expected_format="json",
        parser_hint="json_records",
        access_class="free",
        authentication_requirement="none",
        rationale="Valid instruction",
        evidence_ids=[uuid4()],
    )
    assert RetrievalInstruction(**base).http_method == "GET"

    # Non-HTTPS rejected
    with pytest.raises(ValidationError, match="https"):
        RetrievalInstruction(**{**base, "source_url": "http://api.example.org/v1/obs"})

    # Userinfo credentials in URL rejected
    with pytest.raises(ValidationError, match="credentials"):
        RetrievalInstruction(**{**base, "source_url": "https://user:pass@api.example.org/v1/obs"})

    # Secret query param in URL or query_params rejected
    with pytest.raises(ValidationError, match="credential parameter"):
        RetrievalInstruction(**{**base, "source_url": "https://api.example.org/v1/obs?api_key=secret123"})
    with pytest.raises(ValidationError, match="credential_env_var"):
        RetrievalInstruction(**{**base, "query_params": {"access_token": "abc"}})

    # Secret header rejected
    with pytest.raises(ValidationError, match="headers"):
        RetrievalInstruction(**{**base, "headers": {"Authorization": "Bearer secret"}})

    # Credential env var pairing rules
    with pytest.raises(ValidationError, match="must not be set"):
        RetrievalInstruction(**{
            **base,
            "authentication_requirement": "none",
            "credential_env_var": "MY_KEY",
            "credential_location": "header",
            "credential_field_name": "X-API-Key",
        })
    with pytest.raises(ValidationError, match="requires credential_env_var"):
        RetrievalInstruction(**{**base, "authentication_requirement": "api_key"})
    with pytest.raises(ValidationError, match="uppercase"):
        RetrievalInstruction(**{
            **base,
            "authentication_requirement": "api_key",
            "credential_env_var": "my_lowercase_key",
            "credential_location": "header",
            "credential_field_name": "X-API-Key",
        })

    # Runtime freshness threshold pairing
    with pytest.raises(ValidationError, match="requires max_data_age_hours"):
        RetrievalInstruction(**{**base, "requires_runtime_freshness_verification": True})
    with pytest.raises(ValidationError, match="may only be set"):
        RetrievalInstruction(**{**base, "max_data_age_hours": 1})


def test_connector_not_required_requires_evidence_backed_connector_finding():
    ev = _evidence()
    f_unknown = SourceFinding(
        dimension="connector",
        claim="Unknown",
        value=None,
        evidence_ids=[],
        confidence=0.0,
        status="unknown",
        explanation="Not investigated.",
    )
    assessment = SourceAssessment(
        required_dimensions=["connector"],
        findings_by_dimension={"connector": [f_unknown.finding_id]},
        connector=ConnectorAssessment(status="not_required", finding_ids=[f_unknown.finding_id]),
    )
    with pytest.raises(ValidationError, match="evidence-backed connector findings"):
        CandidateSource(
            source_url="https://data.example.org/",
            title="Test",
            discovery_method="web_search",
            discovered_at=_NOW,
            evidence=[ev],
            findings=[f_unknown],
            assessment=assessment,
            evaluation_status="complete",
        )


def test_replan_attempt_model_validation():
    cid = uuid4()
    attempt = ReplanAttempt(
        attempt_number=1,
        triggered_at=_NOW,
        reason="Primary endpoint timed out",
        failed_candidate_id=cid,
        failed_endpoint_url="https://api.example.org/v1/obs",
        excluded_candidate_ids=[cid],
        excluded_urls=["https://api.example.org/v1/obs"],
    )
    assert attempt.attempt_number == 1

    with pytest.raises(ValidationError, match="exclude at least one"):
        ReplanAttempt(attempt_number=1, triggered_at=_NOW, reason="Empty")

    with pytest.raises(ValidationError, match="failed_candidate_id"):
        ReplanAttempt(
            attempt_number=1,
            triggered_at=_NOW,
            reason="Mismatch",
            failed_candidate_id=cid,
            excluded_urls=["https://api.example.org/v1/obs"],
        )

    with pytest.raises(ValidationError, match="failed_endpoint_url"):
        ReplanAttempt(
            attempt_number=1,
            triggered_at=_NOW,
            reason="Mismatch",
            failed_endpoint_url="https://api.example.org/v1/obs",
            excluded_candidate_ids=[cid],
            excluded_urls=["https://data.example.org/landing"],
        )


# ---------------------------------------------------------------------------
# 2. All 10 validate_spec() invariants
# ---------------------------------------------------------------------------

def test_valid_increment4_plan_passes_validate_spec():
    spec = _build_valid_plan_spec(
        freshness_need="live",
        usage_rights=["use", "modify"],
        conditional_modify=True,
    )
    valid, errors = validate_spec(spec)
    assert valid, errors


def test_invariant_1_and_2_cardinality_and_unique_fallback_ranks():
    spec = _build_valid_plan_spec()
    c1 = spec.plan.candidate_sources[0]
    c2 = CandidateSource(
        source_url="https://b.example.org/",
        title="B",
        discovery_method="web_search",
        discovered_at=_NOW,
        evaluation_status="not_started",
        selection_status="selected",
    )
    spec.plan.candidate_sources.append(c2)
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("At most one CandidateSource" in e for e in errors)

    # Duplicate fallback ranks
    spec = _build_valid_plan_spec()
    d1 = CandidateSource(
        source_url="https://d1.example.org/",
        title="D1",
        discovery_method="web_search",
        discovered_at=_NOW,
        selection_status="deferred",
        fallback_rank=1,
    )
    d2 = CandidateSource(
        source_url="https://d2.example.org/",
        title="D2",
        discovery_method="web_search",
        discovered_at=_NOW,
        selection_status="deferred",
        fallback_rank=1,
    )
    spec.plan.candidate_sources.extend([d1, d2])
    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("fallback_rank" in e for e in errors)


def test_invariant_4_landing_page_cannot_authorize_unrelated_endpoint():
    spec = _build_valid_plan_spec(endpoint_url="https://api.example.org/v1/obs")
    # Change instruction URL to an unrelated endpoint on the same host (or landing page)
    spec.plan.retrieval_instructions[0].source_url = "https://data.example.org/unrelated.csv"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("endpoint_url" in e for e in errors)

    # Method mismatch also fails
    spec2 = _build_valid_plan_spec(retrieval_method="http_api")
    spec2.plan.retrieval_instructions[0].retrieval_method = "direct_download"
    valid2, errors2 = validate_spec(spec2)
    assert not valid2
    assert any("retrieval_methods" in e for e in errors2)

    # Unsupported connector status fails
    spec3 = _build_valid_plan_spec(connector_status="not_implemented")
    valid3, errors3 = validate_spec(spec3)
    assert not valid3
    assert any("connector assessment status" in e for e in errors3)


def test_invariant_4_endpoint_query_strings_filters_and_fixed_params():
    # 1. Base endpoint + legitimate request filters in query_params -> valid
    spec_filter = _build_valid_plan_spec(endpoint_url="https://api.example.org/v1/obs")
    spec_filter.plan.retrieval_instructions[0].query_params = {"city": "Delhi", "limit": "100"}
    valid, errors = validate_spec(spec_filter)
    assert valid, errors

    # 2. Arbitrary URL query string when documented endpoint has no query string -> rejected
    spec_bypass = _build_valid_plan_spec(endpoint_url="https://api.example.org/v1/obs")
    spec_bypass.plan.retrieval_instructions[0].source_url = (
        "https://api.example.org/v1/obs?undocumented=1"
    )
    valid_b, errors_b = validate_spec(spec_bypass)
    assert not valid_b
    assert any("endpoint_url" in e for e in errors_b)

    # 3. Documented endpoint with fixed query params + non-colliding query_params -> valid
    fixed_ep = "https://data.example.org/export?dataset=aq&format=csv"
    spec_fixed = _build_valid_plan_spec(endpoint_url=fixed_ep, retrieval_method="direct_download")
    spec_fixed.plan.retrieval_instructions[0].query_params = {"region": "north"}
    valid_f, errors_f = validate_spec(spec_fixed)
    assert valid_f, errors_f

    # 4. Altered fixed query param on instruction URL -> rejected
    spec_fixed_bad = _build_valid_plan_spec(endpoint_url=fixed_ep, retrieval_method="direct_download")
    spec_fixed_bad.plan.retrieval_instructions[0].source_url = (
        "https://data.example.org/export?dataset=other&format=csv"
    )
    valid_fb, errors_fb = validate_spec(spec_fixed_bad)
    assert not valid_fb
    assert any("endpoint_url" in e for e in errors_fb)

    # 5. Overriding fixed query param via query_params rejected by RetrievalInstruction
    with pytest.raises(ValidationError, match="duplicate or override fixed query parameters"):
        RetrievalInstruction(
            candidate_id=uuid4(),
            source_url=fixed_ep,
            retrieval_method="direct_download",
            query_params={"dataset": "hijacked"},
            expected_format="csv",
            parser_hint="csv_table",
            access_class="free",
            authentication_requirement="none",
            rationale="Override attempt",
            evidence_ids=[uuid4()],
        )

    # 6. observed_response at one URL cannot authorize an unrelated endpoint URL
    spec_obs = _build_valid_plan_spec(endpoint_url="https://api.example.org/v1/obs")
    cand = spec_obs.plan.candidate_sources[0]
    ev_other_obs = cand.evidence[3]  # observed_response for https://api.example.org/v1/obs
    # Point retrieval_methods finding ONLY to an observed_response at a different endpoint
    ev_other_obs.source_url = "https://api.example.org/v1/different"
    for f in cand.findings:
        if f.dimension == "retrieval_methods":
            f.evidence_ids = [ev_other_obs.evidence_id]
    spec_obs.plan.retrieval_instructions[0].evidence_ids = [ev_other_obs.evidence_id]
    valid_o, errors_o = validate_spec(spec_obs)
    assert not valid_o
    assert any("endpoint_url" in e for e in errors_o)


def test_invariant_5_and_6_access_and_per_right_license_conditions():
    spec = _build_valid_plan_spec(usage_rights=["use", "modify"], conditional_modify=True)
    # Tamper with conditional text on instruction
    spec.plan.retrieval_instructions[0].license_conditions_by_right["modify"] = "Altered text"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("license_conditions_by_right['modify']" in e for e in errors)

    # Missing required conditional right from instruction
    spec2 = _build_valid_plan_spec(usage_rights=["use", "modify"], conditional_modify=True)
    spec2.plan.retrieval_instructions[0].license_conditions_by_right = {}
    valid2, errors2 = validate_spec(spec2)
    assert not valid2
    assert any("license_conditions_by_right['modify']" in e for e in errors2)

    # Allowed right cannot carry condition in instruction
    spec3 = _build_valid_plan_spec(usage_rights=["use"])
    spec3.plan.retrieval_instructions[0].license_conditions_by_right = {"use": "Fake condition"}
    valid3, errors3 = validate_spec(spec3)
    assert not valid3
    assert any("must not carry conditional restrictions" in e for e in errors3)


def test_invariant_7_freshness_threshold_and_claim_type():
    # Explicit max_data_age_hours preserved
    spec = _build_valid_plan_spec(freshness_need="live", max_data_age_hours=6)
    assert spec.plan.retrieval_instructions[0].max_data_age_hours == 6
    valid, errors = validate_spec(spec)
    assert valid, errors

    # Matching textual and numeric thresholds pass
    spec_match = _build_valid_plan_spec(
        freshness_need="recent", max_data_age="2 days", max_data_age_hours=48
    )
    assert spec_match.plan.retrieval_instructions[0].max_data_age_hours == 48
    valid_m, errors_m = validate_spec(spec_match)
    assert valid_m, errors_m

    # Mismatched instruction threshold rejected
    spec.plan.retrieval_instructions[0].max_data_age_hours = 1
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("preserve requirement threshold (6)" in e for e in errors)

    # Recognized text-only durations resolve accurately
    for text, expected_h in [
        ("6 hours", 6),
        ("2 days", 48),
        ("1 week", 168),
        ("120 minutes", 2),
    ]:
        req = RequirementSpec(
            dataset_name="AQ",
            description="AQ",
            freshness_need="live",
            max_data_age=text,
            max_data_age_hours=None,
        )
        hours, err = resolve_freshness_threshold_hours(req)
        assert err is None
        assert hours == expected_h

    # Conflicting text and numeric thresholds must raise a clear error, never silently pick one
    conflict_req = RequirementSpec(
        dataset_name="AQ",
        description="AQ",
        freshness_need="recent",
        max_data_age="24 hours",
        max_data_age_hours=6,
    )
    hours_c, err_c = resolve_freshness_threshold_hours(conflict_req)
    assert hours_c is None
    assert err_c is not None and "Conflicting freshness thresholds" in err_c

    # Ambiguous or sub-hour textual durations require clarification rather than guessing
    for bad_text in ("very recent", "30 minutes", "90 minutes", "0 hours"):
        ambiguous_req = RequirementSpec(
            dataset_name="AQ",
            description="AQ",
            freshness_need="live",
            max_data_age=bad_text,
            max_data_age_hours=None,
        )
        hours_b, err_b = resolve_freshness_threshold_hours(ambiguous_req)
        assert hours_b is None
        assert err_b is not None

    # Defaults apply when no explicit threshold is supplied
    assert resolve_freshness_threshold_hours(
        RequirementSpec(dataset_name="AQ", description="AQ", freshness_need="live")
    ) == (1, None)
    assert resolve_freshness_threshold_hours(
        RequirementSpec(dataset_name="AQ", description="AQ", freshness_need="recent")
    ) == (24, None)

    # Boundary consistency: 0 and negative values rejected on RequirementSpec & RetrievalInstruction
    for bad_val in (0, -1):
        with pytest.raises(ValidationError):
            RequirementSpec(
                dataset_name="AQ",
                description="AQ",
                freshness_need="live",
                max_data_age_hours=bad_val,
            )
        with pytest.raises(ValidationError):
            RetrievalInstruction(
                candidate_id=uuid4(),
                source_url="https://api.example.org/v1/obs",
                retrieval_method="http_api",
                expected_format="json",
                parser_hint="json_records",
                access_class="free",
                authentication_requirement="none",
                requires_runtime_freshness_verification=True,
                max_data_age_hours=bad_val,
                rationale="Test",
                evidence_ids=[uuid4()],
            )

    # Observed claim_type without observed_response evidence rejected
    spec_bad_claim = _build_valid_plan_spec(freshness_need="live")
    cand = spec_bad_claim.plan.candidate_sources[0]
    for f in cand.findings:
        if f.dimension == "freshness":
            f.evidence_ids = [cand.evidence[1].evidence_id]  # dataset_page, not observed_response
    valid, errors = validate_spec(spec_bad_claim)
    assert not valid
    assert any("requires observed_response evidence" in e for e in errors)


def test_invariant_8_and_9_research_status_and_blocked_fetch_refs():
    spec = _build_valid_plan_spec()
    spec.plan.research_status = "not_started"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("research_status='not_started'" in e for e in errors)

    spec2 = _build_valid_plan_spec()
    spec2.plan.blocked_fetches.append(
        BlockedFetchRecord(
            url="https://blocked.example.org/",
            candidate_id=uuid4(),
            reason_code="allowlist_blocked",
            network_attempted=False,
            block_stage="pre_request_policy",
            detail="Not allowlisted",
            recorded_at=_NOW,
        )
    )
    valid2, errors2 = validate_spec(spec2)
    assert not valid2
    assert any("unknown candidate_id" in e for e in errors2)


def test_invariant_8_provider_aware_research_status_cases():
    base_req = RequirementSpec(dataset_name="AQ", description="AQ", is_supported=True)

    def make_plan_spec(**plan_kwargs) -> DatasetSpec:
        s = create_initial_spec("AQ", "AQ")
        s.requirement = base_req
        s.plan = PlanSpec(**plan_kwargs)
        return s

    # 1. Provider error with zero candidates and zero blocked URLs:
    #    - With another succeeded provider -> partial is valid
    partial_err = make_plan_spec(
        research_status="partial",
        provider_statuses={"zenodo": "error", "ckan": "empty"},
    )
    assert validate_spec(partial_err) == (True, [])
    #    - Single failing configured provider with 0 candidates -> partial is invalid, failed is valid
    only_err_partial = make_plan_spec(
        research_status="partial",
        provider_statuses={"zenodo": "error"},
    )
    ok, errs = validate_spec(only_err_partial)
    assert not ok
    assert any("partial" in e for e in errs)

    # 2. Invalid response with zero candidates:
    inv_partial = make_plan_spec(
        research_status="partial",
        provider_statuses={"zenodo": "invalid_response", "ckan": "empty"},
    )
    assert validate_spec(inv_partial) == (True, [])
    inv_failed = make_plan_spec(
        research_status="failed",
        provider_statuses={"zenodo": "invalid_response"},
    )
    assert validate_spec(inv_failed) == (True, [])

    # 3. Successful empty discovery (ignoring unconfigured web_search) -> complete is valid
    empty_complete = make_plan_spec(
        research_status="complete",
        provider_statuses={"zenodo": "empty", "ckan": "empty", "web_search": "not_configured"},
    )
    assert validate_spec(empty_complete) == (True, [])

    # 4. All configured providers failing -> failed is valid; complete & partial are invalid
    for bad_status in ("complete", "partial"):
        all_fail_bad = make_plan_spec(
            research_status=bad_status,
            provider_statuses={
                "zenodo": "error",
                "ckan": "invalid_response",
                "web_search": "not_configured",
            },
        )
        ok, _ = validate_spec(all_fail_bad)
        assert not ok
    all_fail_ok = make_plan_spec(
        research_status="failed",
        provider_statuses={
            "zenodo": "error",
            "ckan": "invalid_response",
            "web_search": "not_configured",
        },
    )
    assert validate_spec(all_fail_ok) == (True, [])

    # 5. Mixture of successful and failed providers with evaluated candidate -> partial valid, complete invalid
    mix_spec = _build_valid_plan_spec()
    mix_spec.plan.provider_statuses = {"zenodo": "results", "ckan": "error"}
    mix_spec.plan.research_status = "complete"
    ok, errs = validate_spec(mix_spec)
    assert not ok
    assert any("failed providers" in e for e in errs)
    mix_spec.plan.research_status = "partial"
    assert validate_spec(mix_spec) == (True, [])

    # 6. Only unconfigured optional providers (and empty dict) -> not_started valid;
    #    complete, partial, and failed all invalid (not_configured is not a failure)
    assert validate_spec(
        make_plan_spec(
            research_status="not_started",
            provider_statuses={"web_search": "not_configured"},
        )
    ) == (True, [])
    for invalid_status in ("complete", "partial", "failed"):
        only_unconf = make_plan_spec(
            research_status=invalid_status,
            provider_statuses={"web_search": "not_configured"},
        )
        ok, _ = validate_spec(only_unconf)
        assert not ok
    ok_empty, _ = validate_spec(make_plan_spec(research_status="complete", provider_statuses={}))
    assert not ok_empty


def test_invariant_8_candidate_completion_vs_failure_combinations():
    # 1. One complete candidate and one failed candidate -> complete INVALID, partial VALID
    spec_mixed = _build_valid_plan_spec()
    failed_cand = CandidateSource(
        source_url="https://failed.example.org/",
        title="Failed candidate",
        discovery_method="repository_search",
        discovered_at=_NOW,
        evaluation_status="failed",
        selection_status="rejected",
    )
    spec_mixed.plan.candidate_sources.append(failed_cand)
    spec_mixed.plan.research_status = "complete"
    ok_complete, errs_complete = validate_spec(spec_mixed)
    assert not ok_complete
    assert any("failed candidates" in e for e in errs_complete)

    spec_mixed.plan.research_status = "partial"
    assert validate_spec(spec_mixed) == (True, [])

    # 2. All candidates failed (even when provider returned results) ->
    #    complete INVALID, partial INVALID, failed VALID
    base_req = RequirementSpec(dataset_name="AQ", description="AQ", is_supported=True)
    all_failed_spec = create_initial_spec("AQ", "AQ")
    all_failed_spec.requirement = base_req
    f1 = CandidateSource(
        source_url="https://failed-a.example.org/",
        title="Failed A",
        discovery_method="repository_search",
        discovered_at=_NOW,
        evaluation_status="failed",
        selection_status="rejected",
    )
    f2 = CandidateSource(
        source_url="https://failed-b.example.org/",
        title="Failed B",
        discovery_method="repository_search",
        discovered_at=_NOW,
        evaluation_status="failed",
        selection_status="rejected",
    )
    for bad_status in ("complete", "partial"):
        all_failed_spec.plan = PlanSpec(
            candidate_sources=[f1, f2],
            provider_statuses={"zenodo": "results"},
            research_status=bad_status,
        )
        ok, _ = validate_spec(all_failed_spec)
        assert not ok

    all_failed_spec.plan = PlanSpec(
        candidate_sources=[f1, f2],
        provider_statuses={"zenodo": "results"},
        research_status="failed",
    )
    assert validate_spec(all_failed_spec) == (True, [])

    # 3. All candidates complete -> complete VALID, partial INVALID
    spec_all_ok = _build_valid_plan_spec()
    assert spec_all_ok.plan.research_status == "complete"
    assert validate_spec(spec_all_ok) == (True, [])
    spec_all_ok.plan.research_status = "partial"
    ok_partial, _ = validate_spec(spec_all_ok)
    assert not ok_partial

    # 4. Successful empty discovery -> complete VALID
    empty_ok = create_initial_spec("AQ", "AQ")
    empty_ok.requirement = base_req
    empty_ok.plan = PlanSpec(
        research_status="complete",
        provider_statuses={"zenodo": "empty", "web_search": "not_configured"},
    )
    assert validate_spec(empty_ok) == (True, [])

    # 5. Failed provider alongside successful provider -> complete INVALID, partial VALID
    prov_mix = create_initial_spec("AQ", "AQ")
    prov_mix.requirement = base_req
    prov_mix.plan = PlanSpec(
        research_status="complete",
        provider_statuses={"zenodo": "empty", "ckan": "error"},
    )
    ok_pm, _ = validate_spec(prov_mix)
    assert not ok_pm
    prov_mix.plan.research_status = "partial"
    assert validate_spec(prov_mix) == (True, [])


def test_invariant_10_replan_history_and_failed_endpoint_exclusion():
    spec = _build_valid_plan_spec(endpoint_url="https://api.example.org/v1/obs")
    primary = spec.plan.candidate_sources[0]

    # Case A: Replan excludes failed endpoint URL, instruction cannot reuse it
    spec.replan_count = 1
    spec.replan_history = [
        ReplanAttempt(
            attempt_number=1,
            triggered_at=_NOW,
            reason="Endpoint timed out",
            failed_candidate_id=primary.candidate_id,
            failed_endpoint_url="https://api.example.org/v1/obs",
            excluded_candidate_ids=[primary.candidate_id],
            excluded_urls=["https://api.example.org/v1/obs"],
        )
    ]
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("selection_status='rejected'" in e for e in errors)
    assert any("excluded candidate" in e for e in errors)
    assert any("excluded endpoint URL" in e for e in errors)

    # Case B: Non-sequential attempt_number and count mismatch
    spec.replan_count = 2
    spec.replan_history[0].attempt_number = 2
    valid2, errors2 = validate_spec(spec)
    assert not valid2
    assert any("replan_count" in e for e in errors2)
    assert any("sequential" in e for e in errors2)


# ---------------------------------------------------------------------------
# 3. Backward compatibility with persisted runs
# ---------------------------------------------------------------------------

def test_backward_compatibility_loads_persisted_current_json():
    current_path = Path(__file__).resolve().parent.parent / "runs" / "current.json"
    if not current_path.exists():
        pytest.skip("runs/current.json not present")

    raw = json.loads(current_path.read_text(encoding="utf-8"))
    spec = DatasetSpec.model_validate(raw)
    assert spec.replan_count == 0
    assert spec.replan_history == []
    if spec.plan is not None:
        assert spec.plan.retrieval_instructions == []
        assert spec.plan.blocked_fetches == []
        assert spec.plan.provider_statuses == {}
    valid, errors = validate_spec(spec)
    assert valid, errors


# ---------------------------------------------------------------------------
# 4. Final research-status semantics: complete / partial / failed
# ---------------------------------------------------------------------------

def _rejected_complete_candidate(source_url: str, title: str) -> CandidateSource:
    """A fully evaluated candidate that was cleanly rejected.

    evaluation_status='complete' + selection_status='rejected' is a finished
    evaluation, not an evaluation failure.
    """
    finding = SourceFinding(
        dimension="purpose",
        claim="Investigated; not a usable match for the requirement.",
        value=None,
        evidence_ids=[],
        confidence=0.0,
        status="unknown",
        explanation="Dimension investigated; no usable value established.",
    )
    assessment = SourceAssessment(
        required_dimensions=["purpose"],
        findings_by_dimension={"purpose": [finding.finding_id]},
    )
    return CandidateSource(
        source_url=source_url,
        title=title,
        discovery_method="repository_search",
        discovered_at=_NOW,
        findings=[finding],
        assessment=assessment,
        evaluation_status="complete",
        selection_status="rejected",
        selection_reason="Fully evaluated and rejected.",
    )


def _failed_candidate(source_url: str, title: str) -> CandidateSource:
    """A candidate whose evaluation failed (distinct from a clean rejection)."""
    return CandidateSource(
        source_url=source_url,
        title=title,
        discovery_method="repository_search",
        discovered_at=_NOW,
        evaluation_status="failed",
        selection_status="rejected",
        selection_reason="Evaluation failed: source unreachable.",
    )


def _deferred_candidate(
    source_url: str = "https://fallback.example.org/",
    title: str = "Fallback",
    *,
    quota_status: str = "not_applicable",
    license_status: str = "allowed",
    connector_status: str = "planned",
    retrieval_methods: tuple = ("http_api", "manual"),
) -> CandidateSource:
    """A fully evaluated, deferred fallback candidate with a configurable assessment.

    Default configuration is usable: quota within limits, 'use' right allowed,
    connector planned (implementation pending), and a documented manual path.
    """
    ev_docs = _evidence(
        source_url="https://fallback.example.org/docs",
        title="Fallback API docs",
        excerpt="GET https://fallback.example.org/v1/data returns JSON.",
    )
    ev_license = _evidence(
        source_url="https://fallback.example.org/license",
        title="License page",
        evidence_type="license_page",
        excerpt="CC-BY-4.0; attribution required.",
    )
    ev_conn = _evidence(
        source_url="https://fallback.example.org/v1/data",
        title="Connector test",
        evidence_type="connector_test",
        capture_method="observed_response",
        authority="observed",
    )

    f_access = SourceFinding(
        dimension="access",
        claim="Free access without authentication.",
        value={"access_class": "free", "auth": "none"},
        evidence_ids=[ev_docs.evidence_id],
        confidence=0.9,
        status="verified",
        explanation="Documented access terms.",
    )
    f_license = SourceFinding(
        dimension="license_usage",
        claim="Licensed under CC-BY-4.0.",
        value={"spdx": "CC-BY-4.0"},
        evidence_ids=[ev_license.evidence_id],
        confidence=0.9,
        status="verified",
        explanation="License page states the terms.",
    )
    f_retrieval = SourceFinding(
        dimension="retrieval_methods",
        claim="Data is retrievable over HTTP.",
        value={
            "methods": list(retrieval_methods),
            "endpoint_url": "https://fallback.example.org/v1/data",
        },
        evidence_ids=[ev_docs.evidence_id],
        confidence=0.9,
        status="verified",
        explanation="Documented retrieval endpoint.",
    )
    f_connector = SourceFinding(
        dimension="connector",
        claim=f"Connector status: {connector_status}.",
        value={"status": connector_status},
        evidence_ids=[ev_conn.evidence_id],
        confidence=0.9,
        status="verified",
        explanation="Observed connector behaviour.",
    )

    if quota_status == "exceeds_limit":
        quota = QuotaAssessment(
            status="exceeds_limit",
            expected_usage=200.0,
            expected_usage_unit="requests/day",
            limit_value=100.0,
            limit_unit="requests/day",
            finding_ids=[f_access.finding_id],
        )
    else:
        quota = QuotaAssessment(
            status=quota_status,
            finding_ids=[f_access.finding_id],
        )

    if license_status == "unknown":
        license_perm = LicensePermissionAssessment(
            status="unknown",
            verification_status="unknown",
        )
    else:
        license_perm = LicensePermissionAssessment(
            status=license_status,
            verification_status="verified",
            finding_ids=[f_license.finding_id],
        )

    assessment = SourceAssessment(
        required_dimensions=[
            "access",
            "license_usage",
            "retrieval_methods",
            "connector",
        ],
        findings_by_dimension={
            "access": [f_access.finding_id],
            "license_usage": [f_license.finding_id],
            "retrieval_methods": [f_retrieval.finding_id],
            "connector": [f_connector.finding_id],
        },
        access_class="free",
        access_class_finding_ids=[f_access.finding_id],
        authentication_requirement="none",
        authentication_finding_ids=[f_access.finding_id],
        quota=quota,
        license_permissions={"use": license_perm},
        connector=ConnectorAssessment(
            status=connector_status,
            finding_ids=[f_connector.finding_id],
        ),
    )

    return CandidateSource(
        source_url=source_url,
        title=title,
        discovery_method="repository_search",
        discovered_at=_NOW,
        evidence=[ev_docs, ev_license, ev_conn],
        findings=[f_access, f_license, f_retrieval, f_connector],
        assessment=assessment,
        evaluation_status="complete",
        selection_status="deferred",
        fallback_rank=1,
        selection_reason="Valid match; connector implementation pending.",
    )


def _spec_with_plan(
    candidates,
    *,
    research_status: str,
    provider_statuses: dict,
    blocked_fetches=None,
) -> DatasetSpec:
    spec = create_initial_spec("AQ", "Air quality observations")
    spec.plan = PlanSpec(
        candidate_sources=list(candidates),
        research_status=research_status,
        provider_statuses=dict(provider_statuses),
        blocked_fetches=list(blocked_fetches or []),
    )
    return spec


def _blocked_fetch_for(candidate: CandidateSource) -> BlockedFetchRecord:
    return BlockedFetchRecord(
        url="https://api.primary.example.org/v1/obs",
        candidate_id=candidate.candidate_id,
        reason_code="allowlist_blocked",
        network_attempted=False,
        block_stage="pre_request_policy",
        detail="Endpoint not on the fetch allowlist; no network request was made.",
        recorded_at=_NOW,
    )


def test_research_status_selected_plus_failed_plus_provider_failure_is_partial():
    # Scenario 1: selected complete candidate + failed candidate + provider failure
    spec = _build_valid_plan_spec()
    spec.plan.provider_statuses = {"zenodo": "results", "ckan": "error"}
    spec.plan.candidate_sources.append(
        _failed_candidate("https://failed.example.org/", "Failed candidate")
    )
    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("selected or deferred candidate" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed providers" in e for e in errors)


def test_research_status_deferred_plus_failed_plus_provider_failure_is_partial():
    # Scenario 2: valid deferred candidate + failed candidate + provider failure
    deferred = _deferred_candidate()
    spec = _spec_with_plan(
        [deferred, _failed_candidate("https://failed.example.org/", "Failed candidate")],
        research_status="partial",
        provider_statuses={"zenodo": "results", "ckan": "error"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("selected or deferred candidate" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed providers" in e for e in errors)


def test_research_status_all_rejected_complete_with_successful_providers_is_complete():
    # Scenario 3: clean evaluations, clean rejections, successful providers
    candidates = [
        _rejected_complete_candidate("https://a.example.org/", "Rejected A"),
        _rejected_complete_candidate("https://b.example.org/", "Rejected B"),
    ]
    spec = _spec_with_plan(
        candidates,
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("requires a recoverable" in e for e in errors)

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("requires a total failure" in e for e in errors)


def test_research_status_rejected_plus_failed_without_provider_failure_is_partial():
    # Scenario 4: one clean rejection, one failed evaluation, no provider failure
    candidates = [
        _rejected_complete_candidate("https://a.example.org/", "Rejected A"),
        _failed_candidate("https://failed.example.org/", "Failed candidate"),
    ]
    spec = _spec_with_plan(
        candidates,
        research_status="partial",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed candidates" in e for e in errors)

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("requires a total failure" in e for e in errors)


def test_research_status_all_candidates_failed_is_failed():
    # Scenario 5: every discovered candidate failed evaluation
    candidates = [
        _failed_candidate("https://failed-a.example.org/", "Failed A"),
        _failed_candidate("https://failed-b.example.org/", "Failed B"),
    ]
    spec = _spec_with_plan(
        candidates,
        research_status="failed",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("all candidates failed" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed candidates" in e for e in errors)


def test_research_status_mixed_empty_error_providers_zero_candidates_is_partial():
    # Scenario 6: one provider succeeded (empty), one failed -> partial, not failed
    spec = _spec_with_plan(
        [],
        research_status="partial",
        provider_statuses={"zenodo": "empty", "ckan": "error"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("requires a total failure" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed providers" in e for e in errors)


def test_research_status_all_providers_failed_zero_candidates_is_failed():
    # Scenario 7: total discovery failure
    spec = _spec_with_plan(
        [],
        research_status="failed",
        provider_statuses={"zenodo": "error", "ckan": "invalid_response"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("all candidates failed with no usable results" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("failed providers" in e for e in errors)


def test_research_status_blocked_primary_with_valid_deferred_fallback_is_partial():
    # Scenario 8: blocked primary, usable deferred fallback -> partial, never failed
    primary = _rejected_complete_candidate("https://primary.example.org/", "Primary")
    fallback = _deferred_candidate()
    spec = _spec_with_plan(
        [primary, fallback],
        research_status="partial",
        provider_statuses={"zenodo": "results"},
        blocked_fetches=[_blocked_fetch_for(primary)],
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "failed"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("selected or deferred candidate" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("blocked fetches" in e for e in errors)


def test_research_status_blocked_primary_with_no_usable_fallback_is_failed():
    # Scenario 9: blocked primary, no selected or deferred fallback -> failed
    primary = _rejected_complete_candidate("https://primary.example.org/", "Primary")
    spec = _spec_with_plan(
        [primary],
        research_status="failed",
        provider_statuses={"zenodo": "results"},
        blocked_fetches=[_blocked_fetch_for(primary)],
    )
    valid, errors = validate_spec(spec)
    assert valid, errors

    spec.plan.research_status = "partial"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("requires a recoverable" in e for e in errors)

    spec.plan.research_status = "complete"
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("blocked fetches" in e for e in errors)


# ---------------------------------------------------------------------------
# 5. Deferred-candidate usability guards
# ---------------------------------------------------------------------------

def test_usable_deferred_candidate_is_a_valid_fallback():
    # Positive control: the default deferred candidate passes every usability guard
    spec = _spec_with_plan(
        [_deferred_candidate()],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert valid, errors


def test_deferred_candidate_with_exceeds_limit_quota_is_invalid():
    spec = _spec_with_plan(
        [_deferred_candidate(quota_status="exceeds_limit")],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("exceeds_limit" in e and "usable fallback" in e for e in errors)


def test_deferred_candidate_with_prohibited_required_license_right_is_invalid():
    spec = _spec_with_plan(
        [_deferred_candidate(license_status="prohibited")],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("required usage right 'use'" in e for e in errors)


def test_deferred_candidate_with_unknown_required_license_right_is_invalid():
    spec = _spec_with_plan(
        [_deferred_candidate(license_status="unknown")],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("required usage right 'use'" in e for e in errors)


def test_deferred_candidate_with_unsupported_connector_and_no_manual_path_is_invalid():
    spec = _spec_with_plan(
        [_deferred_candidate(connector_status="unsupported", retrieval_methods=("http_api",))],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("no documented manual retrieval path" in e for e in errors)

    # A documented manual retrieval path keeps the same candidate usable
    spec_manual = _spec_with_plan(
        [_deferred_candidate(connector_status="unsupported", retrieval_methods=("http_api", "manual"))],
        research_status="complete",
        provider_statuses={"zenodo": "results"},
    )
    valid, errors = validate_spec(spec_manual)
    assert valid, errors
