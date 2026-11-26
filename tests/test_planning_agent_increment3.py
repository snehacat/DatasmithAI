"""
Focused tests for Planning Agent Increment 3 integration.

Covers:
- ApplicationPlanningPolicy is derived from the actual RequirementSpec
- LLM output cannot override application-owned evaluation policy
- Required dimensions and criticality levels reach the Planner
- CandidateSource objects and their evidence/assessments can be stored in PlanSpec
- Old PlanSpec JSON without the new fields still validates (backward compatibility)
- research_status and selection_status remain independent
- Existing Planning Agent behaviour remains compatible
- Missing or invalid policy is handled safely
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agents import planning_agent as planning_module
from agents.planning_agent import PlanningAgent
from agents.source_catalog import SourceCatalogEntry
from core.dataset_spec import (
    ApplicationPlanningPolicy,
    CandidateSelectionStatus,
    CandidateSource,
    ConnectorAssessment,
    DatasetSpec,
    EvaluationRequirements,
    PlanSpec,
    RequirementSpec,
    SourceAssessment,
    SourceEvidence,
    SourceEvaluationStatus,
    SourceFinding,
    UsageIntent,
    create_initial_spec,
    derive_evaluation_requirements,
    validate_spec,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

def _minimal_complete_candidate(source_url: str, title: str, sel: str) -> CandidateSource:
    """Minimal CandidateSource with evaluation_status='complete' (needs an assessment)."""
    finding = SourceFinding(
        dimension="purpose",
        claim="Relevant to the request.",
        value={"relevant": True},
        evidence_ids=[],
        confidence=0.5,
        status="unverified",
        explanation="Catalog hint suggests relevance.",
    )
    assessment = SourceAssessment(
        required_dimensions=["purpose"],
        findings_by_dimension={"purpose": [finding.finding_id]},
    )
    return CandidateSource(
        source_url=source_url,
        title=title,
        discovery_method="web_search",
        discovered_at=_NOW,
        findings=[finding],
        assessment=assessment,
        evaluation_status="complete",
        selection_status=sel,
    )


def _minimal_partial_candidate(source_url: str, title: str, sel: str) -> CandidateSource:
    """Minimal CandidateSource with evaluation_status='partial' (needs at least one assessed
    dimension and at least one unassessed required dimension)."""
    finding = SourceFinding(
        dimension="purpose",
        claim="Relevant to the request.",
        value={"relevant": True},
        evidence_ids=[],
        confidence=0.5,
        status="unverified",
        explanation="Catalog hint suggests relevance.",
    )
    assessment = SourceAssessment(
        required_dimensions=["purpose", "access"],  # 'access' not yet assessed
        findings_by_dimension={"purpose": [finding.finding_id]},
    )
    return CandidateSource(
        source_url=source_url,
        title=title,
        discovery_method="web_search",
        discovered_at=_NOW,
        findings=[finding],
        assessment=assessment,
        evaluation_status="partial",
        selection_status=sel,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_requirement(**overrides) -> RequirementSpec:
    values = dict(
        dataset_name="Air quality study",
        description="live air quality data for Delhi",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5", "pm10"],
        geography="Delhi",
        freshness_need="live",
        time_window_days=7,
        is_supported=True,
    )
    values.update(overrides)
    return RequirementSpec(**values)


def _make_spec(**req_overrides) -> DatasetSpec:
    spec = create_initial_spec("Air quality study", "live air quality data for Delhi")
    spec.requirement = _make_requirement(**req_overrides)
    return spec


def _planner_with_empty_llm() -> PlanningAgent:
    agent = PlanningAgent()
    agent.llm = Mock()
    agent.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": [],
        "reasoning": "Rule-based.",
    })
    return agent


# ---------------------------------------------------------------------------
# 1. Policy is derived from the actual requirement
# ---------------------------------------------------------------------------

class TestPolicyDerivation:

    def test_plan_stores_planning_policy(self):
        """plan() must store a non-None planning_policy in the resulting PlanSpec."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert spec.plan is not None
        assert spec.plan.planning_policy is not None

    def test_planning_policy_is_application_planning_policy_instance(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert isinstance(spec.plan.planning_policy, ApplicationPlanningPolicy)

    def test_planning_policy_required_dimensions_come_from_requirement(self):
        """Freshness should be essential when freshness_need='live'."""
        planner = _planner_with_empty_llm()
        spec = _make_spec(freshness_need="live")
        planner.plan(spec)
        policy = spec.plan.planning_policy
        assert "freshness" in policy.evaluation_requirements.required_dimensions
        assert policy.evaluation_requirements.dimension_criticality["freshness"] == "essential"

    def test_planning_policy_geography_dimension_added_for_geographic_requirement(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec(geography="Delhi")
        planner.plan(spec)
        policy = spec.plan.planning_policy
        assert "geography" in policy.evaluation_requirements.required_dimensions

    def test_planning_policy_variables_dimension_added_when_features_given(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec(expected_features=["pm2.5", "pm10"])
        planner.plan(spec)
        policy = spec.plan.planning_policy
        assert "variables" in policy.evaluation_requirements.required_dimensions

    def test_planning_policy_always_includes_access_and_license_as_essential(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        policy = spec.plan.planning_policy
        reqs = policy.evaluation_requirements
        assert "access" in reqs.required_dimensions
        assert reqs.dimension_criticality["access"] == "essential"
        assert "license_usage" in reqs.required_dimensions
        assert reqs.dimension_criticality["license_usage"] == "essential"

    def test_planning_policy_publisher_reliability_is_advisory(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        reqs = spec.plan.planning_policy.evaluation_requirements
        assert "publisher_reliability" in reqs.required_dimensions
        assert reqs.dimension_criticality["publisher_reliability"] == "advisory"

    def test_policy_stored_even_when_no_sources_found(self, monkeypatch):
        """Policy must be stored even when no catalog sources match."""
        monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [])
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert spec.plan.planning_policy is not None

    def test_explicit_caller_policy_is_used_verbatim(self):
        """When a caller passes policy=..., the planner must store that exact object."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        caller_policy = derive_evaluation_requirements(
            spec.requirement,
            UsageIntent(required_rights=["use", "redistribute"]),
        )
        planner.plan(spec, policy=caller_policy)
        assert spec.plan.planning_policy is caller_policy

    def test_caller_policy_usage_intent_is_preserved(self):
        """The usage_intent inside caller-supplied policy must not be overwritten."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        caller_policy = derive_evaluation_requirements(
            spec.requirement,
            UsageIntent(required_rights=["use", "modify", "redistribute"]),
        )
        planner.plan(spec, policy=caller_policy)
        stored = spec.plan.planning_policy
        assert stored.usage_intent.required_rights == ["use", "modify", "redistribute"]


# ---------------------------------------------------------------------------
# 2. LLM output cannot override application-owned evaluation policy
# ---------------------------------------------------------------------------

class TestLLMCannotOverridePolicy:

    def test_llm_response_does_not_alter_planning_policy(self):
        """Even if the LLM response somehow included policy keys, the stored
        planning_policy must come from the application, not the LLM."""
        planner = _planner_with_empty_llm()
        # Inject a malicious LLM response that tries to override policy fields.
        planner.llm.call.return_value = SimpleNamespace(content={
            "extra_queries": ["pm2.5 Delhi live data"],
            "reasoning": "LLM reasoning.",
            # LLM should not be able to inject policy through these keys:
            "planning_policy": {"evil": "override"},
            "required_dimensions": ["connector"],
            "dimension_criticality": {},
        })
        spec = _make_spec(
            # Use a requirement that generates < 4 rule queries so the LLM fires
            geography=None,
            expected_features=["pm2.5"],
            freshness_need="unspecified",
        )
        planner.plan(spec, use_llm=True)
        policy = spec.plan.planning_policy
        # Policy must be an ApplicationPlanningPolicy from the application, not a dict
        assert isinstance(policy, ApplicationPlanningPolicy)
        # Must have the proper required_dimensions from the requirement, not LLM injection
        assert "access" in policy.evaluation_requirements.required_dimensions
        assert "license_usage" in policy.evaluation_requirements.required_dimensions

    def test_llm_exception_does_not_affect_stored_policy(self):
        """An LLM exception must not corrupt the planning_policy."""
        from core.llm_wrapper import LLMConnectionError
        planner = _planner_with_empty_llm()
        planner.llm.call.side_effect = LLMConnectionError("offline")
        spec = _make_spec(geography=None, expected_features=["pm2.5"], freshness_need="unspecified")
        planner.plan(spec, use_llm=True)
        assert isinstance(spec.plan.planning_policy, ApplicationPlanningPolicy)

    def test_planning_policy_is_identical_regardless_of_llm_content(self):
        """Policy derived from the same requirement must be identical whether or not
        the LLM call succeeds."""
        requirement = _make_requirement(geography=None, expected_features=["pm2.5"], freshness_need="unspecified")

        planner_ok = _planner_with_empty_llm()
        spec_ok = create_initial_spec("AQ", "pm2.5 data")
        spec_ok.requirement = requirement
        planner_ok.plan(spec_ok, use_llm=True)

        from core.llm_wrapper import LLMConnectionError
        planner_fail = _planner_with_empty_llm()
        planner_fail.llm.call.side_effect = LLMConnectionError("offline")
        spec_fail = create_initial_spec("AQ", "pm2.5 data")
        spec_fail.requirement = requirement
        planner_fail.plan(spec_fail, use_llm=True)

        assert (
            spec_ok.plan.planning_policy.evaluation_requirements.required_dimensions
            == spec_fail.plan.planning_policy.evaluation_requirements.required_dimensions
        )


# ---------------------------------------------------------------------------
# 3. Required dimensions and criticality levels reach the Planner
# ---------------------------------------------------------------------------

class TestDimensionsReachPlanner:

    @pytest.mark.parametrize("freshness_need,expected_dim", [
        ("live",    "freshness"),
        ("recent",  "freshness"),
        ("historical", "temporal_coverage"),
    ])
    def test_freshness_and_temporal_dimensions_derived_correctly(self, freshness_need, expected_dim):
        planner = _planner_with_empty_llm()
        spec = _make_spec(freshness_need=freshness_need, time_range=None, time_window_days=None)
        if freshness_need == "historical":
            spec.requirement.time_range = "2020 to 2023"
        planner.plan(spec)
        assert expected_dim in spec.plan.planning_policy.evaluation_requirements.required_dimensions

    def test_resolution_dimension_added_when_granularity_given(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec(time_granularity="Hourly")
        planner.plan(spec)
        assert "resolution" in spec.plan.planning_policy.evaluation_requirements.required_dimensions

    def test_all_essential_dimensions_have_essential_criticality(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        reqs = spec.plan.planning_policy.evaluation_requirements
        for dim in reqs.required_dimensions:
            criticality = reqs.dimension_criticality[dim]
            assert criticality in {"essential", "advisory"}, (
                f"Dimension {dim!r} has invalid criticality {criticality!r}"
            )

    def test_policy_version_is_set(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert spec.plan.planning_policy.evaluation_requirements.policy_version


# ---------------------------------------------------------------------------
# 4. CandidateSource objects can be stored in PlanSpec
# ---------------------------------------------------------------------------

class TestCandidateSourceStorage:

    def _make_candidate(self, source_url: str = "https://data.example.org/") -> CandidateSource:
        return CandidateSource(
            source_url=source_url,
            title="Example dataset",
            publisher="Example Provider",
            discovery_method="web_search",
            discovered_at=_NOW,
            evaluation_status="not_started",
            selection_status="not_selected",
        )

    def test_plan_spec_accepts_candidate_sources(self):
        candidate = self._make_candidate()
        plan = PlanSpec(candidate_sources=[candidate])
        assert len(plan.candidate_sources) == 1
        assert plan.candidate_sources[0].title == "Example dataset"

    def test_candidate_source_with_evidence_and_assessment_can_be_stored(self):
        """A fully assessed candidate with evidence and findings can be stored."""
        evidence = SourceEvidence(
            source_url="https://data.example.org/api",
            title="API documentation",
            discovery_method="provider_documentation",
            evidence_type="api_documentation",
            capture_method="fetched_content",
            authority="official_provider",
            retrieved_at=_NOW,
        )
        finding = SourceFinding(
            dimension="access",
            claim="Free tier access is available.",
            value={"access_class": "free_tier"},
            evidence_ids=[evidence.evidence_id],
            confidence=0.9,
            status="verified",
            explanation="Official documentation confirms free tier.",
        )
        assessment = SourceAssessment(
            required_dimensions=["access"],
            findings_by_dimension={"access": [finding.finding_id]},
            access_class="free_tier",
            access_class_finding_ids=[finding.finding_id],
        )
        candidate = CandidateSource(
            source_url="https://data.example.org/",
            title="Example dataset",
            discovery_method="web_search",
            discovered_at=_NOW,
            evidence=[evidence],
            findings=[finding],
            assessment=assessment,
            evaluation_status="complete",
            selection_status="selected",
        )
        plan = PlanSpec(
            candidate_sources=[candidate],
            research_status="complete",
        )
        assert plan.candidate_sources[0].assessment.access_class == "free_tier"
        assert plan.candidate_sources[0].selection_status == "selected"
        assert plan.research_status == "complete"

    def test_multiple_candidates_with_different_statuses(self):
        c1 = _minimal_complete_candidate("https://source-a.example.org/", "Source A", "selected")
        c2 = _minimal_partial_candidate("https://source-b.example.org/", "Source B", "deferred")
        c3 = CandidateSource(
            source_url="https://source-c.example.org/",
            title="Source C",
            discovery_method="repository_search",
            discovered_at=_NOW,
            evaluation_status="not_started",
            selection_status="rejected",
        )
        plan = PlanSpec(candidate_sources=[c1, c2, c3], research_status="partial")
        statuses = {c.title: c.selection_status for c in plan.candidate_sources}
        assert statuses == {"Source A": "selected", "Source B": "deferred", "Source C": "rejected"}

    def test_plan_stores_candidates_from_planner_run(self):
        """plan() does not populate candidates itself yet (discovery not implemented),
        but the field must be present and empty by default."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert spec.plan.candidate_sources == []
        assert spec.plan.research_status == "not_started"
        assert spec.plan.candidate_sources == []
        assert spec.plan.research_status == "not_started"


# ---------------------------------------------------------------------------
# 5. Backward compatibility: old PlanSpec JSON without new fields still validates
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_planspec_without_planning_policy_validates(self):
        """Old serialised PlanSpec that has no planning_policy key must still load."""
        old_json = {
            "source_requirements": [],
            "search_queries": [],
            "primary_source_types": [],
            "fallback_source_types": [],
            "confidence": 0.4,
        }
        plan = PlanSpec.model_validate(old_json)
        assert plan.planning_policy is None
        assert plan.candidate_sources == []
        assert plan.research_status == "not_started"
        assert plan.confidence == 0.4

    def test_full_datasetspec_without_new_fields_validates(self):
        """A DatasetSpec serialised before Increment 3 must still round-trip."""
        spec = create_initial_spec("Old spec", "old description")
        spec_json = spec.model_dump(mode="json")
        # Simulate an old JSON that has no new PlanSpec keys
        spec_json.pop("plan", None)
        restored = DatasetSpec.model_validate(spec_json)
        assert restored.plan is None

    def test_planspec_json_round_trip_with_policy(self):
        """A PlanSpec with planning_policy must survive a JSON round-trip."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        serialised = spec.plan.model_dump(mode="json")
        restored = PlanSpec.model_validate(serialised)
        assert restored.planning_policy is not None
        orig_dims = set(spec.plan.planning_policy.evaluation_requirements.required_dimensions)
        restored_dims = set(restored.planning_policy.evaluation_requirements.required_dimensions)
        assert orig_dims == restored_dims

    def test_existing_planspec_fields_unchanged(self):
        """The existing PlanSpec fields must remain accessible after Increment 3."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        plan = spec.plan
        # All pre-existing fields must still be present
        assert hasattr(plan, "source_requirements")
        assert hasattr(plan, "search_queries")
        assert hasattr(plan, "primary_source_types")
        assert hasattr(plan, "fallback_source_types")
        assert hasattr(plan, "reasoning")
        assert hasattr(plan, "confidence")
        assert hasattr(plan, "uncovered_features")


# ---------------------------------------------------------------------------
# 6. research_status and selection_status remain independent
# ---------------------------------------------------------------------------

class TestStatusIndependence:

    def test_fully_evaluated_candidate_can_be_deferred(self):
        """A candidate that has been fully evaluated may still be deferred if
        no connector is available.  evaluation_status and selection_status are
        independent fields."""
        candidate = _minimal_complete_candidate(
            "https://data.example.org/", "Evaluated but no connector", "deferred"
        )
        assert candidate.evaluation_status == "complete"
        assert candidate.selection_status == "deferred"

    def test_not_started_candidate_is_not_selected(self):
        """A candidate that has not been researched yet must default to
        not_selected and not_started independently."""
        candidate = CandidateSource(
            source_url="https://data.example.org/",
            title="Not yet investigated",
            discovery_method="catalog_hint",
            discovered_at=_NOW,
        )
        assert candidate.evaluation_status == "not_started"
        assert candidate.selection_status == "not_selected"

    @pytest.mark.parametrize("eval_status,sel_status", [
        ("not_started",  "not_selected"),
        ("in_progress",  "not_selected"),
        ("in_progress",  "deferred"),
        ("complete",     "selected"),
        ("complete",     "rejected"),
        ("complete",     "deferred"),
        ("partial",      "not_selected"),
        ("partial",      "deferred"),
        ("failed",       "rejected"),
    ])
    def test_all_valid_status_combinations_accepted(self, eval_status, sel_status):
        """Every sensible combination of evaluation and selection status must be valid."""
        if eval_status == "complete":
            candidate = _minimal_complete_candidate(
                "https://data.example.org/", "Candidate", sel_status
            )
        elif eval_status == "partial":
            candidate = _minimal_partial_candidate(
                "https://data.example.org/", "Candidate", sel_status
            )
        else:
            candidate = CandidateSource(
                source_url="https://data.example.org/",
                title="Candidate",
                discovery_method="web_search",
                discovered_at=_NOW,
                evaluation_status=eval_status,
                selection_status=sel_status,
            )
        assert candidate.evaluation_status == eval_status
        assert candidate.selection_status == sel_status

    def test_research_status_on_plan_is_independent_of_candidate_statuses(self):
        """PlanSpec.research_status summarises overall progress and is set
        independently of individual candidate evaluation or selection statuses."""
        c1 = _minimal_complete_candidate("https://a.example.org/", "A", "selected")
        c2 = CandidateSource(
            source_url="https://b.example.org/",
            title="B",
            discovery_method="web_search",
            discovered_at=_NOW,
            evaluation_status="not_started",
            selection_status="not_selected",
        )
        # research_status="partial" because c2 has not been evaluated yet —
        # independent of c1 being selected.
        plan = PlanSpec(candidate_sources=[c1, c2], research_status="partial")
        assert plan.research_status == "partial"
        assert plan.candidate_sources[0].selection_status == "selected"
        assert plan.candidate_sources[1].evaluation_status == "not_started"


# ---------------------------------------------------------------------------
# 7. Existing Planning Agent behaviour remains compatible
# ---------------------------------------------------------------------------

class TestExistingBehaviourCompatible:

    def test_plan_still_produces_source_requirements(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert len(spec.plan.source_requirements) > 0

    def test_plan_still_produces_search_queries(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert len(spec.plan.search_queries) > 0

    def test_plan_still_sets_primary_source_types(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert isinstance(spec.plan.primary_source_types, list)

    def test_unsupported_requirement_still_skips_plan(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec(is_supported=False)
        planner.plan(spec)
        assert spec.plan is None

    def test_plan_called_without_policy_kwarg_still_works(self):
        """Callers who do not pass policy= must still get a valid plan."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)         # no policy kwarg
        assert spec.plan is not None
        assert spec.plan.planning_policy is not None

    def test_validate_spec_passes_after_increment3_plan(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        valid, errors = validate_spec(spec)
        assert valid, errors

    def test_plan_json_round_trip_unchanged_for_existing_fields(self):
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        dumped = spec.model_dump(mode="json")
        restored = DatasetSpec.model_validate(dumped)
        assert restored.plan.confidence == spec.plan.confidence
        assert [s.source_type for s in restored.plan.source_requirements] == [
            s.source_type for s in spec.plan.source_requirements
        ]


# ---------------------------------------------------------------------------
# 8. Missing or invalid policy is handled safely
# ---------------------------------------------------------------------------

class TestPolicyErrorHandling:

    def test_unsupported_requirement_does_not_derive_policy(self):
        """plan() returns early for unsupported requirements without touching policy."""
        planner = _planner_with_empty_llm()
        spec = _make_spec(is_supported=False)
        spec.plan = None
        planner.plan(spec)
        # plan is not set for unsupported requirements
        assert spec.plan is None

    def test_plan_with_no_sources_stores_policy_and_returns(self, monkeypatch):
        """Even when the catalog is empty, the policy must be stored."""
        monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [])
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec)
        assert spec.plan is not None
        assert spec.plan.planning_policy is not None

    def test_explicit_none_policy_falls_back_to_derived(self):
        """Passing policy=None explicitly must trigger internal derivation."""
        planner = _planner_with_empty_llm()
        spec = _make_spec()
        planner.plan(spec, policy=None)
        assert isinstance(spec.plan.planning_policy, ApplicationPlanningPolicy)

    def test_candidate_source_selection_status_defaults_to_not_selected(self):
        """New CandidateSource instances default to not_selected."""
        c = CandidateSource(
            source_url="https://data.example.org/",
            title="Candidate",
            discovery_method="web_search",
            discovered_at=_NOW,
        )
        assert c.selection_status == "not_selected"

    def test_plan_spec_research_status_defaults_to_not_started(self):
        plan = PlanSpec()
        assert plan.research_status == "not_started"


# ---------------------------------------------------------------------------
# Policy consistency validation (Increment 3 correction)
# ---------------------------------------------------------------------------

class TestPolicyConsistencyValidation:
    """The planner must reject any caller-supplied policy whose evaluation
    requirements do not exactly match what is derived from the actual
    RequirementSpec.  Rejection must happen before any LLM interaction."""

    def _fresh_spec(self, **req_overrides) -> DatasetSpec:
        return _make_spec(**req_overrides)

    # ---- Negative cases: inconsistent policies are rejected ----

    def test_policy_missing_required_freshness_dimension_is_rejected(self):
        """A policy that omits a dimension required by the requirement (e.g.,
        freshness when freshness_need='live') must be rejected with a clear
        ValueError before any LLM call."""
        planner = _planner_with_empty_llm()
        spec = self._fresh_spec(freshness_need="live")

        # Derive a correct policy, then manually remove 'freshness'.
        good_policy = derive_evaluation_requirements(spec.requirement, UsageIntent())
        reqs = good_policy.evaluation_requirements
        dims_without_freshness = [d for d in reqs.required_dimensions if d != "freshness"]
        crit_without_freshness = {
            d: c for d, c in reqs.dimension_criticality.items() if d != "freshness"
        }
        bad_policy = ApplicationPlanningPolicy(
            usage_intent=good_policy.usage_intent,
            evaluation_requirements=EvaluationRequirements(
                policy_version=reqs.policy_version,
                required_dimensions=dims_without_freshness,
                dimension_criticality=crit_without_freshness,
                required_variables=reqs.required_variables,
                required_formats=reqs.required_formats,
                required_retrieval_methods=reqs.required_retrieval_methods,
            ),
        )

        with pytest.raises(ValueError, match="missing required dimensions"):
            planner.plan(spec, policy=bad_policy)

        # LLM must not have been called
        planner.llm.call.assert_not_called()

    def test_policy_essential_dimension_changed_to_advisory_is_rejected(self):
        """Changing a dimension's criticality from essential to advisory must
        be caught as a policy mismatch."""
        planner = _planner_with_empty_llm()
        spec = self._fresh_spec()

        good_policy = derive_evaluation_requirements(spec.requirement, UsageIntent())
        reqs = good_policy.evaluation_requirements

        # Find an essential dimension and demote it to advisory.
        essential_dims = [
            d for d, c in reqs.dimension_criticality.items() if c == "essential"
        ]
        assert essential_dims, "Need at least one essential dimension for this test"
        demoted_dim = essential_dims[0]

        bad_criticality = dict(reqs.dimension_criticality)
        bad_criticality[demoted_dim] = "advisory"

        bad_policy = ApplicationPlanningPolicy(
            usage_intent=good_policy.usage_intent,
            evaluation_requirements=EvaluationRequirements(
                policy_version=reqs.policy_version,
                required_dimensions=reqs.required_dimensions,
                dimension_criticality=bad_criticality,
                required_variables=reqs.required_variables,
                required_formats=reqs.required_formats,
                required_retrieval_methods=reqs.required_retrieval_methods,
            ),
        )

        with pytest.raises(ValueError, match="criticality"):
            planner.plan(spec, policy=bad_policy)

        planner.llm.call.assert_not_called()

    def test_policy_derived_from_different_requirement_is_rejected(self):
        """A policy derived from a different requirement (different geography,
        freshness, features) must be rejected even if it is structurally valid."""
        planner = _planner_with_empty_llm()

        # Spec A: air quality, Delhi, live.
        spec_a = self._fresh_spec(
            geography="Delhi",
            freshness_need="live",
            expected_features=["pm2.5", "pm10"],
        )
        # Spec B: weather, global, historical – produces a different policy.
        spec_b = _make_spec(
            geography=None,
            freshness_need="historical",
            expected_features=[],
            subdomain="weather",
            domain="meteorological",
        )
        spec_b.requirement.time_range = "2010 to 2020"

        policy_for_b = derive_evaluation_requirements(spec_b.requirement, UsageIntent())

        # Try to run plan for spec_a with policy derived from spec_b.
        with pytest.raises(ValueError):
            planner.plan(spec_a, policy=policy_for_b)

        planner.llm.call.assert_not_called()

    def test_policy_with_wrong_policy_version_is_rejected(self):
        """A policy whose policy_version differs from the derived one must be
        rejected — policy_version is part of the evaluation contract."""
        planner = _planner_with_empty_llm()
        spec = self._fresh_spec()

        good_policy = derive_evaluation_requirements(spec.requirement, UsageIntent())
        reqs = good_policy.evaluation_requirements

        bad_policy = ApplicationPlanningPolicy(
            usage_intent=good_policy.usage_intent,
            evaluation_requirements=EvaluationRequirements(
                policy_version="99",           # wrong version
                required_dimensions=reqs.required_dimensions,
                dimension_criticality=reqs.dimension_criticality,
                required_variables=reqs.required_variables,
                required_formats=reqs.required_formats,
                required_retrieval_methods=reqs.required_retrieval_methods,
            ),
        )

        with pytest.raises(ValueError, match="policy_version"):
            planner.plan(spec, policy=bad_policy)

        planner.llm.call.assert_not_called()

    # ---- Positive case: consistent policy with custom UsageIntent is accepted ----

    def test_correct_policy_with_custom_usage_intent_is_accepted(self):
        """A policy whose evaluation_requirements exactly match the derived ones
        must be accepted even when it carries a custom UsageIntent.  The stored
        policy must be the caller-supplied object (UsageIntent preserved)."""
        planner = _planner_with_empty_llm()
        spec = self._fresh_spec()

        custom_usage = UsageIntent(required_rights=["use", "modify", "redistribute"])
        correct_policy = derive_evaluation_requirements(spec.requirement, custom_usage)

        planner.plan(spec, policy=correct_policy)

        assert spec.plan is not None
        stored = spec.plan.planning_policy
        # The stored policy is the caller-supplied object
        assert stored is correct_policy
        # UsageIntent is preserved
        assert stored.usage_intent.required_rights == ["use", "modify", "redistribute"]

    def test_invalid_policy_rejected_before_llm_interaction(self):
        """Rejection must occur before any LLM call, ensuring LLM tokens are
        not wasted on a plan that will ultimately be discarded."""
        planner = _planner_with_empty_llm()
        spec = self._fresh_spec(freshness_need="live")

        good_policy = derive_evaluation_requirements(spec.requirement, UsageIntent())
        reqs = good_policy.evaluation_requirements
        # Remove 'freshness' to make it invalid
        dims = [d for d in reqs.required_dimensions if d != "freshness"]
        crit = {d: c for d, c in reqs.dimension_criticality.items() if d != "freshness"}
        bad_policy = ApplicationPlanningPolicy(
            usage_intent=good_policy.usage_intent,
            evaluation_requirements=EvaluationRequirements(
                policy_version=reqs.policy_version,
                required_dimensions=dims,
                dimension_criticality=crit,
                required_variables=reqs.required_variables,
                required_formats=reqs.required_formats,
                required_retrieval_methods=reqs.required_retrieval_methods,
            ),
        )

        with pytest.raises(ValueError):
            planner.plan(spec, policy=bad_policy)

        # Critical assertion: LLM was not touched
        planner.llm.call.assert_not_called()
        # Plan must not have been stored
        assert spec.plan is None
