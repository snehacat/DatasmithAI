from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import planning_agent as planning_module
from agents.planning_agent import PlanningAgent
from agents.source_catalog import SourceCatalogEntry
from core.dataset_spec import (
    DatasetSpec,
    PlanSpec,
    PipelineStatus,
    RequirementSpec,
    SearchQuery,
    SourceRequirement,
    create_initial_spec,
    validate_spec,
)


def make_requirement(**overrides):
    values = {
        "dataset_name": "Weather study",
        "description": "daily temperature data for Delhi",
        "domain": "meteorological",
        "subdomain": "weather",
        "expected_features": ["temperature"],
        "geography": "Delhi, India",
        "freshness_need": "recent",
        "is_supported": True,
    }
    values.update(overrides)
    return RequirementSpec(**values)


def make_spec(**requirement_overrides):
    spec = create_initial_spec("Weather study", "daily temperature data for Delhi")
    spec.requirement = make_requirement(**requirement_overrides)
    return spec


@pytest.fixture
def planner():
    agent = PlanningAgent()
    agent.llm = Mock()
    agent.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": [],
        "reasoning": "Rule-based providers fit the requested topic and region.",
    })
    return agent


def test_unsupported_requirement_is_unchanged_and_skips_llm(planner):
    spec = make_spec(is_supported=False)
    existing_plan = PlanSpec(reasoning="keep")
    spec.plan = existing_plan
    result = planner.plan(spec)
    assert result is spec
    assert result.plan is existing_plan
    planner.llm.call.assert_not_called()


def test_missing_requirement_raises(planner):
    spec = create_initial_spec("Missing", "No requirement section")
    spec.requirement = None
    with pytest.raises(ValueError, match="spec.requirement must be set"):
        planner.plan(spec)
    planner.llm.call.assert_not_called()


@pytest.mark.parametrize(("subdomain", "domain", "expected"), [
    ("weather", "meteorological", "open_meteo_weather"),
    ("air quality", "environmental", "openaq_air_quality"),
    ("earthquakes", "geological", "usgs_earthquake_catalog"),
    ("forest fire monitoring", "environmental", "nasa_firms_fire"),
    ("flood monitoring", "hydrological", "india_cwc_flood_portal"),
    ("ocean data", "oceanographic", "copernicus_marine"),
    ("satellite data", "environmental", "sentinel_satellite"),
    ("general", "environmental", "nasa_earthdata_general"),
    ("climate", "meteorological", "era5_reanalysis"),
    ("pollution", "environmental", "cpcb_namp_air_quality"),
    ("seismic", "geological", "usgs_earthquake_catalog"),
    ("river", "hydrological", "copernicus_glofas_flood"),
    ("water quality", "environmental", "nasa_earthdata_general"),
    ("natural disasters", "geological", "usgs_earthquake_catalog"),
])
def test_catalog_covers_supported_topics(planner, subdomain, domain, expected):
    spec = make_spec(subdomain=subdomain, domain=domain, geography="India")
    planner.plan(spec)
    assert expected in {source.source_type for source in spec.plan.source_requirements}


def test_unknown_subdomain_falls_back_to_domain_sources_with_warning(planner):
    spec = make_spec(subdomain="unknown phenomenon", domain="meteorological")
    planner.plan(spec)
    assert spec.plan.source_requirements
    assert any("Unknown subdomain" in warning for warning in spec.warnings)


def test_unspecified_geography_selects_global_sources_and_warns(planner):
    spec = make_spec(geography=None)
    planner.plan(spec)
    assert spec.plan.source_requirements
    catalog = {entry.source_type: entry for entry in planning_module.SOURCE_CATALOG}
    assert all(
        catalog[source.source_type].geography == "global"
        for source in spec.plan.source_requirements
    )
    assert any("Geography is unspecified" in warning for warning in spec.warnings)


def test_india_specific_provider_is_ranked_before_global_provider(planner):
    spec = make_spec(subdomain="air quality", domain="environmental", geography="Delhi")
    planner.plan(spec)
    assert spec.plan.source_requirements[0].source_type == "cpcb_namp_air_quality"


def test_live_request_with_historical_only_sources_warns_and_penalizes_confidence(
    planner, monkeypatch
):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "historical_weather", "Historical Weather",
            ["weather", "meteorological"], ["temperature"], "download",
            False, False, True, "global", "historical", "Test catalog entry.",
        )
    ])
    spec = make_spec(freshness_need="live", time_range=None, geography="India")
    planner.plan(spec)
    assert spec.plan.fallback_source_types == ["historical_weather"]
    assert spec.plan.confidence < 0.8
    assert any("freshness" in warning for warning in spec.warnings)


def test_rerunning_replaces_plan_without_duplicate_queries(planner):
    spec = make_spec()
    planner.plan(spec)
    first_plan = spec.plan
    planner.plan(spec)
    assert spec.plan is not first_plan
    queries = [query.query.casefold() for query in spec.plan.search_queries]
    assert len(queries) == len(set(queries))


@pytest.mark.parametrize("content", [
    None,
    [],
    {"extra_queries": "not a list", "reasoning": 42},
    {"extra_queries": [4, None], "reasoning": ""},
    {"extra_queries": [], "reasoning": "x" * 121},
])
def test_unusable_llm_content_uses_rules_and_warns(planner, content, monkeypatch):
    # Use a single-source catalog so rule queries stay < 4 (LLM is not auto-skipped)
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
        ),
    ])
    planner.llm.call.return_value = SimpleNamespace(content=content)
    spec = make_spec(geography=None)
    planner.plan(spec)
    assert spec.plan.reasoning
    assert "LLM unavailable, planner used rules only" in spec.warnings


@pytest.mark.parametrize("bad_query", [
    "https://example.test/search",
    "www.example.test query",
    19,
    "x" * 121,
])
def test_invalid_llm_queries_are_ignored(planner, bad_query, monkeypatch):
    # Single-source catalog keeps rule queries < 4 so LLM is called
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
        ),
    ])
    planner.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": [bad_query],
        "reasoning": "",
    })
    spec = make_spec(geography=None)
    planner.plan(spec)
    assert all(
        bad_query not in query.query for query in spec.plan.search_queries
        if isinstance(bad_query, str)
    )
    assert "LLM unavailable, planner used rules only" in spec.warnings


def test_llm_exception_does_not_fail_supported_planning(planner, monkeypatch):
    # Single-source catalog keeps rule queries < 4 so LLM is called
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
        ),
    ])
    planner.llm.call.side_effect = RuntimeError("offline")
    spec = make_spec(geography=None)
    assert planner.plan(spec) is spec
    assert spec.plan.source_requirements
    assert "LLM unavailable, planner used rules only" in spec.warnings
    assert planner.stats["llm_calls"] == 1


def test_valid_llm_extra_queries_are_added(planner, monkeypatch):
    # Single-source catalog keeps rule queries < 4 so LLM is called
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
        ),
    ])
    planner.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": ["Delhi daily temperature archive"],
        "reasoning": "Search global and regional providers.",
    })
    spec = make_spec(geography=None)
    planner.plan(spec)
    assert "Delhi daily temperature archive" in {
        query.query for query in spec.plan.search_queries
    }
    assert spec.plan.reasoning == "Search global and regional providers."


def test_llm_extra_queries_are_deduplicated(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
        ),
    ])
    planner.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": ["Single Weather weather global documentation"],
        "reasoning": "Use public sources.",
    })
    spec = make_spec(geography=None)
    planner.plan(spec)
    queries = [query.query.casefold() for query in spec.plan.search_queries]
    assert len(queries) == len(set(queries))


def test_llm_refinement_keeps_required_source_queries_at_six_max(planner, monkeypatch):
    """LLM may add up to 3 extra queries per required source (total ≤ 6, global cap ≤ 12)."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_weather", "Single Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.", connector_available=True,
        ),
    ])
    planner.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": ["first extra", "second extra", "third extra"],
        "reasoning": "Use the best-fit source.",
    })
    spec = make_spec(geography=None)
    planner.plan(spec)
    required_types = {
        source.source_type for source in spec.plan.source_requirements
        if source.importance == "required"
    }
    for source_type in required_types:
        assert sum(
            query.related_source_type == source_type
            for query in spec.plan.search_queries
        ) <= 6  # 3 rule + 3 LLM


def test_source_and_query_priorities_are_unique_and_increasing(planner):
    spec = make_spec(subdomain="weather", domain="meteorological")
    planner.plan(spec)
    source_priorities = [item.priority for item in spec.plan.source_requirements]
    query_priorities = [item.priority for item in spec.plan.search_queries]
    assert source_priorities == sorted(set(source_priorities))
    assert query_priorities == sorted(set(query_priorities))


def test_every_query_references_an_existing_source_requirement(planner):
    spec = make_spec()
    planner.plan(spec)
    source_types = {item.source_type for item in spec.plan.source_requirements}
    assert all(query.related_source_type in source_types for query in spec.plan.search_queries)


def test_free_sources_constraint_filters_nonfree_entries(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "paid_weather", "Paid Weather", ["weather"], ["temperature"], "api",
            False, False, False, "global", "all", "Test catalog entry.",
        ),
        SourceCatalogEntry(
            "free_weather", "Free Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test catalog entry.",
        ),
    ])
    spec = make_spec(constraints=["free sources only"])
    planner.plan(spec)
    assert {item.source_type for item in spec.plan.source_requirements} == {"free_weather"}


def test_free_sources_constraint_excludes_catalog_commercial_hint(planner):
    spec = make_spec(constraints=["free sources only"])
    planner.plan(spec)
    assert "commercial_weather_api" not in {
        item.source_type for item in spec.plan.source_requirements
    }


def test_constraints_filter_before_source_limit(planner, monkeypatch):
    entries = [
        SourceCatalogEntry(
            f"restricted_{index}", f"Restricted {index}", ["weather"], ["temperature"],
            "api", False, False, False, "global", "all", "Test catalog entry.",
        )
        for index in range(7)
    ]
    entries.append(SourceCatalogEntry(
        "available_free", "Available Free", ["weather"], ["temperature"], "api",
        False, False, True, "global", "all", "Test catalog entry.",
    ))
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", entries)
    spec = make_spec(constraints=["free sources only"])
    planner.plan(spec)
    assert [item.source_type for item in spec.plan.source_requirements] == ["available_free"]


def test_no_login_constraint_filters_login_sources(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "login_weather", "Login Weather", ["weather"], ["temperature"], "download",
            False, True, True, "global", "all", "Test catalog entry.",
        ),
        SourceCatalogEntry(
            "open_weather", "Open Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test catalog entry.",
        ),
    ])
    spec = make_spec(constraints=["no login required"])
    planner.plan(spec)
    assert {item.source_type for item in spec.plan.source_requirements} == {"open_weather"}


def test_public_data_constraint_filters_nonpublic_sources(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "private_weather", "Private Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test catalog entry.", public=False,
        ),
        SourceCatalogEntry(
            "public_weather", "Public Weather", ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test catalog entry.",
        ),
    ])
    spec = make_spec(constraints=["public data only"])
    planner.plan(spec)
    assert {item.source_type for item in spec.plan.source_requirements} == {"public_weather"}


def test_constraints_removing_every_source_keeps_none(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "restricted_weather", "Restricted Weather", ["weather"], ["temperature"], "portal",
            False, True, False, "global", "all", "Test catalog entry.",
        )
    ])
    spec = make_spec(constraints=["free sources only", "no login required"])
    planner.plan(spec)
    assert spec.plan.source_requirements == []
    assert spec.plan.search_queries == []
    assert any("no excluded source was retained" in warning for warning in spec.warnings)


def test_satellite_constraint_adds_required_satellite_source(planner):
    spec = make_spec(
        constraints=["satellite-derived data"],
        freshness_need="historical",
        time_range="2020 to 2021",
    )
    planner.plan(spec)
    required_types = {
        item.source_type for item in spec.plan.source_requirements
        if item.importance == "required"
    }
    assert required_types & {"sentinel_satellite", "landsat_satellite", "modis_satellite"}


def test_historical_satellite_source_stays_fallback_for_live_request(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "historical_satellite", "Historical Satellite",
            ["satellite", "satellite data"], ["satellite imagery"],
            "download", False, False, True, "global", "historical", "Test.",
        ),
    ])
    spec = make_spec(
        constraints=["satellite-derived data"],
        freshness_need="live",
        time_range=None,
        geography=None,
    )
    planner.plan(spec, use_llm=False)

    assert len(spec.plan.source_requirements) == 1
    assert spec.plan.source_requirements[0].importance == "fallback"
    assert spec.plan.fallback_source_types == ["historical_satellite"]
    assert "historical_satellite" not in spec.plan.primary_source_types


def test_injected_general_fallback_is_not_a_primary_source(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "general_fallback", "General Fallback",
            ["general"], ["environmental observations"],
            "portal", False, False, True, "global", "all", "Test.",
        ),
    ])
    spec = make_spec(
        domain="unclassified",
        subdomain="unlisted topic",
        geography=None,
        expected_features=[],
        freshness_need="unspecified",
    )
    planner.plan(spec, use_llm=False)

    assert [source.source_type for source in spec.plan.source_requirements] == [
        "general_fallback"
    ]
    assert spec.plan.source_requirements[0].importance == "fallback"
    assert spec.plan.primary_source_types == []
    assert spec.plan.fallback_source_types == ["general_fallback"]
    assert any("using a general fallback" in warning for warning in spec.warnings)


@pytest.mark.parametrize(
    ("constraints", "free", "requires_login", "public"),
    [
        (["satellite-derived data", "free sources only"], False, False, True),
        (["satellite-derived data", "no login required"], True, True, True),
        (["satellite-derived data", "public data only"], True, False, False),
    ],
)
def test_satellite_requirement_never_bypasses_hard_constraints(
    planner, monkeypatch, constraints, free, requires_login, public
):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "restricted_satellite", "Restricted Satellite",
            ["satellite", "satellite data"], ["satellite imagery"], "download",
            False, requires_login, free, "global", "historical", "Test.",
            public=public,
        ),
    ])
    spec = make_spec(constraints=constraints)
    planner.plan(spec)

    assert spec.plan.source_requirements == []
    assert spec.plan.primary_source_types == []
    assert spec.plan.search_queries == []
    assert any("no excluded source was retained" in warning for warning in spec.warnings)


def test_source_requirement_contains_explanation_access_and_feature_matches(planner):
    spec = make_spec()
    planner.plan(spec)
    weather = next(
        source for source in spec.plan.source_requirements
        if source.source_type == "open_meteo_weather"
    )
    assert weather.why_needed.endswith(".")
    assert weather.information_expected == ["temperature"]
    assert weather.preferred_access_methods == ["api"]
    assert weather.connector_available is False
    assert "manual discovery/access is required" in weather.why_needed


def test_feature_matching_uses_controlled_aliases_and_preserves_requested_names():
    entry = SourceCatalogEntry(
        "river_source", "River Source", ["hydrology"],
        ["timestamp", "river discharge", "river level"],
        "download", False, False, True, "global", "all", "Test.",
    )
    requested = ["time", "streamflow", "water level"]
    assert PlanningAgent._feature_matches(entry, requested) == requested


def test_feature_matching_does_not_match_unrelated_substrings():
    entry = SourceCatalogEntry(
        "river_source", "River Source", ["hydrology"],
        ["timestamp", "river discharge", "river level"],
        "download", False, False, True, "global", "all", "Test.",
    )
    assert PlanningAgent._feature_matches(
        entry, ["time zone", "river", "level", "flow", "water level sensor"]
    ) == []


def test_flood_prediction_plan_matches_river_feature_aliases(planner):
    spec = make_spec(
        description="flood prediction data",
        domain="hydrological",
        subdomain="flooding",
        expected_features=["timestamp", "streamflow", "water level", "rainfall"],
        geography="Bihar",
        freshness_need="unspecified",
    )
    planner.plan(spec, use_llm=False)

    cwc = next(
        source for source in spec.plan.source_requirements
        if source.source_type == "india_cwc_flood_portal"
    )
    assert cwc.importance == "required"
    assert cwc.information_expected == ["timestamp", "streamflow", "water level"]
    assert cwc.connector_available is False
    assert "manual discovery/access is required" in cwc.why_needed


def test_flood_plan_reports_four_uncovered_of_five_requested_features(planner, monkeypatch):
    features = ["timestamp", "water level", "rainfall", "temperature", "humidity"]
    spec = make_spec(
        description="flood prediction data",
        domain="hydrological",
        subdomain="flood monitoring",
        expected_features=features,
        geography=None,
        freshness_need="unspecified",
    )
    planner.plan(spec, use_llm=False)

    assert [source.source_type for source in spec.plan.source_requirements] == [
        "copernicus_glofas_flood"
    ]
    assert spec.plan.source_requirements[0].information_expected == ["timestamp"]
    assert spec.plan.uncovered_features == features[1:]
    assert any(
        "water level, rainfall, temperature, humidity" in warning
        for warning in spec.warnings
    )
    assert spec.plan.confidence <= 0.3

    monkeypatch_catalog = [
        SourceCatalogEntry(
            "complete_flood", "Complete Flood Source", ["flood"],
            features, "api", False, False, True, "global", "all", "Test.",
            connector_available=True,
        ),
    ]
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", monkeypatch_catalog)
    complete = make_spec(
        description="flood prediction data",
        domain="hydrological",
        subdomain="flood monitoring",
        expected_features=features,
        geography=None,
        freshness_need="unspecified",
    )
    planner.plan(complete, use_llm=False)
    assert complete.plan.uncovered_features == []
    assert complete.plan.confidence > spec.plan.confidence


@pytest.mark.parametrize(
    ("provided", "requested"),
    [
        ("river discharge", "water level"),
        ("river level", "streamflow"),
    ],
)
def test_river_discharge_and_water_level_are_not_equated(provided, requested):
    entry = SourceCatalogEntry(
        "river_source", "River Source", ["hydrology"], [provided],
        "download", False, False, True, "global", "all", "Test.",
    )
    assert PlanningAgent._feature_matches(entry, [requested]) == []


def test_unimplemented_required_source_gets_less_confidence_credit(
    planner, monkeypatch
):
    scores = {}
    for connector_available in (False, True):
        monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
            SourceCatalogEntry(
                "weather_source", "Weather Source", ["weather"], ["temperature"],
                "api", False, False, True, "global", "all", "Test.",
                connector_available=connector_available,
            ),
        ])
        spec = make_spec(expected_features=["temperature"])
        planner.plan(spec, use_llm=False)
        assert spec.plan.source_requirements[0].importance == "required"
        assert spec.plan.source_requirements[0].connector_available is connector_available
        scores[connector_available] = spec.plan.confidence

    assert scores[True] > scores[False]


def test_confidence_reflects_high_partial_and_poor_plan_fit(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "full_weather", "Full Weather", ["weather"],
            ["temperature", "humidity"], "api",
            False, False, True, "global", "all", "Test.",
            connector_available=True,
        ),
    ])
    high = make_spec(expected_features=["temperature", "humidity"])
    planner.plan(high, use_llm=False)
    assert high.plan.confidence == 1.0

    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "partial_weather", "Partial Weather", ["weather"],
            ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
            connector_available=True,
        ),
    ])
    partial = make_spec(expected_features=["temperature", "rainfall"])
    planner.plan(partial, use_llm=False)
    assert 0.75 < partial.plan.confidence < 0.95

    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "poor_weather", "Poor Weather", ["weather"],
            ["wind speed"], "download",
            False, False, True, "global", "historical", "Test.",
        ),
    ])
    poor = make_spec(
        expected_features=["temperature", "rainfall", "humidity"],
        freshness_need="live",
    )
    planner.plan(poor, use_llm=False)
    assert poor.plan.primary_source_types == []
    assert 0.0 <= poor.plan.confidence < 0.5


def test_primary_and_fallback_type_lists_match_importance(planner):
    spec = make_spec()
    planner.plan(spec)
    assert spec.plan.primary_source_types == [
        item.source_type for item in spec.plan.source_requirements
        if item.importance == "required"
    ]
    assert spec.plan.fallback_source_types == [
        item.source_type for item in spec.plan.source_requirements
        if item.importance == "fallback"
    ]


def test_validate_spec_passes_after_planning(planner):
    spec = make_spec()
    planner.plan(spec)
    assert validate_spec(spec) == (True, [])


def test_planned_spec_json_round_trip(planner):
    spec = make_spec()
    planner.plan(spec)
    round_tripped = DatasetSpec(**spec.model_dump(mode="json"))
    assert round_tripped.plan == spec.plan


def test_validate_spec_rejects_query_for_unknown_source_type():
    spec = make_spec()
    spec.plan = PlanSpec(
        source_requirements=[SourceRequirement(
            source_type="known", why_needed="Needed.", importance="required", priority=1,
        )],
        search_queries=[SearchQuery(
            query="weather data", purpose="Find data.", related_source_type="missing", priority=1,
        )],
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert any("does not match a source requirement" in error for error in errors)


def test_validate_spec_rejects_duplicate_source_priorities():
    spec = make_spec()
    spec.plan = PlanSpec(source_requirements=[
        SourceRequirement(source_type="a", why_needed="One.", importance="required", priority=1),
        SourceRequirement(source_type="b", why_needed="Two.", importance="useful", priority=1),
    ])
    valid, errors = validate_spec(spec)
    assert not valid
    assert "Plan source requirement priorities must be unique" in errors


def test_validate_spec_rejects_duplicate_query_priorities():
    spec = make_spec()
    spec.plan = PlanSpec(
        source_requirements=[SourceRequirement(
            source_type="a", why_needed="One.", importance="required", priority=1,
        )],
        search_queries=[
            SearchQuery(query="first", purpose="One.", related_source_type="a", priority=1),
            SearchQuery(query="second", purpose="Two.", related_source_type="a", priority=1),
        ],
    )
    valid, errors = validate_spec(spec)
    assert not valid
    assert "Plan search query priorities must be unique" in errors


def test_plan_has_only_one_summary_print(planner, capsys):
    planner.plan(make_spec())
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert output[0].startswith("✅ Planning complete:")



# ---------------------------------------------------------------------------
# Item 1 – Failing test: air quality / Delhi / live / free / pollutants
# ---------------------------------------------------------------------------

_AIR_QUALITY_POLLUTANTS = ["timestamp", "pm2_5", "pm10", "o3", "no2", "so2"]
_VALID_SOURCE_TYPES = {entry.source_type for entry in planning_module.SOURCE_CATALOG}


def _make_aq_delhi_spec():
    """Reproduce the exact requirement from the bug report."""
    spec = create_initial_spec("Air quality study", "live air quality data for Delhi last 7 days")
    spec.requirement = RequirementSpec(
        dataset_name="Air quality study",
        description="live air quality data for Delhi last 7 days",
        domain="environmental",
        subdomain="air quality",
        expected_features=_AIR_QUALITY_POLLUTANTS,
        geography="Delhi",
        freshness_need="live",
        time_window_days=7,
        constraints=["free sources only"],
        is_supported=True,
    )
    return spec


def test_aq_delhi_has_at_least_one_required_source_with_non_empty_primary(planner):
    """At least 1 source must have importance 'required' and primary_source_types non-empty."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)

    required = [s for s in spec.plan.source_requirements if s.importance == "required"]
    assert required, "Expected at least one source with importance='required'"
    assert spec.plan.primary_source_types, "primary_source_types must not be empty"


def test_aq_delhi_search_query_count_in_range(planner):
    """Must produce 2–12 search queries; zero is a bug."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)

    n = len(spec.plan.search_queries)
    assert 2 <= n <= 12, f"Expected 2–12 queries, got {n}"


def test_aq_delhi_every_query_related_source_type_is_valid(planner):
    """Every query's related_source_type must reference a known source in the plan."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)

    plan_source_types = {s.source_type for s in spec.plan.source_requirements}
    for q in spec.plan.search_queries:
        assert q.related_source_type in plan_source_types, (
            f"Query '{q.query}' references unknown source type '{q.related_source_type}'"
        )
        assert q.related_source_type in _VALID_SOURCE_TYPES, (
            f"Query references source type not in catalog: '{q.related_source_type}'"
        )


def test_aq_delhi_information_expected_covers_provided_pollutants(planner):
    """For each required source, information_expected must include every requested pollutant
    that the catalog entry advertises (matched via normalize_feature_name)."""
    from agents.source_catalog import normalize_feature_name

    spec = _make_aq_delhi_spec()
    planner.plan(spec)

    catalog = {entry.source_type: entry for entry in planning_module.SOURCE_CATALOG}
    requested_norm = {normalize_feature_name(f) for f in _AIR_QUALITY_POLLUTANTS}

    for src in spec.plan.source_requirements:
        if src.importance != "required":
            continue
        entry = catalog[src.source_type]
        provided_norm = {normalize_feature_name(p) for p in entry.provides}
        # The catalog says it provides these pollutants; every one that overlaps with the
        # request must appear in information_expected.
        should_be_there = requested_norm & provided_norm
        expected_norm = {normalize_feature_name(f) for f in src.information_expected}
        missing = should_be_there - expected_norm
        assert not missing, (
            f"Source '{src.source_type}' provides {should_be_there} but "
            f"information_expected only has {expected_norm}; missing: {missing}"
        )


def test_aq_delhi_no_freshness_warning_when_live_source_selected(planner):
    """No freshness warning must be emitted when at least one selected source supports 'live'."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)

    catalog = {entry.source_type: entry for entry in planning_module.SOURCE_CATALOG}
    selected_temporals = {
        catalog[s.source_type].temporal
        for s in spec.plan.source_requirements
        if s.importance in {"required", "useful"}
    }
    has_live_source = bool(selected_temporals & {"live", "all"})

    if has_live_source:
        freshness_warnings = [w for w in spec.warnings if "freshness" in w.lower()]
        assert not freshness_warnings, (
            f"Unexpected freshness warning(s) when a live source is selected: {freshness_warnings}"
        )


# ---------------------------------------------------------------------------
# Item 7 – Catalog tests: free-with-key vs paid, Open-Meteo Air Quality
# ---------------------------------------------------------------------------

def test_free_sources_constraint_keeps_free_key_sources(planner, monkeypatch):
    """'free sources only' must keep sources where free=True even if requires_key=True."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "free_key_source", "Free With Key", ["air quality"], ["pm2.5"], "api",
            True, False, True, "global", "live", "Free tier with API key.",
        ),
        SourceCatalogEntry(
            "paid_source", "Paid Source", ["air quality"], ["pm2.5"], "api",
            True, False, False, "global", "live", "Paid source.",
        ),
    ])
    spec = create_initial_spec("AQ study", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ study",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5"],
        geography=None,
        freshness_need="live",
        constraints=["free sources only"],
        is_supported=True,
    )
    planner.plan(spec)
    source_types = {s.source_type for s in spec.plan.source_requirements}
    assert "free_key_source" in source_types, "Free-with-key source must be kept"
    assert "paid_source" not in source_types, "Paid source must be removed"


def test_open_meteo_air_quality_in_catalog():
    """open_meteo_air_quality must exist in the catalog, be free, have no key/login,
    have temporal='live', and cover Delhi (global geography)."""
    from agents.source_catalog import SOURCE_CATALOG as CAT, normalize_feature_name
    entry = next((e for e in CAT if e.source_type == "open_meteo_air_quality"), None)
    assert entry is not None, "open_meteo_air_quality not found in SOURCE_CATALOG"
    assert entry.free, "open_meteo_air_quality must be free"
    assert not entry.requires_key, "open_meteo_air_quality must not require an API key"
    assert not entry.requires_login, "open_meteo_air_quality must not require login"
    assert entry.temporal == "live", f"Expected temporal='live', got '{entry.temporal}'"
    assert entry.geography == "global", f"Expected geography='global', got '{entry.geography}'"
    # Must provide standard air-quality pollutants
    provided_norm = {normalize_feature_name(p) for p in entry.provides}
    for pollutant in ("pm25", "pm10", "o3", "no2", "so2"):
        assert pollutant in provided_norm, (
            f"open_meteo_air_quality should provide '{pollutant}'; got {provided_norm}"
        )


def test_open_meteo_air_quality_selected_for_free_delhi_live_request(planner):
    """open_meteo_air_quality must appear when planning for Delhi / live / free."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)
    source_types = {s.source_type for s in spec.plan.source_requirements}
    assert "open_meteo_air_quality" in source_types, (
        "open_meteo_air_quality should be selected for a free, live Delhi air-quality request"
    )


# ---------------------------------------------------------------------------
# Item 6 – use_llm flag
# ---------------------------------------------------------------------------

def test_use_llm_false_skips_llm_call(planner):
    """When use_llm=False the LLM must never be called."""
    spec = make_spec()
    planner.plan(spec, use_llm=False)
    planner.llm.call.assert_not_called()


def test_use_llm_skipped_when_rules_give_four_or_more_queries(planner):
    """LLM is skipped automatically when use_llm=False regardless of query count."""
    # Explicitly disable LLM – the stat must be incremented.
    spec = _make_aq_delhi_spec()
    planner.plan(spec, use_llm=False)
    planner.llm.call.assert_not_called()
    assert planner.stats.get("llm_skipped", 0) >= 1


def test_llm_skipped_stat_recorded(planner):
    """planner.stats['llm_skipped'] must be incremented when LLM is skipped."""
    spec = make_spec()
    planner.plan(spec, use_llm=False)
    assert planner.stats.get("llm_skipped", 0) >= 1


# ---------------------------------------------------------------------------
# Leftover fixes – failing tests written first
# ---------------------------------------------------------------------------

# ---- Item 1 (RA): freshness live/recent → granularity "Hourly" -------------

def _make_live_aq_spec_for_ra():
    """Minimal spec that RequirementAnalyzer would produce for a live request
    with no stated granularity – we bypass the LLM by injecting the result."""
    from agents.requirement_analyzer import RequirementAnalyzer
    from unittest.mock import patch, MagicMock
    from types import SimpleNamespace

    analyzer = RequirementAnalyzer(interactive=False)
    # Inject a fake LLM that returns a valid but minimal topic response.
    fake_response = SimpleNamespace(content={
        "domain": "environmental",
        "subdomain": "air quality",
        "problem_type": "monitoring",
        "data_modality": "time_series",
        "expected_features": ["pm2.5", "pm10"],
        "target_variable": None,
        "time_granularity": None,   # model returns nothing → rule must default
    })
    analyzer.llm = MagicMock()
    analyzer.llm.call.return_value = fake_response

    spec = create_initial_spec("AQ test", "live air quality data for Delhi")
    return analyzer.analyze(spec)


def test_live_freshness_defaults_granularity_to_hourly():
    """When freshness='live' and no granularity stated, time_granularity must be 'Hourly'."""
    spec = _make_live_aq_spec_for_ra()
    assert spec.requirement.time_granularity == "Hourly", (
        f"Expected 'Hourly', got {spec.requirement.time_granularity!r}"
    )


def test_live_freshness_granularity_label_in_inferred():
    """The inferred_by_model entry must contain '(default for live data)'."""
    spec = _make_live_aq_spec_for_ra()
    matches = [n for n in spec.requirement.inferred_by_model if "default for live" in n]
    assert matches, (
        f"Expected an inferred_by_model entry with '(default for live data)'. "
        f"Got: {spec.requirement.inferred_by_model}"
    )


def test_recent_freshness_defaults_granularity_to_hourly():
    """When freshness='recent' and no granularity stated, time_granularity must be 'Hourly'."""
    from agents.requirement_analyzer import RequirementAnalyzer
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    analyzer = RequirementAnalyzer(interactive=False)
    fake_response = SimpleNamespace(content={
        "domain": "environmental",
        "subdomain": "air quality",
        "problem_type": "monitoring",
        "data_modality": "time_series",
        "expected_features": ["pm2.5"],
        "target_variable": None,
        "time_granularity": None,
    })
    analyzer.llm = MagicMock()
    analyzer.llm.call.return_value = fake_response

    spec = create_initial_spec("AQ test", "recent air quality data for Delhi")
    spec = analyzer.analyze(spec)
    assert spec.requirement.time_granularity == "Hourly", (
        f"Expected 'Hourly' for recent freshness, got {spec.requirement.time_granularity!r}"
    )


# ---- Item 2: deduplicate missing_or_unclear / "Using defaults for" ----------

def test_missing_or_unclear_has_no_duplicate_size_entries():
    """missing_or_unclear must not contain both 'dataset size' and
    'dataset size depends on number of stations' as separate entries."""
    from agents.requirement_analyzer import RequirementAnalyzer
    from unittest.mock import MagicMock
    from types import SimpleNamespace

    analyzer = RequirementAnalyzer(interactive=False)
    fake_response = SimpleNamespace(content={
        "domain": "environmental",
        "subdomain": "air quality",
        "problem_type": "monitoring",
        "data_modality": "time_series",
        "expected_features": ["pm2.5"],
        "target_variable": None,
        "time_granularity": None,
    })
    analyzer.llm = MagicMock()
    analyzer.llm.call.return_value = fake_response

    spec = create_initial_spec("AQ test", "live air quality data for Delhi last 7 days")
    spec = analyzer.analyze(spec)

    size_entries = [
        e for e in spec.requirement.missing_or_unclear
        if "dataset size" in e or "number of stations" in e
    ]
    assert len(size_entries) == 1, (
        f"Expected exactly one size entry in missing_or_unclear, got {size_entries}"
    )
    # The single entry must carry the informative suffix
    assert "station" in size_entries[0], (
        f"The size entry should mention stations, got {size_entries[0]!r}"
    )


# ---- Item 3: query templates -----------------------------------------------

def test_queries_use_distinct_templates_no_duplicate_api_word(planner):
    """No query may contain the word 'API' more than once."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)
    for q in spec.plan.search_queries:
        count = q.query.lower().count("api")
        assert count <= 1, f"'API' appears {count}x in query: {q.query!r}"


def test_queries_do_not_include_free_data_without_free_constraint(planner):
    """'free data' must not appear in queries unless the user asked for free sources."""
    spec = make_spec()   # weather, no free constraint
    planner.plan(spec)
    for q in spec.plan.search_queries:
        assert "free data" not in q.query.lower(), (
            f"'free data' appeared in query without free constraint: {q.query!r}"
        )


def test_queries_provider_name_has_no_brackets_or_slashes(planner):
    """Provider names embedded in queries must have brackets [] and slashes / stripped."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)
    for q in spec.plan.search_queries:
        assert "[" not in q.query and "]" not in q.query, (
            f"Brackets in query: {q.query!r}"
        )
        assert "/" not in q.query, f"Slash in query: {q.query!r}"


def test_queries_cover_at_least_two_distinct_templates_per_required_source(planner):
    """Each required source must have queries from at least 2 different templates
    (i.e., not all queries identical except for the provider name)."""
    spec = _make_aq_delhi_spec()
    planner.plan(spec)
    required_types = {
        s.source_type for s in spec.plan.source_requirements if s.importance == "required"
    }
    for src_type in required_types:
        src_queries = [q.query for q in spec.plan.search_queries
                       if q.related_source_type == src_type]
        assert len(src_queries) >= 2, (
            f"Required source {src_type!r} has fewer than 2 queries"
        )
        # Check they differ beyond just being identical strings
        assert len(set(src_queries)) >= 2, (
            f"Required source {src_type!r} queries are all identical: {src_queries}"
        )


# ---- Item 4: LLM skip when rule queries >= 4 --------------------------------

def test_llm_not_called_when_rule_queries_reach_four(planner, monkeypatch):
    """When rule-built queries already reach >= 4, llm_calls must stay 0
    and reasoning must contain 'rule-based'."""
    # Use a catalog with 2 required sources so rule-building creates 4+ queries
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "src_a", "Source A", ["air quality"], ["pm2.5", "pm10", "o3", "no2", "so2"],
            "api", False, False, True, "global", "live", "Test.",
        ),
        SourceCatalogEntry(
            "src_b", "Source B", ["air quality"], ["pm2.5", "pm10"],
            "api", False, False, True, "global", "live", "Test.",
        ),
    ])
    spec = _make_aq_delhi_spec()
    planner.plan(spec)
    assert planner.stats["llm_calls"] == 0, (
        f"LLM was called despite >= 4 rule queries; llm_calls={planner.stats['llm_calls']}"
    )
    assert "rule" in (spec.plan.reasoning or "").lower(), (
        f"reasoning should mention 'rule', got: {spec.plan.reasoning!r}"
    )


def test_llm_called_when_rule_queries_fewer_than_four(planner, monkeypatch):
    """When rule queries < 4, the LLM must be called (llm_calls == 1)."""
    # Only 1 source → at most 3 rule queries → LLM fires
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_src", "Single Source", ["weather"], ["temperature"],
            "api", False, False, True, "global", "all", "Test.",
        ),
    ])
    spec = make_spec(geography=None)
    planner.plan(spec)
    assert planner.stats["llm_calls"] == 1, (
        f"Expected exactly 1 LLM call, got {planner.stats['llm_calls']}"
    )


# ---- Connector availability is separate from source importance --------------

def test_catalog_entries_have_connector_available_field():
    """Every catalog entry must have connector_available (bool) and data_origin (str)."""
    from agents.source_catalog import SOURCE_CATALOG as CAT
    for entry in CAT:
        assert hasattr(entry, "connector_available"), (
            f"{entry.source_type} missing connector_available"
        )
        assert isinstance(entry.connector_available, bool), (
            f"{entry.source_type}.connector_available must be bool"
        )
        assert hasattr(entry, "data_origin"), (
            f"{entry.source_type} missing data_origin"
        )
        assert entry.data_origin in {"measured", "modeled", "mixed"}, (
            f"{entry.source_type}.data_origin={entry.data_origin!r} not in allowed set"
        )


def test_relevant_source_without_connector_can_still_be_required(planner, monkeypatch):
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "relevant_no_connector", "Relevant Source",
            ["weather"], ["temperature"], "api",
            False, False, True, "global", "all", "Test.",
            connector_available=False,
        ),
    ])
    spec = make_spec()
    planner.plan(spec)
    source = spec.plan.source_requirements[0]
    assert source.importance == "required"
    assert source.connector_available is False
    assert spec.plan.primary_source_types == ["relevant_no_connector"]
    assert "manual discovery/access is required" in source.why_needed
    assert any("without an implemented connector" in warning for warning in spec.warnings)


def test_openaq_and_open_meteo_aq_are_connector_available():
    """openaq_air_quality and open_meteo_air_quality must have connector_available=True."""
    from agents.source_catalog import SOURCE_CATALOG as CAT
    catalog = {e.source_type: e for e in CAT}
    for key in ("openaq_air_quality", "open_meteo_air_quality"):
        assert catalog[key].connector_available, f"{key} should have connector_available=True"


def test_source_requirement_has_data_origin_field():
    """SourceRequirement must carry a data_origin field copied from the catalog."""
    spec = _make_aq_delhi_spec()
    planner = PlanningAgent()
    planner.llm = __import__("unittest.mock", fromlist=["Mock"]).Mock()
    planner.llm.call.return_value = __import__("types").SimpleNamespace(content={
        "extra_queries": [], "reasoning": "rule-based.",
    })
    planner.plan(spec)
    catalog = {entry.source_type: entry for entry in planning_module.SOURCE_CATALOG}
    for src in spec.plan.source_requirements:
        assert hasattr(src, "data_origin"), (
            f"SourceRequirement for {src.source_type} missing data_origin"
        )
        assert src.data_origin in {"measured", "modeled", "mixed"}, (
            f"{src.source_type}.data_origin={src.data_origin!r}"
        )
        assert src.connector_available is catalog[src.source_type].connector_available


# ---- Item 6: confidence field description -----------------------------------

def test_plan_confidence_field_description_says_match():
    """PlanSpec.confidence Field description must say 'match confidence', not 'data availability'."""
    from core.dataset_spec import PlanSpec
    import inspect
    # Pydantic v2: model_fields
    field_info = PlanSpec.model_fields["confidence"]
    desc = (field_info.description or "").lower()
    assert "match" in desc, (
        f"confidence description should say 'match confidence'; got: {field_info.description!r}"
    )
    assert "data availability" not in desc, (
        f"confidence description must not say 'data availability'; got: {field_info.description!r}"
    )


# ---- Item 7: welcome banner -------------------------------------------------

def test_welcome_banner_says_dataset_builder(capsys):
    """main.py banner must read 'DataSmith AI - Dataset Builder'."""
    import importlib
    import main as main_module
    # Patch all I/O so main() doesn't block
    from unittest.mock import patch, MagicMock
    with patch("main.warm_up_ollama"), \
         patch("main.get_run_manager", return_value=MagicMock()), \
         patch("builtins.input", side_effect=["Test Dataset", ""]):
        try:
            main_module.main()
        except (SystemExit, StopIteration, Exception):
            pass
    out = capsys.readouterr().out
    assert "DataSmith AI - Dataset Builder" in out, (
        f"Banner not found. Output started with: {out[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Regression: Uttarakhand / air quality / unspecified freshness
# ---------------------------------------------------------------------------

import itertools as _itertools

_FRESHNESS_VALUES = ["live", "recent", "historical", "unspecified", None]
_GEOGRAPHY_VALUES = ["Delhi", "Uttarakhand", None]
_AQ_CONNECTOR_TYPES = {"openaq_air_quality", "open_meteo_air_quality"}

# Build all 15 combinations as explicit parametrize IDs
_AQ_MATRIX = list(_itertools.product(_FRESHNESS_VALUES, _GEOGRAPHY_VALUES))


def _make_aq_matrix_spec(freshness, geography):
    """Build a RequirementSpec for the air-quality matrix test."""
    spec = create_initial_spec("AQ matrix", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ matrix",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5", "pm10", "no2"],
        geography=geography,
        freshness_need=freshness if freshness is not None else "unspecified",
        time_range=None,
        constraints=[],
        is_supported=True,
    )
    return spec


def _aq_matrix_planner():
    """Return a PlanningAgent with the LLM fully mocked (always returns empty suggestions)."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    agent = PlanningAgent()
    agent.llm = Mock()
    agent.llm.call.return_value = SimpleNamespace(content={
        "extra_queries": [], "reasoning": "rule-based mock.",
    })
    return agent


@pytest.mark.parametrize("freshness,geography", _AQ_MATRIX)
def test_aq_matrix_no_exception(freshness, geography):
    """plan() must never raise for any freshness × geography combination."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(freshness, geography)
    planner.plan(spec)  # must not raise


@pytest.mark.parametrize("freshness,geography", _AQ_MATRIX)
def test_aq_matrix_at_least_one_required_when_connector_matches(freshness, geography):
    """When a connector_available source matches air quality AND satisfies the requested
    freshness, at least one source must be 'required'."""
    from agents.source_catalog import SOURCE_CATALOG as CAT
    catalog = {e.source_type: e for e in CAT}

    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(freshness, geography)
    planner.plan(spec)

    # Determine whether a connector-available AQ source is geo+temporal-reachable
    country = PlanningAgent._country(geography)
    effective_freshness = freshness if freshness is not None else "unspecified"

    def geo_ok(e):
        return e.geography == "global" or (country and e.geography == country)

    def temporal_ok_for(e):
        return PlanningAgent._temporal_match(e, effective_freshness, None)

    reachable = any(
        geo_ok(catalog[st]) and temporal_ok_for(catalog[st])
        for st in _AQ_CONNECTOR_TYPES
        if st in catalog
    )
    if reachable:
        required = [s for s in spec.plan.source_requirements if s.importance == "required"]
        assert required, (
            f"freshness={freshness!r} geography={geography!r}: "
            f"expected ≥1 required source, got importances="
            f"{[s.importance for s in spec.plan.source_requirements]}"
        )


@pytest.mark.parametrize("freshness,geography", _AQ_MATRIX)
def test_aq_matrix_at_least_four_rule_queries(freshness, geography):
    """Every combination must produce ≥ 4 rule-built queries total (LLM is mocked empty)."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(freshness, geography)
    planner.plan(spec)
    n = len(spec.plan.search_queries)
    assert n >= 4, (
        f"freshness={freshness!r} geography={geography!r}: "
        f"expected ≥4 queries, got {n}"
    )


@pytest.mark.parametrize("freshness,geography", _AQ_MATRIX)
def test_aq_matrix_query_priorities_unique(freshness, geography):
    """Query priorities must be unique across the whole plan."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(freshness, geography)
    planner.plan(spec)
    prios = [q.priority for q in spec.plan.search_queries]
    assert len(prios) == len(set(prios)), (
        f"freshness={freshness!r} geography={geography!r}: duplicate priorities {prios}"
    )


@pytest.mark.parametrize("freshness,geography", _AQ_MATRIX)
def test_aq_matrix_every_query_source_type_valid(freshness, geography):
    """Every query's related_source_type must exist in the plan's source_requirements."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(freshness, geography)
    planner.plan(spec)
    plan_types = {s.source_type for s in spec.plan.source_requirements}
    for q in spec.plan.search_queries:
        assert q.related_source_type in plan_types, (
            f"freshness={freshness!r} geography={geography!r}: "
            f"query '{q.query}' references unknown source '{q.related_source_type}'"
        )


@pytest.mark.parametrize("geography", _GEOGRAPHY_VALUES)
def test_aq_unspecified_freshness_no_freshness_warning(geography):
    """unspecified freshness must never produce a freshness warning."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec("unspecified", geography)
    planner.plan(spec)
    freshness_warnings = [w for w in spec.warnings if "freshness" in w.lower()]
    assert not freshness_warnings, (
        f"geography={geography!r}: unexpected freshness warnings: {freshness_warnings}"
    )


@pytest.mark.parametrize("geography", _GEOGRAPHY_VALUES)
def test_aq_none_freshness_no_freshness_warning(geography):
    """None freshness (stored as 'unspecified') must never produce a freshness warning."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec(None, geography)
    planner.plan(spec)
    freshness_warnings = [w for w in spec.warnings if "freshness" in w.lower()]
    assert not freshness_warnings, (
        f"geography={geography!r}: unexpected freshness warnings: {freshness_warnings}"
    )


@pytest.mark.parametrize("geography", _GEOGRAPHY_VALUES)
def test_aq_unspecified_freshness_no_fallback_demotion_for_connector_sources(geography):
    """unspecified freshness: connector_available sources must NOT be demoted to 'fallback'."""
    from agents.source_catalog import SOURCE_CATALOG as CAT
    catalog = {e.source_type: e for e in CAT}

    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec("unspecified", geography)
    planner.plan(spec)

    for src in spec.plan.source_requirements:
        if src.source_type in _AQ_CONNECTOR_TYPES and catalog[src.source_type].source_type in {
            s.source_type for s in spec.plan.source_requirements
        }:
            assert src.importance != "fallback", (
                f"geography={geography!r}: {src.source_type} was demoted to 'fallback' "
                f"with unspecified freshness"
            )


# Item 3: rule queries for every source including fallback
def test_rule_queries_built_for_fallback_sources(planner, monkeypatch):
    """Fallback sources must also get rule-built queries (2 per fallback source)."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "historical_src", "Historical Source", ["air quality"], ["pm2.5", "pm10"],
            "api", False, False, True, "global", "historical", "Test.",
        ),
    ])
    spec = create_initial_spec("AQ test", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ test",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5"],
        geography=None,
        freshness_need="live",   # source is historical → fallback
        constraints=[],
        is_supported=True,
    )
    planner.plan(spec)
    fallback_types = {s.source_type for s in spec.plan.source_requirements
                      if s.importance == "fallback"}
    for ft in fallback_types:
        count = sum(q.related_source_type == ft for q in spec.plan.search_queries)
        assert count >= 2, (
            f"Fallback source {ft!r} has only {count} rule queries; expected ≥ 2"
        )


def test_plan_never_contains_only_llm_queries(planner, monkeypatch):
    """Even when the LLM is active, rule queries must exist before LLM adds extras."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_aq", "Single AQ", ["air quality"], ["pm2.5"],
            "api", False, False, True, "global", "live", "Test.",
        ),
    ])
    planner.llm.call.return_value = __import__("types").SimpleNamespace(content={
        "extra_queries": ["extra llm query one", "extra llm query two"],
        "reasoning": "LLM reasoning.",
    })
    spec = create_initial_spec("AQ test", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ test",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5"],
        geography=None,
        freshness_need="live",
        constraints=[],
        is_supported=True,
    )
    planner.plan(spec)
    # Must have at least one rule-built query (purpose does not say "Additional search")
    rule_queries = [q for q in spec.plan.search_queries
                    if "Additional search" not in q.purpose]
    assert rule_queries, "Plan must contain at least one rule-built query"


# Item 4: LLM query relevance filtering
def test_llm_query_irrelevant_to_subdomain_is_discarded(planner, monkeypatch):
    """LLM queries that contain no word from the subdomain synonyms must be dropped."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_aq", "Single AQ", ["air quality"], ["pm2.5"],
            "api", False, False, True, "global", "live", "Test.",
        ),
    ])
    planner.llm.call.return_value = __import__("types").SimpleNamespace(content={
        "extra_queries": [
            "completely unrelated weather forecast query",   # no AQ synonym → discard
            "pm2.5 monitoring station data",                  # has AQ synonym → keep
        ],
        "reasoning": "LLM reasoning.",
    })
    spec = create_initial_spec("AQ test", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ test",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5"],
        geography=None,
        freshness_need="live",
        constraints=[],
        is_supported=True,
    )
    planner.plan(spec)
    query_texts = {q.query for q in spec.plan.search_queries}
    assert "completely unrelated weather forecast query" not in query_texts, (
        "Irrelevant LLM query was not discarded"
    )


def test_llm_query_relevant_to_subdomain_is_kept(planner, monkeypatch):
    """LLM queries that contain a subdomain synonym must be accepted (up to cap)."""
    monkeypatch.setattr(planning_module, "SOURCE_CATALOG", [
        SourceCatalogEntry(
            "single_aq", "Single AQ", ["air quality"], ["pm2.5"],
            "api", False, False, True, "global", "live", "Test.",
        ),
    ])
    planner.llm.call.return_value = __import__("types").SimpleNamespace(content={
        "extra_queries": ["pm2.5 monitoring station data"],
        "reasoning": "AQ reasoning.",
    })
    spec = create_initial_spec("AQ test", "air quality data")
    spec.requirement = RequirementSpec(
        dataset_name="AQ test",
        description="air quality data",
        domain="environmental",
        subdomain="air quality",
        expected_features=["pm2.5"],
        geography=None,
        freshness_need="live",
        constraints=[],
        is_supported=True,
    )
    planner.plan(spec)
    query_texts = {q.query for q in spec.plan.search_queries}
    assert "pm2.5 monitoring station data" in query_texts, (
        "Relevant LLM query was incorrectly discarded"
    )


# Item 5: reasoning must not mention "time period" when none was given
def test_reasoning_omits_time_period_when_none_given():
    """When no time_range and freshness is unspecified, reasoning must not say 'time period'."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec("unspecified", "Delhi")
    planner.plan(spec)
    reasoning = (spec.plan.reasoning or "").lower()
    assert "time period" not in reasoning, (
        f"reasoning mentions 'time period' when none was given: {spec.plan.reasoning!r}"
    )


def test_reasoning_mentions_geography():
    """Reasoning must include the geography when one was given."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec("unspecified", "Uttarakhand")
    planner.plan(spec)
    reasoning = spec.plan.reasoning or ""
    assert "Uttarakhand" in reasoning or "uttarakhand" in reasoning.lower(), (
        f"reasoning does not mention geography: {reasoning!r}"
    )


def test_reasoning_mentions_source_counts():
    """Reasoning must state how many required/useful/fallback sources were found."""
    planner = _aq_matrix_planner()
    spec = _make_aq_matrix_spec("unspecified", "Delhi")
    planner.plan(spec)
    reasoning = spec.plan.reasoning or ""
    # Should contain at least one number (source count) or the word "source"
    has_count = any(ch.isdigit() for ch in reasoning)
    has_word = "source" in reasoning.lower()
    assert has_count or has_word, (
        f"reasoning doesn't mention source counts or 'source': {reasoning!r}"
    )
