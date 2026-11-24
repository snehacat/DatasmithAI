"""
Regression tests for the Step 1 + Step 2 fixes in agents/requirement_analyzer.py.
None of these need Ollama: the LLM is mocked or never reached.

Put this file in tests/ and run:  python -m pytest tests/test_requirement_analyzer_fixes.py -q
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from core.dataset_spec import RequirementSpec, PipelineStatus, create_initial_spec
from agents.requirement_analyzer import RequirementAnalyzer, LLMRequirementAnalysis


def make_analyzer(interactive=False):
    analyzer = RequirementAnalyzer(interactive=interactive)
    analyzer.llm = Mock()  # any accidental LLM call is visible
    return analyzer


def llm_result(**kw):
    base = dict(domain="environmental", problem_type="analysis", data_modality="tabular",
                expected_features=[], topic_description="summary")
    base.update(kw)
    return LLMRequirementAnalysis(**base)


# ---------------------------------------------------------------- Bug 1
def test_user_description_is_never_overwritten():
    a = make_analyzer()
    user_text = "rainfall data for kerala"
    with patch.object(a, '_analyze_with_llm',
                      return_value=llm_result(domain="healthcare", topic_description="LLM summary")):
        req = a._analyze_requirement("Kerala Rain", user_text)
    assert req.description == user_text
    assert req.topic_summary == "LLM summary"


# ---------------------------------------------------------------- Bug 2
def test_topic_question_only_when_no_domain():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="rainfall csv", domain="environmental",
                          geography="Delhi", time_range="last 7 days", expected_size="500 rows",
                          output_format="csv", completeness=0.1,
                          explicitly_stated=[
                              "geography: Delhi", "time_range: last 7 days", "dataset_size: 500"
                          ])
    with patch('builtins.input', side_effect=AssertionError("no question should be asked")):
        out = a._bounded_clarification(req, "X", "rainfall csv")
    assert out is req  # nothing asked, nothing changed


def test_topic_question_asked_when_no_domain():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="stuff csv", domain=None,
                          geography="Delhi", time_range="last 7 days", expected_size="500 rows",
                          output_format="csv",
                          explicitly_stated=[
                              "geography: Delhi", "time_range: last 7 days", "dataset_size: 500"
                          ])
    inputs = iter(["rainfall data for kerala"])
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)), \
         patch.object(a, '_analyze_with_llm', return_value=llm_result(domain="meteorological")):
        out = a._bounded_clarification(req, "X", "stuff csv")
    assert out.domain == "meteorological"


# ---------------------------------------------------------------- Bug 3
@pytest.mark.parametrize("text,expected", [
    ("thousand rows of rainfall", None),            # 'usa' inside 'thousand'
    ("rainfall in ukraine", None),                  # 'uk' inside 'ukraine'
    ("rainfall in bihar", "Bihar"),
    ("weather in usa", "USA"),
    ("worldwide earthquake data", "Worldwide"),
    ("global warming trends", None),                # topic, not a place
    ("air quality in delhi", "Delhi"),
    ("temperature in tamil nadu", "Tamil Nadu"),
])
def test_geography(text, expected):
    a = make_analyzer()
    assert a._extract_geography(text, [], []) == expected


# ---------------------------------------------------------------- Bug 4
@pytest.mark.parametrize("text,expected", [
    ("i need 1,000 rows of data", "1000 rows"),
    ("about 10 years of rainfall", None),
    ("2000 rows of temperature", "2000 rows"),
    ("10k records please", "10000 rows"),
    ("about 1000 samples", "1000 rows"),
    ("give me weather data", None),
])
def test_size(text, expected):
    a = make_analyzer()
    assert a._extract_size(text, [], []) == expected


def test_typed_size_in_clarification_is_accepted():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="rainfall csv", domain="environmental",
                          geography="Delhi", time_range="last 7 days", expected_size=None,
                          output_format="csv", missing_or_unclear=["dataset size"],
                          explicitly_stated=["geography: Delhi", "time_range: last 7 days"])
    inputs = iter(["2000 rows"])
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
        out = a._bounded_clarification(req, "X", "rainfall csv")
    assert out.expected_size == "2000 rows"
    assert "dataset size" not in out.missing_or_unclear


# ---------------------------------------------------------------- Bug 6 / 7
def test_llm_nulls_do_not_fail():
    a = make_analyzer()
    result = a._parse_llm_payload({"domain": "environmental", "problem_type": None,
                                   "data_modality": None, "expected_features": None,
                                   "topic_description": None})
    assert result.data_modality == "tabular"
    assert result.problem_type == "analysis"
    assert result.expected_features == []


def test_rules_fallback_with_no_domain_does_not_crash():
    a = make_analyzer()
    result = a._extract_topic_with_rules("nothing useful here", "nothing useful here")
    assert result.domain is None


def test_llm_exception_falls_back_to_rules():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm', side_effect=TimeoutError("timed out")):
        req = a._analyze_requirement("Air Quality Delhi", "air quality in Delhi")
    assert req.domain == "environmental" and req.subdomain == "air quality"


def test_llm_fallback_is_reported_on_spec():
    a = make_analyzer()
    spec = create_initial_spec("Air Quality", "air quality in Delhi")
    with patch.object(a, '_analyze_with_llm', side_effect=TimeoutError("timed out")):
        out = a.analyze(spec)
    warning = "LLM unavailable, rule-based fallback used; expected_features may be empty"
    assert warning in out.warnings
    assert "llm_fallback_used" in out.requirement.inferred_by_model


# ---------------------------------------------------------------- Bug 8
def test_option_4_for_granularity_means_yearly_not_custom():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="weather csv", domain="meteorological",
                          geography="Delhi", time_range="2010 to 2020", expected_size="500 rows",
                          output_format="csv",
                          explicitly_stated=[
                              "geography: Delhi", "time_range: 2010 to 2020", "dataset_size: 500"
                          ])
    inputs = iter(["4"])   # a second input() call would raise StopIteration
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
        out = a._bounded_clarification(req, "X", "weather csv")
    assert "clarified_granularity: Yearly" in out.explicitly_stated


def test_option_4_for_size_is_custom_and_reads_one_line():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="weather csv", domain="meteorological",
                          geography="Delhi", time_range="last 7 days", output_format="csv",
                          explicitly_stated=["geography: Delhi", "time_range: last 7 days"])
    inputs = iter(["4", "3,500"])
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
        out = a._bounded_clarification(req, "X", "weather csv")
    assert out.expected_size == "3500 rows"


# ---------------------------------------------------------------- Bug 9
def test_answered_questions_are_removed():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="weather csv", domain="meteorological",
                          geography=None, output_format="csv",
                          missing_or_unclear=["geographic area"],
                          clarifying_questions=["Which geographic area or location do you need?"])
    inputs = iter(["delhi", "skip"])
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
        out = a._bounded_clarification(req, "X", "weather csv")
    assert out.geography == "Delhi"
    assert out.clarifying_questions == []
    assert "geographic area" not in out.missing_or_unclear


def test_derivation_note_is_a_warning_not_missing_info():
    a = make_analyzer()
    spec = create_initial_spec("Flood Risk", "flood risk prediction in rishikesh")
    with patch.object(a, '_analyze_with_llm',
                      return_value=llm_result(target_variable="flood_occurrence")):
        out = a.analyze(spec)
    note = "target must be derived from real events or thresholds"
    assert out.requirement.target_requires_derivation is True
    assert note not in out.requirement.missing_or_unclear
    assert note in out.warnings


def test_current_year_comes_from_the_clock():
    a = make_analyzer()
    year = datetime.now().year
    with patch.object(a, '_analyze_with_llm', return_value=llm_result()):
        req = a._analyze_requirement("Flood", f"flood dataset for rishikesh from 2020 to {year}")
    assert any("latest available" in c for c in req.constraints)


def test_old_range_does_not_add_latest_constraint():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm', return_value=llm_result()):
        req = a._analyze_requirement("Rain", "rainfall in kerala from 2010 to 2015")
    assert not any("latest available" in c for c in req.constraints)


def test_granularity_not_asked_when_user_stated_it():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm', return_value=llm_result()):
        req = a._analyze_requirement("Rain", "monthly rainfall in kerala from 2010 to 2020")
    assert not any("granularity" in q for q in req.clarifying_questions)


# ---------------------------------------------------------------- Bug 10 / Step 2
@pytest.mark.parametrize("text", [
    "live weather data for delhi",
    "recent earthquakes worldwide",                       # plural
    "rainfall data for livestock grazing areas in rajasthan",
    "glacier retreat in the himalayas",
    "thick sea ice extent",                               # 'ice' as a whole word
    "rainfall by crop product region",                    # 'product' no longer rejects
    "review of climate records",                          # 'review' no longer rejects
    "weather data, free sources only, no paid data",      # negated restriction
    "weather data without login required",                # negated restriction
])
def test_supported(text):
    a = make_analyzer()
    ok, reason = a._check_support(text)
    assert ok is True and reason is None, text


@pytest.mark.parametrize("text", [
    "stock prices for tesla",
    "air pollution impact on stock market",
    "twitter sentiment about floods",
    "weather data from private sources requiring login",
    "police station locations",                           # no 'ice' false match, no env keyword
    "accident risk prediction dataset from year 2020 to 2025",
    "price list for groceries",
])
def test_rejected_instantly_without_llm(text):
    a = make_analyzer()
    ok, reason = a._check_support(text)
    assert ok is False
    assert "not supported yet" in reason.lower()
    assert "outside our focus" in reason.lower()
    a.llm.call.assert_not_called()


def test_rejected_request_sets_status_and_makes_no_llm_call():
    a = make_analyzer()
    spec = create_initial_spec("Tesla Stock", "stock prices for tesla")
    out = a.analyze(spec)
    assert out.requirement.is_supported is False
    assert out.requirement.completeness == 0.0
    assert out.pipeline_status == PipelineStatus.REJECTED
    a.llm.call.assert_not_called()


# ---------------------------------------------------------------- Bug 11
def test_llm_calls_are_counted_and_timed():
    a = make_analyzer()
    a.llm.call.return_value = Mock(content={"domain": "environmental", "topic_description": "x"})
    a._analyze_with_llm("Rain", "rainfall in kerala", "rain rainfall in kerala")
    assert a.stats['llm_calls'] == 1
    assert a.stats['llm_seconds'] >= 0.0


def test_user_wait_is_recorded_separately():
    a = make_analyzer()
    with patch('builtins.input', return_value="hello"):
        assert a._read_input("? ") == "hello"
    assert a.stats['user_wait_seconds'] >= 0.0
    assert a.stats['llm_calls'] == 0


# ---------------------------------------------------------------- small extras
def test_excel_is_not_matched_inside_excellent():
    a = make_analyzer()
    assert a._extract_output_format("excellent rainfall data", [], []) == "csv"
    assert a._extract_output_format("export to excel file", [], []) == "excel"


# ---------------------------------------------------------------- subdomain / target placeholder
def test_subdomain_with_place_is_replaced_by_keyword_subdomain():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm',
                      return_value=llm_result(subdomain="Rishikesh Rainfall", problem_type="prediction")):
        req = a._analyze_requirement("Rainfall Rishikesh from 2010 to 2026", "Flood risk dataset")
    assert req.subdomain == "flood monitoring"


def test_clean_subdomain_keeps_good_value():
    a = make_analyzer()
    assert a._clean_subdomain("air quality", "air quality in delhi") == "air quality"


def test_clean_subdomain_strips_place_and_year_when_no_keyword_fallback():
    a = make_analyzer()
    assert a._clean_subdomain("rainfall delhi 2015", "nothing useful here") == "rainfall"
    assert a._clean_subdomain(None, "x") is None


def test_prediction_with_derived_target_gets_placeholder():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm',
                      return_value=llm_result(problem_type="prediction", target_variable=None)):
        req = a._analyze_requirement("Flood", "flood risk dataset for rishikesh")
    assert req.target_requires_derivation is True
    assert req.target_variable == "to be defined from real data"


def test_no_placeholder_for_monitoring_request():
    a = make_analyzer()
    with patch.object(a, '_analyze_with_llm',
                      return_value=llm_result(problem_type="monitoring", target_variable=None)):
        req = a._analyze_requirement("Flood", "flood risk monitoring for rishikesh")
    assert req.target_variable is None

def _range(text):
    inferred = []
    result = RequirementAnalyzer()._extract_time_range(text, [], inferred)
    return result, inferred


def test_reversed_year_range_is_swapped():
    result, inferred = _range("rainfall data from 2020 to 2010")
    assert result == "2010 to 2020"
    assert any("reversed" in note for note in inferred)


def test_future_end_year_is_capped():
    now = datetime.now().year
    result, inferred = _range(f"rainfall from 2015 to {now + 5}")
    assert result == f"2015 to {now}"
    assert any("capped" in note for note in inferred)


def test_entirely_future_range_is_rejected():
    now = datetime.now().year
    result, _ = _range(f"rainfall from {now + 2} to {now + 4}")
    assert result is None


def test_zero_day_window_is_ignored():
    result, _ = _range("rainfall for the last 0 days")
    assert result is None


def test_normal_ranges_still_work():
    assert _range("rainfall from 2010 to 2020")[0] == "2010 to 2020"
    assert _range("rainfall for the last 7 days")[0] == "last 7 days"


def test_abbreviated_month_range_is_expanded():
    assert _range("jan 2020 to march 2021")[0] == "January 2020 to March 2021"


def test_future_range_adds_missing_question_and_warning():
    a = make_analyzer()
    spec = create_initial_spec("Future Rain", "rainfall in Delhi from 2030 to 2035")
    with patch.object(a, '_analyze_with_llm', return_value=llm_result()):
        out = a.analyze(spec)
    assert out.requirement.time_range is None
    assert "historical time range" in out.requirement.missing_or_unclear
    assert any("entirely in the future" in q for q in out.requirement.clarifying_questions)


def test_expected_rows_are_capped_and_zero_is_rejected():
    a = make_analyzer()
    explicitly = []
    inferred = []
    assert a._extract_size("50000000 rows", explicitly, inferred) == "10000000 rows"
    assert "capped" in inferred[0]
    assert a._extract_size("0 rows", [], []) is None


def test_more_than_four_places_keeps_four_and_reports_dropped():
    a = make_analyzer()
    spec = create_initial_spec(
        "Rain",
        "rainfall in Delhi, Mumbai, Pune, Chennai, Kolkata",
    )
    with patch.object(a, '_analyze_with_llm', return_value=llm_result()):
        out = a.analyze(spec)
    assert out.requirement.geography == "Delhi, Mumbai, Pune, Chennai"
    assert any("Kolkata" in q for q in out.requirement.clarifying_questions)


@pytest.mark.parametrize("text", [
    "rainfall for financial year 2020-21 in Maharashtra",
    "cloud cover and visibility data for Delhi",
    "pm2.5 data (paid sources not allowed)",
    "I don't want paid APIs for temperature",
    "fish stocks and sea temperature in Maharashtra",
    "private weather stations in Delhi",
    "private sensors for rainfall in Kerala",
])
def test_scope_allow_and_negation_phrases(text):
    a = make_analyzer()
    assert a._check_support(text)[0] is True


@pytest.mark.parametrize("text", [
    "price of a sea food restaurant",
    "ice cream sales",
])
def test_deny_phrases_keep_requests_out_of_scope(text):
    a = make_analyzer()
    assert a._check_support(text)[0] is False


def test_short_typo_fir_is_suggested():
    assert ("fir", "fire") in RequirementAnalyzer()._find_typos("Fir hazard in mp")

def test_clean_subdomain_rejects_feature_list():
    a = RequirementAnalyzer()
    # general weather: no keyword subdomain exists, so the result is None
    assert a._clean_subdomain(
        "temperature, humidity, wind_speed", "weather prediction for rishikesh"
    ) is None
    assert a._clean_subdomain("wind_speed", "weather data for delhi") is None


def test_clean_subdomain_feature_list_falls_back_to_keyword_subdomain():
    a = RequirementAnalyzer()
    assert a._clean_subdomain(
        "rainfall, river_level", "flood data for assam"
    ) == "flood monitoring"

def test_completeness_is_rounded_to_two_decimals():
    a = RequirementAnalyzer()
    # 0.50 + 0.30 (4+ explicit) - 3 * 0.05 (missing) is 0.6499999999999999 without rounding
    score = a._calculate_confidence(['a', 'b', 'c', 'd'], ['x', 'y', 'z'], True)
    assert score == 0.65

def test_granularity_default_shown_matches_model_guess(monkeypatch, capsys):
    a = RequirementAnalyzer()
    monkeypatch.setattr(a, "_read_input", lambda prompt: "")   # user just presses Enter
    req = RequirementSpec(
        dataset_name="Flood",
        description="flood risk in bihar from 2020 to 2026 csv 1000 rows",
        domain="hydrological", geography="Bihar",
        time_range="2020 to 2026", time_granularity="Yearly",
        expected_size="1000 rows", output_format="csv",
    )
    result = a._bounded_clarification(req, "Flood", req.description)

    out = capsys.readouterr().out
    assert "Default: Yearly" in out
    assert "Using default: Yearly" in out
    assert result.time_granularity == "Yearly"

def test_granularity_asked_after_multi_year_range_chosen_in_clarification(monkeypatch, capsys):
    a = RequirementAnalyzer()
    # answers: option 4 (custom range), the range itself, then granularity option 3 (Monthly)
    replies = iter(["4", "2010 to 2019", "3"])
    monkeypatch.setattr(a, "_read_input", lambda prompt: next(replies))
    req = RequirementSpec(
        dataset_name="Flood",
        description="flood data for bihar csv 1000 rows",
        domain="hydrological", geography="Bihar",
        expected_size="1000 rows", output_format="csv",
        explicitly_stated=["geography: Bihar", "dataset_size: 1000"],
    )
    result = a._bounded_clarification(req, "Flood", req.description)

    assert "Time granularity for 2010 to 2019" in capsys.readouterr().out
    assert result.time_range == "2010 to 2019"
    assert result.time_granularity == "Monthly"

def test_prompt_does_not_ask_for_topic_description():
    a = RequirementAnalyzer()
    seen = {}

    class FakeResponse:
        content = {"domain": "meteorological"}

    class FakeLLM:
        def call(self, **kwargs):
            seen["prompt"] = kwargs["prompt"]
            return FakeResponse()

    a.llm = FakeLLM()
    a._analyze_with_llm("Rain", "rainfall in kerala", "rain rainfall in kerala")

    assert "topic_description" not in seen["prompt"]
    assert "rainfall in kerala" in seen["prompt"]

_NOW = datetime.now().year


@pytest.mark.parametrize("text, expected", [
    ("flood data in assam between 2018 and 2022", "2018 to 2022"),
    ("glacier retreat in himalayas since 2000", f"2000 to {_NOW}"),
    ("rainfall 2010 to present", f"2010 to {_NOW}"),
    ("soil moisture in punjab for 2019", "2019 to 2019"),
    ("daily temperature of delhi in january 2024", "January 2024"),
    ("pm2.5 in lucknow during winter 2023", "Winter 2023"),
    ("january 2023 to march 2024 temperature", "January 2023 to March 2024"),
    ("rainfall in bengaluru 10 years", "last 10 years"),
    ("past decade rainfall", "last 10 years"),
    ("heavy rainfall alerts for mumbai today", "today"),
    ("rainfall data 2020 to 2010", "2010 to 2020"),
])
def test_time_range_patterns(text, expected):
    assert _range(text)[0] == expected


@pytest.mark.parametrize("text", [
    "need 2000 rows of rainfall",            # 2000 is a size, not a year
    "rainfall for 2000 rows in kerala",
    "temperature data every 6 hours",        # a granularity, not a window
    "10 day forecast for delhi",             # the future, not history
    "from 2030 to 2035 rainfall",            # entirely in the future
    "rainfall last 0 days",
])
def test_time_range_false_positives(text):
    assert _range(text)[0] is None


def test_today_means_recent_data():
    freshness, max_age = RequirementAnalyzer()._extract_freshness(
        "heavy rainfall alerts for mumbai today", [], [], [], []
    )
    assert freshness == "recent"
    assert max_age == "24 hours"

@pytest.mark.parametrize("text, expected", [
    ("Fllod risk prediction for Nepal", ("fllod", "flood")),
    ("air polution in delhi", ("polution", "pollution")),
    ("forest fire data for uttrakhand", ("uttrakhand", "uttarakhand")),
    ("temprature of mumbai", ("temprature", "temperature")),
])
def test_typos_are_detected(text, expected):
    assert expected in RequirementAnalyzer()._find_typos(text)


@pytest.mark.parametrize("text", [
    "rainfall forecast for Kochi",
    "temperature in Indore and Raipur",
    "floods and flooding in Bihar",
    "weather prediction data for uttarakhand",
    "10 day forecast for gurgaon",
    "student project on climate",
])
def test_valid_words_are_not_flagged(text):
    assert RequirementAnalyzer()._find_typos(text) == []


def _rules_only(a):
    """Keep tests off the real LLM: use the rule-based topic extraction."""
    a._analyze_with_llm = lambda name, desc, combined: a._extract_topic_with_rules(combined, desc)


def test_user_confirms_typo_and_request_is_accepted(monkeypatch):
    a = RequirementAnalyzer(interactive=True)
    _rules_only(a)
    monkeypatch.setattr(a, "_read_input", lambda prompt: "")   # Enter = yes, then defaults
    spec = a.analyze(create_initial_spec("Test", "Fllod risk prediction for Nepal"))

    assert spec.requirement.is_supported
    assert spec.requirement.description == "Fllod risk prediction for Nepal"   # kept as typed
    assert any("fllod" in warning for warning in spec.warnings)


def test_user_rejects_typo_suggestion(monkeypatch):
    a = RequirementAnalyzer(interactive=True)
    _rules_only(a)
    replies = iter(["n"])
    monkeypatch.setattr(a, "_read_input", lambda prompt: next(replies, ""))
    spec = a.analyze(create_initial_spec("Test", "Fllod risk prediction for Nepal"))

    assert not spec.requirement.is_supported
    assert spec.requirement.description == "Fllod risk prediction for Nepal"


def test_typo_hint_added_when_not_interactive():
    spec = RequirementAnalyzer().analyze(create_initial_spec("Test", "air polution in delhi"))
    assert not spec.requirement.is_supported
    assert "Did you mean 'pollution' instead of 'polution'?" in spec.requirement.unsupported_reason

@pytest.mark.parametrize("text, expected", [
    ("5 lakh rows", 500_000),
    ("2 million records", 2_000_000),
    ("1.5 crore samples", 10_000_000),
    ("3 thousand entries", 3_000),
    ("10k rows", 10_000),
    ("2000 rows", 2_000),
    ("1,000 rows", 1_000),
])
def test_size_with_word_multipliers(text, expected):
    assert RequirementAnalyzer()._extract_size(text.lower(), [], []) == f"{expected} rows"


@pytest.mark.parametrize("text", ["about 10 years", "5 months of data", "last 7 days"])
def test_non_sizes_are_not_sizes(text):
    assert RequirementAnalyzer()._extract_size(text.lower(), [], []) is None


@pytest.mark.parametrize("typed, expected", [
    ("2 lakh", 200_000), ("3 thousand", 3_000), ("2k", 2_000),
    ("2000", 2_000), ("5 million rows", 5_000_000),
])
def test_typed_size_answers(typed, expected):
    assert RequirementAnalyzer()._parse_size_text(typed) == expected