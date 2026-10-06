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
                          output_format="csv", completeness=0.1)
    with patch('builtins.input', side_effect=AssertionError("no question should be asked")):
        out = a._bounded_clarification(req, "X", "rainfall csv")
    assert out is req  # nothing asked, nothing changed


def test_topic_question_asked_when_no_domain():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="stuff csv", domain=None,
                          geography="Delhi", time_range="last 7 days", expected_size="500 rows",
                          output_format="csv")
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
                          output_format="csv", missing_or_unclear=["dataset size"])
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


# ---------------------------------------------------------------- Bug 8
def test_option_4_for_granularity_means_yearly_not_custom():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="weather csv", domain="meteorological",
                          geography="Delhi", time_range="2010 to 2020", expected_size="500 rows",
                          output_format="csv")
    inputs = iter(["4"])   # a second input() call would raise StopIteration
    with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
        out = a._bounded_clarification(req, "X", "weather csv")
    assert "clarified_granularity: Yearly" in out.explicitly_stated


def test_option_4_for_size_is_custom_and_reads_one_line():
    a = make_analyzer(interactive=True)
    req = RequirementSpec(dataset_name="X", description="weather csv", domain="meteorological",
                          geography="Delhi", time_range="last 7 days", output_format="csv")
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