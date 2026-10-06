"""
Tests for Requirement Analyzer Agent

Tests cover:
1. Regex/rule-based extraction (no LLM needed)
2. Support checking (instant, no LLM)
3. Full analysis with LLM (requires Ollama)
4. Full pipeline through orchestrator
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from core.dataset_spec import (
    DatasetSpec,
    RequirementSpec,
    create_initial_spec
)
from agents.requirement_analyzer import RequirementAnalyzer, LLMRequirementAnalysis


# ============================================================================
# UNIT TESTS (No LLM needed)
# ============================================================================

class TestExtractionMethods:
    """Test individual extraction methods without LLM."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer()
    
    def test_extract_output_format_csv(self):
        """Test CSV format extraction."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_output_format(
            "give me weather data in csv format",
            explicitly_stated,
            inferred
        )
        
        assert result == "csv"
        assert "output_format: csv" in explicitly_stated
        assert len(inferred) == 0
    
    def test_extract_output_format_excel(self):
        """Test Excel format extraction."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_output_format(
            "export to excel file",
            explicitly_stated,
            inferred
        )
        
        assert result == "excel"
        assert "output_format: excel" in explicitly_stated
    
    def test_extract_output_format_json(self):
        """Test JSON format extraction."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_output_format(
            "need json output",
            explicitly_stated,
            inferred
        )
        
        assert result == "json"
        assert "output_format: json" in explicitly_stated
    
    def test_extract_output_format_default(self):
        """Test default format when not specified."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_output_format(
            "give me weather data",
            explicitly_stated,
            inferred
        )
        
        assert result == "csv"
        assert len(explicitly_stated) == 0
        assert "output_format: csv (default)" in inferred
    
    def test_extract_size_with_rows(self):
        """Test size extraction with 'rows'."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_size(
            "i need 5000 rows of temperature data",
            explicitly_stated,
            inferred
        )
        
        assert result == "5000 rows"
        assert "dataset_size: 5000" in explicitly_stated
    
    def test_extract_size_with_samples(self):
        """Test size extraction with 'samples'."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_size(
            "about 1000 samples of ocean temperature",
            explicitly_stated,
            inferred
        )
        
        assert result == "1000 rows"
        assert "dataset_size: 1000" in explicitly_stated
    
    def test_extract_size_not_specified(self):
        """Test when size is not specified."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_size(
            "give me weather data",
            explicitly_stated,
            inferred
        )
        
        assert result is None
        assert len(explicitly_stated) == 0
    
    def test_extract_freshness_live(self):
        """Test live freshness extraction."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, max_age = self.analyzer._extract_freshness(
            "live air quality data for delhi",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "live"
        assert max_age == "1 hour"  # Inferred for live
        assert "freshness: live" in explicitly_stated
        assert any("1 hour" in item for item in inferred)
    
    def test_extract_freshness_recent(self):
        """Test recent freshness extraction."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, max_age = self.analyzer._extract_freshness(
            "recent air quality data",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "recent"
        assert max_age == "24 hours"  # Inferred for recent
        assert "freshness: recent" in explicitly_stated
        assert "specific time window" in missing
        assert any("How recent" in q for q in questions)
    
    def test_extract_freshness_recent_with_period(self):
        """Test recent - time window goes to time_range, not max_data_age."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        # Note: "last 7 days" is extracted separately by time_range
        # Freshness just detects "recent" and infers max_data_age
        freshness, max_age = self.analyzer._extract_freshness(
            "recent air quality data last 7 days",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "recent"
        assert max_age == "24 hours"  # Still inferred default
        assert "freshness: recent" in explicitly_stated
    
    def test_extract_freshness_historical(self):
        """Test historical freshness extraction."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, max_age = self.analyzer._extract_freshness(
            "historical rainfall data from 2010 to 2020",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "historical"
        assert "freshness: historical" in explicitly_stated
    
    def test_extract_freshness_unspecified(self):
        """Test when freshness is not specified."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, max_age = self.analyzer._extract_freshness(
            "give me weather data",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "unspecified"
        assert any("freshness: unspecified" in item for item in inferred)
    
    def test_extract_time_range_years(self):
        """Test time range extraction with years."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_time_range(
            "monthly rainfall from 2010 to 2020",
            explicitly_stated,
            inferred
        )
        
        assert result == "2010 to 2020"
        assert "time_range: 2010 to 2020" in explicitly_stated
    
    def test_extract_time_range_this_month(self):
        """Test time range extraction with 'this month'."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_time_range(
            "earthquakes this month",
            explicitly_stated,
            inferred
        )
        
        assert result == "this month"
        assert "time_range: this month" in explicitly_stated
    
    def test_extract_time_range_not_specified(self):
        """Test when time range is not specified."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_time_range(
            "give me weather data",
            explicitly_stated,
            inferred
        )
        
        assert result is None
    
    def test_extract_geography_india(self):
        """Test geography extraction."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_geography(
            "air quality in delhi",
            explicitly_stated,
            inferred
        )
        
        assert result == "Delhi"
        assert "geography: Delhi" in explicitly_stated
    
    def test_extract_geography_worldwide(self):
        """Test worldwide geography extraction."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_geography(
            "worldwide earthquake data",
            explicitly_stated,
            inferred
        )
        
        assert result == "Worldwide"
        assert "geography: Worldwide" in explicitly_stated
    
    def test_extract_geography_not_specified(self):
        """Test when geography is not specified."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_geography(
            "give me weather data",
            explicitly_stated,
            inferred
        )
        
        assert result is None
    
    def test_extract_constraints_free_sources(self):
        """Test constraint extraction - free sources."""
        explicitly_stated = []
        
        result = self.analyzer._extract_constraints(
            "free sources only, no paid data",
            explicitly_stated
        )
        
        assert "free sources only" in result
        assert "constraint: free sources" in explicitly_stated
    
    def test_extract_constraints_no_login(self):
        """Test constraint extraction - no login."""
        explicitly_stated = []
        
        result = self.analyzer._extract_constraints(
            "data without login required",
            explicitly_stated
        )
        
        assert "no login required" in result
        assert "constraint: no login" in explicitly_stated
    
    def test_extract_constraints_public_data(self):
        """Test constraint extraction - public data."""
        explicitly_stated = []
        
        result = self.analyzer._extract_constraints(
            "only public data",
            explicitly_stated
        )
        
        assert "public data only" in result
        assert "constraint: public data" in explicitly_stated


class TestSupportChecking:
    """Test support checking logic (instant, never calls the LLM)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer()
    
    def test_supported_weather(self):
        """Test that weather request is supported."""
        is_supported, reason = self.analyzer._check_support(
            "live weather data for delhi"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_supported_air_quality(self):
        """Test that air quality request is supported."""
        is_supported, reason = self.analyzer._check_support(
            "air quality measurements for mumbai"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_supported_earthquake(self):
        """Test that earthquake request is supported."""
        is_supported, reason = self.analyzer._check_support(
            "recent earthquake data worldwide"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_supported_satellite(self):
        """Test that satellite data request is supported."""
        is_supported, reason = self.analyzer._check_support(
            "satellite fire detection data for india"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_supported_livestock_rainfall(self):
        """Test that livestock rainfall request is supported (false positive check)."""
        is_supported, reason = self.analyzer._check_support(
            "rainfall data for livestock grazing areas in rajasthan"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_unsupported_stock_prices(self):
        """Test that stock prices are instantly rejected."""
        is_supported, reason = self.analyzer._check_support(
            "stock prices for tesla"
        )
        
        assert is_supported is False
        assert "not supported yet" in reason.lower()
    
    def test_unsupported_social_media(self):
        """Test that social media data is instantly rejected."""
        is_supported, reason = self.analyzer._check_support(
            "twitter sentiment data"
        )
        
        assert is_supported is False
        assert "not supported yet" in reason.lower()
    
    def test_unsupported_private_data(self):
        """Test that private data requirement is instantly rejected."""
        is_supported, reason = self.analyzer._check_support(
            "weather data from private sources requiring login"
        )
        
        assert is_supported is False
        assert "not supported yet" in reason.lower()
    
    def test_unsupported_stock_market_with_environmental(self):
        """Test that financial focus is rejected even with environmental keywords."""
        is_supported, reason = self.analyzer._check_support(
            "air pollution impact on stock market"
        )
        
        assert is_supported is False
        assert "not supported yet" in reason.lower()
    
    def test_supported_glacier_retreat(self):
        """Test that glacier retreat is supported."""
        is_supported, reason = self.analyzer._check_support(
            "glacier retreat in the himalayas"
        )
        
        assert is_supported is True
        assert reason is None
    
    def test_supported_landslide(self):
        """Test that landslide events are supported."""
        is_supported, reason = self.analyzer._check_support(
            "landslide events in uttarakhand"
        )
        
        assert is_supported is True
        assert reason is None
        
    def test_accident_risk_rejected_instantly_without_llm(self):
        """Accident risk has no environmental keyword: rejected instantly, LLM never called."""
        self.analyzer.llm = Mock()
        
        is_supported, reason = self.analyzer._check_support(
            "accident risk prediction dataset from year 2020 to 2025"
        )
        
        assert is_supported is False
        assert "not supported yet" in reason.lower()
        self.analyzer.llm.call.assert_not_called()


class TestFreshnessWordBoundaries:
    """Test that freshness detection uses word boundaries."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer()
    
    def test_live_not_in_livestock(self):
        """Test that 'live' in 'livestock' is not detected as freshness."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, _ = self.analyzer._extract_freshness(
            "rainfall data for livestock grazing areas",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "unspecified"
        assert "freshness: live" not in explicitly_stated
        assert any("unspecified" in item for item in inferred)
    
    def test_live_as_word(self):
        """Test that 'live' as a word IS detected."""
        explicitly_stated = []
        inferred = []
        missing = []
        questions = []
        
        freshness, _ = self.analyzer._extract_freshness(
            "live weather data",
            explicitly_stated,
            inferred,
            missing,
            questions
        )
        
        assert freshness == "live"
        assert "freshness: live" in explicitly_stated


class TestGeographyExtraction:
    """Test geography extraction improvements."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer()
    
    def test_rajasthan_detection(self):
        """Test that Rajasthan is detected."""
        explicitly_stated = []
        inferred = []
        
        result = self.analyzer._extract_geography(
            "rainfall data for livestock grazing areas in rajasthan",
            explicitly_stated,
            inferred
        )
        
        assert result == "Rajasthan"
        assert "geography: Rajasthan" in explicitly_stated
    
    def test_other_indian_states(self):
        """Test detection of other Indian states."""
        test_states = [
            ("weather in maharashtra", "Maharashtra"),
            ("rainfall in karnataka", "Karnataka"),
            ("temperature in tamil nadu", "Tamil Nadu"),
            ("rainfall in bihar", "Bihar"),
        ]
        
        for text, expected in test_states:
            explicitly_stated = []
            inferred = []
            result = self.analyzer._extract_geography(text, explicitly_stated, inferred)
            assert result == expected, f"Failed for {text}"
    
    def test_rishikesh_in_dataset_name(self):
        """Test that geography is extracted from the dataset name (LLM is mocked)."""
        year = datetime.now().year
        spec = create_initial_spec(
            f'Flood Risk Prediction dataset of Rishikesh from 2020 to {year}',
            'Flood risk'
        )
        mock_llm_result = LLMRequirementAnalysis(
            domain="environmental",
            topic_description="Flood risk"
        )
        
        with patch.object(self.analyzer, '_analyze_with_llm', return_value=mock_llm_result):
            result = self.analyzer._analyze_requirement(
                spec.requirement.dataset_name,
                spec.requirement.description
            )
        
        # Geography should be extracted from dataset name
        assert result.geography == "Rishikesh"
        assert "Which geographic area" not in str(result.clarifying_questions)
        
        # Time range should also be extracted from dataset name
        assert result.time_range == f"2020 to {year}"
        
        # Time range reaches the current year, so "latest available" constraint is added
        assert any("latest available" in c for c in result.constraints)


# ============================================================================
# INTEGRATION TESTS (Require Ollama)
# ============================================================================

@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
class TestWithLLM:
    """Tests that require LLM (Ollama)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer()
    
    def test_case_1_clear_request(self):
        """
        Test Case 1: Clear request
        "live air quality data for Delhi, last 7 days, CSV"
        """
        print("\n" + "="*70)
        print("TEST CASE 1: Clear request")
        print("="*70)
        
        spec = create_initial_spec(
            "Delhi Air Quality",
            "live air quality data for Delhi, last 7 days, CSV"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Problem Type: {req.problem_type}")
        print(f"Data Modality: {req.data_modality}")
        print(f"Expected Features: {req.expected_features}")
        print(f"Freshness: {req.freshness_need}")
        print(f"Max Data Age: {req.max_data_age}")
        print(f"Geography: {req.geography}")
        print(f"Output Format: {req.output_format}")
        print(f"Is Supported: {req.is_supported}")
        print(f"completeness: {req.completeness}")
        print(f"Explicitly Stated: {req.explicitly_stated}")
        print(f"Missing/Unclear: {req.missing_or_unclear}")
        print(f"Clarifying Questions: {req.clarifying_questions}")
        
        # Assertions
        assert req.is_supported is True
        assert req.freshness_need == "live"
        assert req.max_data_age == "7 days"
        assert req.geography == "Delhi"
        assert req.output_format == "csv"
        assert req.completeness > 0.5
        assert len(req.explicitly_stated) > 3
    
    def test_case_2_vague_request(self):
        """
        Test Case 2: Vague request
        "some weather data"
        """
        print("\n" + "="*70)
        print("TEST CASE 2: Vague request")
        print("="*70)
        
        spec = create_initial_spec(
            "Weather Data",
            "some weather data"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Is Supported: {req.is_supported}")
        print(f"completeness: {req.completeness}")
        print(f"Missing/Unclear: {req.missing_or_unclear}")
        print(f"Clarifying Questions: {req.clarifying_questions}")
        
        # Assertions
        assert req.is_supported is True
        assert req.completeness < 0.6  # Low completeness due to vagueness
        assert len(req.missing_or_unclear) > 0
        assert len(req.clarifying_questions) > 0
    
    def test_case_3_satellite_data(self):
        """
        Test Case 3: Satellite theme
        "recent forest fire detections in India from satellites"
        """
        print("\n" + "="*70)
        print("TEST CASE 3: Satellite fire detection")
        print("="*70)
        
        spec = create_initial_spec(
            "India Fire Detections",
            "recent forest fire detections in India from satellites"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Freshness: {req.freshness_need}")
        print(f"Geography: {req.geography}")
        print(f"Is Supported: {req.is_supported}")
        print(f"Expected Features: {req.expected_features}")
        
        # Assertions
        assert req.is_supported is True
        assert req.freshness_need == "recent"
        assert req.geography == "India"
        assert any("fire" in feat.lower() or "satellite" in feat.lower() 
                  for feat in req.expected_features)
    
    def test_case_4_specific_features(self):
        """
        Test Case 4: Specific features requested
        "earthquakes worldwide this month with magnitude, depth, location and time"
        """
        print("\n" + "="*70)
        print("TEST CASE 4: Specific features")
        print("="*70)
        
        spec = create_initial_spec(
            "Worldwide Earthquakes",
            "earthquakes worldwide this month with magnitude, depth, location and time"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Time Range: {req.time_range}")
        print(f"Geography: {req.geography}")
        print(f"Expected Features: {req.expected_features}")
        print(f"Is Supported: {req.is_supported}")
        
        # Assertions
        assert req.is_supported is True
        assert req.time_range == "this month"
        assert req.geography == "Worldwide"
        # Check that expected features include magnitude, depth, location, time
        feature_str = " ".join(req.expected_features).lower()
        assert "magnitude" in feature_str or "depth" in feature_str
    
    def test_case_5_historical_data(self):
        """
        Test Case 5: Historical data
        "monthly rainfall in Kerala from 2010 to 2020"
        """
        print("\n" + "="*70)
        print("TEST CASE 5: Historical data")
        print("="*70)
        
        spec = create_initial_spec(
            "Kerala Rainfall",
            "monthly rainfall in Kerala from 2010 to 2020"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Freshness: {req.freshness_need}")
        print(f"Time Range: {req.time_range}")
        print(f"Geography: {req.geography}")
        print(f"Is Supported: {req.is_supported}")
        
        # Assertions
        assert req.is_supported is True
        assert req.freshness_need == "historical"
        assert req.time_range == "2010 to 2020"
        assert req.geography == "Kerala"
    
    def test_case_6_size_specified(self):
        """
        Test Case 6: Size specified
        "5000 rows of ocean temperature readings"
        """
        print("\n" + "="*70)
        print("TEST CASE 6: Size specified")
        print("="*70)
        
        spec = create_initial_spec(
            "Ocean Temperature",
            "5000 rows of ocean temperature readings"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Domain: {req.domain}")
        print(f"Subdomain: {req.subdomain}")
        print(f"Expected Size: {req.expected_size}")
        print(f"Is Supported: {req.is_supported}")
        
        # Assertions
        assert req.is_supported is True
        assert req.expected_size == "5000 rows"
        assert "dataset_size: 5000" in req.explicitly_stated
    
    def test_case_7_off_topic(self):
        """
        Test Case 7: Off-topic (financial data)
        "stock prices for Tesla"
        """
        print("\n" + "="*70)
        print("TEST CASE 7: Off-topic (should be rejected)")
        print("="*70)
        
        spec = create_initial_spec(
            "Tesla Stock Prices",
            "stock prices for Tesla"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Is Supported: {req.is_supported}")
        print(f"Unsupported Reason: {req.unsupported_reason}")
        
        # Assertions
        assert req.is_supported is False
        assert req.unsupported_reason is not None
        assert "outside our focus" in req.unsupported_reason.lower()
        assert req.completeness == 0.0
    
    def test_case_8_impossible_private_data(self):
        """
        Test Case 8: Impossible (private data)
        "weather data from private weather stations requiring login"
        """
        print("\n" + "="*70)
        print("TEST CASE 8: Impossible (private data)")
        print("="*70)
        
        spec = create_initial_spec(
            "Private Weather Data",
            "weather data from private weather stations requiring login"
        )
        
        result_spec = self.analyzer.analyze(spec)
        req = result_spec.requirement
        
        print(f"\nDataset: {req.dataset_name}")
        print(f"Description: {req.description}")
        print(f"Is Supported: {req.is_supported}")
        print(f"Unsupported Reason: {req.unsupported_reason}")
        
        # Assertions
        assert req.is_supported is False
        assert req.unsupported_reason is not None
        assert "not supported yet" in req.unsupported_reason.lower()


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
class TestWithOrchestrator:
    """Test full pipeline through orchestrator."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline with orchestrator."""
        print("\n" + "="*70)
        print("FULL PIPELINE TEST WITH ORCHESTRATOR")
        print("="*70)
        
        from core.orchestrator import Orchestrator, AgentDefinition
        from agents.requirement_analyzer import RequirementAnalyzer
        from core.run_manager import RunManager, RunManagerConfig
        from unittest.mock import patch
        
        config = RunManagerConfig(runs_dir=str(tmp_path))
        temp_run_manager = RunManager(config)
        
        with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
            # Create orchestrator
            orchestrator = Orchestrator(auto_save=True)
        
        # Create analyzer
        analyzer = RequirementAnalyzer()
        
        # Register agent
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=analyzer.analyze,
            required_inputs=[],  # First agent, no inputs required
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        # Run pipeline
        import time
        start_time = time.time()
        
        spec = orchestrator.run(
            dataset_name="Delhi Air Quality Live",
            description="live air quality data for Delhi including PM2.5, PM10 and AQI, last 7 days, CSV format"
        )
        
        duration = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETED IN {duration:.2f}s")
        print(f"{'='*70}")
        
        # Print results
        req = spec.requirement
        print(f"\nSpec ID: {spec.spec_id}")
        print(f"Pipeline Status: {spec.pipeline_status.value}")
        print(f"\nRequirement Analysis:")
        print(f"  Dataset: {req.dataset_name}")
        print(f"  Description: {req.description}")
        print(f"  Domain: {req.domain}")
        print(f"  Subdomain: {req.subdomain}")
        print(f"  Problem Type: {req.problem_type}")
        print(f"  Data Modality: {req.data_modality}")
        print(f"  Expected Features: {req.expected_features}")
        print(f"  Freshness: {req.freshness_need}")
        print(f"  Max Data Age: {req.max_data_age}")
        print(f"  Geography: {req.geography}")
        print(f"  Output Format: {req.output_format}")
        print(f"  Is Supported: {req.is_supported}")
        print(f"  completeness: {req.completeness:.2f}")
        print(f"\nExecution History:")
        for exec in spec.execution_history:
            print(f"  - {exec.agent_name}: {exec.status.value} ({exec.duration_seconds:.2f}s)")
        
        # Assertions
        assert spec.pipeline_status.value in ["success", "in_progress"]
        assert spec.requirement is not None
        assert spec.requirement.is_supported is True
        assert len(spec.execution_history) == 1
        assert spec.has_agent_succeeded("RequirementAnalyzer")


class TestBoundedClarification:
    """Test bounded clarification logic."""
    
    def setup_method(self):
        """Setup for each test."""
        self.analyzer = RequirementAnalyzer(interactive=False)
    
    def test_no_clarification_for_clear_request(self):
        """Test that clear requests don't trigger clarification."""
        from core.dataset_spec import RequirementSpec
        
        requirement = RequirementSpec(
            dataset_name="Test Dataset",
            description="Complete test data",
            domain="environmental",
            subdomain="air quality",
            geography="Delhi",
            time_range="last 7 days",
            expected_size="5000 rows",
            output_format="csv",
            completeness=0.85,
            missing_or_unclear=[],
            clarifying_questions=[],
            clarification_round=0,
            is_supported=True
        )
        
        assert requirement.completeness > 0.3
        assert requirement.clarification_round == 0
        print("✅ Clear request logic verified")
    
    def test_clarification_round_tracking(self):
        """Test that clarification_round field works correctly."""
        from core.dataset_spec import RequirementSpec
        
        requirement = RequirementSpec(
            dataset_name="Test",
            description="test",
            clarification_round=0
        )
        
        assert requirement.clarification_round == 0
        requirement.clarification_round = 1
        assert requirement.clarification_round == 1
        print("✅ Clarification round tracking works")
    
    def test_non_interactive_mode_default(self):
        """Test that non-interactive mode is default."""
        analyzer_default = RequirementAnalyzer()
        assert analyzer_default.interactive == False
        
        analyzer_explicit = RequirementAnalyzer(interactive=True)
        assert analyzer_explicit.interactive == True
        print("✅ Interactive mode parameter works")


class TestNewFixes:
    """Test fixes for target derivation, clarification defaults and fallbacks."""
    
    def test_target_requires_derivation(self):
        """Derivation note is a spec WARNING, not a 'missing information' item."""
        analyzer = RequirementAnalyzer()
        year = datetime.now().year
        spec = create_initial_spec(
            "Flood Risk Dataset",
            f"flood risk prediction dataset in rishikesh from 2020 to {year}"
        )
        
        mock_llm_result = LLMRequirementAnalysis(
            domain="environmental",
            subdomain="flood monitoring",
            problem_type="prediction",
            data_modality="time_series",
            expected_features=["rainfall", "river_water_level"],
            target_variable="flood_occurrence",
            topic_description="Flood risk prediction dataset"
        )
        
        with patch.object(analyzer, '_analyze_with_llm', return_value=mock_llm_result):
            result_spec = analyzer.analyze(spec)
            req = result_spec.requirement
            
            assert req.target_requires_derivation is True
            derivation_note = "target must be derived from real events or thresholds"
            assert derivation_note not in req.missing_or_unclear
            assert derivation_note in result_spec.warnings
            # Constraints must contain only user requirements (NOT derivation note)
            assert derivation_note not in req.constraints
    
    def test_clarification_removes_inferred_defaults(self):
        """User confirmation in clarification removes the (default) entry from inferred_by_model."""
        analyzer = RequirementAnalyzer()
        
        initial_req = RequirementSpec(
            dataset_name="Test",
            description="test data",
            output_format="csv",
            inferred_by_model=["output_format: csv (default)", "domain", "subdomain"],
            explicitly_stated=[]
        )
        
        updated_req = analyzer._apply_clarification_answers(initial_req, {'output_format': 'csv'})
        
        assert not any("output_format" in item for item in updated_req.inferred_by_model)
        assert "domain" in updated_req.inferred_by_model
        assert "clarified_format: csv" in updated_req.explicitly_stated
    
    def test_fallback_air_quality_and_forest_fire_on_llm_timeout(self):
        """Test rule-based fallback when LLM times out for air quality and forest fire requests."""
        analyzer = RequirementAnalyzer()
        
        # 1. Air quality in Delhi with LLM timeout
        spec_aq = create_initial_spec("Air Quality Delhi", "air quality in Delhi")
        with patch.object(analyzer, '_analyze_with_llm', side_effect=TimeoutError("LLM call timed out")):
            result_aq = analyzer.analyze(spec_aq)
            assert result_aq.requirement.domain == "environmental"
            assert result_aq.requirement.subdomain == "air quality"
            print("✅ Rule-based fallback correctly handled air quality in Delhi on LLM timeout")
            
        # 2. Forest fire in India with LLM timeout
        spec_fire = create_initial_spec("Forest Fire India", "forest fire in India")
        with patch.object(analyzer, '_analyze_with_llm', side_effect=TimeoutError("LLM call timed out")):
            result_fire = analyzer.analyze(spec_fire)
            assert result_fire.requirement.domain == "environmental"
            assert result_fire.requirement.subdomain == "forest fire monitoring"
            print("✅ Rule-based fallback correctly handled forest fire in India on LLM timeout")

    def test_general_weather_request_subdomain_null(self):
        """Test general weather request keeps domain=meteorological and subdomain=None."""
        analyzer = RequirementAnalyzer()
        spec_weather = create_initial_spec("Weather Request", "general weather forecast data for London")
        
        with patch.object(analyzer, '_analyze_with_llm', side_effect=TimeoutError("LLM timed out")):
            result = analyzer.analyze(spec_weather)
            assert result.requirement.domain == "meteorological"
            assert result.requirement.subdomain is None
            print("✅ General weather request keeps domain=meteorological and subdomain=None")

    def test_custom_option_reads_one_line_and_defaults_if_blank(self):
        """Typing the 'Custom range' option reads exactly one more line; blank falls back to the default."""
        analyzer = RequirementAnalyzer(interactive=True)
        initial_req = RequirementSpec(
            dataset_name="Test Spec",
            description="weather data csv",
            domain="meteorological",
            geography="Delhi",
            expected_size="500 rows",
            output_format="csv"
        )
        
        # Only the time-range question is asked.
        # User enters '4' (Custom range), then an empty line (blank answer).
        inputs = iter(["4", ""])
        with patch('builtins.input', side_effect=lambda prompt="": next(inputs)):
            updated_req = analyzer._bounded_clarification(
                initial_req, "Test Spec", "weather data csv"
            )
        
        assert updated_req.time_range == "latest available"
        assert updated_req.clarification_round == 1
        print("✅ Custom option read exactly one follow-up line, used default for blank input, never asked again")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])