"""
Requirement Analyzer Agent

Turns user's free-text request into structured requirement.
Focus: Real, current data about nature and environment (weather, air quality, 
fire, earthquakes, ocean, satellite-derived tabular data).
"""

import re
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel

from core.dataset_spec import DatasetSpec, RequirementSpec, DataModality
from core.llm_wrapper import get_llm

logger = logging.getLogger(__name__)


# LLM output schema
class LLMRequirementAnalysis(BaseModel):
    """Schema for LLM analysis output."""
    domain: str
    subdomain: Optional[str] = None  # Allow null for general/vague requests
    problem_type: str
    data_modality: str
    expected_features: List[str]  # Input measurements only (real, measurable)
    target_variable: Optional[str] = None  # For prediction: what to predict
    topic_description: str
    time_granularity: Optional[str] = None  # hourly, daily, monthly, etc.


class RequirementAnalyzer:
    """
    Requirement Analyzer Agent.
    
    Analyzes user's free-text request and fills RequirementSpec.
    Uses plain Python for extractables (dates, sizes, formats).
    Uses LLM only for topic understanding (domain, subdomain, features).
    
    Standard entry point: analyze(spec: DatasetSpec) -> DatasetSpec
    """
    
    # FIX 2: Supported domains in one config location
    SUPPORTED_DOMAINS = [
        'weather', 'climate', 'meteorological',
        'air quality', 'pollution', 'atmospheric',
        'fire', 'wildfire',
        'earthquake', 'seismic',
        'ocean', 'marine',
        'satellite', 'remote sensing',
        'temperature', 'rainfall', 'wind', 'humidity', 'pressure',
        'flood', 'drought', 'storm', 'cyclone', 'tsunami', 'volcano',
        'environmental', 'ecological', 'biodiversity',
        'forest', 'vegetation', 'glacier', 'ice', 'snow',
        'water quality', 'soil', 'land use',
        'landslide', 'geological hazards'
    ]
    
    # Clearly unsupported topics (instant rejection)
    INSTANTLY_REJECTED_KEYWORDS = {
        # Financial
        r'\bstock\b', r'\bstocks\b', r'\bprice\b', r'\bprices\b',
        r'\bmarket\b', r'\bmarkets\b', r'\bfinance\b', r'\bfinancial\b',
        r'\btrading\b', r'\bcrypto\b', r'\bcryptocurrency\b',
        # Social media
        r'\bsocial media\b', r'\btwitter\b', r'\bfacebook\b',
        r'\binstagram\b', r'\btiktok\b',
        # Entertainment
        r'\bmovie\b', r'\bfilm\b', r'\breview\b', r'\brating\b',
        # Commerce
        r'\bproduct\b', r'\bshopping\b', r'\becommerce\b', r'\bretail\b',
        # Medical
        r'\bmedical\b', r'\bpatient\b', r'\bhospital\b',
        # Private/restricted
        r'\bprivate\b', r'\blogin required\b', r'\bpaid\b',
        r'\bcredential\b', r'\bconfidential\b'
    }
    
    def __init__(self, interactive: bool = False):
        """
        Initialize analyzer.
        
        Args:
            interactive: Whether to prompt user for clarifications (default: False for tests/orchestrator)
        """
        self.llm = get_llm()
        self.interactive = interactive
    
    def analyze(self, spec: DatasetSpec) -> DatasetSpec:
        """
        Standard orchestrator entry point.
        
        Args:
            spec: DatasetSpec with user input in spec.requirement
            
        Returns:
            Updated DatasetSpec with filled RequirementSpec
        """
        print("🔍 Analyzing requirement...")
        start_time = time.time()
        
        # Extract user input
        if not spec.requirement:
            raise ValueError("spec.requirement must be set with dataset_name and description")
        
        dataset_name = spec.requirement.dataset_name
        description = spec.requirement.description
        
        # Initial analysis
        requirement = self._analyze_requirement(dataset_name, description)
        
        # Bounded clarification (ONE round max, if interactive)
        if self.interactive and requirement.is_supported and requirement.clarification_round == 0:
            requirement = self._bounded_clarification(requirement, dataset_name, description)
        
        # Update spec
        spec.requirement = requirement
        if requirement.target_requires_derivation:
            derivation_note = "target must be derived from real events or thresholds"
            if derivation_note not in spec.warnings:
                spec.add_warning(derivation_note)
        
        duration = time.time() - start_time
        print(f"✅ Requirement analyzed in {duration:.2f}s")
        logger.info(f"Requirement analysis completed in {duration:.2f}s")
        
        return spec
    
    def _analyze_requirement(self, dataset_name: str, description: str) -> RequirementSpec:
        """
        Main analysis logic.
        
        Args:
            dataset_name: Name of dataset
            description: User's free-text description
            
        Returns:
            Filled RequirementSpec
        """
        # FIX 1: Combine dataset_name and description for extraction
        combined = f"{dataset_name} {description}".lower()
        
        # Track what's explicit vs inferred
        explicitly_stated = []
        inferred_by_model = []
        missing_or_unclear = []
        clarifying_questions = []
        
        # 1. Extract simple fields with regex/rules
        output_format = self._extract_output_format(combined, explicitly_stated, inferred_by_model)
        expected_size = self._extract_size(combined, explicitly_stated, inferred_by_model)
        freshness_need, max_data_age = self._extract_freshness(
            combined, explicitly_stated, inferred_by_model, missing_or_unclear, clarifying_questions
        )
        time_range = self._extract_time_range(combined, explicitly_stated, inferred_by_model)
        geography = self._extract_geography(combined, explicitly_stated, inferred_by_model)
        constraints = self._extract_constraints(combined, explicitly_stated)
        
        # 2. Check if request is in scope (environmental/nature focus)
        is_supported, unsupported_reason = self._check_support(combined)
        
        if not is_supported:
            # Return early with unsupported flag
            return RequirementSpec(
                dataset_name=dataset_name,
                description=description,
                is_supported=False,
                unsupported_reason=unsupported_reason,
                output_format=output_format,
                explicitly_stated=explicitly_stated,
                inferred_by_model=inferred_by_model
            )
        
        # 3. Use LLM for topic understanding
        llm_result = self._analyze_with_llm(dataset_name, description, combined)
        
        # Map data modality
        data_modality = self._map_data_modality(llm_result.data_modality)
        
        # Build expected features (input measurements only)
        expected_features = llm_result.expected_features
        target_variable = llm_result.target_variable
        
        # Check if target variable requires derivation (not directly measurable)
        target_requires_derivation = False
        target_terms = ['risk', 'occurrence', 'category', 'index', 'severity', 'defined', 'derived', 'prediction']
        if target_variable and any(k in target_variable.lower() for k in target_terms):
            target_requires_derivation = True
        elif any(k in combined for k in ['flood risk', 'flood occurrence', 'fire risk', 'fire occurrence', 'landslide risk']):
            target_requires_derivation = True
        
        if target_requires_derivation:
            derivation_note = "target must be derived from real events or thresholds"
            if derivation_note not in missing_or_unclear:
                missing_or_unclear.append(derivation_note)
        
        # Add to tracking
        inferred_by_model.extend(['domain', 'subdomain', 'problem_type', 'data_modality'])
        if expected_features:
            inferred_by_model.append('expected_features')
        if target_variable:
            inferred_by_model.append('target_variable')
        
        # FIX 2: Add clarifying question about time granularity for long ranges
        if time_range and not llm_result.time_granularity:
            # Check if it's a multi-year range
            if 'to' in time_range and any(char.isdigit() for char in time_range):
                # Extract years if possible
                years = [int(s) for s in time_range.split() if s.isdigit() and len(s) == 4]
                if len(years) >= 2 and (years[-1] - years[0]) > 1:
                    clarifying_questions.append(
                        "What time granularity do you need? (e.g., hourly, daily, monthly)"
                    )
                    missing_or_unclear.append('time granularity')
        
        # FIX 3: Check if time range extends to present/future
        current_year = 2026  # Based on system date
        if time_range and str(current_year) in time_range:
            # Time range includes current year or future
            constraints.append('must cover up to latest available date')
            explicitly_stated.append('constraint: must cover up to latest available date')
        
        # 4. Check for missing critical information
        if not geography and 'location' not in missing_or_unclear:
            missing_or_unclear.append('geographic area')
            clarifying_questions.append("Which geographic area or location do you need?")
        
        if not expected_size:
            missing_or_unclear.append('dataset size')
        
        # 5. Calculate completeness
        completeness = self._calculate_confidence(
            explicitly_stated, missing_or_unclear, is_supported
        )
        
        # Build RequirementSpec
        return RequirementSpec(
            dataset_name=dataset_name,
            description=llm_result.topic_description or description,
            domain=llm_result.domain,
            subdomain=llm_result.subdomain,
            problem_type=llm_result.problem_type,
            data_modality=data_modality,
            expected_features=expected_features,
            target_variable=target_variable,
            target_requires_derivation=target_requires_derivation,
            expected_size=expected_size,
            freshness_need=freshness_need,
            max_data_age=max_data_age,
            geography=geography,
            time_range=time_range,
            output_format=output_format,
            constraints=constraints,
            is_supported=is_supported,
            unsupported_reason=unsupported_reason,
            explicitly_stated=explicitly_stated,
            inferred_by_model=inferred_by_model,
            missing_or_unclear=missing_or_unclear,
            clarifying_questions=clarifying_questions,
            completeness=completeness
        )
    
    def _extract_output_format(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> str:
        """Extract output format from text."""
        if 'excel' in text or '.xlsx' in text or '.xls' in text:
            explicitly_stated.append('output_format: excel')
            return 'excel'
        elif 'json' in text:
            explicitly_stated.append('output_format: json')
            return 'json'
        elif 'csv' in text:
            explicitly_stated.append('output_format: csv')
            return 'csv'
        else:
            inferred.append('output_format: csv (default)')
            return 'csv'
    
    def _extract_size(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> str:
        """Extract expected dataset size."""
        # Look for patterns like "5000 rows", "1000 samples", "10k records"
        patterns = [
            r'(\d+k?)\s*(rows|samples|records|entries|observations)',
            r'about\s+(\d+k?)\s+',
            r'(\d+k?)\s+rows'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                size = match.group(1)
                explicitly_stated.append(f'dataset_size: {size}')
                return f"{size} rows"
        
        return None
    
    def _extract_freshness(
        self, text: str, explicitly_stated: List[str], inferred: List[str],
        missing: List[str], questions: List[str]
    ) -> Tuple[str, str]:
        """
        Extract freshness need and max data age.
        
        max_data_age = how old the NEWEST record may be
        - For "live": infer "1 hour" (data must be very fresh)
        - For "recent": infer "24 hours" (data should be recent)
        - For historical/unspecified: leave empty
        
        Note: Time windows like "last 7 days" go to time_range, not max_data_age
        """
        freshness = "unspecified"
        max_age = None
        
        # Check for live/real-time using word boundaries
        live_patterns = [r'\blive\b', r'\breal-time\b', r'\brealtime\b', r'\breal time\b']
        if any(re.search(pattern, text) for pattern in live_patterns):
            freshness = "live"
            explicitly_stated.append('freshness: live')
            # For live, infer max_data_age if not specified
            max_age = "1 hour"
            inferred.append('max_data_age: 1 hour (inferred for live data)')
        # Check for recent/current/latest
        elif any(re.search(rf'\b{word}\b', text) for word in ['recent', 'current', 'latest', 'now']):
            freshness = "recent"
            explicitly_stated.append('freshness: recent')
            # For recent, infer max_data_age if not specified
            max_age = "24 hours"
            inferred.append('max_data_age: 24 hours (inferred for recent data)')
            missing.append('specific time window')
            questions.append("How recent? (e.g., last 24 hours, last 7 days)")
        # Check for historical/past/archive
        elif any(re.search(rf'\b{word}\b', text) for word in ['historical', 'past', 'archive']):
            freshness = "historical"
            explicitly_stated.append('freshness: historical')
        # Check if there's a year range (like 2010 to 2020) - implies historical
        elif 'from' in text and 'to' in text:
            if re.search(r'from\s+\d{4}\s+to\s+\d{4}', text):
                freshness = "historical"
                explicitly_stated.append('freshness: historical (inferred from year range)')
        
        if freshness == "unspecified":
            inferred.append('freshness: unspecified (not specified)')
        
        return freshness, max_age
    
    def _extract_time_range(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> str:
        """
        Extract time range (history window).
        
        Examples:
        - "last 7 days" → time_range
        - "from 2010 to 2020" → time_range  
        - "this month" → time_range
        - "past 30 days" → time_range
        """
        # Time window patterns (last X days/weeks/months/years)
        window_patterns = [
            r'last\s+(\d+)\s+(hour|day|week|month|year)s?',
            r'past\s+(\d+)\s+(hour|day|week|month|year)s?',
            r'previous\s+(\d+)\s+(hour|day|week|month|year)s?',
        ]
        
        for pattern in window_patterns:
            match = re.search(pattern, text)
            if match:
                num = match.group(1)
                unit = match.group(2)
                time_range = f"last {num} {unit}s" if int(num) > 1 else f"last {num} {unit}"
                explicitly_stated.append(f'time_range: {time_range}')
                return time_range
        
        # Year range patterns
        year_patterns = [
            r'from\s+(\d{4})\s+to\s+(\d{4})',
            r'(\d{4})\s*-\s*(\d{4})',
            r'(\d{4})\s+to\s+(\d{4})',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_range = f"{match.group(1)} to {match.group(2)}"
                explicitly_stated.append(f'time_range: {time_range}')
                return time_range
        
        # Relative patterns
        relative_patterns = [
            (r'this\s+(week|month|year)', lambda m: f"this {m.group(1)}"),
            (r'last\s+(week|month|year)', lambda m: f"last {m.group(1)}"),
            (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+to\s+(january|february|march|april|may|june|july|august|september|october|november|december)', lambda m: f"{m.group(1)} to {m.group(2)}"),
        ]
        
        for pattern, formatter in relative_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_range = formatter(match)
                explicitly_stated.append(f'time_range: {time_range}')
                return time_range
        
        return None
    
    def _extract_geography(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> str:
        """Extract geographic area."""
        # Common patterns
        geo_patterns = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "in Delhi", "in New York"
            r'for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "for India"
            r'of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "of Rishikesh"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+region',  # "Kerala region"
        ]
        
        # Extended list of locations (countries, Indian states, major cities)
        locations = [
            # India and states
            'delhi', 'india', 'kerala', 'mumbai', 'bangalore', 'chennai', 'kolkata',
            'rajasthan', 'maharashtra', 'karnataka', 'tamil nadu', 'gujarat', 
            'west bengal', 'uttar pradesh', 'punjab', 'haryana', 'uttarakhand',
            'rishikesh', 'haridwar', 'dehradun', 'nainital', 'mussoorie',
            # Global
            'usa', 'china', 'europe', 'asia', 'africa', 'worldwide', 'global',
            'australia', 'canada', 'brazil', 'japan', 'uk', 'russia'
        ]
        
        for loc in locations:
            if loc in text:
                geography = loc.title()
                explicitly_stated.append(f'geography: {geography}')
                return geography
        
        # Try patterns
        for pattern in geo_patterns:
            match = re.search(pattern, text)
            if match:
                geography = match.group(1)
                explicitly_stated.append(f'geography: {geography}')
                return geography
        
        return None
    
    def _extract_constraints(self, text: str, explicitly_stated: List[str]) -> List[str]:
        """Extract constraints."""
        constraints = []
        
        if 'free' in text and ('source' in text or 'data' in text):
            constraints.append('free sources only')
            explicitly_stated.append('constraint: free sources')
        
        if 'no login' in text or 'without login' in text:
            constraints.append('no login required')
            explicitly_stated.append('constraint: no login')
        
        if 'public' in text and 'data' in text:
            constraints.append('public data only')
            explicitly_stated.append('constraint: public data')
        
        # Satellite/remote sensing data
        satellite_keywords = ['satellite', 'satellites', 'remote sensing', 'imagery-derived', 
                             'satellite-derived', 'orbital', 'space-based']
        if any(keyword in text for keyword in satellite_keywords):
            constraints.append('satellite-derived data')
            explicitly_stated.append('constraint: satellite-derived data')
        
        return constraints
    
    def _check_support(self, text: str) -> Tuple[bool, str]:
        """
        Check if request is in scope.
        
        FIX 2: Two-tier checking:
        1. Instant rejection for clearly unsupported (finance, social, private)
        2. LLM check for ambiguous cases (no env keywords, but not clearly rejected)
        
        Returns:
            (is_supported, unsupported_reason)
        """
        # Check for instant rejection keywords
        for pattern in self.INSTANTLY_REJECTED_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                # Special case: allow "livestock" even though it has "stock"
                if 'livestock' in text and pattern in [r'\bstock\b', r'\bstocks\b']:
                    continue
                
                return False, (
                    f"Not supported yet. "
                    f"Currently supported: {', '.join(sorted(set([d.split()[0] for d in self.SUPPORTED_DOMAINS[:10]])))}, "
                    f"and other environmental/nature data."
                )
        
        # Check for environmental keywords
        has_env_keyword = any(keyword in text for keyword in self.SUPPORTED_DOMAINS)
        
        if has_env_keyword:
            return True, None
        
        # FIX 2b: No env keywords and not instantly rejected -> Ask LLM
        logger.info("No environmental keywords found. Asking LLM to check scope...")
        try:
            return self._check_support_with_llm(text)
        except Exception as e:
            logger.error(f"LLM scope check failed: {e}. Defaulting to rejection.")
            return False, (
                f"Not supported yet. "
                f"Currently supported: weather, climate, air quality, natural disasters, "
                f"water/ocean data, forests, glaciers, and other environmental/nature data."
            )
    
    def _check_support_with_llm(self, text: str) -> Tuple[bool, str]:
        """
        Use LLM to determine if ambiguous request is related to supported domains.
        
        Args:
            text: The request text
            
        Returns:
            (is_supported, unsupported_reason)
        """
        prompt = f"""Is this dataset request about NATURAL environmental or earth science phenomena?

Request: "{text}"

SUPPORTED domains (natural/environmental):
- Weather, climate, atmospheric conditions (rain, wind, temperature)
- Natural disasters: earthquakes, volcanic eruptions, tsunamis, floods, wildfires, landslides, droughts
- Water bodies: rivers, oceans, lakes, water quality
- Land features: forests, vegetation, soil, glaciers, snow, ice
- Satellite/remote sensing of NATURAL features (not human activities)
- Ecological data: biodiversity, wildlife, ecosystems

NOT SUPPORTED (human-focused):
- Human safety/accidents (traffic, workplace, industrial accidents)
- Human infrastructure (buildings, roads, urban planning)
- Human health/medical data
- Economic/financial data
- Social data

Key question: Is the request about NATURAL phenomena or about HUMAN activities/safety?

Answer in JSON with:
1. "is_related": true ONLY if clearly about natural/environmental phenomena, false otherwise
2. "confidence": "high", "medium", or "low"
3. "reasoning": one sentence explaining

Examples:
- "accident risk prediction" -> {{"is_related": false, "confidence": "high", "reasoning": "Accidents are human safety events, not natural environmental phenomena"}}
- "glacier retreat" -> {{"is_related": true, "confidence": "high", "reasoning": "Glaciers are natural earth features studied in climate science"}}
- "landslide events" -> {{"is_related": true, "confidence": "high", "reasoning": "Landslides are natural geological hazards"}}
- "traffic accidents" -> {{"is_related": false, "confidence": "high", "reasoning": "Traffic accidents are human safety, not environmental data"}}

Return JSON only:"""
        
        try:
            response = self.llm.call(
                prompt=prompt,
                expected_schema=None,
                force_json=True
            )
            
            result = response.content
            is_related = result.get('is_related', False)
            confidence = result.get('confidence', 'low')
            reasoning = result.get('reasoning', '')
            
            logger.info(f"LLM scope check: is_related={is_related}, confidence={confidence}")
            logger.info(f"LLM reasoning: {reasoning}")
            
            if is_related:
                return True, None
            else:
                return False, (
                    f"Not supported yet. "
                    f"Currently supported: weather, climate, air quality, natural disasters "
                    f"(earthquakes, floods, fires, landslides), water/ocean data, glaciers, "
                    f"forests, vegetation, satellite/remote sensing of nature, and other "
                    f"environmental data. {reasoning}"
                )
                
        except Exception as e:
            logger.error(f"LLM scope check failed: {e}")
            # On error, reject conservatively but mention it's uncertain
            return False, (
                f"Could not determine if this request is supported. "
                f"Currently supported: environmental and nature data (weather, climate, "
                f"natural disasters, water, land, forests, etc.)."
            )
    
    def _analyze_with_llm(self, dataset_name: str, description: str, combined: str) -> LLMRequirementAnalysis:
        """Use LLM for topic understanding."""
        prompt = f"""Analyze this dataset request and return ONLY JSON:

Dataset: {dataset_name}
Description: {description}

Determine:
1. domain: broad category (e.g., "environmental", "meteorological", "seismic")
2. subdomain: specific area if clearly stated (e.g., "air quality", "earthquakes", "flood monitoring"). Return null if not specific or just general weather/environmental data.
3. problem_type: what will be done with data (e.g., "monitoring", "prediction", "analysis")
4. data_modality: "tabular", "text", or "time_series"
5. expected_features: list of 3-6 INPUT MEASUREMENTS that real public data sources actually record
   - ONLY suggest directly measurable variables (rainfall, temperature, river discharge, water level, soil moisture, wind speed, etc.)
   - DO NOT invent computed indices or risk scores
   - DO NOT list year/month/day/hour separately - use "timestamp" for temporal data
   - Return EMPTY LIST if you cannot suggest realistic measurements
6. target_variable: for prediction problems, what is being predicted (e.g., "flood occurrence", "air quality category"). Set to null for monitoring/analysis problems or if the target cannot be directly measured (then note "to be defined from real data").
7. time_granularity: if a multi-year range is mentioned, suggest appropriate granularity (hourly/daily/monthly). Otherwise null.
8. topic_description: clear 1-sentence description

Example response for prediction:
{{
  "domain": "environmental",
  "subdomain": "flood monitoring",
  "problem_type": "prediction",
  "data_modality": "time_series",
  "expected_features": ["timestamp", "rainfall", "river_water_level", "discharge", "temperature"],
  "target_variable": "to be defined from real data",
  "time_granularity": "daily",
  "topic_description": "Flood risk prediction based on meteorological and hydrological measurements"
}}

Example for monitoring:
{{
  "domain": "environmental",
  "subdomain": "air quality",
  "problem_type": "monitoring",
  "data_modality": "time_series",
  "expected_features": ["timestamp", "pm2.5", "pm10", "aqi", "location"],
  "target_variable": null,
  "time_granularity": null,
  "topic_description": "Real-time air quality measurements including PM2.5 and AQI values"
}}

For vague requests like "some weather data", return subdomain as null.

Return JSON only:"""
        
        try:
            response = self.llm.call(
                prompt=prompt,
                expected_schema=LLMRequirementAnalysis
            )
            
            result = LLMRequirementAnalysis(**response.content)
            
            # Clean up subdomain - replace "general", "weather" or empty with None
            if result.subdomain and result.subdomain.lower() in ['general', 'unspecified', 'various', 'weather']:
                result.subdomain = None
            
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed or timed out: {e}. Using rule-based fallback.")
            print(f"⚠️  LLM analysis unavailable: {e}. Using rule-based fallback.")
            return self._extract_topic_with_rules(combined, description)
    
    def _extract_topic_with_rules(self, text: str, description: str) -> LLMRequirementAnalysis:
        """
        Rule-based topic understanding when LLM call fails or times out.
        Reuses SUPPORTED_DOMAINS with whole-word regex matching.
        """
        text_lower = text.lower()
        
        domain = None
        subdomain = None
        
        def has_word(kw: str) -> bool:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            return bool(re.search(pattern, text_lower))
        
        # Whole-word domain & subdomain mapping using SUPPORTED_DOMAINS
        if any(has_word(kw) for kw in ['air quality', 'pollution', 'water quality']):
            domain = "environmental"
            subdomain = "air quality" if (has_word('air quality') or has_word('pollution')) else "water quality"
        elif any(has_word(kw) for kw in ['fire', 'wildfire', 'forest']):
            domain = "environmental"
            subdomain = "forest fire monitoring"
        elif any(has_word(kw) for kw in ['earthquake', 'seismic', 'volcano', 'tsunami', 'landslide', 'geological hazards']):
            domain = "geological"
            subdomain = "earthquakes" if (has_word('earthquake') or has_word('seismic')) else "natural disasters"
        elif has_word('flood'):
            domain = "environmental"
            subdomain = "flood monitoring"
        elif any(has_word(kw) for kw in ['ocean', 'marine']):
            domain = "oceanographic"
            subdomain = "ocean data"
        elif any(has_word(kw) for kw in ['satellite', 'remote sensing']):
            domain = "environmental"
            subdomain = "satellite data"
        elif any(has_word(kw) for kw in ['weather', 'climate', 'meteorological', 'temperature', 'rainfall', 'wind', 'humidity', 'pressure', 'storm', 'cyclone', 'drought']):
            domain = "meteorological"
            subdomain = None  # General weather keeps domain=meteorological and leave subdomain empty (null)
        elif any(has_word(kw) for kw in self.SUPPORTED_DOMAINS):
            domain = "environmental"
            subdomain = None
        
        # Problem type whole-word matching
        problem_type = "analysis"
        if any(has_word(kw) for kw in ['predict', 'prediction', 'forecast', 'forecasting', 'model', 'modeling']):
            problem_type = "prediction"
        elif any(has_word(kw) for kw in ['monitor', 'monitoring', 'real-time', 'live', 'track', 'tracking']):
            problem_type = "monitoring"
            
        data_modality = "tabular"
        if any(has_word(kw) for kw in ['time', 'daily', 'hourly', 'live', 'recent', 'series']):
            data_modality = "time_series"
            
        return LLMRequirementAnalysis(
            domain=domain,
            subdomain=subdomain,
            problem_type=problem_type,
            data_modality=data_modality,
            expected_features=[],
            topic_description=description
        )
    
    def _map_data_modality(self, llm_modality: str) -> DataModality:
        """Map LLM modality string to DataModality enum."""
        modality_map = {
            'tabular': DataModality.TABULAR,
            'text': DataModality.TEXT,
            'time_series': DataModality.TIME_SERIES,
            'time-series': DataModality.TIME_SERIES,
            'timeseries': DataModality.TIME_SERIES,
        }
        
        return modality_map.get(llm_modality.lower().replace(' ', '_'), DataModality.TABULAR)
    
    def _calculate_confidence(
        self, explicitly_stated: List[str], missing: List[str], is_supported: bool
    ) -> float:
        """
        Calculate completeness score (how much detail the user provided).
        
        This measures request completeness, NOT model accuracy.
        
        Formula:
        - Base: 0.50 (has identifiable domain)
        - 4+ explicit: +0.30 → 0.75-0.80 (clear, detailed request)
        - 3 explicit: +0.20 → 0.65-0.70 (good detail)
        - 2 explicit: +0.10 → 0.50-0.60 (some detail)
        - 1 explicit: +0.05 → 0.45-0.55 (minimal detail)
        - 0 explicit: -0.30 → 0.10-0.20 (very vague)
        - Each missing: -0.05 (small penalty for gaps)
        
        Args:
            explicitly_stated: List of explicitly stated items
            missing: List of missing items
            is_supported: Whether request is supported
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not is_supported:
            return 0.0
        
        # Start with base (request is understandable)
        score = 0.50
        
        # Increase for explicit information
        num_explicit = len(explicitly_stated)
        if num_explicit >= 4:
            score += 0.30  # Clear, detailed
        elif num_explicit >= 3:
            score += 0.20  # Good detail
        elif num_explicit >= 2:
            score += 0.10  # Some detail  
        elif num_explicit >= 1:
            score += 0.05  # Minimal detail
        else:
            score -= 0.30  # Very vague
        
        # Small penalty for missing non-critical information
        score -= len(missing) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _bounded_clarification(
        self, 
        requirement: RequirementSpec, 
        dataset_name: str, 
        description: str
    ) -> RequirementSpec:
        """
        Ask user for missing critical information (ONE round, max 4 questions).
        
        Priority:
        1. Topic/target if unclear
        2. Location
        3. Time range or freshness (+ granularity for long ranges)
        4. Dataset size
        5. Features
        6. Output format (only if nothing else)
        
        Args:
            requirement: Initial requirement analysis
            dataset_name: Original dataset name
            description: Original description
            
        Returns:
            Updated RequirementSpec with user answers
        """
        combined = f"{dataset_name} {description}".lower()
        
        # Collect questions (max 4, prioritized)
        questions = []
        
        # Priority 1: Topic/target unclear
        if requirement.completeness < 0.3 or not requirement.domain:
            questions.append({
                'field': 'topic',
                'prompt': 'What topic or phenomenon do you want data about?',
                'default': 'environmental data',
                'type': 'text'
            })
        
        # Priority 2: Location
        if not requirement.geography:
            questions.append({
                'field': 'geography',
                'prompt': 'Which location or geographic area?',
                'default': 'global',
                'type': 'text'
            })
        
        # Priority 3a: Time range (if not stated)
        if not requirement.time_range and requirement.freshness_need == "unspecified":
            questions.append({
                'field': 'time_range',
                'prompt': 'What time period do you need?',
                'default': 'latest available',
                'choices': [
                    ('1', 'Last 7 days'),
                    ('2', 'Last month'),
                    ('3', 'Last year'),
                    ('4', 'Custom range')
                ],
                'type': 'choice'
            })
        
        # Priority 3b: Time granularity (if range is multi-year and no granularity given)
        if requirement.time_range and 'to' in requirement.time_range:
            years = [int(s) for s in requirement.time_range.split() if s.isdigit() and len(s) == 4]
            if len(years) >= 2 and (years[-1] - years[0]) > 1:
                questions.append({
                    'field': 'time_granularity',
                    'prompt': f'Time granularity for {requirement.time_range}?',
                    'default': 'daily',
                    'choices': [
                        ('1', 'Hourly'),
                        ('2', 'Daily'),
                        ('3', 'Monthly'),
                        ('4', 'Yearly')
                    ],
                    'type': 'choice'
                })
        
        # Priority 4: Dataset size
        if not requirement.expected_size and len(questions) < 4:
            questions.append({
                'field': 'expected_size',
                'prompt': 'How many rows/records do you need?',
                'default': '1000 rows',
                'choices': [
                    ('1', '1,000 rows'),
                    ('2', '5,000 rows'),
                    ('3', '10,000 rows'),
                    ('4', 'Custom amount')
                ],
                'type': 'choice'
            })
        
        # Priority 5: Output format (only if < 4 questions and format not specified)
        if len(questions) < 4 and requirement.output_format == 'csv' and 'csv' not in combined:
            questions.append({
                'field': 'output_format',
                'prompt': 'Preferred output format?',
                'default': 'csv',
                'choices': [
                    ('1', 'CSV'),
                    ('2', 'Excel'),
                    ('3', 'JSON')
                ],
                'type': 'choice'
            })
        
        # Limit to 4 questions
        questions = questions[:4]
        
        # If no questions, return as-is
        if not questions:
            return requirement
        
        # Ask questions
        print("\n" + "="*70)
        print("📋 A few quick questions to refine your requirements")
        print("="*70)
        print("(Press Enter to use default, or type 'skip' to accept all defaults)\n")
        
        answers = {}
        skip_all = False
        
        for i, q in enumerate(questions, 1):
            if skip_all:
                answers[q['field']] = q['default']
                continue
            
            # Display question
            print(f"\n{i}. {q['prompt']}")
            
            if q['type'] == 'choice':
                for choice_num, choice_text in q['choices']:
                    print(f"   {choice_num}) {choice_text}")
            
            print(f"   Default: {q['default']}")
            
            # Get answer
            try:
                answer = input(f"   Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            
            if answer.lower() == 'skip':
                skip_all = True
                answers[q['field']] = q['default']
                print("   → Skipping remaining questions, using defaults")
                continue
            
            if not answer:
                answers[q['field']] = q['default']
                print(f"   → Using default: {q['default']}")
            elif q['type'] == 'choice':
                # Handle choice
                choice_map = {num: text for num, text in q['choices']}
                if answer in choice_map:
                    if answer == '4':  # Custom option
                        try:
                            custom = input(f"   Enter {q['field']}: ").strip()
                            answers[q['field']] = custom if custom else q['default']
                        except (EOFError, KeyboardInterrupt):
                            answers[q['field']] = q['default']
                    else:
                        answers[q['field']] = choice_map[answer]
                else:
                    print(f"   → Invalid choice, using default: {q['default']}")
                    answers[q['field']] = q['default']
            else:
                answers[q['field']] = answer
        
        # Update requirement with answers
        requirement = self._apply_clarification_answers(requirement, answers)
        requirement.clarification_round = 1
        
        # Show confirmation
        print("\n" + "="*70)
        print("✅ Requirements confirmed")
        print("="*70)
        self._print_confirmation_summary(requirement)
        
        return requirement
    
    def _apply_clarification_answers(
        self, 
        requirement: RequirementSpec, 
        answers: Dict[str, str]
    ) -> RequirementSpec:
        """Apply user answers to requirement spec."""
        
        for field, value in answers.items():
            if field == 'topic':
                # Use LLM to understand the topic
                if value and value != 'environmental data':
                    try:
                        llm_result = self._analyze_with_llm("Topic", value, value.lower())
                        requirement.domain = llm_result.domain
                        requirement.subdomain = llm_result.subdomain
                        requirement.problem_type = llm_result.problem_type
                        requirement.explicitly_stated.append(f'clarified_topic: {value}')
                    except:
                        pass
            
            elif field == 'geography':
                requirement.geography = value.title()
                requirement.explicitly_stated.append(f'clarified_geography: {value}')
                # Remove location from missing
                if 'geographic area' in requirement.missing_or_unclear:
                    requirement.missing_or_unclear.remove('geographic area')
                requirement.clarifying_questions = [
                    q for q in requirement.clarifying_questions 
                    if 'location' not in q.lower() and 'geographic' not in q.lower()
                ]
            
            elif field == 'time_range':
                requirement.time_range = value
                requirement.explicitly_stated.append(f'clarified_time_range: {value}')
            
            elif field == 'time_granularity':
                # Store in description or as a note
                granularity_note = f"Time granularity: {value}"
                if granularity_note not in requirement.explicitly_stated:
                    requirement.explicitly_stated.append(f'clarified_granularity: {value}')
                # Remove from missing
                if 'time granularity' in requirement.missing_or_unclear:
                    requirement.missing_or_unclear.remove('time granularity')
            
            elif field == 'expected_size':
                requirement.expected_size = value
                requirement.explicitly_stated.append(f'clarified_size: {value}')
                if 'dataset size' in requirement.missing_or_unclear:
                    requirement.missing_or_unclear.remove('dataset size')
            
            elif field == 'output_format':
                format_map = {'CSV': 'csv', 'Excel': 'excel', 'JSON': 'json'}
                requirement.output_format = format_map.get(value, value.lower())
                requirement.explicitly_stated.append(f'clarified_format: {requirement.output_format}')
        
        # Deduplicate inferred_by_model defaults if user confirmed/clarified values
        if requirement.inferred_by_model:
            clean_inferred = []
            for item in requirement.inferred_by_model:
                item_lower = item.lower()
                should_remove = False
                for field in answers:
                    field_lower = field.lower()
                    if field_lower == 'output_format' and 'output_format' in item_lower:
                        should_remove = True
                    elif field_lower == 'expected_size' and ('size' in item_lower or 'dataset_size' in item_lower):
                        should_remove = True
                    elif field_lower == 'geography' and ('geography' in item_lower or 'location' in item_lower):
                        should_remove = True
                    elif field_lower == 'time_range' and ('time_range' in item_lower or 'freshness' in item_lower or 'max_data_age' in item_lower):
                        should_remove = True
                    elif field_lower == 'time_granularity' and ('granularity' in item_lower or 'time' in item_lower):
                        should_remove = True
                    elif field_lower == 'topic' and any(k in item_lower for k in ['domain', 'subdomain', 'topic', 'problem_type']):
                        should_remove = True
                if not should_remove:
                    clean_inferred.append(item)
            requirement.inferred_by_model = clean_inferred
        
        # Recalculate completeness
        requirement.completeness = self._calculate_confidence(
            requirement.explicitly_stated,
            requirement.missing_or_unclear,
            requirement.is_supported
        )
        
        return requirement
    
    def _print_confirmation_summary(self, requirement: RequirementSpec):
        """Print a summary of what was understood."""
        print(f"\n📊 What we understood:")
        if requirement.domain:
            print(f"   • Domain: {requirement.domain}")
        if requirement.subdomain:
            print(f"   • Subdomain: {requirement.subdomain}")
        if requirement.geography:
            print(f"   • Location: {requirement.geography}")
        if requirement.time_range:
            print(f"   • Time range: {requirement.time_range}")
        if requirement.expected_size:
            print(f"   • Size: {requirement.expected_size}")
        if requirement.output_format:
            print(f"   • Format: {requirement.output_format}")
        
        # Show what's still missing (using defaults)
        if requirement.missing_or_unclear:
            print(f"\n📝 Using defaults for:")
            for item in requirement.missing_or_unclear:
                print(f"   • {item}")
        
        print(f"\n✨ Completeness: {requirement.completeness:.0%}")
        print()
