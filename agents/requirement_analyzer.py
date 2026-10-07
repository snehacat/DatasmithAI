"""
Requirement Analyzer Agent

Turns the user's free-text request into a structured requirement.

PILOT DOMAIN: real, current data about nature and environment (weather, air
quality, fire, earthquakes, ocean, satellite-derived tabular data).
The pilot domain is controlled by the config constants right below the imports
(PILOT_DOMAIN_LABEL, PILOT_DOMAIN_KEYWORDS, REJECTED_* lists). To widen the
scope later, change those constants, nothing else.
"""

import re
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

from core.dataset_spec import DatasetSpec, RequirementSpec, DataModality, PipelineStatus
from core.llm_wrapper import get_llm

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG: pilot domain ( what is in scope)
# ============================================================================

PILOT_DOMAIN_LABEL = "environmental and natural data"

# A request is in scope if it contains at least one of these WHOLE words/phrases
# (plural -s/-es is also accepted). Matching is whole-word, so 'ice' does not
# match 'price' or 'police'.
PILOT_DOMAIN_KEYWORDS = [
    # weather / climate / atmosphere
    'weather', 'climate', 'meteorological', 'meteorology', 'atmospheric', 'atmosphere',
    'temperature', 'rain', 'rainfall', 'precipitation', 'monsoon', 'wind', 'humidity',
    'atmospheric pressure', 'air pressure', 'barometric pressure',
    'snow', 'snowfall', 'ice', 'heatwave', 'heat wave', 'cold wave', 'lightning',
    'solar radiation', 'sunshine',
    # air quality / pollution / gases
    'air quality', 'pollution', 'pollutant', 'pm2.5', 'pm10', 'aqi', 'smog', 'dust',
    'ozone', 'carbon dioxide', 'co2', 'methane', 'greenhouse gas', 'emissions',
    # fire
    'fire', 'wildfire', 'forest fire',
    # earth / geology / hazards
    'earthquake', 'seismic', 'tsunami', 'volcano', 'volcanic', 'landslide', 'avalanche',
    'geological', 'geological hazards', 'erosion',
    'flood', 'drought', 'storm', 'thunderstorm', 'cyclone', 'hurricane', 'typhoon', 'tornado',
    # water / ocean
    'ocean', 'marine', 'sea', 'sea level', 'sea surface', 'coral', 'river', 'lake',
    'groundwater', 'water level', 'water quality', 'hydrological', 'hydrology', 'wetland',
    'mangrove',
    # land / vegetation / ecology
    'forest', 'vegetation', 'ndvi', 'deforestation', 'land use', 'land cover', 'land surface',
    'soil', 'soil moisture', 'glacier', 'glacial', 'permafrost',
    'environment', 'environmental', 'ecological', 'ecosystem', 'biodiversity', 'wildlife',
    'habitat', 'nature',
    # remote sensing
    'satellite', 'remote sensing', 'earth observation', 'modis', 'viirs', 'sentinel',
    'landsat', 'era5',
]

# Clearly out-of-scope topics. Rejected immediately (no LLM call) even if an
# environmental word is also present, e.g. "air pollution impact on stock market".
# Kept short on purpose: anything with NO environmental keyword is rejected anyway.
REJECTED_TOPIC_WORDS = [
    'stock market', 'stock price', 'stock exchange', 'stocks', 'share price',
    'finance', 'financial', 'trading', 'crypto', 'cryptocurrency',
    'social media', 'twitter', 'facebook', 'instagram', 'tiktok',
    'movie', 'film', 'shopping', 'ecommerce', 'e-commerce', 'retail',
    'patient', 'hospital', 'blood pressure', 'medical record',
]

# Access restrictions we cannot work with. These are skipped when the user
# NEGATES them ("no paid data", "without login required").
REJECTED_ACCESS_WORDS = [
    'private', 'paid', 'credential', 'confidential', 'paywall',
    'login required', 'requires login', 'requiring login', 'password protected',
]
_NEGATORS = {'no', 'not', 'without', 'never', 'non', 'avoid', 'exclude', 'excluding'}
ALLOWED_DOMAINS = {"environmental", "meteorological", "geological", "oceanographic", "hydrological"}

# Locations the rule-based extractor knows (lowercase -> display name).
# Unknown places are NOT guessed; the user is asked instead.
LOCATIONS = {
    # India: country, states, UTs
    'india': 'India', 'andhra pradesh': 'Andhra Pradesh', 'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam', 'bihar': 'Bihar', 'chhattisgarh': 'Chhattisgarh', 'goa': 'Goa',
    'gujarat': 'Gujarat', 'haryana': 'Haryana', 'himachal pradesh': 'Himachal Pradesh',
    'jharkhand': 'Jharkhand', 'karnataka': 'Karnataka', 'kerala': 'Kerala',
    'madhya pradesh': 'Madhya Pradesh', 'maharashtra': 'Maharashtra', 'manipur': 'Manipur',
    'meghalaya': 'Meghalaya', 'mizoram': 'Mizoram', 'nagaland': 'Nagaland', 'odisha': 'Odisha',
    'punjab': 'Punjab', 'rajasthan': 'Rajasthan', 'sikkim': 'Sikkim', 'tamil nadu': 'Tamil Nadu',
    'telangana': 'Telangana', 'tripura': 'Tripura', 'uttar pradesh': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand', 'west bengal': 'West Bengal',
    'jammu and kashmir': 'Jammu and Kashmir', 'ladakh': 'Ladakh',
    # India: cities / places
    'delhi': 'Delhi', 'new delhi': 'New Delhi', 'mumbai': 'Mumbai', 'bangalore': 'Bangalore',
    'bengaluru': 'Bengaluru', 'chennai': 'Chennai', 'kolkata': 'Kolkata', 'hyderabad': 'Hyderabad',
    'pune': 'Pune', 'ahmedabad': 'Ahmedabad', 'jaipur': 'Jaipur', 'lucknow': 'Lucknow',
    'patna': 'Patna', 'bhopal': 'Bhopal', 'chandigarh': 'Chandigarh', 'shimla': 'Shimla',
    'srinagar': 'Srinagar', 'guwahati': 'Guwahati', 'agra': 'Agra', 'varanasi': 'Varanasi',
    'rishikesh': 'Rishikesh', 'haridwar': 'Haridwar', 'dehradun': 'Dehradun',
    'nainital': 'Nainital', 'mussoorie': 'Mussoorie', 'himalayas': 'Himalayas',
    # Countries / regions
    'usa': 'USA', 'united states': 'United States', 'uk': 'UK', 'united kingdom': 'United Kingdom',
    'china': 'China', 'japan': 'Japan', 'russia': 'Russia', 'australia': 'Australia',
    'canada': 'Canada', 'brazil': 'Brazil', 'pakistan': 'Pakistan', 'nepal': 'Nepal',
    'bangladesh': 'Bangladesh', 'sri lanka': 'Sri Lanka', 'bhutan': 'Bhutan',
    'indonesia': 'Indonesia', 'germany': 'Germany', 'france': 'France', 'italy': 'Italy',
    'spain': 'Spain', 'mexico': 'Mexico', 'south africa': 'South Africa', 'egypt': 'Egypt',
    'europe': 'Europe', 'asia': 'Asia', 'africa': 'Africa', 'antarctica': 'Antarctica',
    'arctic': 'Arctic',
    # Cities abroad
    'london': 'London', 'paris': 'Paris', 'new york': 'New York', 'tokyo': 'Tokyo',
    'beijing': 'Beijing', 'sydney': 'Sydney',
    # Water bodies
    'indian ocean': 'Indian Ocean', 'pacific ocean': 'Pacific Ocean',
    'atlantic ocean': 'Atlantic Ocean', 'bay of bengal': 'Bay of Bengal',
    'arabian sea': 'Arabian Sea',
    # Worldwide
    'worldwide': 'Worldwide', 'global': 'Global',
}


def _keyword_regex(words: List[str], allow_plural: bool = True) -> "re.Pattern":
    """Whole-word regex for a list of words/phrases (longest first)."""
    parts = []
    for w in sorted({w.lower() for w in words}, key=len, reverse=True):
        parts.append(r'\s+'.join(re.escape(tok) for tok in w.split()))
    plural = r'(?:s|es)?' if allow_plural else ''
    return re.compile(r'\b(?:' + '|'.join(parts) + r')' + plural + r'\b')


def _build_location_patterns() -> List[Tuple["re.Pattern", str]]:
    patterns = []
    for loc in sorted(LOCATIONS, key=len, reverse=True):
        body = r'\s+'.join(re.escape(tok) for tok in loc.split())
        if loc == 'global':
            # "global warming" / "global climate" is a topic, not a location
            body += r'(?!\s+(?:warming|climate|temperature))'
        patterns.append((re.compile(r'\b' + body + r'\b'), LOCATIONS[loc]))
    return patterns


_ENV_RE = _keyword_regex(PILOT_DOMAIN_KEYWORDS)
_REJECT_TOPIC_RE = _keyword_regex(REJECTED_TOPIC_WORDS)
_REJECT_ACCESS_RE = _keyword_regex(REJECTED_ACCESS_WORDS)
_LOCATION_PATTERNS = _build_location_patterns()

_SIZE_UNITS = r'(?:rows?|samples?|records?|entries|entry|observations?|data\s*points?)'
_NUMBER = r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)'
_SIZE_RE = re.compile(r'(?<![\d,.])' + _NUMBER + r'\s*([km])?\s*' + _SIZE_UNITS + r'\b')
_BARE_SIZE_RE = re.compile(r'^\s*' + _NUMBER + r'\s*([km])?\s*(?:' + _SIZE_UNITS + r')?\s*$')

_GRANULARITY_WORDS = {
    'hourly': 'Hourly', 'daily': 'Daily', 'weekly': 'Weekly', 'monthly': 'Monthly',
    'quarterly': 'Quarterly', 'yearly': 'Yearly', 'annual': 'Yearly', 'annually': 'Yearly',
}


# ============================================================================
# LLM OUTPUT SCHEMA (every field optional/defaulted so a null from the small
# model can never fail the whole call)
# ============================================================================

class LLMRequirementAnalysis(BaseModel):
    """Schema for LLM analysis output."""
    domain: Optional[str] = None
    subdomain: Optional[str] = None  # null for general/vague requests
    problem_type: str = "analysis"
    data_modality: str = "tabular"
    expected_features: List[str] = Field(default_factory=list)  # input measurements only
    target_variable: Optional[str] = None  # for prediction: what to predict
    topic_description: str = ""
    time_granularity: Optional[str] = None  # hourly, daily, monthly, etc.


class RequirementAnalyzer:
    """
    Requirement Analyzer Agent.

    Analyzes user's free-text request and fills RequirementSpec.
    Uses plain Python for extractables (dates, sizes, formats, scope).
    Uses the LLM only for topic understanding (domain, subdomain, features).

    Standard entry point: analyze(spec: DatasetSpec) -> DatasetSpec
    """

    # The one config constant for the pilot domain (kept under the old name too)
    SUPPORTED_DOMAINS = PILOT_DOMAIN_KEYWORDS

    REJECTION_MESSAGE = (
        f"Not supported yet: your request seems outside our focus on {PILOT_DOMAIN_LABEL} "
        "(weather, climate, air quality, natural disasters, water/ocean, forests, glaciers, "
        "satellite-derived data). Please describe a dataset in one of these areas."
    )

    def __init__(self, interactive: bool = False):
        """
        Args:
            interactive: Whether to prompt the user for clarifications
                         (default False for tests/orchestrator).
        """
        self.llm = get_llm()
        self.interactive = interactive
        self.stats: Dict[str, float] = {}
        self._reset_stats()
        self._used_default = False  

    # ------------------------------------------------------------------
    # Timing / counters (LLM time and user-wait time are kept separate)
    # ------------------------------------------------------------------

    def _reset_stats(self):
        self.stats = {'llm_calls': 0, 'llm_seconds': 0.0, 'user_wait_seconds': 0.0}

    def _call_llm(self, **kwargs):
        """Every LLM call goes through here so it is counted and timed (also on failure)."""
        start = time.time()
        self.stats['llm_calls'] += 1
        try:
            return self.llm.call(**kwargs)
        finally:
            self.stats['llm_seconds'] += time.time() - start

    def _read_input(self, prompt: str) -> str:
        """Read one line from the user; time spent waiting is recorded separately."""
        start = time.time()
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        finally:
            self.stats['user_wait_seconds'] += time.time() - start

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def analyze(self, spec: DatasetSpec) -> DatasetSpec:
        """
        Standard orchestrator entry point.

        Args:
            spec: DatasetSpec with user input in spec.requirement

        Returns:
            Updated DatasetSpec with filled RequirementSpec
        """
        print("🔍 Analyzing requirement...")
        self._reset_stats()
        start_time = time.time()

        if not spec.requirement:
            raise ValueError("spec.requirement must be set with dataset_name and description")

        dataset_name = spec.requirement.dataset_name
        description = spec.requirement.description

        requirement = self._analyze_requirement(dataset_name, description)

        # Bounded clarification (ONE round max, if interactive)
        if self.interactive and requirement.is_supported and requirement.clarification_round == 0:
            requirement = self._bounded_clarification(requirement, dataset_name, description)

        spec.requirement = requirement

        if not requirement.is_supported:
            spec.pipeline_status = PipelineStatus.REJECTED
            spec.current_stage = "requirement_rejected"
            spec.add_warning(f"Request rejected: {requirement.unsupported_reason}")
            print(f"🚫 {requirement.unsupported_reason}")
        elif requirement.target_requires_derivation:
            derivation_note = "target must be derived from real events or thresholds"
            if derivation_note not in spec.warnings:
                spec.add_warning(derivation_note)

        total = time.time() - start_time
        wait = self.stats['user_wait_seconds']
        llm_s = self.stats['llm_seconds']
        processing = max(0.0, total - wait - llm_s)
        print(
            f"✅ Requirement analyzed in {total:.2f}s "
            f"(LLM {llm_s:.2f}s over {int(self.stats['llm_calls'])} call(s), "
            f"waiting for you {wait:.2f}s, other processing {processing:.2f}s)"
        )
        logger.info(
            f"Requirement analysis: total={total:.2f}s llm={llm_s:.2f}s "
            f"llm_calls={int(self.stats['llm_calls'])} user_wait={wait:.2f}s"
        )

        return spec

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def _analyze_requirement(self, dataset_name: str, description: str) -> RequirementSpec:
        """Main analysis logic. Returns a filled RequirementSpec."""
        combined = f"{dataset_name} {description}".lower()

        explicitly_stated: List[str] = []
        inferred_by_model: List[str] = []
        missing_or_unclear: List[str] = []
        clarifying_questions: List[str] = []

        # 1. Simple fields with regex/rules
        output_format = self._extract_output_format(combined, explicitly_stated, inferred_by_model)
        expected_size = self._extract_size(combined, explicitly_stated, inferred_by_model)
        freshness_need, max_data_age = self._extract_freshness(
            combined, explicitly_stated, inferred_by_model, missing_or_unclear, clarifying_questions
        )
        time_range = self._extract_time_range(combined, explicitly_stated, inferred_by_model)
        geography = self._extract_geography(combined, explicitly_stated, inferred_by_model)
        constraints = self._extract_constraints(combined, explicitly_stated)

        # 2. Scope check (instant, no LLM)
        is_supported, unsupported_reason = self._check_support(combined)

        if not is_supported:
            return RequirementSpec(
                dataset_name=dataset_name,
                description=description,
                is_supported=False,
                unsupported_reason=unsupported_reason,
                output_format=output_format,
                explicitly_stated=explicitly_stated,
                inferred_by_model=inferred_by_model
            )

        # 3. LLM for topic understanding (rule-based fallback on ANY failure)
        try:
            llm_result = self._analyze_with_llm(dataset_name, description, combined)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}. Using rule-based fallback.")
            print(f"⚠️  LLM analysis unavailable: {e}. Using rule-based fallback.")
            llm_result = self._extract_topic_with_rules(combined, description)
                # The model may return no usable domain: let the rules decide
        if not llm_result.domain:
            rules = self._extract_topic_with_rules(combined, description)
            llm_result.domain = rules.domain
            if not llm_result.subdomain:
                llm_result.subdomain = rules.subdomain
        llm_result.subdomain = self._clean_subdomain(llm_result.subdomain, combined)
        data_modality = self._map_data_modality(llm_result.data_modality)

        expected_features = llm_result.expected_features
        target_variable = llm_result.target_variable

        # Does the target need to be derived (not directly measurable)?
        target_requires_derivation = False
        target_terms = ['risk', 'occurrence', 'category', 'index', 'severity', 'defined', 'derived', 'prediction']
        if target_variable and any(k in target_variable.lower() for k in target_terms):
            target_requires_derivation = True
        elif any(k in combined for k in ['flood risk', 'flood occurrence', 'fire risk', 'fire occurrence', 'landslide risk']):
            target_requires_derivation = True
        # NOTE: the derivation note is a warning on the spec (added in analyze()),
        # NOT a "missing information" item.

        inferred_by_model.extend(['domain', 'subdomain', 'problem_type', 'data_modality'])
        if expected_features:
            inferred_by_model.append('expected_features')
        if target_variable:
            inferred_by_model.append('target_variable')

        # Prediction request whose target must be derived but the model named none:
        # keep an explicit placeholder so later agents know it still has to be defined.
        if target_requires_derivation and not target_variable and llm_result.problem_type == "prediction":
            target_variable = "to be defined from real data"
            inferred_by_model.append('target_variable: to be defined from real data (placeholder)')

        # Ask about granularity for multi-year ranges, unless the user already said it
        span = self._year_span(time_range)
        if span and (span[1] - span[0]) > 1:
            if not llm_result.time_granularity and not self._extract_granularity(combined):
                clarifying_questions.append(
                    "What time granularity do you need? (e.g., hourly, daily, monthly)"
                )
                missing_or_unclear.append('time granularity')

        # Time range reaches the present -> must include the latest available data
        current_year = datetime.now().year
        if span and span[1] >= current_year:
            constraints.append('must cover up to latest available date')
            explicitly_stated.append('constraint: must cover up to latest available date')

        # Missing critical information
        if not geography and 'geographic area' not in missing_or_unclear:
            missing_or_unclear.append('geographic area')
            clarifying_questions.append("Which geographic area or location do you need?")

        if not expected_size:
            missing_or_unclear.append('dataset size')

        completeness = self._calculate_confidence(
            explicitly_stated, missing_or_unclear, is_supported
        )

        topic_summary = (llm_result.topic_description or "").strip() or None
        if topic_summary and topic_summary == description.strip():
            topic_summary = None  # nothing new (rule-based fallback echoes the user's text)

        return RequirementSpec(
            dataset_name=dataset_name,
            description=description,          # the USER's text is never overwritten
            topic_summary=topic_summary,      # the LLM's summary lives in its own field
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

    # ------------------------------------------------------------------
    # Rule-based extraction
    # ------------------------------------------------------------------

    def _extract_output_format(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> str:
        """Extract output format from text."""
        if re.search(r'\bexcel\b|\bxlsx?\b', text):
            explicitly_stated.append('output_format: excel')
            return 'excel'
        elif re.search(r'\bjson\b', text):
            explicitly_stated.append('output_format: json')
            return 'json'
        elif re.search(r'\bcsv\b', text):
            explicitly_stated.append('output_format: csv')
            return 'csv'
        else:
            inferred.append('output_format: csv (default)')
            return 'csv'

    @staticmethod
    def _to_count(number: str, suffix: Optional[str]) -> int:
        """'1,000' -> 1000, '10' + 'k' -> 10000, '1.5' + 'k' -> 1500."""
        value = float(number.replace(',', ''))
        if suffix == 'k':
            value *= 1_000
        elif suffix == 'm':
            value *= 1_000_000
        return int(round(value))

    def _extract_size(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> Optional[str]:
        """
        Extract expected dataset size. A number only counts when it is directly
        followed by a size word (rows, samples, records, entries, observations,
        data points), so "about 10 years" is NOT a size.
        """
        match = _SIZE_RE.search(text)
        if match:
            count = self._to_count(match.group(1), match.group(2))
            if count > 0:
                explicitly_stated.append(f'dataset_size: {count}')
                return f"{count} rows"
        return None

    def _parse_size_text(self, text: str) -> Optional[int]:
        """Parse a size the user TYPED: '2000', '2,000', '2k', '2000 rows'."""
        text = text.lower().strip()
        match = _BARE_SIZE_RE.match(text) or _SIZE_RE.search(text)
        if match:
            count = self._to_count(match.group(1), match.group(2))
            return count if count > 0 else None
        return None

    def _extract_granularity(self, text: str) -> Optional[str]:
        """Time granularity the user already stated (hourly, daily, monthly, ...)."""
        for word, label in _GRANULARITY_WORDS.items():
            if re.search(rf'\b{word}\b', text):
                return label
        return None

    @staticmethod
    def _year_span(time_range: Optional[str]) -> Optional[Tuple[int, int]]:
        """(first_year, last_year) of a range like '2020 to 2026', else None."""
        if not time_range:
            return None
        years = [int(y) for y in re.findall(r'\b(\d{4})\b', time_range)]
        if len(years) >= 2:
            return years[0], years[-1]
        return None

    def _extract_freshness(
        self, text: str, explicitly_stated: List[str], inferred: List[str],
        missing: List[str], questions: List[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Extract freshness need and max data age.

        max_data_age = how old the NEWEST record may be
        - "live": infer "1 hour"
        - "recent": infer "24 hours"
        - historical/unspecified: empty

        Time windows like "last 7 days" go to time_range, not max_data_age.
        """
        freshness = "unspecified"
        max_age = None

        live_patterns = [r'\blive\b', r'\breal-time\b', r'\brealtime\b', r'\breal time\b']
        if any(re.search(pattern, text) for pattern in live_patterns):
            freshness = "live"
            explicitly_stated.append('freshness: live')
            max_age = "1 hour"
            inferred.append('max_data_age: 1 hour (inferred for live data)')
        elif any(re.search(rf'\b{word}\b', text) for word in ['recent', 'current', 'latest', 'now']):
            freshness = "recent"
            explicitly_stated.append('freshness: recent')
            max_age = "24 hours"
            inferred.append('max_data_age: 24 hours (inferred for recent data)')
            missing.append('specific time window')
            questions.append("How recent? (e.g., last 24 hours, last 7 days)")
        elif any(re.search(rf'\b{word}\b', text) for word in ['historical', 'past', 'archive']):
            freshness = "historical"
            explicitly_stated.append('freshness: historical')
        elif re.search(r'from\s+\d{4}\s+to\s+\d{4}', text):
            freshness = "historical"
            explicitly_stated.append('freshness: historical (inferred from year range)')

        if freshness == "unspecified":
            inferred.append('freshness: unspecified (not specified)')

        return freshness, max_age

    def _extract_time_range(
        self, text: str, explicitly_stated: List[str], inferred: List[str]
    ) -> Optional[str]:
        """
        Extract time range (history window).

        Examples: "last 7 days", "from 2010 to 2020", "this month", "past 30 days".
        """
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

        months = ('january|february|march|april|may|june|july|august|'
                  'september|october|november|december')
        relative_patterns = [
            (r'this\s+(week|month|year)', lambda m: f"this {m.group(1)}"),
            (r'last\s+(week|month|year)', lambda m: f"last {m.group(1)}"),
            (rf'({months})\s+to\s+({months})', lambda m: f"{m.group(1)} to {m.group(2)}"),
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
    ) -> Optional[str]:
        """
        Extract geographic area by WHOLE-WORD match against LOCATIONS
        ("usa" does not match "thousand", "uk" does not match "ukraine").
        Unknown places are not guessed; the user is asked instead.
        """
        text = text.lower()
        for pattern, display in _LOCATION_PATTERNS:   # longest names first
            if pattern.search(text):
                explicitly_stated.append(f'geography: {display}')
                return display
        return None

    def _extract_constraints(self, text: str, explicitly_stated: List[str]) -> List[str]:
        """Extract constraints."""
        constraints = []

        if re.search(r'\bfree\b', text) and ('source' in text or 'data' in text):
            constraints.append('free sources only')
            explicitly_stated.append('constraint: free sources')

        if 'no login' in text or 'without login' in text:
            constraints.append('no login required')
            explicitly_stated.append('constraint: no login')

        if re.search(r'\bpublic\b', text) and 'data' in text:
            constraints.append('public data only')
            explicitly_stated.append('constraint: public data')

        if re.search(r'\bsatellites?\b|\bremote sensing\b|\bimagery-derived\b|'
                     r'\bsatellite-derived\b|\borbital\b|\bspace-based\b', text):
            constraints.append('satellite-derived data')
            explicitly_stated.append('constraint: satellite-derived data')

        return constraints

    # ------------------------------------------------------------------
    # Scope check (pilot domain). Instant, never calls the LLM.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        """True if one of the 3 words before position `start` is a negator."""
        words = re.findall(r"[a-z']+", text[:start])[-3:]
        return any(w in _NEGATORS for w in words)

    def _check_support(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Is the request inside the pilot domain?

        1. clearly out-of-scope topic (finance, social media, ...) -> reject
        2. restricted access (private/paid/login) unless negated    -> reject
        3. at least one pilot-domain keyword (whole word)           -> supported
        4. otherwise                                                -> reject

        Returns: (is_supported, unsupported_reason)
        """
        text = text.lower()

        if _REJECT_TOPIC_RE.search(text):
            return False, self.REJECTION_MESSAGE

        for match in _REJECT_ACCESS_RE.finditer(text):
            if not self._is_negated(text, match.start()):
                return False, self.REJECTION_MESSAGE

        if _ENV_RE.search(text):
            return True, None

        return False, self.REJECTION_MESSAGE

    # ------------------------------------------------------------------
    # LLM topic understanding
    # ------------------------------------------------------------------

    def _parse_llm_payload(self, content: Any) -> LLMRequirementAnalysis:
        """Turn the LLM's JSON into LLMRequirementAnalysis; nulls never fail the call."""
        if isinstance(content, LLMRequirementAnalysis):
            return content
        if not isinstance(content, dict):
            raise ValueError(f"LLM returned {type(content).__name__}, expected a JSON object")

        data = dict(content)

        # null in a field that has a default -> use the default
        for key in ('problem_type', 'data_modality', 'topic_description'):
            if data.get(key) is None:
                data.pop(key, None)

        # the model sometimes writes the string "null" instead of a real null
        for key in ('subdomain', 'target_variable', 'time_granularity'):
            value = data.get(key)
            if isinstance(value, str) and value.strip().lower() in ('', 'null', 'none'):
                data[key] = None

        # expected_features: accept null, a comma string, or a list
        features = data.get('expected_features')
        if features is None:
            features = []
        elif isinstance(features, str):
            features = [f.strip() for f in features.split(',') if f.strip()]
        else:
            features = [str(f) for f in features if f]
        # drop unfilled placeholders such as "<measurement>"
        data['expected_features'] = [f for f in features if not re.fullmatch(r'\s*<.*>\s*', f)]

        # domain must be one of the allowed values, otherwise None (rules decide later)
        domain = data.get('domain')
        domain = domain.strip().lower() if isinstance(domain, str) else None
        data['domain'] = domain if domain in ALLOWED_DOMAINS else None

        result = LLMRequirementAnalysis(**data)

        if result.subdomain and result.subdomain.lower() in ['general', 'unspecified', 'various', 'weather']:
            result.subdomain = None
        return result

    def _clean_subdomain(self, subdomain: Optional[str], combined: str) -> Optional[str]:
        """
        A subdomain names a phenomenon ("flood monitoring"), never a place or a year.
        If the model copied one in ("Rishikesh Rainfall"), use the keyword-based
        subdomain instead, or strip the place/year words.
        """
        if not subdomain:
            return None
        lowered = subdomain.lower()
        has_place = any(pattern.search(lowered) for pattern, _ in _LOCATION_PATTERNS)
        has_year = re.search(r'\b\d{4}\b', lowered) is not None
        if not (has_place or has_year):
            return subdomain

        fallback = self._extract_topic_with_rules(combined, "").subdomain
        if fallback:
            return fallback

        stripped = lowered
        for pattern, _ in _LOCATION_PATTERNS:
            stripped = pattern.sub('', stripped)
        stripped = re.sub(r'\b\d{4}\b', '', stripped)
        stripped = ' '.join(stripped.split())
        return stripped or None

    def _analyze_with_llm(self, dataset_name: str, description: str, combined: str) -> LLMRequirementAnalysis:
        """Use the LLM for topic understanding (falls back to rules on failure)."""
        prompt = f"""Analyze this dataset request and return ONLY a JSON object.

Dataset name: {dataset_name}
Description: {description}

Return exactly these keys:
{{
  "domain": "<ONE word chosen from: environmental, meteorological, geological, oceanographic, hydrological>",
  "subdomain": "<the natural phenomenon or topic only, never a place name or a year; null if the request is general>",
  "problem_type": "<one of: monitoring, prediction, analysis>",
  "data_modality": "<one of: tabular, time_series, text>",
  "expected_features": ["<measurement>", "<measurement>"],
  "target_variable": "<what is predicted, or null>",
  "time_granularity": "<one of: hourly, daily, monthly, yearly, or null>",
  "topic_description": "<one sentence in your own words>"
}}

Rules:
- expected_features: 3 to 6 INPUT MEASUREMENTS that real public data sources record directly.
  Do not invent computed indices or risk scores. Do not list year, month, day or hour separately;
  use "timestamp" for time. Use an empty list if you cannot name realistic measurements.
- target_variable: only for prediction problems. If the target cannot be measured directly,
  write exactly: to be defined from real data. Otherwise null.
- time_granularity: only if the request covers several years, otherwise null.
- subdomain: name the phenomenon or topic only. Never put a place name or a year in it.

Return JSON only:"""

        try:
            response = self._call_llm(prompt=prompt, expected_schema=None)
            return self._parse_llm_payload(response.content)
        except Exception as e:
            logger.error(f"LLM analysis failed or unusable: {e}. Using rule-based fallback.")
            print(f"⚠️  LLM analysis unavailable: {e}. Using rule-based fallback.")
            return self._extract_topic_with_rules(combined, description)

    def _extract_topic_with_rules(self, text: str, description: str) -> LLMRequirementAnalysis:
        """
        Rule-based topic understanding when the LLM call fails or times out.
        Whole-word matching only.
        """
        text_lower = text.lower()

        domain = None
        subdomain = None

        def has_word(kw: str) -> bool:
            return bool(re.search(r'\b' + re.escape(kw.lower()) + r'(?:s|es)?\b', text_lower))

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
        elif any(has_word(kw) for kw in ['weather', 'climate', 'meteorological', 'temperature', 'rain', 'rainfall',
                                         'precipitation', 'monsoon', 'wind', 'humidity', 'storm', 'cyclone', 'drought']):
            domain = "meteorological"
            subdomain = None  # general weather: domain only, subdomain stays null
        elif any(has_word(kw) for kw in self.SUPPORTED_DOMAINS):
            domain = "environmental"
            subdomain = None
        # domain may stay None: the schema allows it and the clarification step asks for the topic

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

    def _map_data_modality(self, llm_modality: Optional[str]) -> DataModality:
        """Map LLM modality string to DataModality enum (default: tabular)."""
        modality_map = {
            'tabular': DataModality.TABULAR,
            'text': DataModality.TEXT,
            'time_series': DataModality.TIME_SERIES,
            'time-series': DataModality.TIME_SERIES,
            'timeseries': DataModality.TIME_SERIES,
        }
        if not llm_modality:
            return DataModality.TABULAR
        return modality_map.get(llm_modality.lower().strip().replace(' ', '_'), DataModality.TABULAR)

    def _calculate_confidence(
        self, explicitly_stated: List[str], missing: List[str], is_supported: bool
    ) -> float:
        """
        Completeness score (how much detail the user provided, NOT model accuracy).

        - Base 0.50
        - 4+ explicit +0.30, 3 explicit +0.20, 2 explicit +0.10, 1 explicit +0.05, 0 explicit -0.30
        - each missing item -0.05
        """
        if not is_supported:
            return 0.0

        score = 0.50

        num_explicit = len(explicitly_stated)
        if num_explicit >= 4:
            score += 0.30
        elif num_explicit >= 3:
            score += 0.20
        elif num_explicit >= 2:
            score += 0.10
        elif num_explicit >= 1:
            score += 0.05
        else:
            score -= 0.30

        score -= len(missing) * 0.05

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Clarification (ONE round, max 4 questions)
    # ------------------------------------------------------------------

    def _bounded_clarification(
        self,
        requirement: RequirementSpec,
        dataset_name: str,
        description: str
    ) -> RequirementSpec:
        """
        Ask the user for missing critical information (ONE round, max 4 questions).

        Priority:
        1. Topic, ONLY if no domain was found
        2. Location
        3. Time range or freshness (+ granularity for long ranges)
        4. Dataset size
        5. Output format (only if nothing else)

        A choice question has a 'custom_choice' key only if one of its options
        means "type your own value".
        """
        combined = f"{dataset_name} {description}".lower()
        questions: List[Dict[str, Any]] = []

        # Priority 1: topic, only if we could not work out any domain
        if not requirement.domain:
            questions.append({
                'field': 'topic',
                'prompt': 'What topic or phenomenon do you want data about?',
                'default': 'environmental data',
                'type': 'text'
            })

        # Priority 2: location
        if not requirement.geography:
            questions.append({
                'field': 'geography',
                'prompt': 'Which location or geographic area?',
                'default': 'global',
                'type': 'text'
            })

        # Priority 3a: time range
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
                'custom_choice': '4',
                'type': 'choice'
            })

        # Priority 3b: granularity for multi-year ranges (unless already stated)
        span = self._year_span(requirement.time_range)
        if span and (span[1] - span[0]) > 1 and not self._extract_granularity(combined):
            questions.append({
                'field': 'time_granularity',
                'prompt': f'Time granularity for {requirement.time_range}?',
                'default': 'Daily',
                'choices': [
                    ('1', 'Hourly'),
                    ('2', 'Daily'),
                    ('3', 'Monthly'),
                    ('4', 'Yearly')
                ],
                # no custom_choice: "4" here means Yearly
                'type': 'choice'
            })

        # Priority 4: dataset size
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
                'custom_choice': '4',
                'type': 'choice'
            })

        # Priority 5: output format
        if len(questions) < 4 and requirement.output_format == 'csv' and not re.search(r'\bcsv\b', combined):
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

        questions = questions[:4]

        if not questions:
            return requirement

        print("\n" + "=" * 70)
        print("📋 A few quick questions to refine your requirements")
        print("=" * 70)
        print("(Press Enter to use default, or type 'skip' to accept all defaults)\n")

        answers: Dict[str, str] = {}
        defaulted = set()          # fields where the user did NOT give a usable answer
        skip_all = False

        for i, q in enumerate(questions, 1):
            field = q['field']
            if skip_all:
                answers[field] = q['default']
                defaulted.add(field)
                continue

            print(f"\n{i}. {q['prompt']}")
            if q['type'] == 'choice':
                for choice_num, choice_text in q['choices']:
                    print(f"   {choice_num}) {choice_text}")
            print(f"   Default: {q['default']}")

            answer = self._read_input("   Your answer: ")

            if answer.lower() == 'skip':
                skip_all = True
                answers[field] = q['default']
                defaulted.add(field)
                print("   → Skipping remaining questions, using defaults")
                continue

            if not answer:
                answers[field] = q['default']
                defaulted.add(field)
                print(f"   → Using default: {q['default']}")
                continue

            answers[field] = self._resolve_answer(q, answer)
            if self._used_default:
                defaulted.add(field)

        requirement = self._apply_clarification_answers(requirement, answers, defaulted)
        requirement.clarification_round = 1

        print("\n" + "=" * 70)
        print("✅ Requirements confirmed")
        print("=" * 70)
        self._print_confirmation_summary(requirement)

        return requirement

    def _resolve_answer(self, q: Dict[str, Any], answer: str) -> str:
        """Turn what the user typed into a value for question q.
        Sets self._used_default = True if the typed text was unusable and the default was used."""
        self._used_default = False
        default = q['default']

        if q['type'] != 'choice':
            return answer

        choice_map = dict(q['choices'])
        custom_key = q.get('custom_choice')

        if custom_key and answer == custom_key:
            typed = self._read_input(f"   Enter {q['field']}: ")
            return self._normalise_free_text(q['field'], typed, default)

        if answer in choice_map:
            return choice_map[answer]

        # not an option number: treat it as the user typing their own value
        return self._normalise_free_text(q['field'], answer, default)

    def _normalise_free_text(self, field: str, text: str, default: str) -> str:
        """Make a typed answer usable for `field`, or fall back to the default."""
        text = (text or "").strip()
        if not text:
            self._used_default = True
            return default

        if field == 'expected_size':
            count = self._parse_size_text(text)
            if count:
                return f"{count} rows"
            print(f"   → Could not read a number from '{text}', using default: {default}")
            self._used_default = True
            return default

        if field == 'output_format':
            lowered = text.lower()
            if lowered in ('csv', 'excel', 'json'):
                return lowered
            if lowered in ('xlsx', 'xls'):
                return 'excel'
            print(f"   → Unknown format '{text}', using default: {default}")
            self._used_default = True
            return default

        if field == 'time_granularity':
            found = self._extract_granularity(text.lower())
            if found:
                return found
            print(f"   → Unknown granularity '{text}', using default: {default}")
            self._used_default = True
            return default

        if field == 'time_range':
            canonical = self._extract_time_range(text.lower(), [], [])
            return canonical or text

        return text

    # Words that identify the question text / missing item belonging to each field
    _QUESTION_KEYWORDS = {
        'topic': ['topic'],
        'geography': ['geographic', 'location'],
        'time_range': ['how recent', 'time period', 'time range', 'time window'],
        'time_granularity': ['granularity'],
        'expected_size': ['how many', 'rows'],
        'output_format': ['format'],
    }
    _MISSING_LABELS = {
        'geography': ['geographic area'],
        'time_range': ['specific time window'],
        'time_granularity': ['time granularity'],
        'expected_size': ['dataset size'],
    }

    def _apply_clarification_answers(
        self,
        requirement: RequirementSpec,
        answers: Dict[str, str],
        defaulted: Optional[set] = None
    ) -> RequirementSpec:
        """
        Apply user answers to the requirement spec.

        Fields in `defaulted` were NOT answered by the user. The default is used as the
        working value, but it is recorded in inferred_by_model (marked "(default)"), not in
        explicitly_stated, and the field stays open in missing_or_unclear / clarifying_questions.
        """
        defaulted = defaulted or set()

        def record(field: str, label: str, value: str):
            if field in defaulted:
                note = f'{label}: {value} (default)'
                if note not in requirement.inferred_by_model:
                    requirement.inferred_by_model.append(note)
            else:
                requirement.explicitly_stated.append(f'clarified_{label}: {value}')

        for field, value in answers.items():
            if field == 'topic':
                if value and value != 'environmental data':
                    ok, reason = self._check_support(value.lower())
                    if not ok:
                        requirement.is_supported = False
                        requirement.unsupported_reason = reason
                        requirement.completeness = 0.0
                        record(field, 'topic', value)
                        return requirement
                    try:
                        llm_result = self._analyze_with_llm("Topic", value, value.lower())
                        rules = self._extract_topic_with_rules(value.lower(), value)
                        requirement.domain = llm_result.domain or rules.domain
                        requirement.subdomain = self._clean_subdomain(llm_result.subdomain, value.lower())
                        requirement.problem_type = llm_result.problem_type
                        record(field, 'topic', value)
                    except Exception as e:
                        logger.error(f"Could not analyse clarified topic '{value}': {e}")

            elif field == 'geography':
                requirement.geography = value.title()
                record(field, 'geography', value)

            elif field == 'time_range':
                requirement.time_range = value
                record(field, 'time_range', value)

            elif field == 'time_granularity':
                record(field, 'granularity', value)

            elif field == 'expected_size':
                count = self._parse_size_text(value)
                requirement.expected_size = f"{count} rows" if count else value
                record(field, 'size', requirement.expected_size)

            elif field == 'output_format':
                fmt = {'csv': 'csv', 'excel': 'excel', 'json': 'json'}.get(value.lower(), 'csv')
                requirement.output_format = fmt
                record(field, 'format', fmt)

            # A defaulted field stays open; an answered one is closed
            if field in defaulted:
                continue

            keywords = self._QUESTION_KEYWORDS.get(field, [])
            requirement.clarifying_questions = [
                q for q in requirement.clarifying_questions
                if not any(k in q.lower() for k in keywords)
            ]
            for label in self._MISSING_LABELS.get(field, []):
                if label in requirement.missing_or_unclear:
                    requirement.missing_or_unclear.remove(label)

        # Remove earlier model guesses ONLY for fields the user really answered
        answered = [f for f in answers if f not in defaulted]
        if answered and requirement.inferred_by_model:
            markers = {
                'output_format': ['output_format'],
                'expected_size': ['size'],
                'geography': ['geography', 'location'],
                'time_range': ['time_range', 'freshness', 'max_data_age'],
                'time_granularity': ['granularity'],
            }
            requirement.inferred_by_model = [
                item for item in requirement.inferred_by_model
                if not any(
                    any(m in item.lower() for m in markers.get(field, []))
                    for field in answered
                )
            ]

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

        if requirement.missing_or_unclear:
            print(f"\n📝 Using defaults for:")
            for item in requirement.missing_or_unclear:
                print(f"   • {item}")

        print(f"\n✨ Completeness: {requirement.completeness:.0%}")
        print()