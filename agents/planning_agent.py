"""Rule-first planning for finding public dataset sources."""

import logging
import re
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.dataset_spec import (
    ApplicationPlanningPolicy,
    CandidateSource,
    DatasetSpec,
    PlanSpec,
    RequirementSpec,
    SearchQuery,
    SourceRequirement,
    UsageIntent,
    derive_evaluation_requirements,
)
from core.llm_wrapper import LLMError, LLMResponseError, get_llm

from .source_catalog import SOURCE_CATALOG, SourceCatalogEntry, normalize_feature_name

logger = logging.getLogger(__name__)

_INDIA_LOCATIONS = {
    "india", "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal", "delhi", "mumbai",
    "bangalore", "bengaluru", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
    "jaipur", "lucknow", "patna", "bhopal", "chandigarh", "shimla", "srinagar",
    "guwahati", "kochi", "thiruvananthapuram", "nagpur", "nashik",
}
_US_LOCATIONS = {
    "us", "usa", "united states", "texas", "california", "florida", "alaska",
    "hawaii", "louisiana", "oklahoma", "nevada", "oregon", "washington state",
    "arizona", "colorado", "new york",
}
_OTHER_KNOWN_LOCATIONS = {
    "united kingdom", "uk", "england", "scotland", "wales", "ireland",
    "canada", "mexico", "brazil", "argentina", "chile", "france", "germany",
    "spain", "italy", "portugal", "netherlands", "china", "japan", "south korea",
    "north korea", "singapore", "thailand", "vietnam", "indonesia", "philippines",
    "australia", "new zealand", "south africa", "egypt", "nigeria", "kenya",
    "saudi arabia", "united arab emirates", "uae", "russia",
}
_KNOWN_LOCATIONS = _INDIA_LOCATIONS | _US_LOCATIONS | _OTHER_KNOWN_LOCATIONS
_ALIASES = {
    "air quality": {"air quality", "pollution", "pm2.5", "pm10", "aqi", "ozone", "no2", "so2"},
    "fire": {"fire", "wildfire", "forest fire", "forest fire monitoring"},
    "earthquake": {
        "earthquake", "earthquakes", "seismic", "geological", "natural disasters",
    },
    "flood": {"flood", "flood monitoring", "hydrology", "hydrological"},
    "water quality": {"water quality"},
    "ocean": {"ocean", "oceanographic", "marine", "sea"},
    "satellite": {"satellite", "satellite data", "remote sensing", "earth observation"},
    "weather": {"weather", "meteorological", "climate", "temperature", "rainfall", "precipitation"},
    "general": {"general", "environmental", "nature"},
}
_SATELLITE_CONSTRAINT = "satellite-derived data"

# Freshness compatibility: a source's temporal value satisfies a request's freshness when it
# appears in the corresponding set.
_FRESHNESS_COMPAT: Dict[str, Set[str]] = {
    "live":       {"live", "all"},
    "recent":     {"live", "recent", "all"},
    "historical": {"historical", "all"},
    "unspecified": {"live", "recent", "historical", "all"},
}


class PlanningAgent:
    """Create a source-search strategy from a supported requirement."""

    def __init__(self):
        self.llm = get_llm()
        self.stats: Dict[str, float] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "llm_calls": 0,
            "llm_seconds": 0.0,
            "user_wait_seconds": 0.0,
            "llm_skipped": 0,
        }

    def _call_llm(self, **kwargs):
        start = time.time()
        self.stats["llm_calls"] += 1
        try:
            return self.llm.call(**kwargs)
        finally:
            self.stats["llm_seconds"] += time.time() - start

    @staticmethod
    def _warn_once(spec: DatasetSpec, message: str) -> None:
        if message not in spec.warnings:
            spec.add_warning(message)

    @staticmethod
    def _country(geography: Optional[str]) -> Optional[str]:
        if not geography:
            return None
        lowered = geography.lower()
        if any(re.search(rf"\b{re.escape(place)}\b", lowered) for place in _INDIA_LOCATIONS):
            return "india"
        if any(re.search(rf"\b{re.escape(place)}\b", lowered) for place in _US_LOCATIONS):
            return "us"
        return None

    @staticmethod
    def _topics(subdomain: Optional[str], domain: Optional[str]) -> Tuple[Set[str], Set[str]]:
        sub = (subdomain or "").lower().strip()
        dom = (domain or "").lower().strip()
        specific: Set[str] = set()
        for key, aliases in _ALIASES.items():
            if any(PlanningAgent._phrase_present(alias, sub) for alias in aliases):
                specific.update(aliases)
                specific.add(key)
        domain_terms: Set[str] = set()
        for key, aliases in _ALIASES.items():
            if any(PlanningAgent._phrase_present(term, dom) for term in aliases | {key}):
                domain_terms.update(aliases)
                domain_terms.add(key)
        if dom == "environmental":
            domain_terms.update(_ALIASES["general"])
        return specific, domain_terms

    @staticmethod
    def _overlaps(left: Sequence[str], right: Set[str]) -> bool:
        normalized = [value.lower().strip() for value in left]
        for value in normalized:
            if value in right:
                return True
            if any(PlanningAgent._phrase_present(term, value) or PlanningAgent._phrase_present(value, term)
                   for term in right):
                return True
        return False

    @staticmethod
    def _phrase_present(phrase: str, text: str) -> bool:
        """Match an alias or topic as a normalized word/phrase, never as an arbitrary substring."""
        if not phrase or not text:
            return False
        normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not normalized_phrase or not normalized_text:
            return False
        return re.search(rf"(?:^| ){re.escape(normalized_phrase)}(?: |$)", f" {normalized_text} ") is not None

    @staticmethod
    def _feature_matches(entry: SourceCatalogEntry, features: Sequence[str]) -> List[str]:
        """Return the subset of *features* that the catalog entry advertises.

        Matching uses exact canonical names and explicit aliases; requested feature
        strings are returned unchanged.
        """
        provided_norm = [normalize_feature_name(item) for item in entry.provides]
        matched = []
        for feature in features:
            norm = normalize_feature_name(feature)
            if norm and norm in provided_norm:
                matched.append(feature)
        return matched

    @staticmethod
    def _temporal_match(entry: SourceCatalogEntry, freshness: str, time_range: Optional[str]) -> bool:
        """True when the entry's ``temporal`` value satisfies the requested freshness.

        "unspecified" (or None) means the user gave no freshness requirement — every
        source satisfies it.  A non-empty *time_range* does **not** escalate an
        unspecified freshness to "historical"; that caused good sources to be wrongly
        demoted when the user provided a vague phrase like "latest available".
        """
        required = (freshness or "unspecified").lower()
        # Only escalate to historical when the user gave a concrete date/year range,
        # not for free-text phrases like "latest available".
        if required == "unspecified" and time_range:
            import re as _re
            if _re.search(r'\b(?:19|20)\d{2}\b', time_range):
                required = "historical"
        compat = _FRESHNESS_COMPAT.get(required, _FRESHNESS_COMPAT["unspecified"])
        return entry.temporal in compat

    def _candidate_sources(self, spec: DatasetSpec) -> Tuple[List[SourceCatalogEntry], bool]:
        requirement = spec.requirement
        assert requirement is not None
        specific, domain_terms = self._topics(requirement.subdomain, requirement.domain)
        selected_terms = specific or domain_terms
        if requirement.subdomain and not specific:
            self._warn_once(
                spec,
                f"Unknown subdomain '{requirement.subdomain}'; using domain-level sources.",
            )
        if not requirement.geography:
            self._warn_once(spec, "Geography is unspecified; using global sources only.")
        is_satellite = any(_SATELLITE_CONSTRAINT in c.lower() for c in requirement.constraints)
        if is_satellite:
            selected_terms.update(_ALIASES["satellite"])

        candidates = [
            entry for entry in SOURCE_CATALOG
            if self._overlaps(entry.covers, selected_terms)
        ]
        if is_satellite:
            candidates.extend(
                entry for entry in SOURCE_CATALOG
                if self._overlaps(entry.covers, _ALIASES["satellite"]) and entry not in candidates
            )

        country = self._country(requirement.geography)
        if requirement.geography:
            candidates = [
                entry for entry in candidates
                if entry.geography == "global" or (country and entry.geography == country)
            ]
        else:
            candidates = [entry for entry in candidates if entry.geography == "global"]

        features = requirement.expected_features
        fresh = requirement.freshness_need
        temporal_matches = any(
            self._temporal_match(entry, fresh, requirement.time_range) for entry in candidates
        )

        def rank(entry: SourceCatalogEntry):
            geography_score = 2 if country and entry.geography == country else 1 if entry.geography == "global" else 0
            feature_score = len(self._feature_matches(entry, features))
            temporal_score = int(self._temporal_match(entry, fresh, requirement.time_range))
            access_score = int(fresh in {"live", "recent"} and entry.access == "api")
            topic_score = int(self._overlaps(entry.covers, specific or domain_terms))
            connector_score = int(entry.connector_available)
            return (
                topic_score,
                feature_score,
                temporal_score,
                geography_score,
                connector_score,
                access_score,
            )

        candidates.sort(key=rank, reverse=True)
        return candidates, temporal_matches

    @staticmethod
    def _apply_constraints(
        entries: Sequence[SourceCatalogEntry], constraints: Sequence[str]
    ) -> List[SourceCatalogEntry]:
        constrained = list(entries)
        lowered = {constraint.lower() for constraint in constraints}
        if "free sources only" in lowered:
            # Keep sources that are free. A source that requires a free key (requires_key=True,
            # free=True) is kept; only paid sources (free=False) are removed.
            constrained = [entry for entry in constrained if entry.free]
        if "no login required" in lowered:
            constrained = [entry for entry in constrained if not entry.requires_login]
        if "public data only" in lowered:
            constrained = [entry for entry in constrained if entry.public]
        return constrained

    @staticmethod
    def _matched_phenomenon(requirement) -> str:
        return requirement.subdomain or requirement.domain or "environmental data"

    @staticmethod
    def _clean_provider_name(name: str) -> str:
        """Strip brackets, slashes and trailing parenthetical junk from a catalog name.

        ``"CPCB / NAMP"`` → ``"CPCB NAMP"``
        ``"Open-Meteo Air Quality API"`` → ``"Open-Meteo Air Quality"``
        ``"Commercial weather API (example)"`` → ``"Commercial weather"``
        """
        # Remove parenthetical suffixes like " (example)"
        name = re.sub(r'\s*\([^)]*\)', '', name)
        # Replace slashes (with optional surrounding spaces) with a space
        name = re.sub(r'\s*/\s*', ' ', name)
        # Drop lone "API" words at the end (already captured in template)
        name = re.sub(r'\bAPI\b', '', name)
        return ' '.join(name.split())

    @staticmethod
    def _build_queries(
        source_requirements: Sequence[SourceRequirement],
        entries: Dict[str, SourceCatalogEntry],
        phenomenon: str,
        geography: Optional[str],
        has_free_constraint: bool = False,
    ) -> List[SearchQuery]:
        """Build 2–3 distinct rule-based queries for every selected source.

        - required / useful: 3 queries each
        - fallback: 2 queries each (documentation + hourly data)

        Templates per source (provider-specific, so shared phrases don't collide):
          1. ``<provider> <phenomenon> <place> documentation``
          2. ``<provider> <phenomenon> <place> hourly data``
          3. ``<provider> <place> <phenomenon> station data``  (required/useful only)

        "free data" replaces template 3 only when the user asked for free sources.
        No query contains "API" more than once.
        Provider names have brackets and slashes stripped.
        """
        place = geography or "global"
        queries: List[SearchQuery] = []
        seen: Set[str] = set()

        for source in source_requirements:
            entry = entries[source.source_type]
            provider = PlanningAgent._clean_provider_name(entry.name)

            if source.importance == "fallback":
                templates = [
                    f"{provider} {phenomenon} {place} documentation",
                    f"{provider} {phenomenon} {place} hourly data",
                ]
            else:  # required / useful
                t3 = (
                    f"{provider} {place} {phenomenon} free data"
                    if has_free_constraint
                    else f"{provider} {place} {phenomenon} station data"
                )
                templates = [
                    f"{provider} {phenomenon} {place} documentation",
                    f"{provider} {phenomenon} {place} hourly data",
                    t3,
                ]

            for query in templates:
                key = query.casefold()
                if key in seen or len(queries) >= 12:
                    break
                seen.add(key)
                queries.append(SearchQuery(
                    query=query[:120],
                    purpose=f"Find {phenomenon} data from {provider}.",
                    related_source_type=source.source_type,
                    priority=len(queries) + 1,
                ))

        return queries

    @staticmethod
    def _subdomain_synonyms(phenomenon: str) -> Set[str]:
        """Return a set of lowercase synonyms for the phenomenon / subdomain.

        Used to decide whether an LLM-suggested query is relevant enough to keep.
        """
        base = phenomenon.lower()
        words: Set[str] = {base}
        for key, aliases in _ALIASES.items():
            if base in aliases or key == base or any(a in base for a in aliases):
                words.update(aliases)
                words.add(key)
        # Always include individual words from the phenomenon itself
        words.update(base.split())
        return words

    @classmethod
    def _normalize_search_query(cls, query: str) -> str:
        normalized = re.sub(r"\s+", " ", query.strip()).casefold()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @classmethod
    def _mentioned_locations(cls, text: str) -> Set[str]:
        """Find explicitly known geographic names using word/phrase boundaries."""
        matches: Set[str] = set()
        for place in sorted(_KNOWN_LOCATIONS, key=len, reverse=True):
            if cls._phrase_present(place, text):
                matches.add(place)
        return matches

    @staticmethod
    def _country_aliases() -> Set[str]:
        return {"india", "us", "usa", "united states"}

    @classmethod
    def _location_country(cls, location: str) -> Optional[str]:
        if location in {"india"} or location in _INDIA_LOCATIONS:
            return "india"
        if location in {"us", "usa", "united states"} or location in _US_LOCATIONS:
            return "us"
        return location

    @classmethod
    def _locations_compatible(
        cls,
        requested_text: str,
        query_text: str,
    ) -> bool:
        """Validate known location names without assuming complete geocoding."""
        requested = cls._mentioned_locations(requested_text)
        queried = cls._mentioned_locations(query_text)
        if not requested:
            return not queried
        if not queried:
            return True

        country_aliases = cls._country_aliases()
        requested_countries = {
            cls._location_country(location)
            for location in requested
            if cls._location_country(location)
        }
        queried_countries = {
            cls._location_country(location)
            for location in queried
            if cls._location_country(location)
        }
        if not requested_countries.intersection(queried_countries):
            return False

        requested_exact = requested - country_aliases
        queried_exact = queried - country_aliases
        requested_is_country_only = bool(requested & country_aliases) and not requested_exact

        if requested_is_country_only:
            return all(
                cls._location_country(location) in requested_countries
                for location in queried_exact
            )

        if requested_exact:
            return all(location in requested_exact for location in queried_exact)

        return True

    @classmethod
    def _llm_query_relevant(
        cls,
        query: str,
        requirement: RequirementSpec,
        phenomenon: str,
    ) -> bool:
        """Deterministic relevance checks for LLM-generated search suggestions."""
        q_low = cls._normalize_search_query(query)
        if not q_low or "http" in q_low or "www." in q_low:
            return False

        topic_terms = cls._subdomain_synonyms(phenomenon)

        topic_match = any(cls._phrase_present(str(term), q_low) for term in topic_terms)
        if not topic_match:
            return False

        # Reject a query that explicitly identifies a different top-level topic.
        requirement_topics = cls._topics(requirement.subdomain, requirement.domain)[0]
        for topic_key, aliases in _ALIASES.items():
            if topic_key in requirement_topics:
                continue
            if any(cls._phrase_present(alias, q_low) for alias in aliases):
                # A generic alias such as "temperature" can legitimately belong to
                # multiple domains, so require an explicit strong topic phrase.
                if any(cls._phrase_present(alias, q_low) for alias in (
                    "earthquake", "wildfire", "forest fire", "flood", "ocean", "satellite",
                    "air quality", "pollution",
                ) if alias in aliases):
                    return False

        # If features are explicitly requested, a generated query must retain at
        # least one exact requested feature or a controlled alias. Topic terms
        # alone are not enough to prove feature preservation.
        if requirement.expected_features:
            for feature in requirement.expected_features:
                feature_norm = normalize_feature_name(feature)
                if feature_norm and cls._phrase_present(feature_norm, q_low):
                    break
                if cls._phrase_present(feature, q_low):
                    break
            else:
                return False

        if requirement.geography:
            if not cls._locations_compatible(requirement.geography, q_low):
                return False

        return True

    def _llm_refinement(self, plan: PlanSpec, spec: DatasetSpec, use_llm: bool = True) -> None:
        fallback_reasoning = self._rule_reasoning(spec)

        # Skip LLM when explicitly disabled.
        if not use_llm:
            self.stats["llm_skipped"] = self.stats.get("llm_skipped", 0) + 1
            if not plan.reasoning:
                plan.reasoning = fallback_reasoning
            return

        # Auto-skip: when rule-built queries already reach 4 or more, calling the LLM
        # for more queries adds marginal value and would only approach the 12-cap.
        if len(plan.search_queries) >= 4:
            self.stats["llm_skipped"] = self.stats.get("llm_skipped", 0) + 1
            if not plan.reasoning:
                plan.reasoning = f"rule-based: {fallback_reasoning}"
            elif not plan.reasoning.startswith("rule-based"):
                plan.reasoning = f"rule-based: {plan.reasoning}"
            return

        phenomenon = self._matched_phenomenon(spec.requirement)
        prompt = (
            "Refine this source-search plan. Return JSON with keys "
            '"extra_queries" (up to 3 short strings) and "reasoning" (one sentence). '
            "Do not include URLs. Requirement: "
            f"{spec.requirement.description}"
        )
        try:
            response = self._call_llm(prompt=prompt, expected_schema=None)
            content = getattr(response, "content", None)
            if not isinstance(content, dict):
                raise LLMResponseError("planner response was not a JSON object")
            raw_queries = content.get("extra_queries", [])
            raw_reasoning = content.get("reasoning")
            queries: List[str] = []
            if isinstance(raw_queries, list):
                for q in raw_queries:
                    if not (
                        isinstance(q, str)
                        and q.strip()
                        and len(q.strip()) <= 120
                        and "http" not in q.lower()
                        and "www." not in q.lower()
                    ):
                        continue
                    normalized_query = self._normalize_search_query(q)
                    if not self._llm_query_relevant(normalized_query, spec.requirement, phenomenon):
                        continue
                    compact_query = re.sub(r"[^a-z0-9]", "", normalized_query)
                    duplicate = any(
                        re.sub(r"[^a-z0-9]", "", self._normalize_search_query(existing))
                        == compact_query
                        for existing in queries
                    ) or any(
                        re.sub(r"[^a-z0-9]", "", self._normalize_search_query(existing.query))
                        == compact_query
                        for existing in plan.search_queries
                    )
                    if duplicate:
                        continue
                    queries.append(re.sub(r"\s+", " ", q.strip()))
                queries = queries[:3]
            reasoning = (
                raw_reasoning.strip()
                if isinstance(raw_reasoning, str)
                and raw_reasoning.strip()
                and len(raw_reasoning.strip()) <= 120
                and "http" not in raw_reasoning.lower()
                and "www." not in raw_reasoning.lower()
                else None
            )
            if not queries and not reasoning:
                raise LLMResponseError("planner response had no usable suggestions")
            if reasoning:
                plan.reasoning = reasoning
            added = 0
            for query in queries:
                if not plan.source_requirements or len(plan.search_queries) >= 12:
                    break
                normalized = query.casefold()
                if any(existing.query.casefold() == normalized for existing in plan.search_queries):
                    continue
                # Attach to the source whose catalog name appears in the query,
                # else the highest-priority required source, else any source.
                q_low = query.lower()
                target = next(
                    (
                        source for source in plan.source_requirements
                        if source.importance == "required"
                        and self._clean_provider_name(
                            # entries_by_type not available here; use source_type as proxy
                            source.source_type.replace("_", " ")
                        ).lower() in q_low
                    ),
                    None,
                )
                if target is None:
                    target = next(
                        (s for s in plan.source_requirements if s.importance == "required"),
                        None,
                    )
                if target is None:
                    target = next(
                        (s for s in plan.source_requirements if s.importance == "useful"),
                        None,
                    )
                if target is None:
                    target = next(iter(plan.source_requirements), None)
                if target is None:
                    break
                plan.search_queries.append(SearchQuery(
                    query=query,
                    purpose="Additional search suggestion from the planner.",
                    related_source_type=target.source_type,
                    priority=len(plan.search_queries) + 1,
                ))
                added += 1
                if added >= 3:
                    break
        except LLMError as exc:
            logger.warning("Planner LLM refinement unavailable; using rules: %s", exc)
            plan.reasoning = fallback_reasoning
            self._warn_once(spec, "LLM unavailable, planner used rules only")

    @staticmethod
    def _rule_reasoning(spec: DatasetSpec) -> str:
        """Build a one-sentence reasoning string from facts actually in the plan.

        Only mentions freshness when it was explicitly stated; only mentions time
        window when one was given; never says "time period" when nothing was provided.
        """
        requirement = spec.requirement
        assert requirement is not None
        topic = requirement.subdomain or requirement.domain or "environmental data"
        location = requirement.geography or "global coverage"

        parts: List[str] = [f"Searching for {topic} data in {location}"]

        if requirement.freshness_need not in {"unspecified", None}:
            parts.append(f"freshness={requirement.freshness_need}")

        if requirement.time_window_days:
            parts.append(f"window={requirement.time_window_days}d")
        elif requirement.time_range and requirement.time_range != "latest available":
            parts.append(f"range={requirement.time_range}")

        # Count sources in the plan if already built (called after source loop)
        if spec.plan:
            reqs = spec.plan.source_requirements
            n_req = sum(1 for s in reqs if s.importance == "required")
            n_use = sum(1 for s in reqs if s.importance == "useful")
            n_fall = sum(1 for s in reqs if s.importance == "fallback")
            source_summary = f"{n_req} required, {n_use} useful, {n_fall} fallback source(s)"
            parts.append(source_summary)

        return "; ".join(parts) + "."

    def plan(
        self,
        spec: DatasetSpec,
        use_llm: bool = True,
        policy: Optional[ApplicationPlanningPolicy] = None,
    ) -> DatasetSpec:
        """Create or replace the data-source plan for ``spec.requirement``.

        Parameters
        ----------
        spec:
            The dataset spec to plan for.
        use_llm:
            When ``True`` (default) the LLM may add extra queries; when ``False``
            or when rule-built queries already number >= 4, the LLM call is skipped.
        policy:
            Application-owned evaluation policy derived from the actual requirement.
            If ``None``, the planner derives policy internally from ``spec.requirement``
            using ``derive_evaluation_requirements()``.  The LLM never authors or
            modifies this policy regardless of how it is supplied.
        """
        self._reset_stats()
        if spec.requirement is None:
            raise ValueError("spec.requirement must be set with dataset_name and description")
        if not spec.requirement.is_supported:
            return spec

        # Derive the expected evaluation requirements from the actual requirement.
        # This is the ground truth; it must happen before any LLM interaction.
        try:
            derived_policy = derive_evaluation_requirements(
                spec.requirement,
                UsageIntent(),
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to derive evaluation policy from requirement: {exc}"
            ) from exc

        if policy is not None:
            # A caller-supplied policy is accepted only when its evaluation
            # requirements exactly match the derived requirements.  The
            # UsageIntent may differ (e.g., the caller adds "redistribute").
            # Reject before any LLM interaction so errors are surfaced early.
            supplied = policy.evaluation_requirements
            expected = derived_policy.evaluation_requirements

            # Compare every aspect of the evaluation requirements that affects
            # which dimensions are investigated and how they are prioritised.
            mismatches: List[str] = []

            supplied_dims = set(supplied.required_dimensions)
            expected_dims = set(expected.required_dimensions)
            missing = expected_dims - supplied_dims
            extra   = supplied_dims - expected_dims
            if missing:
                mismatches.append(
                    f"supplied policy is missing required dimensions: {sorted(missing)}"
                )
            if extra:
                mismatches.append(
                    f"supplied policy has unexpected dimensions: {sorted(extra)}"
                )

            # Criticality must match for every shared dimension.
            for dim in supplied_dims & expected_dims:
                s_crit = supplied.dimension_criticality.get(dim)
                e_crit = expected.dimension_criticality.get(dim)
                if s_crit != e_crit:
                    mismatches.append(
                        f"dimension '{dim}' has criticality {s_crit!r} in supplied "
                        f"policy but {e_crit!r} is required"
                    )

            # Required variables, formats, and retrieval methods must match.
            def _norm_list(values: List[str]) -> List[str]:
                return sorted(v.strip().lower() for v in values)

            if _norm_list(supplied.required_variables) != _norm_list(expected.required_variables):
                mismatches.append(
                    f"required_variables mismatch: supplied {supplied.required_variables!r}, "
                    f"expected {expected.required_variables!r}"
                )
            if _norm_list(supplied.required_formats) != _norm_list(expected.required_formats):
                mismatches.append(
                    f"required_formats mismatch: supplied {supplied.required_formats!r}, "
                    f"expected {expected.required_formats!r}"
                )
            if sorted(supplied.required_retrieval_methods) != sorted(
                expected.required_retrieval_methods
            ):
                mismatches.append(
                    f"required_retrieval_methods mismatch: "
                    f"supplied {supplied.required_retrieval_methods!r}, "
                    f"expected {expected.required_retrieval_methods!r}"
                )
            if supplied.policy_version != expected.policy_version:
                mismatches.append(
                    f"policy_version mismatch: supplied {supplied.policy_version!r}, "
                    f"expected {expected.policy_version!r}"
                )

            if mismatches:
                detail = "; ".join(mismatches)
                raise ValueError(
                    f"Supplied policy is inconsistent with the actual requirement "
                    f"({detail}). Derive the policy from the requirement using "
                    f"derive_evaluation_requirements() before passing it to the planner."
                )

            # Evaluation requirements are consistent; preserve the caller's UsageIntent.
            effective_policy: ApplicationPlanningPolicy = policy
        else:
            effective_policy = derived_policy

        try:
            requirement = spec.requirement
            candidates, _ = self._candidate_sources(spec)
            general_fallback_used = not candidates
            if not candidates:
                candidates = [
                    entry for entry in SOURCE_CATALOG
                    if entry.geography == "global"
                ][:1]
                self._warn_once(spec, "No directly matching source was found; using a general fallback.")

            constrained = self._apply_constraints(candidates, requirement.constraints)
            if not constrained and candidates:
                self._warn_once(
                    spec,
                    "Hard user constraints excluded all matching sources; no excluded source was retained.",
                )

            if not constrained:
                plan = PlanSpec(
                    uncovered_features=list(dict.fromkeys(requirement.expected_features)),
                    reasoning=self._rule_reasoning(spec),
                    confidence=0.0,
                    planning_policy=effective_policy,
                )
                if requirement.expected_features:
                    self._warn_once(
                        spec,
                        "No selected source covers requested feature(s): "
                        + ", ".join(plan.uncovered_features) + ".",
                    )
                if not candidates:
                    self._warn_once(spec, "No catalog sources are available for this requirement.")
                spec.plan = plan
                spec.current_stage = "planning"
                print("✅ Planning complete: 0 source type(s), 0 search query(ies)")
                return spec

            source_candidates = constrained[:6]
            if any(_SATELLITE_CONSTRAINT in c.lower() for c in requirement.constraints):
                satellite_entry = next(
                    (
                        entry for entry in constrained
                        if self._overlaps(entry.covers, _ALIASES["satellite"])
                    ),
                    None,
                )
                if satellite_entry and satellite_entry not in source_candidates:
                    source_candidates[-1:] = [satellite_entry]

            # Check whether at least one candidate satisfies the requested freshness.
            temporal_ok = any(
                self._temporal_match(entry, requirement.freshness_need, requirement.time_range)
                for entry in source_candidates
            )
            if requirement.freshness_need in {"live", "recent"} and not temporal_ok:
                self._warn_once(
                    spec,
                    "No selected source advertises support for the requested freshness.",
                )

            features = requirement.expected_features
            source_requirements: List[SourceRequirement] = []
            entries_by_type: Dict[str, SourceCatalogEntry] = {}

            # Importance reflects requirement fit; connector availability is separate metadata.
            required_count = 0
            for index, entry in enumerate(source_candidates, 1):
                feature_matches = self._feature_matches(entry, features)
                time_match = self._temporal_match(
                    entry, requirement.freshness_need, requirement.time_range
                )
                geo_match = (
                    entry.geography == "global"
                    or entry.geography == self._country(requirement.geography)
                )

                if general_fallback_used or not time_match or not geo_match:
                    importance = "fallback"
                elif required_count < 2:
                    importance = "required"
                    required_count += 1
                else:
                    importance = "useful"

                # Only already constraint-filtered satellite candidates can be promoted.
                if (
                    not general_fallback_used
                    and time_match
                    and geo_match
                    and any(
                        _SATELLITE_CONSTRAINT in c.lower()
                        for c in requirement.constraints
                    )
                    and self._overlaps(entry.covers, _ALIASES["satellite"])
                ):
                    importance = "required"

                location = requirement.geography or "the requested area"
                topic = self._matched_phenomenon(requirement)
                why = f"Prioritize {entry.name} for {topic} data for {location}."
                if not entry.connector_available:
                    why += " No implemented connector is available; manual discovery/access is required."
                source_requirements.append(SourceRequirement(
                    source_type=entry.source_type,
                    why_needed=why,
                    information_expected=feature_matches,
                    importance=importance,
                    preferred_access_methods=[entry.access],
                    priority=index,
                    data_origin=entry.data_origin,
                    connector_available=entry.connector_available,
                ))
                entries_by_type[entry.source_type] = entry

            unavailable_connectors = [
                source.source_type for source in source_requirements
                if not source.connector_available
            ]
            if unavailable_connectors:
                self._warn_once(
                    spec,
                    "Selected source types without an implemented connector require manual "
                    "discovery/access: " + ", ".join(unavailable_connectors) + ".",
                )

            queries = self._build_queries(
                source_requirements,
                entries_by_type,
                self._matched_phenomenon(requirement),
                requirement.geography,
                has_free_constraint="free sources only" in {c.lower() for c in requirement.constraints},
            )
            primary = [s.source_type for s in source_requirements if s.importance == "required"]
            fallback = [s.source_type for s in source_requirements if s.importance == "fallback"]
            matched_geography = bool(requirement.geography) and any(
                entry.geography == "global"
                or entry.geography == self._country(requirement.geography)
                for entry in source_candidates
            )
            requested_features = {
                normalize_feature_name(feature)
                for feature in features
                if normalize_feature_name(feature)
            }
            matched_features = {
                normalize_feature_name(feature)
                for source in source_requirements
                for feature in source.information_expected
            }
            feature_coverage = (
                len(requested_features & matched_features) / len(requested_features)
                if requested_features else 0.5
            )
            feature_coverage = max(0.0, min(1.0, feature_coverage))
            uncovered_features: List[str] = []
            seen_uncovered: Set[str] = set()
            for feature in features:
                normalized = normalize_feature_name(feature)
                if (
                    normalized
                    and normalized not in matched_features
                    and normalized not in seen_uncovered
                ):
                    uncovered_features.append(feature)
                    seen_uncovered.add(normalized)
            if uncovered_features:
                self._warn_once(
                    spec,
                    "No selected source covers requested feature(s): "
                    + ", ".join(uncovered_features) + ".",
                )
            has_usable_primary = any(
                source.importance == "required" and source.connector_available
                for source in source_requirements
            )
            only_fallback = bool(source_requirements) and all(
                source.importance == "fallback" for source in source_requirements
            )
            confidence = 0.20
            confidence += 0.15 if matched_geography else 0.0
            confidence += 0.40 * feature_coverage
            confidence += 0.10 if temporal_ok else 0.0
            confidence += 0.15 if has_usable_primary else 0.0
            confidence -= 0.15 if not has_usable_primary else 0.0
            confidence -= 0.10 if only_fallback else 0.0
            if requirement.freshness_need in {"live", "recent"} and not temporal_ok:
                confidence -= 0.1

            plan = PlanSpec(
                source_requirements=source_requirements,
                search_queries=queries,
                uncovered_features=uncovered_features,
                primary_source_types=primary,
                fallback_source_types=fallback,
                reasoning=self._rule_reasoning(spec),
                confidence=round(max(0.0, min(1.0, confidence)), 2),
                planning_policy=effective_policy,
            )
            # Assign plan first so _rule_reasoning can count sources from spec.plan
            spec.plan = plan
            plan.reasoning = self._rule_reasoning(spec)
            self._llm_refinement(plan, spec, use_llm=use_llm)
            spec.plan = plan
            spec.current_stage = "planning"
        except (ArithmeticError, KeyError, TypeError, ValueError, AttributeError):
            logger.exception("Unexpected programming or source-matching error while planning")
            raise

        print(
            f"✅ Planning complete: {len(spec.plan.source_requirements)} source type(s), "
            f"{len(spec.plan.search_queries)} search query(ies)"
        )
        return spec
