"""Canonical feature-name normalization shared by planning and schema validation."""

import re
import unicodedata


def normalize_feature_name(name: str) -> str:
    """Canonicalize spelling and explicitly supported feature aliases.

    >>> normalize_feature_name("PM2.5") == "pm25"
    >>> normalize_feature_name("PM 2.5") == "pm25"
    >>> normalize_feature_name("rainfall") == "precipitation"
    >>> normalize_feature_name("O3") == "ozone"
    """
    compact = re.sub(
        r"[\s_\-\.]",
        "",
        unicodedata.normalize("NFKC", name).casefold(),
    )
    aliases = {
        "timestamp": "time",
        "riverdischarge": "streamflow",
        "riverlevel": "waterlevel",
        "waterlevel": "waterlevel",
        "rainfall": "precipitation",
        "precipitation": "precipitation",
        "ozone": "ozone",
        "o3": "ozone",
    }
    return aliases.get(compact, compact)
