"""Static provider hints for planning; verify capabilities and access before use."""

import re
from dataclasses import dataclass
from typing import List


def normalize_feature_name(name: str) -> str:
    """Canonicalize spelling and explicitly supported feature aliases.

    Examples
    --------
    >>> normalize_feature_name("PM2.5")   == "pm25"
    >>> normalize_feature_name("pm2_5")   == "pm25"
    >>> normalize_feature_name("PM 2.5")  == "pm25"
    >>> normalize_feature_name("no2")     == "no2"
    >>> normalize_feature_name("timestamp") == "time"
    >>> normalize_feature_name("river discharge") == "streamflow"
    >>> normalize_feature_name("river level") == "waterlevel"
    >>> normalize_feature_name("rainfall") == "precipitation"
    >>> normalize_feature_name("ozone") == "ozone"
    >>> normalize_feature_name("O3") == "ozone"
    """
    compact = re.sub(r"[\s_\-\.]", "", name.lower())
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


@dataclass(frozen=True)
class SourceCatalogEntry:
    source_type: str
    name: str
    covers: List[str]
    provides: List[str]
    access: str
    requires_key: bool
    requires_login: bool
    free: bool
    geography: str
    temporal: str
    notes: str
    public: bool = True
    # Connector status is separate from source relevance and importance.
    # Only set True once a connector is implemented and tested.
    connector_available: bool = False
    # Origin of the data: "measured" = real sensor/station readings,
    # "modeled" = reanalysis / forecast model output,
    # "mixed" = blend of measured and modeled (verify before use).
    data_origin: str = "measured"


SOURCE_CATALOG = [
    SourceCatalogEntry(
        "open_meteo_weather", "Open-Meteo",
        ["weather", "meteorological", "temperature", "rainfall", "precipitation", "wind", "humidity"],
        ["temperature", "precipitation", "wind speed", "wind direction", "humidity", "cloud cover"],
        "api", False, False, True, "global", "all",
        "Provider coverage and historical limits vary; verify before use.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "commercial_weather_api", "Commercial weather API (example)",
        ["weather", "meteorological", "temperature", "rainfall"],
        ["temperature", "precipitation", "wind speed", "humidity"],
        "api", True, True, False, "global", "all",
        "Generic category only, not a named provider; verify terms and pricing before use.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "nasa_power_climate", "NASA POWER",
        ["climate", "weather", "solar", "solar radiation", "temperature", "meteorological"],
        ["solar radiation", "temperature", "precipitation", "wind speed", "humidity"],
        "api", False, False, True, "global", "historical",
        "Temporal resolution and variables vary by product; verify before use.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "era5_reanalysis", "ERA5 reanalysis",
        ["climate", "weather", "meteorological", "temperature", "precipitation", "wind"],
        ["temperature", "precipitation", "wind speed", "humidity", "surface pressure"],
        "download", True, True, True, "global", "historical",
        "Copernicus account and data access terms may apply; verify before use.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "noaa_weather", "NOAA",
        ["weather", "meteorological", "climate", "ocean", "temperature"],
        ["temperature", "precipitation", "wind speed", "humidity", "sea surface temperature"],
        "api", True, False, True, "us", "all",
        "Products and access requirements vary; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "imd_weather_india", "India Meteorological Department",
        ["weather", "meteorological", "climate", "rainfall", "temperature"],
        ["rainfall", "temperature", "humidity", "wind speed"],
        "portal", False, False, True, "india", "historical",
        "Portal access, coverage, and download formats may vary; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "openaq_air_quality", "OpenAQ",
        ["air quality", "pollution", "pm2.5", "pm10", "ozone", "aqi", "no2", "so2"],
        ["pm2.5", "pm10", "o3", "no2", "so2", "timestamp"],
        "api", True, False, True, "global", "live",
        "OpenAQ streams station readings in near-real-time and also archives history. "
        "A free API key is sufficient; no paid tier is required for basic access. "
        "Verify station density and rate limits before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "cpcb_namp_air_quality", "CPCB NAMP",
        ["air quality", "pollution", "pm2.5", "pm10", "aqi", "no2", "so2"],
        ["pm2.5", "pm10", "no2", "so2", "air quality index", "timestamp"],
        "portal", False, False, True, "india", "live",
        "CPCB publishes near-real-time hourly readings from CAAQMS stations across India, "
        "including Delhi. Historical data is also available via the portal. "
        "Verify portal availability and station coverage before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "open_meteo_air_quality", "Open-Meteo Air Quality",
        ["air quality", "pollution", "pm2.5", "pm10", "ozone", "aqi", "no2", "so2"],
        ["pm2.5", "pm10", "o3", "no2", "so2", "timestamp"],
        "api", False, False, True, "global", "live",
        "Completely free, no API key required. Provides hourly air-quality forecasts "
        "and recent historical values via a simple REST endpoint. "
        "Verify before use: coverage and variables may change between API versions.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "usgs_earthquake_catalog", "USGS Earthquake Catalog",
        ["earthquakes", "earthquake", "seismic", "geological", "natural disasters"],
        ["magnitude", "depth", "location", "time"],
        "api", False, False, True, "global", "all",
        "Catalog product and query limits may apply; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "nasa_firms_fire", "NASA FIRMS",
        ["fire", "wildfire", "forest fire", "forest fire monitoring"],
        ["fire detections", "confidence", "brightness", "acquisition time", "location"],
        "api", True, False, True, "global", "live",
        "NASA FIRMS streams MODIS/VIIRS active fire detections within ~3 hours of "
        "satellite overpass. A free MAP key is required; historical archive is also "
        "available. Verify coverage and latency before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "copernicus_glofas_flood", "Copernicus GloFAS",
        ["flood", "flood monitoring", "hydrology", "hydrological", "river"],
        ["river discharge", "flood extent", "forecast", "location", "time"],
        "download", True, True, True, "global", "all",
        "Account, licensing, and product suitability vary; verify before use.",
        connector_available=False, data_origin="modeled",
    ),
    SourceCatalogEntry(
        "india_cwc_flood_portal", "India CWC",
        ["flood", "flood monitoring", "hydrology", "hydrological", "river"],
        ["river level", "river discharge", "flood status", "location", "time"],
        "portal", False, False, True, "india", "live",
        "CWC publishes near-real-time gauge readings and flood bulletins. "
        "Portal availability and downloadable history are uncertain; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "copernicus_marine", "Copernicus Marine",
        ["ocean", "oceanographic", "marine", "sea", "sea level", "sea surface"],
        ["sea surface temperature", "sea level", "salinity", "ocean currents", "wave height"],
        "download", True, True, True, "global", "all",
        "Account and product access requirements vary; verify before use.",
        connector_available=False, data_origin="mixed",
    ),
    SourceCatalogEntry(
        "noaa_ocean_data", "NOAA Ocean Data",
        ["ocean", "oceanographic", "marine", "sea", "sea level", "sea surface"],
        ["sea surface temperature", "sea level", "salinity", "wave height"],
        "download", False, False, True, "us", "historical",
        "Products and geographic coverage vary; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "sentinel_satellite", "Copernicus Sentinel",
        ["satellite", "satellite data", "remote sensing", "earth observation", "land cover"],
        ["satellite imagery", "vegetation", "land cover", "surface reflectance"],
        "download", True, True, True, "global", "historical",
        "Account and product-specific access rules apply; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "landsat_satellite", "USGS Landsat",
        ["satellite", "satellite data", "remote sensing", "earth observation", "land cover"],
        ["satellite imagery", "surface reflectance", "land cover", "temperature"],
        "download", True, True, True, "global", "historical",
        "Account and product-specific access rules apply; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "modis_satellite", "NASA MODIS",
        ["satellite", "satellite data", "remote sensing", "earth observation", "forest fire", "vegetation"],
        ["satellite imagery", "vegetation", "fire detections", "land surface temperature", "aerosol"],
        "download", True, True, True, "global", "historical",
        "NASA Earthdata login and product-specific terms may apply; verify before use.",
        connector_available=False, data_origin="measured",
    ),
    SourceCatalogEntry(
        "nasa_earthdata_general", "NASA Earthdata",
        [
            "environmental", "general", "earth observation", "satellite", "climate",
            "water quality",
        ],
        ["satellite imagery", "climate measurements", "environmental observations"],
        "portal", True, True, True, "global", "historical",
        "Catalog contains many distinct datasets; verify product details before use.",
        connector_available=False, data_origin="mixed",
    ),
]
