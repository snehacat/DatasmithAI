import pytest

from core.feature_names import normalize_feature_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("timestamp", "time"),
        ("river discharge", "streamflow"),
        ("river level", "waterlevel"),
        ("rainfall", "precipitation"),
        ("ozone", "ozone"),
        ("O3", "ozone"),
    ],
)
def test_normalize_feature_name_preserves_aliases(name, expected):
    assert normalize_feature_name(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TIMESTAMP", "time"),
        ("RiVeR DiScHaRgE", "streamflow"),
        ("RAINfALL", "precipitation"),
        ("o3", "ozone"),
    ],
)
def test_normalize_feature_name_is_case_insensitive(name, expected):
    assert normalize_feature_name(name) == expected


def test_normalize_feature_name_applies_unicode_nfkc():
    assert normalize_feature_name("\uff30\uff2d\uff12\uff0e\uff15") == "pm25"


@pytest.mark.parametrize("name", ["PM2.5", "pm2_5", "PM 2.5"])
def test_normalize_feature_name_preserves_pm25_normalization(name):
    assert normalize_feature_name(name) == "pm25"
