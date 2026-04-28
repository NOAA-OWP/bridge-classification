"""Tests for src/lidar_utils shared utilities."""

import pytest

gpd = pytest.importorskip("geopandas", reason="geopandas not installed")

from src.lidar_utils import (
    EPSG,
    DEFAULT_BUFFER,
    safe_source_name,
    bridge_stem,
    parse_bridge_stem,
    stratified_sample,
)


def test_safe_source_name():
    assert safe_source_name("US/Source Name") == "US_Source_Name"
    assert safe_source_name("USGS_LPC_PA:2017") == "USGS_LPC_PA_2017"
    assert safe_source_name("ME_CrownofMaine_B2_2018") == "ME_CrownofMaine_B2_2018"
    assert safe_source_name("path\\with\\backslash") == "path_with_backslash"


def test_bridge_stem():
    assert bridge_stem("123", "ME_Crown/B2 2018") == "bridge_123_ME_Crown_B2_2018"
    assert bridge_stem("456", "USGS_LPC_PA:2017") == "bridge_456_USGS_LPC_PA_2017"
    assert bridge_stem("789", "Simple_Source") == "bridge_789_Simple_Source"


def test_parse_bridge_stem():
    assert parse_bridge_stem("bridge_123_SOURCE_NAME") == ("123", "SOURCE_NAME")
    assert parse_bridge_stem("bridge_456_ME_Crown_B2") == ("456", "ME_Crown_B2")
    assert parse_bridge_stem("not_a_bridge") is None
    assert parse_bridge_stem("bridge_") is None
    assert parse_bridge_stem("") is None


def test_parse_bridge_stem_roundtrip():
    stem = bridge_stem("12345", "USGS_LPC_PA_2017")
    parsed = parse_bridge_stem(stem)
    assert parsed is not None
    assert parsed[0] == "12345"
    assert parsed[1] == "USGS_LPC_PA_2017"


def test_stratified_sample_basic():
    entries = [{"huc_id": f"huc{i % 3}", "osm_id": str(i)} for i in range(20)]
    sample = stratified_sample(entries, 5, 2, seed=27)
    assert len(sample) == 5
    hucs = set(e["huc_id"] for e in sample)
    assert len(hucs) >= 2


def test_stratified_sample_deduplication():
    entries = [
        {"huc_id": "h1", "osm_id": "1"},
        {"huc_id": "h1", "osm_id": "1"},
        {"huc_id": "h2", "osm_id": "2"},
    ]
    sample = stratified_sample(entries, 10, 5, seed=27)
    assert len(sample) == 2


def test_stratified_sample_max_per_group():
    entries = [{"huc_id": "h1", "osm_id": str(i)} for i in range(10)]
    sample = stratified_sample(entries, 10, 2, seed=27)
    assert len(sample) == 2


def test_stratified_sample_custom_keys():
    entries = [{"region": f"r{i % 3}", "bridge_id": str(i)} for i in range(12)]
    sample = stratified_sample(entries, 4, 2, seed=27, group_key="region", id_key="bridge_id")
    assert len(sample) == 4
    regions = set(e["region"] for e in sample)
    assert len(regions) >= 2


def test_stratified_sample_deterministic():
    entries = [{"huc_id": f"huc{i % 5}", "osm_id": str(i)} for i in range(30)]
    s1 = stratified_sample(entries, 10, 3, seed=42)
    s2 = stratified_sample(entries, 10, 3, seed=42)
    assert s1 == s2


def test_constants():
    assert EPSG == 3857
    assert DEFAULT_BUFFER == 10.0
