from pathlib import Path
import sys

from click.testing import CliRunner
import geopandas as gpd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
import warnings
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cerf_data_centers.run_siting as run_siting


warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
    module="pyogrio"
)


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _create_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(0, data.shape[0], 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def minimal_config(tmp_path):
    config = {
        "settings": {
            "region_raster_path": str(tmp_path / "dummy_region.tif"),
            "siting_raster_path": str(tmp_path / "dummy_suit.tif"),
            "output_file": str(tmp_path / "output.geojson"),
        },
        "constraints": {
            "land_cost_per_sqft": str(tmp_path / "land_cost.tif"),
            "electricity_rate_per_kwh": str(tmp_path / "electricity.tif"),
            "personal_prop_tax_rate": str(tmp_path / "personal_prop_tax.tif"),
            "real_property_tax_rate": str(tmp_path / "real_prop_tax.tif"),
            "sales_tax_rate": str(tmp_path / "sales_tax.tif"),
            "interconnection_distance_km": str(tmp_path / "interconnection.tif"),
            "mechanical_cool_fraction": str(tmp_path / "mechanical_cool.tif"),
            "water_cool_fraction": str(tmp_path / "water_cool.tif"),
        },
        "market_gravity": {
            "market_raster_path": str(tmp_path / "market.tif")
        },
        "expansion_plan": {
            "TestRegion": {
                "region_id": 1,
                "campus_size_square_ft": 10000,
                "equipment_capital_expenditure_usd": 1000000,
                "building_capital_expenditure_usd": 2000000,
                "data_center_it_power_mw": 5,
                "data_center_pue": 1.5,
                "interconnection_cost_usd_per_km": 50000,
                "n_sites": 1,
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, config)
    return config_path


@pytest.fixture
def dummy_raster_files(tmp_path):
    raster_names = [
        "dummy_region.tif",
        "dummy_suit.tif",
        "land_cost.tif",
        "electricity.tif",
        "personal_prop_tax.tif",
        "real_prop_tax.tif",
        "sales_tax.tif",
        "interconnection.tif",
        "mechanical_cool.tif",
        "water_cool.tif",
        "market.tif",
    ]
    for name in raster_names:
        _create_raster(tmp_path / name, np.ones((10, 10), dtype=np.float32))
    return tmp_path


@pytest.fixture
def two_region_config(tmp_path):
    region = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    suitability = np.where(region > 0, 1, 0).astype(np.float32)
    land_cost = np.array(
        [
            [1.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 9.0],
            [0.0, 0.0, 8.0, 7.0],
        ],
        dtype=np.float32,
    )
    constant = np.where(region > 0, 1.0, 0.0).astype(np.float32)

    _create_raster(tmp_path / "region.tif", region)
    _create_raster(tmp_path / "suitability.tif", suitability)
    _create_raster(tmp_path / "land_cost.tif", land_cost)
    _create_raster(tmp_path / "electricity.tif", constant * 0.05)
    _create_raster(tmp_path / "personal_tax.tif", constant * 0.01)
    _create_raster(tmp_path / "real_tax.tif", constant * 0.02)
    _create_raster(tmp_path / "sales_tax.tif", constant * 0.03)
    _create_raster(tmp_path / "interconnection.tif", constant)
    _create_raster(tmp_path / "mechanical.tif", constant * 0.5)
    _create_raster(tmp_path / "water.tif", constant * 0.5)
    _create_raster(tmp_path / "market.tif", constant)

    config = {
        "settings": {
            "region_raster_path": str(tmp_path / "region.tif"),
            "siting_raster_path": str(tmp_path / "suitability.tif"),
            "output_file": str(tmp_path / "result.geojson"),
            "cost_weight": 1.0,
            "market_weight": 0.0,
        },
        "constraints": {
            "land_cost_per_sqft": str(tmp_path / "land_cost.tif"),
            "electricity_rate_per_kwh": str(tmp_path / "electricity.tif"),
            "personal_prop_tax_rate": str(tmp_path / "personal_tax.tif"),
            "real_property_tax_rate": str(tmp_path / "real_tax.tif"),
            "sales_tax_rate": str(tmp_path / "sales_tax.tif"),
            "interconnection_distance_km": str(tmp_path / "interconnection.tif"),
            "mechanical_cool_fraction": str(tmp_path / "mechanical.tif"),
            "water_cool_fraction": str(tmp_path / "water.tif"),
        },
        "market_gravity": {
            "market_raster_path": str(tmp_path / "market.tif"),
        },
        "expansion_plan": {
            "region_01": {
                "region_id": 1,
                "campus_size_square_ft": 1,
                "equipment_capital_expenditure_usd": 1000,
                "building_capital_expenditure_usd": 2000,
                "data_center_it_power_mw": 1,
                "data_center_pue": 1.2,
                "interconnection_cost_usd_per_km": 100,
                "n_sites": 1,
            },
            "region_02": {
                "region_id": 2,
                "campus_size_square_ft": 1,
                "equipment_capital_expenditure_usd": 1000,
                "building_capital_expenditure_usd": 2000,
                "data_center_it_power_mw": 1,
                "data_center_pue": 1.2,
                "interconnection_cost_usd_per_km": 100,
                "n_sites": 1,
            },
        },
    }
    config_path = tmp_path / "two_region.yaml"
    _write_yaml(config_path, config)
    return config_path


def _normalize_output(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = gdf.copy()
    normalized["geometry"] = normalized.geometry.to_wkt()
    normalized = normalized[sorted(normalized.columns)]
    normalized = normalized.sort_values(by=["id"]).reset_index(drop=True)
    return normalized


@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_site_command(monkeypatch, minimal_config, dummy_raster_files, tmp_path):
    called = []

    def mock_run(config_path, max_workers=None):
        called.append((config_path, max_workers))
        return gpd.GeoDataFrame(
            {
                "id": ["site_1"],
                "region": ["TestRegion"],
                "campus_size_square_ft": [1],
                "land_building_cost_usd": [1.0],
                "total_capex_cost": [1.0],
            },
            geometry=[Point(0, 0)],
            crs="ESRI:102003",
        )

    monkeypatch.setattr(run_siting, "run", mock_run)

    runner = CliRunner()
    result = runner.invoke(run_siting.site, [str(minimal_config), "--workers", "3"])
    assert result.exit_code == 0
    assert called == [(str(minimal_config), 3)]

    output_path = tmp_path / "cli_output.geojson"
    result2 = runner.invoke(run_siting.site, [str(minimal_config), "--output", str(output_path)])
    assert result2.exit_code == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_group_help():
    runner = CliRunner()
    result = runner.invoke(run_siting.cli, ["--help"])
    assert result.exit_code == 0
    assert "Data Center Siting Tool" in result.output


@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_site_help():
    runner = CliRunner()
    result = runner.invoke(run_siting.site, ["--help"])
    assert result.exit_code == 0
    assert "Run the data center siting process" in result.output


def test_resolve_region_scenarios_uses_global_scenarios():
    region_dict = {"region_id": 1}
    scenarios = {
        "large": {"campus_size_square_ft": 300, "n_sites": 2},
        "small": {"campus_size_square_ft": 100, "n_sites": 5},
    }

    resolved = run_siting._resolve_region_scenarios("region_01", region_dict, scenarios)

    assert [name for name, _ in resolved] == ["large", "small"]
    assert resolved[0][1]["region_id"] == 1
    assert resolved[0][1]["campus_size_square_ft"] == 300
    assert resolved[1][1]["campus_size_square_ft"] == 100


def test_resolve_region_scenarios_supports_legacy_region_config():
    region_dict = {
        "region_id": 1,
        "campus_size_square_ft": 100,
        "equipment_capital_expenditure_usd": 1000,
        "building_capital_expenditure_usd": 500,
        "data_center_it_power_mw": 1,
        "data_center_pue": 1.2,
        "interconnection_cost_usd_per_km": 10,
        "n_sites": 1,
    }

    resolved = run_siting._resolve_region_scenarios("region_01", region_dict, None)

    assert resolved == [("default", region_dict)]


def test_drop_overlapping_smaller_sites_prefers_larger_site():
    gdf = gpd.GeoDataFrame(
        {
            "id": ["small", "large", "other"],
            "campus_size_square_ft": [100, 300, 50],
            "data_center_it_power_mw": [1, 3, 1],
            "total_capex_cost": [10, 30, 5],
            "geometry": [
                box(0, 0, 10, 10),
                box(5, 5, 15, 15),
                box(20, 20, 25, 25),
            ],
        },
        geometry="geometry",
        crs="ESRI:102003",
    )

    filtered = run_siting._drop_overlapping_smaller_sites(gdf)

    assert list(filtered["id"]) == ["large", "other"]


def test_run_max_workers_one_matches_default(two_region_config):
    default_result = run_siting.run(str(two_region_config))
    explicit_result = run_siting.run(str(two_region_config), max_workers=1)

    assert_frame_equal(
        _normalize_output(default_result),
        _normalize_output(explicit_result),
        check_dtype=False,
    )


def test_run_parallel_matches_serial(two_region_config):
    serial_result = run_siting.run(str(two_region_config), max_workers=1)
    parallel_result = run_siting.run(str(two_region_config), max_workers=2)

    assert_frame_equal(
        _normalize_output(serial_result),
        _normalize_output(parallel_result),
        check_dtype=False,
    )


def test_run_parallel_failure_does_not_write_output(monkeypatch, two_region_config):
    output_path = Path(yaml.safe_load(two_region_config.read_text(encoding="utf-8"))["settings"]["output_file"])

    def mock_parallel(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_siting, "_run_regions_in_parallel", mock_parallel)

    with pytest.raises(RuntimeError, match="boom"):
        run_siting.run(str(two_region_config), max_workers=2)

    assert not output_path.exists()
