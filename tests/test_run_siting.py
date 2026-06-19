import pytest
import tempfile
import os
import sys
import yaml
import geopandas as gpd
from pathlib import Path
from click.testing import CliRunner
import types
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import warnings
import cerf_data_centers.run_siting

# Filter out the shapely.geos deprecation warning
warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
    module="pyogrio"
)

run_siting = types.SimpleNamespace(**{
    name: getattr(cerf_data_centers.run_siting, name)
    for name in dir(cerf_data_centers.run_siting)
    if not name.startswith("__")
})

@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal valid config file for testing."""
    config = {
        "settings": {
            "region_raster_path": str(tmp_path / "dummy_region.tif"),
            "siting_raster_path": str(tmp_path / "dummy_suit.tif"),
            "output_file": str(tmp_path / "output.geojson")
        },
        "constraints": {
            "land_cost_per_sqft": str(tmp_path / "land_cost.tif"),
            "electricity_rate_per_kwh": str(tmp_path / "electricity.tif"),
            "personal_prop_tax_rate": str(tmp_path / "personal_prop_tax.tif"),
            "real_property_tax_rate": str(tmp_path / "real_prop_tax.tif"),
            "sales_tax_rate": str(tmp_path / "sales_tax.tif"),
            "interconnection_distance_km": str(tmp_path / "interconnection.tif"),
            "mechanical_cool_fraction": str(tmp_path / "mechanical_cool.tif"),
            "water_cool_fraction": str(tmp_path / "water_cool.tif")
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
                "n_sites": 1
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def create_dummy_raster(path, value=1):
    """Create a valid dummy raster file."""
    # Create a 10x10 array
    data = np.ones((10, 10), dtype=np.float32) * value
    
    # Create a transform
    transform = from_origin(0, 10, 1, 1)
    
    # Create the raster file
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data, 1)

@pytest.fixture
def dummy_raster_files(tmp_path):
    """Create dummy raster files for testing."""
    # Create dummy files for all required rasters
    raster_files = [
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
        "market.tif"
    ]
    for file in raster_files:
        create_dummy_raster(tmp_path / file)
    return tmp_path

@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_site_command(monkeypatch, minimal_config, dummy_raster_files, tmp_path):
    """Test the CLI 'site' command."""
    # Mock the entire run function to avoid complex mocking
    def mock_run(config_path):
        return gpd.GeoDataFrame({'test': [1, 2, 3]})
    
    monkeypatch.setattr(run_siting, "run", mock_run)

    runner = CliRunner()
    result = runner.invoke(run_siting.site, [str(minimal_config)])
    assert result.exit_code == 0

    # Test with output option
    output_path = tmp_path / "cli_output.geojson"
    result2 = runner.invoke(run_siting.site, [str(minimal_config), "--output", str(output_path)])
    assert result2.exit_code == 0

@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_group_help():
    """Test the CLI group help command."""
    runner = CliRunner()
    result = runner.invoke(run_siting.cli, ["--help"])
    assert result.exit_code == 0
    assert "Data Center Siting Tool" in result.output

@pytest.mark.filterwarnings("ignore::DeprecationWarning:pyogrio.*:")
def test_cli_site_help():
    """Test the CLI site command help."""
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
