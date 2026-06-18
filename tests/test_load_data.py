import os
import tempfile
import numpy as np
import pytest
import yaml
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import logging

from cerf_data_centers import load_data

@pytest.fixture
def sample_yaml(tmp_path):
    data = {
        'settings': {
            'output_file': 'output.shp',
            'region_raster_path': 'region.tif',
            'siting_raster_path': 'suitability.tif'
        },
        'constraints': {
            'land_cost': 'land_cost.tif',
            'electricity_rate': 'elec_rate.tif'
        },
        'expansion_plan': {
            'california': {
                'region_id': 6,
                'campus_size_square_ft': 1000000,
                'equipment_capital_expenditure': 201600000,
                'building_capital_expenditure': 262200000,
                'interconnection_cost_km': 1000000,
                'data_center_power_mw': 36
            }
        }
    }
    yaml_path = tmp_path / "config.yml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return str(yaml_path), data

def test_read_yaml(sample_yaml):
    yaml_path, data = sample_yaml
    result = load_data.read_yaml(yaml_path)
    assert isinstance(result, dict)
    assert result['settings']['output_file'] == 'output.shp'

def test_get_yaml_valid(sample_yaml):
    yaml_path, data = sample_yaml
    result = load_data.get_yaml(yaml_path)
    assert result['settings']['region_raster_path'] == 'region.tif'

def test_get_yaml_none():
    with pytest.raises(AttributeError):
        load_data.get_yaml(None)

def test_get_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data.get_yaml("nonexistent_file.yml")

@pytest.fixture
def raster_file(tmp_path):
    # Create a 3x3 raster with values 1-9
    arr = np.arange(1, 10, dtype=np.float32).reshape((3, 3))
    transform = from_origin(0, 3, 1, 1)  # arbitrary
    raster_path = tmp_path / "test_raster.tif"
    with rasterio.open(
        raster_path, 'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(arr, 1)
    return str(raster_path), arr, transform

def test_load_region_raster(raster_file):
    raster_path, arr, transform = raster_file
    region_array, loaded_transform = load_data.load_region_raster(raster_path)
    assert np.array_equal(region_array, arr)
    assert loaded_transform == transform

def test_load_region_raster_with_window(raster_file):
    raster_path, arr, transform = raster_file
    window = Window(col_off=1, row_off=1, width=2, height=2)
    region_array, loaded_transform = load_data.load_region_raster(raster_path, window=window)
    np.testing.assert_array_equal(region_array, arr[1:3, 1:3])
    assert loaded_transform == rasterio.open(raster_path).window_transform(window)

def test_load_suitability_raster(raster_file):
    raster_path, arr, _ = raster_file
    suit_array = load_data.load_raster_array(raster_path)
    assert isinstance(suit_array, np.ndarray)
    assert suit_array.shape == arr.shape
    assert np.allclose(suit_array, arr)

def test_load_suitability_raster_with_window(raster_file):
    raster_path, arr, _ = raster_file
    window = Window(col_off=0, row_off=1, width=2, height=2)
    suit_array = load_data.load_raster_array(raster_path, window=window)
    np.testing.assert_array_equal(suit_array, arr[1:3, 0:2])

@pytest.fixture
def constraint_rasters(tmp_path):
    # Create two 3x3 rasters: one for land_cost, one for electricity_rate
    arr1 = np.ones((3, 3), dtype=np.float32) * 10  # land_cost
    arr2 = np.ones((3, 3), dtype=np.float32) * 5   # electricity_rate
    transform = from_origin(0, 3, 1, 1)
    land_cost_path = tmp_path / "land_cost.tif"
    elec_rate_path = tmp_path / "elec_rate.tif"
    for arr, path in zip([arr1, arr2], [land_cost_path, elec_rate_path]):
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(arr, 1)
    return [str(land_cost_path), str(elec_rate_path)], [arr1, arr2], transform

def test_collect_constraints(constraint_rasters):
    raster_paths, arrs, transform = constraint_rasters
    # Suitability: only (0,0), (1,1), (2,2) are suitable
    suit_array = np.zeros((3, 3), dtype=np.uint8)
    suit_array[0, 0] = 1
    suit_array[1, 1] = 1
    suit_array[2, 2] = 1
    raster_names = ['land_cost', 'electricity_rate']
    logger = logging.getLogger("test_logger")
    node_values = load_data.collect_constraints(
        suit_array, transform, raster_paths, raster_names, logger
    )
    # Should have 3 nodes
    assert len(node_values) == 3
    for node, vals in node_values.items():
        assert set(vals.keys()) == set(raster_names)
        assert vals['land_cost'] == 10
        assert vals['electricity_rate'] == 5

def test_collect_constraints_with_window(tmp_path):
    transform = from_origin(0, 4, 1, 1)
    land = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float32,
    )
    elec = land + 100
    land_path = tmp_path / "land_cost_window.tif"
    elec_path = tmp_path / "elec_rate_window.tif"
    for arr, path in [(land, land_path), (elec, elec_path)]:
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(arr, 1)

    suit_array = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    window = Window(col_off=1, row_off=1, width=2, height=2)
    node_values = load_data.collect_constraints(
        suit_array=suit_array,
        transform=transform,
        raster_paths=[str(land_path), str(elec_path)],
        raster_names=['land_cost', 'electricity_rate'],
        logger=logging.getLogger("test_logger"),
        window=window,
    )

    assert node_values[(0, 1)]['land_cost'] == 7
    assert node_values[(0, 1)]['electricity_rate'] == 107
    assert node_values[(1, 0)]['land_cost'] == 10
    assert node_values[(1, 0)]['electricity_rate'] == 110

def test_collect_constraints_empty_suitability(constraint_rasters):
    raster_paths, arrs, transform = constraint_rasters
    suit_array = np.zeros((3, 3), dtype=np.uint8)
    raster_names = ['land_cost', 'electricity_rate']
    logger = logging.getLogger("test_logger")
    node_values = load_data.collect_constraints(
        suit_array, transform, raster_paths, raster_names, logger
    )
    assert node_values == {}

def test_find_region_window(tmp_path):
    region = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 7, 7, 0, 0],
            [0, 7, 7, 0, 8],
            [0, 0, 0, 0, 8],
        ],
        dtype=np.int16,
    )
    transform = from_origin(0, 4, 1, 1)
    region_path = tmp_path / "regions.tif"
    with rasterio.open(
        region_path, 'w',
        driver='GTiff',
        height=region.shape[0],
        width=region.shape[1],
        count=1,
        dtype=region.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(region, 1)

    window = load_data.find_region_window(str(region_path), [7, 8])
    assert window.row_off == 1
    assert window.col_off == 1
    assert window.height == 3
    assert window.width == 4

def test_validate_raster_alignment_detects_mismatch(tmp_path):
    transform_a = from_origin(0, 3, 1, 1)
    transform_b = from_origin(1, 3, 1, 1)
    ref = np.ones((3, 3), dtype=np.float32)
    other = np.ones((3, 3), dtype=np.float32)
    ref_path = tmp_path / "ref.tif"
    other_path = tmp_path / "other.tif"

    for arr, path, transform in [(ref, ref_path, transform_a), (other, other_path, transform_b)]:
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(arr, 1)

    with pytest.raises(ValueError, match="Raster alignment mismatch"):
        load_data.validate_raster_alignment(str(ref_path), [str(other_path)])
