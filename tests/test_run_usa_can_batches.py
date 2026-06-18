import importlib.util
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_usa_can_batches.py"
SPEC = importlib.util.spec_from_file_location("run_usa_can_batches", SCRIPT_PATH)
run_usa_can_batches = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_usa_can_batches)


@pytest.fixture
def sample_master_config():
    return {
        "settings": {
            "region_raster_path": "region.tif",
            "siting_raster_path": "suitability.tif",
            "output_root": "outputs/usa_can_batches",
            "cost_weight": 0.5,
            "market_weight": 0.5,
        },
        "constraints": {
            "land_cost_per_sqft": "land.tif",
            "electricity_rate_per_kwh": "electricity.tif",
            "personal_prop_tax_rate": "personal_tax.tif",
            "real_property_tax_rate": "real_tax.tif",
            "sales_tax_rate": "sales_tax.tif",
            "interconnection_distance_km": "interconnection.tif",
            "mechanical_cool_fraction": "mechanical.tif",
            "water_cool_fraction": "water.tif",
        },
        "market_gravity": {"market_raster_path": "market.tif"},
        "expansion_plan": {
            "region_01": {
                "region_id": 1,
                "campus_size_square_ft": 150000,
                "equipment_capital_expenditure_usd": 1,
                "building_capital_expenditure_usd": 2,
                "data_center_it_power_mw": 3,
                "data_center_pue": 1.2,
                "interconnection_cost_usd_per_km": 4,
                "n_sites": 5,
            },
            "region_02": {
                "region_id": 2,
                "campus_size_square_ft": 150000,
                "equipment_capital_expenditure_usd": 1,
                "building_capital_expenditure_usd": 2,
                "data_center_it_power_mw": 3,
                "data_center_pue": 1.2,
                "interconnection_cost_usd_per_km": 4,
                "n_sites": 5,
            },
            "region_03": {
                "region_id": 3,
                "campus_size_square_ft": 150000,
                "equipment_capital_expenditure_usd": 1,
                "building_capital_expenditure_usd": 2,
                "data_center_it_power_mw": 3,
                "data_center_pue": 1.2,
                "interconnection_cost_usd_per_km": 4,
                "n_sites": 5,
            },
        },
    }


@pytest.fixture
def sample_batch_config():
    return {
        "batch_defaults": {
            "output_name_template": "batch_{batch_id:02d}.geojson",
            "config_name_template": "batch_{batch_id:02d}.yaml",
            "log_name_template": "batch_{batch_id:02d}.log",
        },
        "batches": [
            {"batch_id": 1, "region_ids": [1, 2]},
            {"batch_id": 2, "region_ids": [3]},
        ],
    }


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def test_write_batch_configs_generates_expected_yaml(tmp_path, sample_master_config, sample_batch_config):
    master_path = tmp_path / "master.yaml"
    batch_path = tmp_path / "batches.yaml"
    output_root = tmp_path / "outputs"
    write_yaml(master_path, sample_master_config)
    write_yaml(batch_path, sample_batch_config)

    summaries = run_usa_can_batches.write_batch_configs(master_path, batch_path, output_root)

    assert len(summaries) == 2

    batch_one_config = run_usa_can_batches.load_yaml(output_root / "configs" / "batch_01.yaml")
    batch_two_config = run_usa_can_batches.load_yaml(output_root / "configs" / "batch_02.yaml")

    assert sorted(batch_one_config["settings"]["selected_region_ids"]) == [1, 2]
    assert sorted(region["region_id"] for region in batch_one_config["expansion_plan"].values()) == [1, 2]
    assert batch_one_config["settings"]["batch_id"] == 1
    assert batch_one_config["settings"]["output_file"].endswith("/outputs/batch_01/batch_01.geojson")

    assert batch_two_config["settings"]["selected_region_ids"] == [3]
    assert sorted(region["region_id"] for region in batch_two_config["expansion_plan"].values()) == [3]
    assert batch_two_config["settings"]["batch_id"] == 2
    assert batch_two_config["settings"]["output_file"].endswith("/outputs/batch_02/batch_02.geojson")


def test_validate_batch_assignments_rejects_duplicate_region_ids(sample_master_config, sample_batch_config):
    sample_batch_config["batches"][1]["region_ids"] = [2]

    with pytest.raises(ValueError, match="assigned more than once"):
        run_usa_can_batches.validate_batch_assignments(sample_master_config, sample_batch_config)


def test_validate_batch_assignments_rejects_missing_region_ids(sample_master_config, sample_batch_config):
    sample_batch_config["batches"][1]["region_ids"] = [99]

    with pytest.raises(ValueError, match="not defined in expansion_plan"):
        run_usa_can_batches.validate_batch_assignments(sample_master_config, sample_batch_config)


def test_validate_batch_assignments_requires_contiguous_batch_ids(sample_master_config, sample_batch_config):
    sample_batch_config["batches"][1]["batch_id"] = 3

    with pytest.raises(ValueError, match="continuous starting at 1"):
        run_usa_can_batches.validate_batch_assignments(sample_master_config, sample_batch_config)
