from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from cerf_data_centers.postprocess_output import generate_output_artifacts
from cerf_data_centers.run_siting import run


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def dump_yaml(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def index_expansion_plan_by_region_id(expansion_plan: dict) -> dict[int, tuple[str, dict]]:
    indexed: dict[int, tuple[str, dict]] = {}
    for region_name, region_config in expansion_plan.items():
        region_id = region_config["region_id"]
        if region_id in indexed:
            raise ValueError(f"Duplicate region_id {region_id} found in expansion_plan.")
        indexed[region_id] = (region_name, region_config)
    return indexed


def validate_batch_assignments(master_config: dict, batch_config: dict) -> None:
    expansion_plan = master_config.get("expansion_plan", {})
    indexed_regions = index_expansion_plan_by_region_id(expansion_plan)
    batches = batch_config.get("batches", [])

    batch_ids = [batch["batch_id"] for batch in batches]
    expected_batch_ids = list(range(1, len(batches) + 1))
    if batch_ids != expected_batch_ids:
        raise ValueError(
            f"Batch IDs must be continuous starting at 1. Found {batch_ids}, expected {expected_batch_ids}."
        )

    assigned_region_ids: set[int] = set()
    for batch in batches:
        batch_id = batch["batch_id"]
        for region_id in batch["region_ids"]:
            if region_id not in indexed_regions:
                raise ValueError(
                    f"Batch {batch_id} references region_id {region_id}, which is not defined in expansion_plan."
                )
            if region_id in assigned_region_ids:
                raise ValueError(
                    f"region_id {region_id} is assigned more than once across batch definitions."
                )
            assigned_region_ids.add(region_id)


def build_batch_config(
    master_config: dict,
    region_ids: list[int],
    batch_id: int,
    output_root: Path,
) -> dict:
    batch_defaults = master_config.get("batch_defaults", {})
    output_name_template = batch_defaults.get("output_name_template", "batch_{batch_id:02d}.geojson")

    indexed_regions = index_expansion_plan_by_region_id(master_config["expansion_plan"])

    batch_config = deepcopy(master_config)
    batch_expansion_plan = {}
    for region_id in region_ids:
        region_name, region_config = indexed_regions[region_id]
        batch_expansion_plan[region_name] = deepcopy(region_config)

    batch_output_dir = output_root / f"batch_{batch_id:02d}"
    batch_output_file = batch_output_dir / output_name_template.format(batch_id=batch_id)

    batch_config["expansion_plan"] = batch_expansion_plan
    batch_config.setdefault("settings", {})
    batch_config["settings"]["batch_id"] = batch_id
    batch_config["settings"]["selected_region_ids"] = region_ids
    batch_config["settings"]["output_file"] = str(batch_output_file.resolve()).replace("\\", "/")
    batch_config["settings"]["output_root"] = str(output_root.resolve()).replace("\\", "/")

    return batch_config


def write_batch_configs(
    master_config_path: str | Path,
    batch_config_path: str | Path,
    output_root: str | Path,
) -> list[dict]:
    master_config = load_yaml(master_config_path)
    batch_plan = load_yaml(batch_config_path)
    validate_batch_assignments(master_config, batch_plan)

    output_root_path = Path(output_root).resolve()
    configs_dir = output_root_path / "configs"
    merged_dir = output_root_path / "merged"
    configs_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    batch_defaults = batch_plan.get("batch_defaults", {})
    config_name_template = batch_defaults.get("config_name_template", "batch_{batch_id:02d}.yaml")
    output_name_template = batch_defaults.get("output_name_template", "batch_{batch_id:02d}.geojson")

    master_config_with_defaults = deepcopy(master_config)
    master_config_with_defaults["batch_defaults"] = batch_defaults

    summaries: list[dict] = []
    for batch in batch_plan["batches"]:
        batch_id = batch["batch_id"]
        region_ids = batch["region_ids"]
        batch_dir = output_root_path / f"batch_{batch_id:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        derived_config = build_batch_config(
            master_config=master_config_with_defaults,
            region_ids=region_ids,
            batch_id=batch_id,
            output_root=output_root_path,
        )

        config_path = configs_dir / config_name_template.format(batch_id=batch_id)
        expected_output_path = batch_dir / output_name_template.format(batch_id=batch_id)
        dump_yaml(config_path, derived_config)

        summary = {
            "batch_id": batch_id,
            "region_ids": region_ids,
            "config_path": config_path,
            "output_path": expected_output_path,
        }
        summaries.append(summary)
        print(
            f"Batch {batch_id:02d}: region_ids={region_ids} "
            f"config={config_path} output={expected_output_path}"
        )

    return summaries


def _execute_batch_config(config_path: str | Path, region_workers: int = 1) -> str:
    run(str(config_path), max_workers=max(1, region_workers))
    return str(config_path)


def execute_batches(
    batch_summaries: list[dict],
    max_workers: int = 1,
    region_workers: int = 1,
) -> None:
    worker_count = max(1, max_workers)
    region_worker_count = max(1, region_workers)

    if worker_count == 1:
        for summary in batch_summaries:
            _execute_batch_config(summary["config_path"], region_workers=region_worker_count)
        return

    future_to_batch = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for summary in batch_summaries:
            future = executor.submit(
                _execute_batch_config,
                str(summary["config_path"]),
                region_worker_count,
            )
            future_to_batch[future] = summary["batch_id"]

        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                future.result()
            except Exception as exc:
                for pending_future in future_to_batch:
                    pending_future.cancel()
                raise RuntimeError(f"Batch {batch_id:02d} execution failed.") from exc
            print(f"Batch {batch_id:02d} complete")


def merge_outputs(output_root: Path) -> None:
    """
    Merge batch GeoJSON outputs into a single GeoJSON, CSV, and HTML map.
    """
    output_root = Path(output_root).resolve()
    merged_dir = output_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    batch_geojsons = sorted(output_root.glob("batch_*/batch_*.geojson"))
    if not batch_geojsons:
        raise FileNotFoundError(f"No batch GeoJSON outputs found under {output_root}")

    frames = [gpd.read_file(path) for path in batch_geojsons]
    merged_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

    if "region" in merged_gdf.columns and "id" in merged_gdf.columns:
        merged_gdf = merged_gdf.sort_values(by=["region", "id"]).reset_index(drop=True)

    merged_geojson = merged_dir / "usa_can_merged.geojson"
    merged_geojson.write_text(merged_gdf.to_json(), encoding="utf-8")
    html_path, csv_path = generate_output_artifacts(merged_geojson)

    print(f"Merged {len(batch_geojsons)} batch outputs into {merged_geojson}")
    print(f"Saved HTML map to {html_path}")
    print(f"Saved CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-batch CERF siting configuration files and optionally execute them."
    )
    parser.add_argument("--master-config", required=True, help="Path to the master USA/CAN configuration YAML.")
    parser.add_argument("--batch-config", required=True, help="Path to the batch definition YAML.")
    parser.add_argument("--output-root", required=True, help="Directory where derived configs will be written.")
    parser.add_argument("--execute", action="store_true", help="Run the generated batch configs after writing them.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel batch workers to use.")
    parser.add_argument(
        "--region-workers",
        type=int,
        default=1,
        help="Number of parallel region workers to use inside each batch.",
    )
    args = parser.parse_args()

    summaries = write_batch_configs(
        master_config_path=args.master_config,
        batch_config_path=args.batch_config,
        output_root=args.output_root,
    )

    if args.execute:
        execute_batches(
            summaries,
            max_workers=args.workers,
            region_workers=args.region_workers,
        )
        merge_outputs(Path(args.output_root))


if __name__ == "__main__":
    main()
