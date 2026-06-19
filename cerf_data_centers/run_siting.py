from tqdm import tqdm
import time
import logging
import geopandas as gpd
import pandas as pd
import click
import os
import sys
from pathlib import Path

import numpy as np

from .load_data import (
    load_region_raster,
    load_raster_array,
    collect_constraints,
    find_region_window,
    get_yaml,
    validate_raster_alignment,
)
from .calculate_locational_cost import calculate_locational_cost
from .determine_sites import get_region_suit_array, build_graph, site_based_on_siting_score
from .calculate_gravity_score import calc_gravity_array_from_distance, calc_gravity_score
from .configure_output import configure_output
from .postprocess_output import generate_output_artifacts
from .utils import convert_sqft_to_grid_cells, get_normalized_value


def _resolve_region_scenarios(
    region_name: str,
    region_dict: dict,
    global_scenarios: dict | None,
) -> list[tuple[str, dict]]:
    """
    Return the scenario configs to run for a region.

    Supports:
    - legacy single-scenario region configs
    - a top-level `scenarios` map shared by all regions
    - per-region `scenarios` overrides
    """
    if 'scenarios' in region_dict:
        scenario_dict = region_dict['scenarios']
    else:
        scenario_dict = global_scenarios

    if not scenario_dict:
        return [('default', region_dict)]

    base_region_dict = {k: v for k, v in region_dict.items() if k != 'scenarios'}
    resolved_scenarios: list[tuple[str, dict]] = []
    for scenario_name, scenario_config in scenario_dict.items():
        merged_config = {**base_region_dict, **scenario_config}
        merged_config['scenario_name'] = scenario_name
        resolved_scenarios.append((scenario_name, merged_config))

    if not resolved_scenarios:
        raise ValueError(f"Region {region_name} does not define any scenarios to run.")

    return resolved_scenarios


def _drop_overlapping_smaller_sites(output_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove smaller sites whose polygons overlap with already-kept larger sites.
    """
    if output_gdf.empty:
        return output_gdf

    sort_columns = ['campus_size_square_ft', 'data_center_it_power_mw', 'total_capex_cost']
    available_columns = [column for column in sort_columns if column in output_gdf.columns]
    ranked_gdf = output_gdf.sort_values(
        by=available_columns,
        ascending=[False] * len(available_columns),
        kind='mergesort',
    ).reset_index(drop=True)

    kept_indices: list[int] = []
    kept_geometries = []
    for idx, row in ranked_gdf.iterrows():
        geometry = row.geometry
        overlaps_larger_site = any(
            geometry.intersects(kept_geometry) and geometry.intersection(kept_geometry).area > 0
            for kept_geometry in kept_geometries
        )
        if overlaps_larger_site:
            continue
        kept_indices.append(idx)
        kept_geometries.append(geometry)

    return ranked_gdf.loc[kept_indices].reset_index(drop=True)


def run(config: str) -> gpd.GeoDataFrame:
    """
    Run the data center siting process based on the provided configuration file.

    Args:
        config (str): Path to the configuration file in YAML format.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing siting results (one or more sites per
        region) produced by the algorithm.

    Raises
    ------
    FileNotFoundError
        If the provided `config` path does not exist.
    KeyError, ValueError
        If required configuration keys or sections are missing or malformed.
    RuntimeError
        For unexpected runtime errors during raster processing or graph
        operations.

    Notes
    -----
    The function performs these high-level steps:
    - Load configuration and raster inputs (region, suitability, market).
    - Collect cost/constraint layers for suitable grid cells.
    - For each region in `expansion_plan`:
      - Extract suitable blocks and build a connected-component graph.
      - Compute locational costs and market-gravity scores per node.
      - Normalize scores, compute combined siting score, and select top
        `n_sites` according to the configuration.
    - Optionally write the result to `settings.output_file` if present.

    This function is intended to be callable programmatically (returns a
    GeoDataFrame) and is also wrapped by a CLI command `site` in this
    module which performs additional logging and CLI-based output options.
    """
    t0 = time.time()
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)

    logger.info("Application started")

    logger.info(f"Using configuration file: {config}")
    config_dict = get_yaml(config)

    settings_dict = config_dict['settings']
    constraint_dict = config_dict['constraints']
    market_dict = config_dict['market_gravity']
    expansion_dict = config_dict['expansion_plan']
    global_scenarios = config_dict.get('scenarios')
    selected_region_ids = settings_dict.get(
        'selected_region_ids',
        [region_cfg['region_id'] for region_cfg in expansion_dict.values()]
    )

    # get cost and constraint layers as lists
    raster_names = list(constraint_dict.keys())
    raster_paths = list(constraint_dict.values())
    all_raster_paths = [
        settings_dict['region_raster_path'],
        settings_dict['siting_raster_path'],
        market_dict['market_raster_path'],
        *raster_paths,
    ]

    logger.info("Validating raster alignment...")
    validate_raster_alignment(settings_dict['region_raster_path'], all_raster_paths[1:])

    logger.info(f"Computing minimum raster window for selected regions: {selected_region_ids}")
    region_window = find_region_window(settings_dict['region_raster_path'], selected_region_ids)
    logger.info(
        "Using raster window row_off=%s col_off=%s height=%s width=%s",
        int(region_window.row_off),
        int(region_window.col_off),
        int(region_window.height),
        int(region_window.width),
    )

    # load region array
    logger.info(f"Loading region raster from {settings_dict['region_raster_path']}")
    region_array, transform = load_region_raster(settings_dict['region_raster_path'], window=region_window)

    # load suitability array
    logger.info(f"Loading siting suitability raster from {settings_dict['siting_raster_path']}")
    suit_array = load_raster_array(raster_fn=settings_dict['siting_raster_path'], window=region_window)
    batch_region_mask = np.isin(region_array, selected_region_ids)
    batch_suit_array = np.where(batch_region_mask, suit_array, 0)

    # load cost and constraint arrays in suitable grid cells
    node_values = collect_constraints(
        batch_suit_array,
        transform,
        raster_paths,
        raster_names,
        logger,
        window=region_window,
    )

    # load market size and distance to market
    market_array = load_raster_array(raster_fn=market_dict['market_raster_path'], window=region_window)

    logger.info(f"Calculating market gravity multipliers...")
    # get the gravity multiplier array based on distance to market and market size
    gravity_multiplier_array = calc_gravity_array_from_distance(market_array, batch_suit_array)

    logger.info("Starting siting process...")
    output_gdf = gpd.GeoDataFrame(geometry=[], crs='ESRI:102003')
    for region_name in list(expansion_dict.keys()):

        logger.info(f"Processing region: {region_name}")

        region_dict = expansion_dict[region_name]
        region_scenarios = _resolve_region_scenarios(region_name, region_dict, global_scenarios)
        region_id = region_dict['region_id']
        region_suit_array = get_region_suit_array(region_array, batch_suit_array, region_id)

        for scenario_name, scenario_dict in region_scenarios:
            logger.info("Processing scenario %s for region %s", scenario_name, region_name)

            campus_size_square_ft = scenario_dict['campus_size_square_ft']
            cooling_water_intensity_gal_per_mwh = scenario_dict.get('cooling_water_intensity_gal_per_mwh', 460)
            cooling_water_consumption_fraction = scenario_dict.get('cooling_water_consumption_fraction', .8)
            facility_overhead_frac = scenario_dict.get('facility_overhead_frac', 0)
            equipment_capex_usd = scenario_dict['equipment_capital_expenditure_usd']
            building_capex_usd = scenario_dict['building_capital_expenditure_usd']
            assessed_real_property_frac = scenario_dict.get('assessed_real_property_frac', .16)
            assessed_personal_property_frac = scenario_dict.get('assessed_personal_property_frac', .8)
            data_center_it_power_mw = scenario_dict['data_center_it_power_mw']
            data_center_pue = scenario_dict['data_center_pue']
            interconnection_cost_usd_per_km = scenario_dict['interconnection_cost_usd_per_km']
            number_of_sites = scenario_dict['n_sites']
            min_block_size = convert_sqft_to_grid_cells(campus_size_square_ft)

            logger.info("Building siting graph for scenario %s...", scenario_name)
            G = build_graph(region_suit_array, min_block_size, raster_names, node_values)
            if G.number_of_nodes() == 0:
                logger.warning(
                    "No suitable connected components met the minimum block size for region %s scenario %s.",
                    region_name,
                    scenario_name,
                )
                continue

            logger.info("Calculating locational costs for scenario %s...", scenario_name)
            for node, attrs in tqdm(list(G.nodes(data=True))):
                G.nodes[node]['locational_cost'], node_parameter_dict = calculate_locational_cost(
                    campus_size_square_ft=campus_size_square_ft,
                    land_cost_usd_per_sqft=attrs['land_cost_per_sqft'],
                    elec_rate_usd_per_kwh=attrs['electricity_rate_per_kwh'],
                    personal_prop_tax_rate=attrs['personal_prop_tax_rate'],
                    real_property_tax_rate=attrs['real_property_tax_rate'],
                    sales_tax_rate=attrs['sales_tax_rate'],
                    interconnection_distance_km=attrs['interconnection_distance_km'],
                    mechanical_cool_fraction=attrs['mechanical_cool_fraction'],
                    water_cool_fraction=attrs['water_cool_fraction'],
                    equipment_capex_usd=equipment_capex_usd,
                    building_capex_usd=building_capex_usd,
                    interconnection_cost_usd_per_km=interconnection_cost_usd_per_km,
                    data_center_it_power_mw=data_center_it_power_mw,
                    data_center_pue=data_center_pue,
                    assessed_real_property_frac=assessed_real_property_frac,
                    assessed_personal_property_frac=assessed_personal_property_frac,
                    cooling_water_intensity_gal_per_mwh=cooling_water_intensity_gal_per_mwh,
                    cooling_water_consumption_fraction=cooling_water_consumption_fraction,
                    facility_overhead_frac=facility_overhead_frac,
                )
                G.nodes[node]['parameters'] = node_parameter_dict

            max_locational_cost = max(data['locational_cost'] for _, data in G.nodes(data=True))
            min_locational_cost = min(data['locational_cost'] for _, data in G.nodes(data=True))

            logger.info("Calculating market gravity score for scenario %s...", scenario_name)
            for node, attrs in tqdm(list(G.nodes(data=True))):
                G.nodes[node]['gravity_score'] = calc_gravity_score(
                    node,
                    gravity_multiplier_array,
                    data_center_it_power_mw,
                    alpha=0.5,
                )

            max_gravity_score = max(data['gravity_score'] for _, data in G.nodes(data=True))
            min_gravity_score = min(data['gravity_score'] for _, data in G.nodes(data=True))

            logger.info("Normalizing cost and market score for scenario %s...", scenario_name)
            cost_weight = settings_dict.get('cost_weight', 1.0)
            market_weight = settings_dict.get('market_weight', 0.0)
            for node, attrs in tqdm(list(G.nodes(data=True))):
                G.nodes[node]['normalized_locational_cost'] = get_normalized_value(
                    G,
                    attribute='locational_cost',
                    node=node,
                    max_value=max_locational_cost,
                    min_value=min_locational_cost,
                )
                G.nodes[node]['normalized_gravity_score'] = get_normalized_value(
                    G,
                    attribute='gravity_score',
                    node=node,
                    max_value=max_gravity_score,
                    min_value=min_gravity_score,
                )

                norm_loc_cost = G.nodes[node]['normalized_locational_cost']
                norm_gravity_score = G.nodes[node]['normalized_gravity_score']
                G.nodes[node]['total_weighted_siting_score'] = (
                    cost_weight * norm_loc_cost
                ) + (
                    market_weight * norm_gravity_score
                )

            result_list = site_based_on_siting_score(
                G,
                number_of_sites,
                min_block_size,
                region_name,
                transform,
            )

            logger.info(
                "Sited %s data center(s) in region %s for scenario %s.",
                len(result_list),
                region_name,
                scenario_name,
            )

            scenario_gdf = configure_output(
                result_list,
                region_id,
                output_prefix=f"{region_id}_{scenario_name}",
                extra_fields={'scenario': scenario_name},
            )
            output_gdf = pd.concat([output_gdf, scenario_gdf], ignore_index=True)

    if 'geometry' in output_gdf.columns:
        output_gdf = _drop_overlapping_smaller_sites(
            gpd.GeoDataFrame(output_gdf, geometry='geometry', crs='ESRI:102003')
        )

    if settings_dict.get('output_file'):
        logger.info(f"Saving output to {settings_dict['output_file']}")
        output_gdf.to_file(settings_dict['output_file'])
        html_path, csv_path = generate_output_artifacts(settings_dict['output_file'])
        logger.info(f"Saved HTML map to {html_path}")
        logger.info(f"Saved CSV to {csv_path}")

    logger.info(f"All regions processed in {round((time.time() - t0)/60, 2)} minutes.")

    return output_gdf


@click.group()
def cli():
    """Data Center Siting Tool - Identifies optimal locations for data centers based on various constraints and costs."""
    pass


@cli.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('-o', '--output', type=click.Path(dir_okay=False, path_type=Path),
              help='Path to save the output GeoJSON file (overrides config file setting)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging output')
@click.option('--log-file', type=click.Path(dir_okay=False, path_type=Path),
              help='Path to save the log file')
def site(config: Path, output: Path, verbose: bool, log_file: Path):
    """Run the data center siting process using the provided configuration file."""
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )

    logger = logging.getLogger(__name__)

    try:
        # Run the siting process
        output_gdf = run(str(config))

        # Save output if specified via CLI
        if output:
            logger.info(f"Saving output to {output}")
            output_gdf.to_file(output)
            html_path, csv_path = generate_output_artifacts(output)
            logger.info(f"Saved HTML map to {html_path}")
            logger.info(f"Saved CSV to {csv_path}")

    except Exception as e:
        logger.error(f"Error running data center siting: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
