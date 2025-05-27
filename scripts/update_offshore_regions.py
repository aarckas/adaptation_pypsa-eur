# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Updates the offshore_regions.geojson with the added transmission project buses as substation vor Voronoi partition.
"""
import logging

import pypsa
import geopandas as gpd
import pandas as pd
from _helpers import REGION_COLS, configure_logging, set_scenario_config
from base_network import build_bus_shapes, process_offshore_regions

#from shapely.geometry import Point
#from pypsa.geo import process_offshore_regions

logger = logging.getLogger(__name__)

def update_offshore_regions(
    n: pypsa.Network, 
    #admin_shapes: str,
    offshore_shapes: str, 
    countries: list[str],
) -> tuple[
    list[gpd.GeoDataFrame], gpd.GeoDataFrame
]:
    """
    Rebuilds only the offshore regions.geojson using updated buses.

    Parameters
    ----------
    n : pypsa.Network
        The full network (with added POC/HUB nodes).
    offshore_shapes_path : str
        Path to the base offshore shapefile.
    countries : list of str
        Countries to include in the update.

    Returns
    -------
    A tuple containing:
            - List of GeoDataFrames for each offshore region
            - Combined GeoDataFrame of all offshore shapes
    """

    offshore_shapes = gpd.read_file(offshore_shapes)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index("name")["geometry"]

    buses = n.buses[["x", "y", "country", "onshore_bus", "substation_lv", "substation_off"]].copy()
    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])
    buses = gpd.GeoDataFrame(buses, geometry="geometry", crs="EPSG:4326")

    # Only build offshore regions
    offshore_regions = process_offshore_regions(
        buses, 
        offshore_shapes, 
        countries, 
        n.crs.name)

    if offshore_regions:
        offshore_shapes = pd.concat(offshore_regions, ignore_index=True).set_crs(n.crs)
    else:
        offshore_shapes = gpd.GeoDataFrame(columns=["name", "geometry"], crs=n.crs).set_index("name")

    logger.info("Generated offshore regions for base network extended by transmission projects.")
    
    return offshore_regions, offshore_shapes


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("update_offshore_regions", configfiles="config/baltic/baltic_test.yaml")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    
    params = snakemake.params
    n = pypsa.Network(snakemake.input.network)
    admin_shapes = snakemake.input.admin_shapes
    offshore_shapes = snakemake.input.offshore_shapes
    countries = snakemake.params.countries


    offshore_regions, offshore_shapes = update_offshore_regions(
        n, 
        offshore_shapes, 
        countries
        )


    offshore_shapes.to_file(snakemake.output.regions_offshore_updated)

    #n.export_to_netcdf(snakemake.output.base_network)


