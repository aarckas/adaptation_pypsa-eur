# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
1. Preprocesses the QGIS exports and stores in transmission_projects folder.
2. Removes unwanted lines/links/buses from base_network (e.g. existing Offshore Buses, Connectors,..)
3. Sets "substation_off" = False for Onshore Buses (except for North Sea Connections)

Inputs
------

- Base Network (base_network=resources("networks/base.nc"))
- Search Graph Tabel from QGIS Pipeline (search_graph="data/QGIS_Export/search_graph_table.csv")
- Bus (HUB/POC) Data from QGIS Pipeline (bus_data="data/QGIS_Export/bus_data.csv")

Outputs
-------

- New Lines (new_lines=directory("data/transmission_projects/manual/new_lines.csv"))
- New Links (new_links=directory("data/transmission_projects/manual/new_links.csv"))
- Bus (HUB/POC) data (bus_data=resources("QGIS_bus_data.csv"))
- Preprocessed Network (network=resources("networks/base_pre.nc"))
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import shapely
from _helpers import configure_logging, set_scenario_config
from pypsa.descriptors import nominal_attrs
from scipy import spatial
from shapely.geometry import LineString, Point
from pyproj import Transformer
import geopy.distance

logger = logging.getLogger(__name__)

import pandas as pd


def process_search_graph(search_graph: str, bus_data: str) -> None:
    """
    Processes the search_graph CSV file, matches it with bus_data, transforms coordinates, 
    and creates the required attributes/columns.

    Parameters
    ----------
    search_graph_path : str
        Path to the input search_graph CSV file.
    bus_data_path : str
        Path to the bus_data CSV file.
    output_path : str
        Path to save the processed CSV file.
    """
    
    # Initialize transformer for CRS transformation
    transformer = Transformer.from_crs("ESRI:102014", "EPSG:4326", always_xy=True)

    # Transform coordinates in bus_data
    bus_data["x_transformed"], bus_data["y_transformed"] = transformer.transform(
        bus_data["X"], bus_data["Y"]
    )

    # Match node_id and node_id_2 with bus_data
    search_graph = search_graph.rename(columns={"node_id": "bus0", "node_id_2": "bus1", "length_km": "length"})
    search_graph = search_graph.merge(
        bus_data[["node_id", "x_transformed", "y_transformed"]],
        left_on="bus0",
        right_on="node_id",
        how="left"
    ).rename(columns={"x_transformed": "x0", "y_transformed": "y0"})
    search_graph = search_graph.merge(
        bus_data[["node_id", "x_transformed", "y_transformed"]],
        left_on="bus1",
        right_on="node_id",
        how="left"
    ).rename(columns={"x_transformed": "x1", "y_transformed": "y1"})

    # Add fixed attributes
    search_graph["p_nom"] = 0
    search_graph["p_nom_max"] = 2000
    search_graph["build_year"] = 2030
    search_graph["project_status"] = "confirmed"
    search_graph["underground"] = True
    search_graph["voltage"] = 525

    required_columns = ["ij", "bus0", "bus1", "p_nom", "p_nom_max", "length", "voltage", "project_status", "build_year", "underground", "x0", "y0", "x1", "y1"]
    search_graph_df = search_graph[required_columns]
    
    search_graph_df = search_graph_df.set_index("ij")
    search_graph_raw = search_graph.set_index("ij")

    logger.info(f"Processed search_graph with bus data saved to Transmission Projects folder.")
    
    return search_graph_df, search_graph_raw


def adapt_offshore_topology(n: pypsa.Network, countries: list) -> None:
    """
    Sets the 'substation_off' attribute to False for buses from specified countries,
    except for those west of x=9.5 (North Sea).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing buses.
    countries_with_adapted_offshore_topology : list
        List of countries for which the offshore topology should be adapted.
    """
    # Get the buses DataFrame
    buses = n.df("Bus")

    # Filter buses where substation_off is True
    buses_sub_off = buses[buses.substation_off]

    # Iterate over buses and apply the condition
    for index, bus in buses_sub_off.iterrows():
        condition_baltic_offshore_bus = all([
            bus.country in countries,
            bus.x > 9.55,  # Exclude North Sea buses
            bus.y > 49, # Exclude Mediterranean buses
            bus.symbol != "Wind farm"  # Exclude wind farm buses
        ])
        if condition_baltic_offshore_bus:
            n.buses.loc[index, "substation_off"] = False

    logger.info("Adapted offshore topology for specified countries.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("preprocess_transmission_projects", configfiles="config/baltic/baltic_test.yaml")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    countries = snakemake.params.countries
    
    search_graph = pd.read_csv(snakemake.input.search_graph)
    bus_data = pd.read_csv(snakemake.input.bus_data)
    #output_path_links = snakemake.output.new_links
    #output_path_lines = snakemake.output.new_lines
    
    search_graph_df, search_graph_raw = process_search_graph(search_graph, bus_data)
    #process_search_graph(search_graph, bus_data, output_path_lines)
        
    bus_data.set_index("node_id", inplace=True)
    bus_data.drop(columns=["X", "Y"], inplace=True, errors='ignore')
    
     # Adapt offshore topology
    adapt_offshore_topology(n, countries)

    # Save the modified network
    n.export_to_netcdf(snakemake.output.network)
    search_graph_df.to_csv(snakemake.output.new_links)
    search_graph_raw.to_csv(snakemake.output.search_graph_data)
    bus_data.to_csv(snakemake.output.bus_data)

    