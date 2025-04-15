# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Adds offshore locations in the Baltic Sea to build regions accordingly.

Relevant Settings
-----------------

.. code:: yaml

    - topology settings

Inputs
------

- topology settings
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``networks/base_baltic.nc``: confer :ref:`base_baltic`

Description
-----------
"""

import pandas as pd
import pypsa
import geopy.distance
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import Point

from _helpers import configure_logging


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("base_network_baltic")
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # load topology, offshore buses and lines to be added
    topology_option = snakemake.config["offshore_topology"]

    if topology_option == 'default-dc':
        buses_offshore = pd.read_pickle(r"E:\Master_Thesis_Setup\pypsa-eur-offshore_adaptions\data\balticoffshoregridkit\default_topology_dc\offshore_buses.pkl")
        lines_offshore = pd.read_pickle(r"E:\Master_Thesis_Setup\pypsa-eur-offshore_adaptions\data\balticoffshoregridkit\default_topology_dc\offshore_lines.pkl")
    elif topology_option == 'meshed-mst-dc':
        buses_offshore = pd.read_pickle(r"data/balticoffshoregridkit/meshed_mst_topology_dc/offshore_buses.pkl")
        lines_offshore = pd.read_pickle(r"data/balticoffshoregridkit/meshed_mst_topology_dc/offshore_lines.pkl")
    else:
        buses_offshore = pd.read_pickle(snakemake.input.eg_buses)
        lines_offshore = pd.read_pickle(snakemake.input.eg_lines)


    # set offshore substations to False for countries with new offshore buses
    # exempt existing wind farms, e.g. keep North Sea Offshore Nodes
    countries_with_adapted_offshore_topology = buses_offshore.country.unique()

    buses = n.df('Bus')
    buses_sub_off = buses[buses.substation_off]

    for index, bus in buses_sub_off.iterrows():
        condition_baltic_offshore_bus = all([
            bus.country in countries_with_adapted_offshore_topology,
            bus.symbol != 'Wind farm',
            bus.x > 9.5])  # North Sea
        if condition_baltic_offshore_bus:
            n.buses.loc[index, 'substation_off'] = False

    # add offshore buses and lines, optionally links
    n.import_components_from_dataframe(buses_offshore, "Bus")
    n.import_components_from_dataframe(lines_offshore, "Line")

    # link POC connections to AC buses in PyPSA
    # only map to bus with the same carrier
    POC_buses = [bus_id for bus_id in buses_offshore.index if 'POC' in bus_id]
    buses['bus_id'] =buses.index
    buses_AC = buses[buses.carrier == 'AC']
    tree = KDTree(buses_AC[['x', 'y']])

    POC_link_counter = 0
    for bus in POC_buses:
        # Find nearest neighbor using KDTree
        coord_POC = buses_offshore.loc[bus, ['x', 'y']].values
        distance_degrees, index = tree.query([coord_POC])

        nearest_bus = buses_AC.index[index[0]]
        coordPyPSA = buses.loc[nearest_bus, ['x', 'y']].values

        # Convert degrees to kilometers
        distance_km = geopy.distance.geodesic(coord_POC, coordPyPSA).km

        # Add line or link to connect POC and PyPSA, high capacity to avoid bottleneck
        # Converter Link
        if topology_option in ['default-dc', 'meshed-mst-dc']:
            n.madd("Link",
                   names=['POC Converter Link ' + str(POC_link_counter)],
                   bus0=nearest_bus,  # AC bud PyPSA
                   bus1=bus,  # DC bus offshore
                   carrier='AC',
                   length=0,
                   p_nom=4000,
                   p_max_pu=0.9,
                   p_min_pu=-0.9,
                   capital_cost=0,
                   efficiency=1.0,
                   p_nom_extendable=True,
                   marginal_cost=0,
                   min_up_time=0,
                   min_down_time=0,
                   up_time_before=1,
                   down_time_before=0,
                   p_nom_opt=4000,
                   type='',
                   lifetime=float('inf'),
                   p_nom_max=float('inf'),
                   p_set=0.0,
                   marginal_cost_quadratic=0.0,
                   stand_by_cost=0.0,
                   terrain_factor=1.0,
                   committable=False,
                   start_up_cost=0.0,
                   shut_down_cost=0.0,
                   ramp_limit_up=float('nan'),
                   ramp_limit_down=float('nan'),
                   ramp_limit_start_up=1.0,
                   ramp_limit_shut_down=1.0
                   )
        # AC line
        else:
            n.madd("Line",
                   names=['POC Link ' + str(POC_link_counter)],
                   v_nom=380,
                   bus0=nearest_bus,
                   bus1=bus,
                   length=distance_km,
                   type='Al/St 240/40 4-bundle 380.0',
                   s_max_pu=0.7,
                   carrier='AC',
                   s_nom=4000,
                   under_construction=False,
                   underground=False,
                   )

        POC_link_counter += 1

        # Print result
        print(f"The nearest neighbor to {coord_POC} is {coordPyPSA} with a distance of {distance_km:.2f} km")

    # export network
    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output[0])



