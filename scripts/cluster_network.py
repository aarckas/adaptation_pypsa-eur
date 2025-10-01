# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated
buses and transmission corridors.

Outputs
-------

- ``resources/regions_onshore_base_s_{clusters}.geojson``:

    .. image:: img/regions_onshore_base_s_X.png
        :scale: 33 %

- ``resources/regions_offshore_base_s_{clusters}.geojson``:

    .. image:: img/regions_offshore_base_s_X.png
        :scale: 33 %

- ``resources/busmap_base_s_{clusters}.csv``: Mapping of buses from ``networks/base.nc`` to ``networks/base_s_{clusters}.nc``;
- ``resources/linemap_base_s_{clusters}.csv``: Mapping of lines from ``networks/base.nc`` to ``networks/base_s_{clusters}.nc``;
- ``networks/base_s_{clusters}.nc``:

    .. image:: img/base_s_X.png
        :scale: 40  %

Description
-----------

.. note::

    **Is it possible to run the model without the** ``simplify_network`` **rule?**

        No, the network clustering methods in the PyPSA module
        `pypsa.clustering.spatial <https://github.com/PyPSA/PyPSA/blob/master/pypsa/clustering/spatial.py>`_
        do not work reliably with multiple voltage levels and transformers.

Exemplary unsolved network clustered to 512 nodes:

.. image:: img/base_s_512.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 256 nodes:

.. image:: img/base_s_256.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 128 nodes:

.. image:: img/base_s_128.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 37 nodes:

.. image:: img/base_s_37.png
    :scale: 40  %
    :align: center
"""

import logging
import warnings
from functools import reduce

import geopandas as gpd
import linopy
import numpy as np
import pandas as pd
import pypsa
import tqdm
import xarray as xr
from _helpers import configure_logging, set_scenario_config
from packaging.version import Version, parse
from pypsa.clustering.spatial import (
    busmap_by_greedy_modularity,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
)
from scipy.sparse.csgraph import connected_components
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import MultiPolygon, Polygon

PD_GE_2_2 = parse(pd.__version__) >= Version("2.2")

warnings.filterwarnings(action="ignore", category=UserWarning)
idx = pd.IndexSlice
logger = logging.getLogger(__name__)

GEO_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3035"
BUS_TOL = 500  # meters


def normed(x):
    return (x / x.sum()).fillna(0.0)


def weighting_for_country(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = normed(weights.reindex(df.index, fill_value=0))
    return (w * (100 / w.max())).clip(lower=1).astype(int)


def get_feature_data_for_hac(fn: str) -> pd.DataFrame:
    ds = xr.open_dataset(fn)
    feature_data = (
        pd.concat([ds[var].to_pandas() for var in ds.data_vars], axis=0).fillna(0.0).T
    )
    feature_data.columns = feature_data.columns.astype(str)
    return feature_data


def fix_country_assignment_for_hac(n: pypsa.Network) -> None:
    # overwrite country of nodes that are disconnected from their country-topology
    for country in n.buses.country.unique():
        m = n[n.buses.country == country].copy()

        _, labels = connected_components(m.adjacency_matrix(), directed=False)

        component = pd.Series(labels, index=m.buses.index)
        component_sizes = component.value_counts()

        if len(component_sizes) > 1:
            disconnected_bus = component[component == component_sizes.index[-1]].index[
                0
            ]

            neighbor_bus = n.lines.query(
                "bus0 == @disconnected_bus or bus1 == @disconnected_bus"
            ).iloc[0][["bus0", "bus1"]]
            new_country = list(set(n.buses.loc[neighbor_bus].country) - {country})[0]

            logger.info(
                f"overwriting country `{country}` of bus `{disconnected_bus}` "
                f"to new country `{new_country}`, because it is disconnected "
                "from its initial inter-country transmission grid."
            )
            n.buses.at[disconnected_bus, "country"] = new_country


def distribute_n_clusters_to_countries(
    n: pypsa.Network,
    n_clusters: int,
    cluster_weights: pd.Series,
    focus_weights: dict | None = None,
    solver_name: str = "scip",
) -> pd.Series:
    """
    Determine the number of clusters per country.
    """
    
    # --- 1) Build per-(country,sub_network) weights only over "base" groups ---
    # Anything with zero cluster_weight is treated as non-base (HUB/POC) and gets weight 0.
    L_raw = cluster_weights.groupby([n.buses.country, n.buses.sub_network]).sum()
    pos = L_raw > 0  # base sub-networks (carry load/weight)
    # Set non-base to 0, then normalize WITHIN each country so country-rows sum to 1 (if any base exists)
    L = (L_raw.where(pos, 0.0)
               .groupby(level=0, group_keys=False)
               .apply(lambda s: s / s.sum() if s.sum() > 0 else s))

    # Upper bounds (how many candidate buses exist per group)
    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]

    # --- 2) Optional: apply focus_weights as country shares (approximate target) ---
    if isinstance(focus_weights, dict) and len(focus_weights) > 0:
        total_focus = float(sum(focus_weights.values()))
        assert total_focus <= 1.0, "Sum of focus_weights must be ≤ 1."

        idx_country = L.index.get_level_values(0)
        L_adj = pd.Series(0.0, index=L.index, dtype=float)

        # Give each focused country a share 'w', distributed across its base sub-networks
        for c, w in focus_weights.items():
            sel = (idx_country == c)
            s = L.loc[sel]
            if s.sum() > 0:
                L_adj.loc[sel] = s / s.sum() * float(w)
            # if a country has no base groups, it simply can't receive weight

        # Distribute the remainder across non-focused countries, proportional to their base weights
        rem = 1.0 - total_focus
        other = ~idx_country.isin(focus_weights.keys())
        if rem > 0 and L.loc[other].sum() > 0:
            L_adj.loc[other] = L.loc[other] / L.loc[other].sum() * rem

        L = L_adj

    # Final sanity
    if not np.isclose(L.sum(), 1.0, rtol=1e-6):
        # Renormalize just in case of tiny numeric drift
        if L.sum() > 0:
            L = L / L.sum()
        else:
            raise ValueError("All weights are zero. Check cluster_weights/focus_weights inputs.")


    m = linopy.Model()
    
    #add dynamic constraint, so that added HUBs and POCs dont get own cluster at this point, but low-load buses from base subnetworks still get one
    #added this change bc otherwise it would allocate all nodes on DE for some reason 
    lower_bounds = (L > 0).astype(int)
    
    clusters = m.add_variables(
        lower=lower_bounds, upper=N, coords=[L.index], name="n", integer=True
    )
    m.add_constraints(clusters.sum() == n_clusters, name="tot")
    # leave out constant in objective (L * n_clusters) ** 2
    m.objective = (clusters * clusters - 2 * clusters * L * n_clusters).sum()
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").propagate = False
    elif solver_name not in ["scip", "cplex", "xpress", "copt", "mosek"]:
        logger.info(
            f"The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `scip`."
        )
        solver_name = "scip"
    m.solve(solver_name=solver_name)
    return m.solution["n"].to_series().astype(int)


def busmap_for_n_clusters(
    n: pypsa.Network,
    n_clusters_c: pd.Series,
    cluster_weights: pd.Series,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    **algorithm_kwds,
) -> pd.Series:
    if algorithm == "hac" and features is None:
        raise ValueError("For HAC clustering, features must be provided.")

    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + " "
        logger.debug(
            f"Determining busmap for country {prefix[:-1]} "
            f"from {len(x)} buses to {n_clusters_c[x.name]}."
        )
        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)
        weight = weighting_for_country(x, cluster_weights)

        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(
                n, weight, n_clusters_c[x.name], buses_i=x.index, **algorithm_kwds
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n,
                n_clusters_c[x.name],
                buses_i=x.index,
                feature=features.reindex(x.index, fill_value=0.0),
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n, n_clusters_c[x.name], buses_i=x.index
            )
        else:
            raise ValueError(
                f"`algorithm` must be one of 'kmeans' or 'hac' or 'modularity'. Is {algorithm}."
            )

    compat_kws = dict(include_groups=False) if PD_GE_2_2 else {}

    return (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, **compat_kws)
        .squeeze()
        .rename("busmap")
    )


def clustering_for_n_clusters(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
) -> pypsa.clustering.spatial.Clustering:
    if aggregation_strategies is None:
        aggregation_strategies = dict()

    line_strategies = aggregation_strategies.get("lines", dict())

    bus_strategies = aggregation_strategies.get("buses", dict())
    bus_strategies.setdefault("substation_lv", lambda x: bool(x.sum()))
    bus_strategies.setdefault("substation_off", lambda x: bool(x.sum()))
    
     # Prevent clustering for POC and HUB
    node_types = n.buses.get("node_type", pd.Series(index=n.buses.index))
    non_clustered_buses = node_types[node_types.isin(["POC", "HUB"])].index
    
     # Override their cluster mapping to their own name
    busmap.loc[non_clustered_buses] = non_clustered_buses

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=bus_strategies,
        line_strategies=line_strategies,
        custom_line_groupers=["build_year"],
    )
    
    return clustering


def cluster_regions(
    busmaps: tuple | list, regions: gpd.GeoDataFrame, with_country: bool = False
) -> gpd.GeoDataFrame:
    """
    Cluster regions based on busmaps and save the results to a file and to the
    network.

    Parameters
    ----------
        - busmaps (list) : A list of busmaps used for clustering.
        - regions (gpd.GeoDataFrame) : The regions to cluster.
        - with_country (bool) : Whether to keep country column.

    Returns
    -------
        None
    """
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])
    columns = ["name", "country", "geometry"] if with_country else ["name", "geometry"]
    regions = regions.reindex(columns=columns).set_index("name")
    regions_c = regions.dissolve(busmap)
    regions_c.index.name = "name"
    return regions_c.reset_index()


def busmap_for_admin_regions(
    n: pypsa.Network,
    admin_shapes: str,
    params: dict,
) -> pd.Series:
    """
    Create a busmap based on administrative regions using the NUTS3 shapefile.

    Parameters
    ----------
        - n (pypsa.Network) : The network to cluster.
        - admin_shapes (str) : The path to the administrative regions.
        - params (dict) : The parameters for clustering.

    Returns
    -------
        busmap (pd.Series): Busmap mapping each bus to an administrative region.
    """
    countries = params.countries
    admin_regions = gpd.read_file(admin_shapes)

    admin_levels = params.administrative
    level = admin_levels.get("level", 0)
    logger.info(f"Clustering at administrative level {level}.")

    # check if BA, MD, UA, or XK are in the network
    adm1_countries = ["BA", "MD", "UA", "XK"]
    buses = n.buses[["x", "y", "country"]].copy()

    # Find the intersection of adm1_countries and n.buses.country
    adm1_countries = list(set(adm1_countries).intersection(buses["country"].unique()))

    if adm1_countries:
        logger.info(
            f"Note that the following countries can only be clustered at a maximum administration level of 1: {adm1_countries}."
        )

    country_level = {
        k: v for k, v in admin_levels.items() if (k != "level") and (k in countries)
    }
    if country_level:
        country_level_list = "\n".join(
            [f"- {k}: level {v}" for k, v in country_level.items()]
        )
        logger.info(
            f"Setting individual administrative levels for:\n{country_level_list}"
        )

    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])
    buses = gpd.GeoDataFrame(buses, geometry="geometry", crs="EPSG:4326")
    buses["busmap"] = ""

    # Map based for each country
    logger.info("Mapping buses to administrative regions.")
    for country in tqdm.tqdm(buses["country"].unique()):
        buses_subset = buses.loc[buses["country"] == country]

        buses.loc[buses_subset.index, "busmap"] = gpd.sjoin_nearest(
            buses_subset.to_crs(epsg=3857),
            admin_regions.loc[admin_regions["country"] == country].to_crs(epsg=3857),
            how="left",
        )["admin"]

    return buses["busmap"]


def keep_largest_polygon(geometry: MultiPolygon) -> Polygon:
    """
    Checks for each MultiPolygon if it contains multiple Polygons and returns the one with the largest area.

    Parameters
    ----------
        geometry (MultiPolygon) : The MultiPolygon to check.

    Returns
    -------
        geometry (Polygon) : The Polygon with the largest area.
    """
    if isinstance(geometry, MultiPolygon):
        # Find the polygon with the largest area in the MultiPolygon
        largest_polygon = max(geometry.geoms, key=lambda poly: poly.area)

        return largest_polygon
    else:
        # If it's a Polygon, return it as is
        return geometry


def update_bus_coordinates(
    n: pypsa.Network,
    busmap: pd.Series,
    admin_shapes: str,
    geo_crs: str = GEO_CRS,
    distance_crs: str = DISTANCE_CRS,
    tol: float = BUS_TOL,
) -> None:
    """
    Updates the x, y coordinates of the buses in the original network based on the busmap and the administrative regions.
    Using the Pole of Inaccessibility (PoI) to determine internal points of the administrative regions.

    Parameters
    ----------
        - n (pypsa.Network) : The original network.
        - busmap (pd.Series) : The busmap mapping each bus to an administrative region.
        - admin_shapes (str) : The path to the administrative regions.
        - geo_crs (str) : The geographic coordinate reference system.
        - distance_crs (str) : The distance coordinate reference system.
        - tol (float) : The tolerance in meters for the PoI calculation.

    Returns
    -------
        None
    """
    logger.info("Updating x, y coordinates of buses based on administrative regions.")
    admin_regions = gpd.read_file(admin_shapes).set_index("admin")
    admin_regions["geometry"] = (
        admin_regions["geometry"]
        .to_crs(distance_crs)
        .apply(keep_largest_polygon)
        .to_crs(geo_crs)
    )
    admin_regions["poi"] = (
        admin_regions["geometry"]
        .to_crs(distance_crs)
        .apply(lambda polygon: polylabel(polygon, tolerance=tol / 2))
        .to_crs(geo_crs)
    )
    admin_regions["x"] = admin_regions["poi"].x
    admin_regions["y"] = admin_regions["poi"].y

    busmap_df = pd.DataFrame(busmap)
    busmap_df = pd.merge(
        busmap_df,
        admin_regions[["x", "y"]],
        left_on="busmap",
        right_index=True,
        how="left",
    )

    # Update x, y coordinates of original network
    n.buses["x"] = busmap_df["x"]
    n.buses["y"] = busmap_df["y"]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_network", clusters=52, configfiles="config/baltic/baltic_sec.yaml")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params
    mode = params.mode
    solver_name = snakemake.config["solving"]["solver"]["name"]

    n = pypsa.Network(snakemake.input.network)
    buses_prev, lines_prev, links_prev = len(n.buses), len(n.lines), len(n.links)

    load = (
        xr.open_dataarray(snakemake.input.load)
        .mean(dim="time")
        .to_pandas()
        .reindex(n.buses.index, fill_value=0.0)
    )

    if snakemake.wildcards.clusters == "all":
        n_clusters = len(n.buses)
    elif mode == "administrative":
        n_clusters = np.nan
    else:
        n_clusters = int(snakemake.wildcards.clusters) #+ len(n.buses.query("node_type == 'POC' or node_type == 'HUB'")) 

    if n_clusters == len(n.buses):
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(n, busmap, linemap)
    else:
        Nyears = n.snapshot_weightings.objective.sum() / 8760

        if mode == "administrative":
            busmap = busmap_for_admin_regions(
                n,
                snakemake.input.admin_shapes,
                params,
            )
            # Update x, y coordinates, ensuring that bus locations are inside the administrative region
            update_bus_coordinates(
                n,
                busmap,
                snakemake.input.admin_shapes,
            )

        elif mode == "custom_busmap":
            custom_busmap = pd.read_csv(
                snakemake.input.custom_busmap, index_col=0
            ).squeeze()
            custom_busmap.index = custom_busmap.index.astype(str)
            logger.info(f"Imported custom busmap from {snakemake.input.custom_busmap}")
            busmap = custom_busmap
        else:
            algorithm = params.cluster_network["algorithm"]
            features = None
            if algorithm == "hac":
                features = get_feature_data_for_hac(snakemake.input.hac_features)
                fix_country_assignment_for_hac(n)

            n.determine_network_topology()

            n_clusters_c = distribute_n_clusters_to_countries(
                n,
                n_clusters,
                load,
                focus_weights=params.focus_weights,
                solver_name=solver_name,
            )

            busmap = busmap_for_n_clusters(
                n,
                n_clusters_c,
                cluster_weights=load,
                algorithm=algorithm,
                features=features,
            )

        clustering = clustering_for_n_clusters(
            n,
            busmap,
            aggregation_strategies=params.aggregation_strategies,
        )
        
        #Restore link lengths from QGIS input that have been altered during the clustering workflow.
        if params.cluster_network["keep_refined_length"]:
            logger.info("Restoring refined link/line lengths for custom links.")
            custom_links = n.links[n.links.bus0.str.contains("HUB") | n.links.bus1.str.contains("HUB")].index
            custom_length = n.links.length
            clustering.n.links.loc[custom_links, "length"] = custom_length.loc[custom_links]
        else:
            logger.info("Using link/line lengths estimated by PyPSA-Eur workflow.")    

    nc = clustering.n

    for attr in ["busmap", "linemap"]:
        getattr(clustering, attr).to_csv(snakemake.output[attr])

    # nc.shapes = n.shapes.copy()
    for which in ["regions_onshore", "regions_offshore"]:
        regions = gpd.read_file(snakemake.input[which])
        clustered_regions = cluster_regions((clustering.busmap,), regions)
        clustered_regions.to_file(snakemake.output[which])
        # append_bus_shapes(nc, clustered_regions, type=which.split("_")[1])

    nc.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    nc.export_to_netcdf(snakemake.output.network)

    logger.info(
        f"Clustered network:\n"
        f"Buses: {buses_prev} to {len(nc.buses)}\n"
        f"Lines: {lines_prev} to {len(nc.lines)}\n"
        f"Links: {links_prev} to {len(nc.links)}"
    )
