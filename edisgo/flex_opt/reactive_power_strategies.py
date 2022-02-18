import logging
import math

import numpy as np
import pandas as pd
from edisgo.network.timeseries import _get_q_sign_generator, _get_q_sign_load
from edisgo.io import pypsa_io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('reactive_power.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

"""
    This function is used for calculating the q_u_curve based on the voltage 
    information of the bus where inverter is connected,
    for this purpose the droop for reactive power calculation is divided 
    in to 5 different reactive power calculation zones. 
    Returned value is the curve_q_set_in_percentage.
    Parameters for Q_U curve optimized for BEV are included, but can be changed
    by specifying the curve_parameter
    curve source: Netzstabilität mit Elektromobilität
"""

# splits df in mv and lv
def get_mv_and_lv_grid_df(bus_df, to_split_df):

    buses_to_split = bus_df.loc[to_split_df.bus.unique()]

    # This Dataframe merges all buses with charging points incl. v_nom
    to_split_with_v_nom_df = to_split_df.merge(
        buses_to_split.v_nom, how="left", left_on="bus", right_index=True)

    # df for all lv charging points
    lv_df = to_split_with_v_nom_df.loc[
        to_split_with_v_nom_df.v_nom < 1].bus

    # df for all mv charging points
    mv_df = to_split_with_v_nom_df.loc[
        to_split_with_v_nom_df.v_nom >= 1].bus

    return lv_df, mv_df

def q_u_curve(
        v_res,
        curve_parameter=None,
):
    if curve_parameter is None:
        curve_parameter = {
            "end_upper": 1.1,
            "start_upper": 1.05,
            "start_lower": 0.95,
            "end_lower": 0.9,
            "max_value": 1,
            "min_value": -1,
        }
    curve_q_set_in_percentage = np.select(
        [
            (v_res > curve_parameter["end_upper"]),
            (v_res <= curve_parameter["end_upper"])
            & (v_res >= curve_parameter["start_upper"]),
            (v_res < curve_parameter["start_upper"])
            & (v_res >= curve_parameter["start_lower"]),
            (v_res < curve_parameter["start_lower"])
            & (v_res >= curve_parameter["end_lower"]),
            (v_res < curve_parameter["end_lower"]),
        ],
        [
            curve_parameter["max_value"],
            curve_parameter["max_value"]
            - curve_parameter["max_value"]
            / (curve_parameter["start_upper"] - curve_parameter["end_upper"])
            * (v_res - curve_parameter["end_upper"]),
            0,
            curve_parameter["min_value"]
            * (v_res - curve_parameter["start_lower"])
            / (curve_parameter["end_lower"] - curve_parameter["start_lower"]),
            curve_parameter["min_value"],
        ],
    )

    # Changing variable in df for future calculation
    curve_q_set_in_percentage = pd.DataFrame(
        curve_q_set_in_percentage,
        index=v_res.index,
        columns=v_res.columns,
    )
    return curve_q_set_in_percentage


"""
    This function calculates the cos_phi_curve.
    Inputs are the 'netto_charging_capacity' and the 'used_charging_capacity'
    of the charging points. 
    Output is a float from 0 to 1 which represents 'curve_q_set_in_percentage'
"""


def cos_phi_p_curve(
        netto_charging_capacity,
        used_charging_capacity,
        curve_parameter=None,
):
    if curve_parameter is None:
        curve_parameter = {
            "end_upper": 1.0,
            "start_upper": 0.5,
            "max_value": 1,
            "min_value": 0,
        }

    # avoid div zero
    netto_charging_capacity = netto_charging_capacity.replace(0, 1)
    p_ac_in_percentage = used_charging_capacity / netto_charging_capacity

    curve_q_set_in_percentage = np.select(
        [
            (curve_parameter["start_upper"] > p_ac_in_percentage),
            (curve_parameter["start_upper"] <= p_ac_in_percentage),
        ],
        [
            curve_parameter["min_value"],
            curve_parameter["max_value"]
            - curve_parameter["max_value"]
            / (curve_parameter["start_upper"] - curve_parameter["end_upper"])
            * (p_ac_in_percentage - curve_parameter["end_upper"]),
        ],
    )

    return curve_q_set_in_percentage


"""
    3 reactive power strategies
     strategy  = str, fixed_cos_phi, cos_phi_p, q_u
     for_cp    = boolean, strategy used on charging points
     for_gen   = boolean, strategy used on generators
     max_trails= int, number of iteration steps
     timesteps = str, using the iteration on specific timesteps
     Threshold = int, digit requirement of voltage needed for iteration to end
     Q_U_STEP  = float, maximum voltage change in one step for q_u iteration        
"""


def reactive_power_strategies(edisgo_obj, strategy="fix_cos_phi", **kwargs):

    # Default 0.95 for LV and 0.9 for MV
    lv = kwargs.get("lv_cos_phi", 0.95)
    mv = kwargs.get("mv_cos_phi", 0.9)
    mv_cos_phi = math.tan(math.acos(mv))
    lv_cos_phi = math.tan(math.acos(lv))

    # bool for charging points and generator,
    # false skips reactive strategy for either of those
    for_cp = kwargs.get("charging_points", True)
    for_gen = kwargs.get("generator", True)

    # max trails or iteration of q_u default 10
    max_trails = kwargs.get("max_trails", 10)

    # THRESHOLD for ending iteration of q_u by matching v_res default 5
    THRESHOLD = kwargs.get("THRESHOLD", 5)

    # Q_U_STEP maximum change of v_res per iteration step of q_u default 0.4
    Q_U_STEP = kwargs.get("Q_U_STEP", 0.4)

    if for_cp:
        # Selecting all buses with an charging point
        cp_buses_df = edisgo_obj.topology.charging_points_df.copy()
        buses_for_cp = edisgo_obj.topology.buses_df

        cp_in_lv, cp_in_mv = get_mv_and_lv_grid_df(buses_for_cp, cp_buses_df)

        # replacing active power with p_nom for q_u
        cp_p_nom_per_timestep = edisgo_obj.timeseries.charging_points_active_power.copy()
        cp_transformed = edisgo_obj.topology.charging_points_df.T
        cp_p_nom_per_timestep.mask(cp_p_nom_per_timestep > 0,
                                   cp_transformed.loc["p_nom", :],
                                   inplace=True, axis=1)

    if for_gen:
        # Selecting all buses with an charging point
        gen_buses_df = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.type.isin(["solar", "wind"])]
        buses_for_gen = edisgo_obj.topology.buses_df

        gen_in_lv, gen_in_mv = get_mv_and_lv_grid_df(buses_for_gen, gen_buses_df)

        # replacing active power with p_nom for q_u
        gen_p_nom_per_timestep = edisgo_obj.timeseries.generators_active_power.loc[
                                 :, gen_buses_df.index].copy()
        gen_transformed = edisgo_obj.topology.generators_df.T
        gen_p_nom_per_timestep.mask(gen_p_nom_per_timestep > 0,
                                    gen_transformed.loc["p_nom", :],
                                    inplace=True, axis=1)

    # Getting df of all buses where reactive power is calculated for q_u and cos_phi_p
    if for_gen and for_cp:
        buses = cp_buses_df.bus.append(
            gen_buses_df.bus).unique()
        buses_to_calculate = edisgo_obj.topology.buses_df.loc[buses, :]

    elif for_gen:

        buses_to_calculate = buses_for_gen.unique()

    elif for_cp:

        buses_to_calculate = buses_for_cp.unique()

    else:

        logger.error(
            "Reactive_strategy has no target. Please specify where to use the reactive_strategy on with for_gen=True or for_cp=True")

    # fixed_cos sets cos_phi to a fixed value depending on the Grid Lvl
    if strategy == "fix_cos_phi":

        if for_cp:
            # Calculating reactive power for lv df
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
            :, cp_in_lv.index
            ] = edisgo_obj.timeseries.charging_points_active_power.loc[
                :, cp_in_lv.index
                ] * lv_cos_phi * _get_q_sign_load("inductive")

            # Calculating reactive power for mv df
            edisgo_obj.timeseries.charging_points_reactive_power.loc[
            :, cp_in_mv.index
            ] = edisgo_obj.timeseries.charging_points_active_power.loc[
                :, cp_in_mv.index
                ] * mv_cos_phi * _get_q_sign_load("inductive")

        # calculating reactive power for generators
        if for_gen:
            # Calculating reactive power for lv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
            :, gen_in_lv.index
            ] = edisgo_obj.timeseries.generators_active_power.loc[
                :, gen_in_lv.index
                ] * lv_cos_phi * _get_q_sign_generator("inductive")

            # Calculating reactive power for mv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
            :, gen_in_mv.index
            ] = edisgo_obj.timeseries.generators_active_power.loc[
                :, gen_in_mv.index
                ] * mv_cos_phi * _get_q_sign_generator("inductive")

    # Calculating the reactive power as a function of the grid voltage(U)
    if strategy == "q_u":

        # try with linear powerflow for convergence problems
        # Filling v_res df
        pypsa_network = edisgo_obj.to_pypsa()

        pypsa_network.lpf(edisgo_obj.timeseries.timeindex)
        pf_results = pypsa_network.pf(
            edisgo_obj.timeseries.timeindex, use_seed=True)

        timesteps_converged = pf_results["converged"][
            pf_results["converged"]["0"]].index
        pypsa_io.process_pfa_results(
            edisgo_obj, pypsa_network, timesteps_converged, dtype="float32")

        # Setting reactive power on not converged timesteps to 0

        timesteps_not_converged = edisgo_obj.timeseries.timeindex.copy()
        timesteps_converged_bool = edisgo_obj.timeseries.timeindex.isin(
            timesteps_converged)
        timesteps_not_converged = timesteps_not_converged[~timesteps_converged_bool]

        logger.info(f"Skipping q_u at {len(timesteps_not_converged)} timesteps.")
        if for_cp:
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
                timesteps_not_converged, :] = 0
        if for_gen:
            edisgo_obj.timeseries._generators_reactive_power.loc[
                timesteps_not_converged, :] = 0

        # iteration of the powerflow
        for n_trials in range(max_trails):

            # Getting the voltage of all buses with charging points
            v_res = edisgo_obj.results.v_res.loc[:, buses_to_calculate.index]

            if n_trials == 0:

                # calculation of maximum q compensation in % based on Q_U_curve
                q_fac = q_u_curve(v_res)

            else:

                q_fac_in_percentage = q_u_curve(v_res)
                q_fac = q_fac_old + Q_U_STEP * (
                        q_fac_in_percentage - q_fac_old)

            if for_cp:

                # Getting q_fac per cp
                groups = cp_buses_df.groupby("bus").groups
                cp_fac_df = pd.concat(
                    [pd.DataFrame({v: q_fac.loc[:, k] for v in groups[k]})
                     for k in groups.keys()], axis=1, )

                # Calculating reactive power for lv df
                edisgo_obj.timeseries._charging_points_reactive_power.loc[timesteps_converged,
                cp_in_lv.index] \
                    = cp_p_nom_per_timestep.loc[timesteps_converged, cp_in_lv.index] \
                      * lv_cos_phi \
                      * cp_fac_df.loc[timesteps_converged, cp_in_lv.index] \
                      * _get_q_sign_load("inductive")

                # Calculating reactive power for mv df
                edisgo_obj.timeseries._charging_points_reactive_power.loc[
                timesteps_converged, cp_in_mv.index
                ] = cp_p_nom_per_timestep.loc[timesteps_converged, cp_in_mv.index] \
                    * mv_cos_phi \
                    * cp_fac_df.loc[timesteps_converged, cp_in_mv.index] \
                    * _get_q_sign_load("inductive")

            # calculating reactive power for generators
            if for_gen:

                # Getting q_fac per generator
                groups = gen_buses_df.groupby("bus").groups
                gen_fac_df = pd.concat(
                    [pd.DataFrame({v: q_fac.loc[:, k] for v in groups[k]})
                     for k in groups.keys()], axis=1, )
                # Calculating reactive power for lv df
                edisgo_obj.timeseries._generators_reactive_power.loc[timesteps_converged,
                gen_in_lv.index] \
                    = gen_p_nom_per_timestep.loc[timesteps_converged, gen_in_lv.index] \
                      * lv_cos_phi \
                      * gen_fac_df.loc[timesteps_converged, gen_in_lv.index] \
                      * _get_q_sign_generator("inductive")

                # Calculating reactive power for mv df
                edisgo_obj.timeseries._generators_reactive_power.loc[
                timesteps_converged, gen_in_mv.index] \
                    = gen_p_nom_per_timestep.loc[timesteps_converged, gen_in_mv.index] \
                      * mv_cos_phi \
                      * gen_fac_df.loc[timesteps_converged, gen_in_mv.index] \
                      * _get_q_sign_generator("inductive")

            # TO-DO: Hier nur über convergierte Zeitschritte laufen lassen?
            # powerflow
            edisgo_obj.analyze(use_seed=True, timesteps=timesteps_converged)

            # getting last q_factor for comparison
            q_fac_old = q_fac.copy()

            # Zeitschritte checken
            if (v_res.round(THRESHOLD) == edisgo_obj.results.v_res[
                buses_to_calculate.index].round(THRESHOLD)).all().all():
                logger.info(
                    f"Stabilized Q(U) control after {n_trials} iterations.")
                break
            else:
                logger.debug(
                    f"Finished Q(U) control iteration {n_trials}.")

                if n_trials == max_trails:
                    logger.info(
                        "Halted Q(U) control after the maximum "
                        f"allowed iterations of {n_trials}.")
        """
        fertige Regelung für ladesäulen nicht sinnvoll, da untersuchte 
        Ladestrategien immer maximale Ladeleistung nutzen 
        (dadurch identisch zu fix cos phi)
        TO_DO: Convertierung zum Dataframe aus fct anpassen
        """
    if strategy == "cos_phi_p":

        if for_cp:
            netto_charging_capacity = cp_p_nom_per_timestep
            used_charging_capacity = edisgo_obj.timeseries.charging_points_active_power

            # calculation of q compensation in % based on cos_phi_p_curve
            cp_cos_phi_p = pd.DataFrame(cos_phi_p_curve(netto_charging_capacity, used_charging_capacity), index=used_charging_capacity.index, columns=used_charging_capacity.columns)
            # Calculating reactive power for lv df
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
            :, cp_in_lv.index] \
                = edisgo_obj.timeseries.charging_points_active_power.loc[
                    :, cp_in_lv.index] \
                    * lv_cos_phi \
                    * cp_cos_phi_p.loc[:, cp_in_lv.index] \
                    * _get_q_sign_load("inductive")

            # Calculating reactive power for mv df
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
                :, cp_in_mv.index] \
                = edisgo_obj.timeseries.charging_points_active_power.loc[
                :, cp_in_mv.index] \
                * mv_cos_phi \
                * cp_cos_phi_p.loc[:, cp_in_mv.index] \
                * _get_q_sign_load("inductive")

                # calculating reactive power for generators
        if for_gen:
            netto_charging_capacity = gen_p_nom_per_timestep
            used_charging_capacity = edisgo_obj.timeseries.generators_active_power

            # calculation of q compensation in % based on cos_phi_p_curve
            gen_cos_phi_p = pd.DataFrame(cos_phi_p_curve(netto_charging_capacity, used_charging_capacity), index=used_charging_capacity.index, columns=used_charging_capacity.columns)

            # Calculating reactive power for lv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
                :, gen_in_lv.index] \
                = edisgo_obj.timeseries.generators_active_power.loc[
                    :, gen_in_lv.index] \
                    * lv_cos_phi \
                    * gen_cos_phi_p.loc[:, gen_in_lv.index] \
                    * _get_q_sign_generator("inductive")

            # Calculating reactive power for mv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
                :, gen_in_mv.index] \
                = edisgo_obj.timeseries.generators_active_power.loc[
                    :, gen_in_mv.index] \
                    * mv_cos_phi \
                    * gen_cos_phi_p.loc[:, gen_in_mv.index] \
                    * _get_q_sign_generator("inductive")

    logger.info(f"Reactive charging strategy {strategy} completed.")
