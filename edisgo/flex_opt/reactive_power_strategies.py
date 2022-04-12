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


def compare_with_fix_cos_df(active_power_df, timesteps_converged, q_u_df, to_insert_df, cos_phi,
                            cap_ind_fac):
    """
        Check if Q in Dataframe is smaller than Q(fix_cos_phi)
        If not replace value with Q(fix_cos_phi)

        Parameters
        ----------
        active_power_df : df
            df for active power, either load or gen
        q_u_df : df
            q_u_df to compare with fix_cos_df
        to_insert_df: df
            Default: used for column selection on q_u_df
        cos_phi: float
            Default: cos_phi factor
        cap_ind_fac: int
            vlaue given by _get_q_sign_load or _get_q_sign_generator
    """

    fix_cos_df = active_power_df.loc[
                     timesteps_converged, to_insert_df.index] * cos_phi \
                 * cap_ind_fac

    # change column order to prevent later mismatching
    columnsTitles = list(fix_cos_df.columns)
    q_u_df = q_u_df.reindex(columns=columnsTitles, )
    q_u_df.mask(
        ((q_u_df < 0) & (abs(q_u_df) > abs(fix_cos_df))), abs(fix_cos_df) * -1,
        inplace=True, axis=1)
    q_u_df.mask(
        ((q_u_df >= 0) & (abs(q_u_df) > abs(fix_cos_df))), abs(fix_cos_df),
        inplace=True, axis=1)

    return q_u_df


# changes columns in dataframe from buses to cp or gen
def group_q_u_per_df(buses_df, q_fac):
    """
            Used for grouping q_u_factor per generator or load
            if buses_df is empty returns q_fac with zeros to prevent errors

            Parameters
            ----------
            buses_df : df
               Dataframe to group by
            q_fac : df
               Dataframe to group
        """
    groups = buses_df.groupby("bus").groups

    q_fac_per_df = {}

    for bus, entries in groups.items():
        for entry in entries:
            q_fac_per_df.update({entry: q_fac.loc[:, bus]})

    q_fac_sorted = pd.DataFrame.from_dict(q_fac_per_df)

    if len(buses_df) == 0:
        q_fac_sorted = pd.DataFrame(0, index=q_fac.index, columns=q_fac.columns)

    return q_fac_sorted

# splits df in mv and lv
def get_mv_and_lv_grid_df(bus_df, to_split_df):

    buses_to_split = bus_df.loc[to_split_df.bus.unique()]
    # This Dataframe merges all buses with charging points incl. v_nom
    to_split_with_v_nom_df = to_split_df.merge(
        buses_to_split.v_nom, how="left", left_on="bus", right_index=True)
    # df for all lv charging points
    lv_df = to_split_with_v_nom_df.loc[
        to_split_with_v_nom_df.v_nom < 1]
    # df for all mv charging points
    mv_df = to_split_with_v_nom_df.loc[
        to_split_with_v_nom_df.v_nom >= 1]

    return lv_df, mv_df

# Q_U_Curve for reactive power calculation
# Input V_res: DataFrame with voltage in p.u.
# Output: Dataframe with q_u value between 1 and -1.
# Curve for loads, to use with generators multiply df with -1
def q_u_curve(
        v_res,
        grid_lvl="lv",
):
    """
        Output: Dataframe with q_u value between 1 and -1.
        Curve for loads, to use with generators multiply df with -1
        Parameters
        ----------
        v_res : Dataframe
            Voltage Dataframe for Q(U)-curve. Units in p.u.
        grid_lvl : str
            Default: "lv"
            other "mv". Different curves for lv and mv to keep in transformer deviations of the grid
    """
    if grid_lvl== "lv":
        curve_parameter={"end_upper": 1.08,
                         "start_upper": 1.05,
                         "start_lower": 0.96,
                         "end_lower": 0.93,
                         "max_value": 1,
                         "min_value": -1}
    if grid_lvl== "mv":
        curve_parameter={"end_upper": 1.055,
                         "start_upper": 1.035,
                         "start_lower": 0.995,
                         "end_lower": 0.975,
                         "max_value": 1,
                         "min_value": -1}
    # If NaN values fill with 1
    v_res.fillna(1)

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
            (curve_parameter["start_upper"] > p_ac_in_percentage).fillna(False),
            (curve_parameter["start_upper"] <= p_ac_in_percentage).fillna(False),
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

    """
    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        The reactive power strategy. Default "fix_cos_phi".
        Other strategies "q_u" and "cos_phi_p"
    kwargs :
        Kwargs may contain any further attributes you want to specify.
        lv_cos_phi : float
            Default: 0.95
         mv_cos_phi : float
            Default: 0.90
        for_cp : bool
            Default: True
            Specifies if reactive strategy is used on charging points
        for_gen : bool
            Default: True
            Specifies if reactive strategy is used on generators
        max_trails : int
            Default: 10
            Max iteration steps for q_u
        THRESHOLD : int
            Default: 4
            The number of digits after the decimal point where v_res
            needs to match to end the iteration
        Q_U_STEP : float
            Default: 0.4
            Steps of change between the iterations for q_u
    """
    # Default 0.95 for LV and 0.9 for MV
    lv = kwargs.get("lv_cos_phi", 0.95)
    mv = kwargs.get("mv_cos_phi", 0.9)
    mv_cos_phi = math.tan(math.acos(mv))
    lv_cos_phi = math.tan(math.acos(lv))

    for_cp = kwargs.get("charging_points", True)
    for_gen = kwargs.get("generator", True)

    max_trails = kwargs.get("max_trails", 10)

    THRESHOLD = kwargs.get("THRESHOLD", 4)

    Q_U_STEP = kwargs.get("Q_U_STEP", 0.4)

    if for_cp:
        # Selecting all buses with an charging point
        cp_buses_df = edisgo_obj.topology.charging_points_df.copy()
        buses_for_cp = edisgo_obj.topology.buses_df

        # replacing active power with p_nom for q_u
        cp_p_nom_per_timestep = edisgo_obj.timeseries.charging_points_active_power.copy()
        cp_p_nom_per_timestep.mask(cp_p_nom_per_timestep > 0,
                                   edisgo_obj.topology.charging_points_df.loc[:, "p_nom"],
                                   inplace=True, axis=1)

        cp_in_lv, cp_in_mv = get_mv_and_lv_grid_df(buses_for_cp, cp_buses_df)
    if for_gen:
        # Selecting all buses with an charging point
        gen_buses_df = edisgo_obj.topology.generators_df.loc[
            edisgo_obj.topology.generators_df.type.isin(["solar", "wind"])]
        buses_for_gen = edisgo_obj.topology.buses_df

        gen_in_lv, gen_in_mv = get_mv_and_lv_grid_df(buses_for_gen, gen_buses_df)

        # replacing active power with p_nom for q_u
        gen_p_nom_per_timestep = edisgo_obj.timeseries.generators_active_power.loc[
                                 :, gen_buses_df.index].copy()
        gen_p_nom_per_timestep.mask(gen_p_nom_per_timestep > 0,
                                    edisgo_obj.topology.generators_df.loc[:, "p_nom"],
                                    inplace=True, axis=1)

    # Getting df of all buses where reactive power is calculated for q_u and cos_phi_p
    if for_gen and for_cp:

        lv_buses = cp_in_lv.bus.append(
            gen_in_lv.bus).unique()
        lv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[lv_buses, :]
        mv_buses = cp_in_mv.bus.append(
            gen_in_mv.bus).unique()
        mv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[mv_buses, :]
    elif ~for_cp:

        lv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[gen_in_lv.bus, :]
        mv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[gen_in_mv.bus, :]
    elif ~for_gen:

        lv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[cp_in_lv.bus, :]
        mv_buses_to_calculate = edisgo_obj.topology.buses_df.loc[cp_in_mv.bus, :]

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
                ] * lv_cos_phi * _get_q_sign_load("capacitive")

            # Calculating reactive power for mv df
            edisgo_obj.timeseries.charging_points_reactive_power.loc[
            :, cp_in_mv.index
            ] = edisgo_obj.timeseries.charging_points_active_power.loc[
                :, cp_in_mv.index
                ] * lv_cos_phi * _get_q_sign_load("inductive") # capacitive Q not allowed in MV, see VDE 4110

        # calculating reactive power for generators
        if for_gen:
            # Calculating reactive power for lv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
            :, gen_in_lv.index
            ] = edisgo_obj.timeseries.generators_active_power.loc[
                :, gen_in_lv.index
                ] * _get_q_sign_generator("inductive") # inductive so Q lowers the voltage

            # cos_phi is 0.95 between 3.68 and 13.68 kVA, 0.9 above 13.68 and 1 lower tahn 3.68
            # checks if p_nom is between 3.68 and 13.68 for cos_phi 0.95
            edisgo_obj.timeseries._generators_reactive_power.mask(
                ((gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 > 3.68)
                & ((gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 <= 13.68),
                edisgo_obj.timeseries._generators_reactive_power * lv_cos_phi,
                inplace=True, axis=1)
            # checks if p_nom is over 13.68 for cos_phi 0.90
            edisgo_obj.timeseries._generators_reactive_power.mask(
                (gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 > 13.68,
                edisgo_obj.timeseries._generators_reactive_power * mv_cos_phi,
                inplace=True, axis=1)

            # Calculating reactive power for mv df
            # cos_phi is set to 0.9 without checking, cause most PV are larger than 13.68 kVA
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

        # Getting df for all buses to calc for THRESHOLD check
        buses_to_calculate = lv_buses_to_calculate.index.append(mv_buses_to_calculate.index).unique()

        # iteration of the powerflow
        for n_trials in range(max_trails):

            # Getting v_res for all buses to calc for THRESHOLD check
            v_res = edisgo_obj.results.v_res.loc[:, buses_to_calculate]

            if n_trials == 0:

                # calculation of maximum q compensation in % based on Q_U_curve for mv and lv
                lv_q_fac = q_u_curve(edisgo_obj.results.v_res.loc[:,
                                     lv_buses_to_calculate.index],
                                     grid_lvl="lv")
                mv_q_fac = q_u_curve(edisgo_obj.results.v_res.loc[:,
                                     mv_buses_to_calculate.index],
                                     grid_lvl="mv")
            else:
                # calculation of maximum q compensation in % based on Q_U_curve for mv and lv
                # and a step response for the Q(U)-curve to prevent oscillation during the iteration
                lv_q_fac = q_u_curve(edisgo_obj.results.v_res.loc[:,
                                     lv_buses_to_calculate.index],
                                     grid_lvl="lv")
                mv_q_fac = q_u_curve(edisgo_obj.results.v_res.loc[:,
                                     mv_buses_to_calculate.index],
                                     grid_lvl="mv")
                lv_q_fac = lv_q_fac_old + Q_U_STEP * (
                        lv_q_fac - lv_q_fac_old)
                mv_q_fac = mv_q_fac_old + Q_U_STEP * (
                        mv_q_fac - mv_q_fac_old)

            # saving q_fac for next iteration step
            lv_q_fac_old = lv_q_fac.copy()
            mv_q_fac_old = mv_q_fac.copy()

            ################
            #### Loads #####
            ################
            if for_cp:

                # Getting q_fac per cp
                lv_q_u_per_cp_df = group_q_u_per_df(cp_in_lv, lv_q_fac)
                mv_q_u_per_cp_df = group_q_u_per_df(cp_in_mv, mv_q_fac)

                # Calculating reactive power for lv df
                cp_lv_result_df = cp_p_nom_per_timestep.loc[
                    timesteps_converged, cp_in_lv.index] \
                    * lv_cos_phi \
                    * lv_q_u_per_cp_df.loc[timesteps_converged, cp_in_lv.index]

                # Check if reactive_power is lower than fix cos
                cp_lv_result_df = compare_with_fix_cos_df(edisgo_obj.timeseries.
                                charging_points_active_power.loc[timesteps_converged,
                                cp_in_lv.index], timesteps_converged,
                                cp_lv_result_df, cp_in_lv, lv_cos_phi,
                                _get_q_sign_load("capacitive"))

                # Calculating reactive power for mv df
                # cos_phi is 0.95 cause VDE 4110 suggests it
                cp_mv_result_df = cp_p_nom_per_timestep.loc[timesteps_converged, cp_in_mv.index] \
                * lv_cos_phi \
                * mv_q_u_per_cp_df.loc[timesteps_converged, cp_in_mv.index]

                # Check if reactive_power is lower than fix cos
                cp_mv_result_df = compare_with_fix_cos_df(edisgo_obj.timeseries.
                                charging_points_active_power.loc[timesteps_converged,
                                cp_in_mv.index], timesteps_converged,
                                cp_mv_result_df, cp_in_mv, mv_cos_phi,
                                _get_q_sign_load("capacitive"))

                # Write result
                edisgo_obj.timeseries._charging_points_reactive_power.loc[
                    timesteps_converged, cp_in_lv.index] = cp_lv_result_df
                edisgo_obj.timeseries._charging_points_reactive_power.loc[
                    timesteps_converged, cp_in_mv.index] = cp_mv_result_df

            ################
            ## Generators ##
            ################
            if for_gen:

                # Getting q_fac per generator
                lv_q_u_per_gen_df = group_q_u_per_df(gen_in_lv, lv_q_fac)
                mv_q_u_per_gen_df = group_q_u_per_df(gen_in_mv, mv_q_fac)

                # Calculating reactive power for lv df
                gen_lv_result_df = gen_p_nom_per_timestep.loc[
                      timesteps_converged, gen_in_lv.index] \
                      * lv_q_u_per_gen_df.loc[timesteps_converged, gen_in_lv.index] \
                      * -1

                # checks if p_nom is between 3.68 and 13.68 for cos_phi 0.95
                gen_lv_result_df.mask(
                    ((gen_p_nom_per_timestep.loc[timesteps_converged,
                      gen_in_lv.index]) * 1000 > 3.68)
                    & ((gen_p_nom_per_timestep.loc[timesteps_converged,
                        gen_in_lv.index]) * 1000 <= 13.68),
                    gen_lv_result_df * lv_cos_phi, inplace=True, axis=1)
                # checks if p_nom is over 13.68 for cos_phi 0.90
                gen_lv_result_df.mask(
                    (gen_p_nom_per_timestep.loc[timesteps_converged,
                     gen_in_lv.index]) * 1000 > 13.68,
                    gen_lv_result_df * mv_cos_phi, inplace=True, axis=1)

                # Check if reactive_power is lower than fix cos
                gen_lv_result_df = compare_with_fix_cos_df(edisgo_obj.timeseries.
                                generators_active_power.loc[timesteps_converged,
                                gen_in_lv.index], timesteps_converged,
                                gen_lv_result_df, gen_in_lv, mv_cos_phi,
                                _get_q_sign_generator("inductive"))

                # Calculating reactive power for mv df
                gen_mv_result_df = gen_p_nom_per_timestep.loc[
                      timesteps_converged, gen_in_mv.index] \
                      * lv_cos_phi \
                      * mv_q_u_per_gen_df.loc[timesteps_converged, gen_in_mv.index] \
                      * -1

                # Check if reactive_power is lower than fix cos
                gen_mv_result_df = compare_with_fix_cos_df(edisgo_obj.timeseries.
                                generators_active_power.loc[timesteps_converged,
                                gen_in_mv.index], timesteps_converged,
                                gen_mv_result_df, gen_in_mv, lv_cos_phi,
                                _get_q_sign_generator("inductive"))

                # Write result
                edisgo_obj.timeseries._generators_reactive_power.loc[
                    timesteps_converged, gen_in_lv.index] = gen_lv_result_df
                edisgo_obj.timeseries._generators_reactive_power.loc[
                    timesteps_converged, gen_in_mv.index] = gen_mv_result_df

            # powerflow
            edisgo_obj.analyze(use_seed=True, timesteps=timesteps_converged)

            # end iteration if changes to volatage are small
            if (v_res.round(THRESHOLD) == edisgo_obj.results.v_res[
                buses_to_calculate].round(THRESHOLD)).all().all():
                logger.info(
                    f"Stabilized Q(U) control after {n_trials} iterations.")
                break
            else:
                logger.info(
                    f"Finished Q(U) control iteration {n_trials}.")

                if n_trials == max_trails:
                    logger.info(
                        "Halted Q(U) control after the maximum "
                        f"allowed iterations of {n_trials}.")


    if strategy == "cos_phi_p":

        if for_cp:

            # calculation of q compensation in % based on cos_phi_p_curve
            cp_cos_phi_p = pd.DataFrame(cos_phi_p_curve(cp_p_nom_per_timestep,
                                        edisgo_obj.timeseries.charging_points_active_power),
                                        index=edisgo_obj.timeseries.charging_points_active_power.index,
                                        columns=edisgo_obj.timeseries.charging_points_active_power.columns)
            # Calculating reactive power for lv df
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
            :, cp_in_lv.index] \
                = edisgo_obj.timeseries.charging_points_active_power.loc[
                    :, cp_in_lv.index] \
                    * lv_cos_phi \
                    * cp_cos_phi_p.loc[:, cp_in_lv.index] \
                    * _get_q_sign_load("capacitive")

            # Calculating reactive power for mv df
            # cos_phi is 0.95 cause VDE 4110 suggests it
            # inductive cause capacitive itsnt allowed in MV
            edisgo_obj.timeseries._charging_points_reactive_power.loc[
                :, cp_in_mv.index] \
                = edisgo_obj.timeseries.charging_points_active_power.loc[
                :, cp_in_mv.index] \
                * lv_cos_phi \
                * cp_cos_phi_p.loc[:, cp_in_mv.index] \
                * _get_q_sign_load("inductive")

                # calculating reactive power for generators
        if for_gen:

            # calculation of q compensation in % based on cos_phi_p_curve
            gen_cos_phi_p = pd.DataFrame(cos_phi_p_curve(gen_p_nom_per_timestep,
                                         edisgo_obj.timeseries.generators_active_power),
                                         index=edisgo_obj.timeseries.generators_active_power.index,
                                         columns=edisgo_obj.timeseries.generators_active_power.columns)

            # Calculating reactive power for lv df
            edisgo_obj.timeseries._generators_reactive_power.loc[
                :, gen_in_lv.index] \
                = edisgo_obj.timeseries.generators_active_power.loc[
                    :, gen_in_lv.index] \
                    * gen_cos_phi_p.loc[:, gen_in_lv.index] \
                    * _get_q_sign_generator("inductive")

            # checks if p_nom is between 3.68 and 13.68 for cos_phi 0.95
            edisgo_obj.timeseries._generators_reactive_power.mask(
                ((gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 > 3.68)
                & ((gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 <= 13.68),
                edisgo_obj.timeseries._generators_reactive_power * lv_cos_phi,
                inplace=True, axis=1)
            # checks if p_nom is over 13.68 for cos_phi 0.90
            edisgo_obj.timeseries._generators_reactive_power.mask(
                (gen_p_nom_per_timestep.loc[:, gen_in_lv.index])*1000 > 13.68,
                edisgo_obj.timeseries._generators_reactive_power * mv_cos_phi,
                inplace=True, axis=1)

            # Calculating reactive power for mv df
            # cos_phi is 0.95 cause VDE 4110 suggests it
            edisgo_obj.timeseries._generators_reactive_power.loc[
                :, gen_in_mv.index] \
                = edisgo_obj.timeseries.generators_active_power.loc[
                    :, gen_in_mv.index] \
                    * lv_cos_phi \
                    * gen_cos_phi_p.loc[:, gen_in_mv.index] \
                    * _get_q_sign_generator("inductive")

    logger.info(f"Reactive charging strategy {strategy} completed.")
